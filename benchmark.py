"""Replay D435 frames, timing detector, initial pose registration and tracking."""
import argparse
import ctypes as C
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from can_detector import detect_can
from mesh import load_scan
from nvml_probe import Probe

RUNTIME_ROOT = Path("/fp")
ROOT = Path(__file__).resolve().parent
SCAN = ROOT / "examples/soda-can/coke_can.glb"
HYPOTHESES = 252

ORIGINAL_FP_REPO = "https://github.com/NVlabs/FoundationPose"
FP_INFER_LIB_REPO = "https://github.com/nvidia-isaac/foundation-pose-inference-library"
RUNTIMES = {
    "pooled-3": dict(label="FP Infer Lib Shared Workspace · 3 refinements",
                     repo=FP_INFER_LIB_REPO, modified=True),
    "pooled": dict(label="FP Infer Lib Shared Workspace · 5 refinements",
                   repo=FP_INFER_LIB_REPO, modified=True),
    "native": dict(label="FP Infer Lib", repo=FP_INFER_LIB_REPO, modified=False),
    "original": dict(label="Original FP", repo=ORIGINAL_FP_REPO, modified=False),
}


class Native:
    def __init__(self, count, mesh_path, mode, iterations):
        from foundation_pose_nvidia import Precision, RuntimeConfig
        from foundation_pose_nvidia.bindings import CreateOptions, load_library

        build = "native" if mode == "native" else "pooled"
        library = load_library(str(ROOT / f".build/{build}/libfoundation_pose_nvidia.so"),
                               print_build_info=False)
        library.bind_group_api(count)
        self.lib = library.cdll
        self.count = count
        self.active_count = count
        self.iterations = iterations
        config = RuntimeConfig(
            n_hypotheses=HYPOTHESES, n_refine_iters=iterations, n_track_iters=2,
            max_image_width=640, max_image_height=480,
            tensorrt_precision=Precision.FP16, capture_cuda_graph=False,
        ).to_ctypes(library)
        options = (CreateOptions * count)()
        engine_cache = os.fsencode(os.environ.get("FP_POSE_ENGINE_CACHE", "/fp/runtime/engines/fp16"))
        for item in options:
            item.cad_path = os.fsencode(mesh_path)
            item.device_id = 0
            item.mesh_unit_scale = 1.0
            item.refine_model_path = b"/fp/runtime/assets/onnx/refiner_net.onnx"
            item.score_model_path = b"/fp/runtime/assets/onnx/score_net.onnx"
            item.engine_cache_dir = engine_cache
        err = C.create_string_buffer(4096)
        self.handle = self.lib.fp_group_create(options, count, C.byref(config), err, len(err))
        if not self.handle:
            raise RuntimeError(err.value.decode())
        try:
            self.prepare(HYPOTHESES)
        except RuntimeError:
            self.close()
            raise

    def prepare(self, batch_size):
        err = C.create_string_buffer(4096)
        if self.lib.fp_group_prepare(self.handle, batch_size, err, len(err)):
            raise RuntimeError(err.value.decode())

    def run(self, rgb, depth, K, masks=None):
        from foundation_pose_nvidia import RgbdFrame
        from foundation_pose_nvidia.bindings import PoseResult

        view = RgbdFrame(rgb, depth, K).to_image_view(with_mask=False)
        results = (PoseResult * self.count)()
        statuses = (C.c_int * self.count)()
        err = C.create_string_buffer(4096)
        tail = (results, statuses, err, len(err))
        if masks is None:
            count = self.active_count
            active = None if count == self.count else (C.c_int * self.count)(*([1] * count))
            rc = self.lib.fp_group_track_frame(self.handle, C.byref(view), active, 2, *tail)
        else:
            count = len(masks)
            if not 1 <= count <= self.count:
                raise ValueError(f"Expected 1 to {self.count} masks, got {count}")
            pointer = C.POINTER(C.c_uint8)
            mask_ptrs = (pointer * self.count)(*[mask.ctypes.data_as(pointer) for mask in masks])
            rc = self.lib.fp_group_register_frame(self.handle, C.byref(view), mask_ptrs,
                                                  self.iterations, HYPOTHESES, *tail)
        if rc or any(statuses[:count]):
            raise RuntimeError((rc, list(statuses[:count]), err.value.decode()))
        self.active_count = count
        return np.array([list(result.pose_row_major) for result in results[:count]]).reshape(-1, 4, 4)

    def register(self, rgb, depth, K, masks):
        """Yield one pose at a time while keeping each estimator in its group."""
        from foundation_pose_nvidia import RgbdFrame
        from foundation_pose_nvidia.bindings import PoseResult

        if not 1 <= len(masks) <= self.count:
            raise ValueError(f"Expected 1 to {self.count} masks, got {len(masks)}")
        self.active_count = len(masks)
        for index, mask in enumerate(masks):
            handle = self.lib.fp_group_handle(self.handle, index)
            view = RgbdFrame(rgb, depth, K, mask).to_image_view(with_mask=True)
            result = PoseResult()
            err = C.create_string_buffer(4096)
            rc = self.lib.fp_register_frame(handle, C.byref(view), self.iterations, HYPOTHESES,
                                            C.byref(result), err, len(err))
            if rc:
                raise RuntimeError(f"Object {index + 1}: {err.value.decode()}")
            yield np.array(result.pose_row_major).reshape(4, 4).copy()

    def close(self):
        self.lib.fp_group_destroy(self.handle)


class Original:
    def __init__(self, count, mesh, debug_dir):
        sys.path[:0] = [str(ROOT / "upstream/original"), str(RUNTIME_ROOT / "original/mycpp/build")]
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        import nvdiffrast.torch as dr

        scorer, refiner = ScorePredictor(), PoseRefinePredictor()
        context = dr.RasterizeCudaContext()
        self.estimators = [
            FoundationPose(mesh.vertices, mesh.vertex_normals, mesh=mesh, scorer=scorer,
                           refiner=refiner, glctx=context, debug=0, debug_dir=str(debug_dir))
            for _ in range(count)
        ]
        self.active_count = count
        if any(len(estimator.rot_grid) != HYPOTHESES for estimator in self.estimators):
            raise RuntimeError("Original rotation grid differs from the comparison configuration")

    def register(self, rgb, depth, K, masks):
        if not 1 <= len(masks) <= len(self.estimators):
            raise ValueError(f"Expected 1 to {len(self.estimators)} masks, got {len(masks)}")
        self.active_count = len(masks)
        for estimator, mask in zip(self.estimators, masks):
            yield estimator.register(K, rgb, depth, mask.astype(bool), iteration=5)

    def run(self, rgb, depth, K, masks=None):
        if masks is not None:
            return np.array(list(self.register(rgb, depth, K, masks)))
        return np.array([estimator.track_one(rgb, depth, K, iteration=2)
                         for estimator in self.estimators[:self.active_count]])


def detect(detector, rgb, labels, score, count):
    outputs, _ = detector.infer(rgb)
    boxes, scores, categories = (outputs[key][0] for key in ("boxes", "scores", "labels"))
    h, w = rgb.shape[:2]
    selected = []
    for index in np.argsort(scores)[::-1]:
        if scores[index] < score or int(categories[index]) not in labels:
            continue
        x0, y0, x1, y1 = boxes[index].round().astype(int)
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        if x1 <= x0 or y1 <= y0:
            continue
        selected.append((int(x0), int(y0), int(x1), int(y1), float(scores[index]), int(categories[index])))
    keep = cv2.dnn.NMSBoxes([(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1, *_ in selected],
                            [box[4] for box in selected], 0.0, 0.5)
    boxes = sorted((selected[int(index)] for index in keep), key=lambda box: box[0])
    masks = []
    if count is None or len(boxes) == count:
        for x0, y0, x1, y1, *_ in boxes:
            mask = np.zeros((h, w), np.uint8)
            mask[y0:y1, x0:x1] = 1
            masks.append(mask)
    return masks, boxes


def configure_runtime(mode):
    os.environ["FP_RENDERER"] = "builtin"
    for switch in ("FP_WORKER_POOL_SIZE", "FP_DYNAMIC_ENQUEUE_MEMORY", "FP_PHASE_SIZED_SCRATCH"):
        os.environ.pop(switch, None)
        if mode.startswith("pooled"):
            os.environ[switch] = "1"
    sys.path[:0] = [str(RUNTIME_ROOT / "runtime"), str(ROOT / "upstream/inference/python/src")]


def make_detector(args):
    if args.detector == "sdetr":
        from detector import Detector

        detector = Detector()

        def infer(rgb, depth, K):
            return detect(detector, rgb, args.labels, args.score, args.objects)
    elif args.detector == "yolo-trt":
        from yolo_trt import YoloTensorRT

        detector = YoloTensorRT(args.engine)
        if args.warm_detector:
            detector.warmup()

        def infer(rgb, depth, K):
            return detector.infer(rgb, args.labels, args.score, args.objects)
    else:
        def infer(rgb, depth, K):
            return detect_can(rgb, depth, K, args.height_m)
    return infer


class MemoryMonitor:
    def __init__(self):
        self.probe = Probe()
        self.samples = []
        self.phase = "startup"
        self.stop = threading.Event()
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def sample(self):
        reading = self.probe.sample()
        others = [item["pid"] for item in reading["processes"] if item["pid"] != os.getpid()]
        own = next((item["used_bytes"] for item in reading["processes"]
                    if item["pid"] == os.getpid()), 0)
        self.samples.append(dict(elapsed_s=time.monotonic() - self.start, phase=self.phase,
                                 process_bytes=own, device_bytes=reading["device_used_bytes"],
                                 foreign_pids=others))

    def poll(self):
        while not self.stop.wait(0.01):
            self.sample()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("original", "native", "pooled", "pooled-3"), required=True)
    parser.add_argument("--live-frames", type=int, default=0)
    parser.add_argument("--objects", type=int, default=1)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--labels", nargs="+", type=int, default=[22])
    parser.add_argument("--detector", choices=("sdetr", "rgbd-can", "yolo-trt"), default="sdetr")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--warm-detector", action="store_true")
    parser.add_argument("--height-m", type=float, required=True)
    args = parser.parse_args()
    if args.warm_detector and args.detector != "yolo-trt":
        parser.error("--warm-detector requires yolo-trt")
    if args.detector == "yolo-trt" and not args.engine:
        parser.error("yolo-trt requires --engine")
    if args.detector == "rgbd-can" and args.objects != 1:
        parser.error("rgbd-can supports one upright red-labeled can")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configure_runtime(args.mode)
    monitor = MemoryMonitor()
    monitor.sample()
    if monitor.samples[-1]["foreign_pids"]:
        raise RuntimeError("GPU already has a compute process")
    monitor.thread.start()
    result = dict(mode=args.mode, objects=args.objects, height_m=args.height_m, score=args.score,
                  labels=args.labels, detector=args.detector, engine=args.engine.name if args.engine else None,
                  detector_warmup=args.warm_detector,
                  capture=args.capture.name, hypotheses=HYPOTHESES,
                  register_iterations=3 if args.mode == "pooled-3" else 5, trials=[])
    camera = ExitStack()
    try:
        started = time.perf_counter()
        import torch

        logging.getLogger().setLevel(logging.ERROR)
        torch.set_num_threads(4)
        detect_objects = make_detector(args)
        mesh = load_scan(SCAN, args.height_m)
        if args.mode == "original":
            runtime = Original(args.objects, mesh, args.output.parent / "original-debug")
        else:
            runtime = Native(args.objects, args.capture.parent / "scan.ply", args.mode,
                             result["register_iterations"])
        torch.cuda.synchronize()
        result["setup_ms"] = (time.perf_counter() - started) * 1000
        monitor.sample()
        logging.getLogger().setLevel(logging.ERROR)
        if args.live_frames:
            from capture import open_camera

            read_frame, result["firmware"] = camera.enter_context(open_camera())
            frames = (read_frame() for _ in range(args.live_frames))
            tracking_frames = (read_frame() for _ in range(5))
        else:
            with np.load(args.capture) as capture:
                images, depths, K = capture["rgb"], capture["depth"], capture["K"]
            if len(images) < 7:
                raise ValueError("Capture must include registration frames and five following tracking frames")
            frames = ((rgb, depth, K, None, None) for rgb, depth in zip(images[:-5], depths[:-5], strict=True))
            tracking_frames = ((rgb, depth, K, None, None) for rgb, depth in zip(images[-5:], depths[-5:], strict=True))
        poses = None
        for i, (rgb, depth, K, camera_timestamp, capture_ms) in enumerate(frames):
            if args.mode == "original":
                K = K.astype(np.float64)
            monitor.phase = "registration"
            start = time.perf_counter()
            masks, boxes = detect_objects(rgb, depth, K)
            detect_ms = (time.perf_counter() - start) * 1000
            trial = dict(frame=i, detector_ms=detect_ms, boxes=boxes, status="detection_miss",
                         camera_timestamp_ms=camera_timestamp, capture_wait_align_ms=capture_ms)
            if masks:
                monitor.phase = "registration"
                pose_start = time.perf_counter()
                first_pose_ms = None
                if args.mode == "original":
                    original_poses = []
                    for pose in runtime.register(rgb, depth, K, masks):
                        original_poses.append(pose)
                        torch.cuda.synchronize()
                        if first_pose_ms is None:
                            first_pose_ms = (time.perf_counter() - start) * 1000
                    poses = np.array(original_poses)
                else:
                    poses = runtime.run(rgb, depth, K, masks)
                    torch.cuda.synchronize()
                end = time.perf_counter()
                all_ms = (end - start) * 1000
                status = "returned" if np.isfinite(poses).all() and np.all(poses[:, 2, 3] > 0) else "invalid"
                trial.update(status=status, registration_ms=(end - pose_start) * 1000,
                             first_pose_ms=first_pose_ms if first_pose_ms is not None else all_ms,
                             all_poses_ms=all_ms, poses=poses.tolist())
                monitor.sample()
            result["trials"].append(trial)
        if poses is not None and result["trials"][-1]["status"] == "returned":
            monitor.phase = "tracking"
            result["tracking"] = []
            for rgb, depth, K, _, _ in tracking_frames:
                if args.mode == "original":
                    K = K.astype(np.float64)
                started = time.perf_counter()
                tracked = runtime.run(rgb, depth, K)
                torch.cuda.synchronize()
                result["tracking"].append(dict(ms=(time.perf_counter() - started) * 1000,
                                               poses=tracked.tolist()))
                monitor.sample()
            result["tracking_retained_bytes"] = monitor.samples[-1]["process_bytes"]
        if args.mode != "original":
            runtime.close()
    finally:
        camera.close()
        monitor.stop.set()
        monitor.thread.join()
        result["memory"] = monitor.samples
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(sample["foreign_pids"] for sample in monitor.samples):
        raise RuntimeError("Foreign GPU usage detected during run")
    print(args.mode, "returned", sum(t["status"] == "returned" for t in result["trials"]),
          "of", len(result["trials"]))


if __name__ == "__main__":
    main()
