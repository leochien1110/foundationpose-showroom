"""Local dashboard for interactive RealSense capture and FoundationPose."""
import gc
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import tempfile
import threading
import time
from urllib.parse import unquote

import cv2
import numpy as np

from benchmark import Native, Original, ROOT, RUNTIME_ROOT, RUNTIMES, SCAN, configure_runtime, detect
from can_detector import detect_can
from capture import open_camera
from mesh import export_ply, load_scan
from mesh_library import MAX_BYTES, MeshLibrary, normalize_geometry
from nvml_probe import Probe
from preview import PreviewHandler, PreviewServer, PreviewState, publish
from scene_geometry import detection_centers, display_mesh
from yolo_prompt import DEFAULT_PROMPT, default_prompt, prompt_engine

DEFAULT_ENGINE = "yoloe-26n-seg-text-320.plan"
DEFAULT_CONFIG = dict(mode="pooled-3", delivery="batch", model=DEFAULT_ENGINE, mesh="coke-can", score=0.25,
                      labels=[0], prompt=DEFAULT_PROMPT, count=None, limit=8, loading="on_demand", preload_count=None)
POSE_DELIVERY = {"batch", "sequential"}
LOAD_MODES = {"on_demand", "preload"}
PAGE = ROOT / "dashboard.html"
VIEWPORT = ROOT / "viewport.js"
STAGES = ("stream", "detect", "register", "track")
TRACKING_WINDOW_S = 1.0
POSE_COOLDOWN_S = 1.0
POSE_FADE_IN_S = 0.18


class DashboardState(PreviewState):
    def __init__(self, library):
        super().__init__()
        self.commands = Queue(maxsize=12)
        self.page = PAGE.read_bytes()
        self.viewport_script = VIEWPORT.read_bytes()
        self.library = library
        self.scene = {}
        self.geometry = {}
        self.models = {p.name: p for p in (ROOT / ".models").glob("yolo*.plan")}
        self.runtimes = RUNTIMES.copy()
        if not ((RUNTIME_ROOT / "original/mycpp/build").is_dir()
                and (ROOT / "upstream/original/weights/2023-10-28-18-33-37").is_dir()
                and (ROOT / "upstream/original/weights/2024-01-11-20-02-45").is_dir()):
            del self.runtimes["original"]
        if not (ROOT / ".build/pooled/libfoundation_pose_nvidia.so").is_file():
            for mode in ("pooled", "pooled-3"):
                del self.runtimes[mode]
        self.default_config = DEFAULT_CONFIG.copy()
        if "pooled-3" not in self.runtimes:
            self.default_config["mode"] = "native"
        if DEFAULT_ENGINE not in self.models:
            self.default_config.update(model="rgbd-can", labels=[], prompt=None)
        self.status.update(detail="Camera stopped", streaming=False, captured=False, detecting=False,
                           registering=False, tracking=False, registered=0, vram_gib=0.0, config=self.default_config.copy())

    def options(self):
        models = [{"id": name, "labels": [0] if name.startswith("yoloe-") else [39],
                   "score": 0.25, "prompt": default_prompt(name)} for name in sorted(self.models)]
        if (RUNTIME_ROOT / "runtime/detector.py").is_file():
            models.append(dict(id="sdetr", labels=[22], score=0.5))
        models.append(dict(id="rgbd-can", labels=[], score=0.25))
        return dict(models=models, meshes=self.library.list(),
                    runtimes=[dict(id=key, **runtime) for key, runtime in self.runtimes.items()])

    def runtime_settings(self, values):
        mode = values["mode"]
        delivery = values.get("delivery", "batch")
        if mode not in self.runtimes or delivery not in POSE_DELIVERY:
            raise ValueError("Select an available runtime and pose update mode")
        return dict(mode=mode, delivery=delivery)

    def configuration(self, values):
        runtime = self.runtime_settings(values)
        model, mesh = values["model"], values["mesh"]
        if model not in self.models and model != "rgbd-can" and (
            model != "sdetr" or not (RUNTIME_ROOT / "runtime/detector.py").is_file()
        ):
            raise ValueError("Select an available detector")
        self.library.paths(mesh)
        prompt = default_prompt(model)
        if prompt is not None:
            prompt = values.get("prompt", prompt)
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 128:
                raise ValueError("Enter a YOLOE prompt of 1 to 128 characters")
            prompt = " ".join(prompt.split())
            if not prompt.isprintable():
                raise ValueError("The YOLOE prompt must contain printable characters")
        score = float(values["score"])
        count, limit = values["count"], values["limit"]
        labels = values["labels"]
        loading = values.get("loading", "on_demand")
        preload_count = values.get("preload_count")
        if loading not in LOAD_MODES:
            raise ValueError("Select On demand or Preload")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Object limit must be between 1 and 32")
        if count is not None and (type(count) is not int or not 1 <= count <= limit):
            raise ValueError("An exact count must be between 1 and the object limit")
        if preload_count is not None and (type(preload_count) is not int or not 1 <= preload_count <= limit):
            raise ValueError("Preload object count must be between 1 and the object limit")
        if not isinstance(labels, list) or any(type(label) is not int or label < 0 for label in labels):
            raise ValueError("Class IDs must be nonnegative integers")
        if prompt is not None and labels != [0]:
            raise ValueError("The YOLOE text prompt uses class ID 0")
        if model != "rgbd-can" and not labels:
            raise ValueError("Enter at least one detector class ID")
        if model == "rgbd-can" and (count not in (None, 1) or mesh != "coke-can" or
                                     (loading == "preload" and (count or preload_count) not in (None, 1))):
            raise ValueError("RGB-D can detector supports one can and the built-in can mesh")
        return dict(**runtime, model=model, mesh=mesh, score=score, labels=labels, prompt=prompt, count=count,
                    limit=limit, loading=loading, preload_count=preload_count)


class DashboardHandler(PreviewHandler):
    def send_json(self, value):
        self.send_body(json.dumps(value).encode(), "application/json")

    def do_GET(self):
        self.path = self.path.split("?", 1)[0]
        if self.path == "/":
            self.send_body(self.server.state.page, "text/html; charset=utf-8")
        elif self.path == "/options.json":
            self.send_json(self.server.state.options())
        elif self.path == "/viewport.js":
            self.send_body(self.server.state.viewport_script, "text/javascript; charset=utf-8")
        elif self.path in {"/scene.json", "/geometry.json"}:
            state = self.server.state
            with state.condition:
                value = state.scene if self.path == "/scene.json" else state.geometry
            self.send_json(value)
        else:
            super().do_GET()

    def do_POST(self):
        if self.headers.get("X-Pose-Dashboard") != "1":
            self.send_error(403)
            return
        self.connection.settimeout(30)
        state = self.server.state
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = MAX_BYTES if self.path == "/upload" else 8192
            if not 0 < length <= maximum:
                raise ValueError("Upload limit is 40 MiB" if self.path == "/upload" else "Invalid request length")
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Request ended before all bytes arrived")
            if self.path == "/upload":
                name = unquote(self.headers.get("X-Mesh-Name", ""))
                height = self.headers.get("X-Mesh-Height", "")
                axis = self.headers.get("X-Mesh-Axis", "Y")
                entry = state.library.upload(name, data, height, axis)
                self.send_json(entry)
                return
            if self.path != "/control":
                self.send_error(404)
                return
            values = json.loads(data)
            action = values["action"]
            if action in STAGES or action == "capture":
                argument = values["enabled"]
                if type(argument) is not bool:
                    raise ValueError("Enabled must be true or false")
            elif action == "apply":
                argument = state.configuration(values["config"])
            elif action == "runtime":
                argument = state.runtime_settings(values["config"])
            elif action in {"reinitialize", "preload"}:
                argument = None
            else:
                raise ValueError("Unknown action")
            state.commands.put_nowait((action, argument))
            self.send_json({"accepted": True})
        except Full:
            self.send_error(409, "Dashboard is busy")
        except (ValueError, TypeError, KeyError, RuntimeError, OSError, IndexError) as exc:
            self.send_error(400, str(exc)[:220])


class Dashboard:
    def __init__(self, state, sync, probe, scratch, empty_cache):
        self.state, self.sync, self.probe = state, sync, probe
        self.scratch, self.empty_cache = Path(scratch), empty_cache
        self.config = state.default_config.copy()
        self.camera = None
        self.next_frame = None
        self.captured_frame = None
        self.capture_pending = False
        self.runtime = None
        self.runtime_count = 0
        self.detector = None
        self.load_timings = {mode: {} for mode in LOAD_MODES}
        self.preloading = False
        self.mesh = None
        self.hull = None
        self.poses = None
        self.detecting = False
        self.registering = False
        self.tracking = False
        self.next_registration = 0.0
        self.pose_recovery_at = 0.0
        self.pose_recovery_alpha = 1.0
        self.operation = "stream"
        self.error = None
        self.last_registration = None
        self.last_object_registration = None
        self.last_registration_at = None
        self.last_detected = 0
        self.boxes = []
        self.markers = []
        self.detail = "Camera stopped"
        self.phase = "STOPPED"
        self.fps = 0.0
        self.frame_time = None
        self.vram = 0.0
        self.vram_time = 0.0
        self.peak_vram = 0.0

    def status(self, *, registered=None, **extra):
        if registered is None:
            registered = len(self.poses) if self.poses is not None else 0
        return dict(streaming=self.camera is not None, captured=self.captured_frame is not None,
                    detecting=self.detecting,
                    registering=self.registering, tracking=self.tracking,
                    registered=registered, poses_ready=self.poses is not None,
                    detected=self.last_detected, detail=self.error or self.detail, state=self.phase,
                    last_registration_ms=self.last_registration, last_registration_at=self.last_registration_at,
                    last_object_registration_ms=self.last_object_registration,
                    config=self.config, vram_gib=self.vram, peak_vram_gib=self.peak_vram,
                    model_loads=self.load_timings.copy(), preloading=self.preloading,
                    detector_ready=self.detector is not None or self.config["model"] == "rgbd-can",
                    prepared_objects=self.runtime_count, **extra)

    def sample_vram(self):
        self.vram = sum(p["used_bytes"] for p in self.probe.sample()["processes"]
                        if p["pid"] == os.getpid()) / 2**30
        self.peak_vram = max(self.peak_vram, self.vram)
        self.vram_time = time.monotonic()

    def publish_status(self, clear=False):
        with self.state.condition:
            if clear:
                self.state.jpeg = None
                self.state.scene = {}
            self.state.status = self.status(frame=0 if clear else self.state.frame)
            self.state.condition.notify_all()

    def release_runtime(self):
        if isinstance(self.runtime, Native):
            self.runtime.close()
        self.runtime = None
        self.runtime_count = 0
        self.poses = None
        gc.collect()
        self.empty_cache()

    def clear_results(self):
        self.poses = None
        self.last_registration = self.last_object_registration = self.last_registration_at = None
        self.next_registration = self.pose_recovery_at = 0.0
        self.pose_recovery_alpha = 1.0

    def close_camera(self):
        if self.camera is not None:
            camera, self.camera = self.camera, None
            self.next_frame = None
            camera.__exit__(None, None, None)
        self.frame_time, self.fps = None, 0.0

    def capture_frame(self):
        if self.captured_frame is not None:
            return
        self.phase, self.detail = "CAPTURING", "Capturing one frame"
        self.publish_status()
        if self.camera is not None:
            frame = self.next_frame()
            self.close_camera()
        else:
            with open_camera() as (next_frame, _):
                frame = next_frame()
        self.tracking = False
        self.clear_results()
        self.boxes, self.markers = [], []
        self.last_detected = 0
        self.captured_frame = frame
        self.capture_pending = self.detecting
        self.phase, self.detail = "CAPTURED", "Captured frame held"
        rgb, _, K, _, camera_ms = frame
        self.publish_frame(rgb, K, None, None, None, camera_ms)

    def stop_from(self, stage, *, keep_preload=True):
        keep_models = keep_preload and self.config["loading"] == "preload"
        for dependent in reversed(STAGES[STAGES.index(stage):]):
            if dependent == "track":
                self.tracking = False
            elif dependent == "register":
                self.registering = False
                self.clear_results()
                if self.runtime is not None and not keep_models:
                    self.release_runtime()
            elif dependent == "detect":
                self.detecting = False
                self.boxes, self.markers = [], []
                self.last_detected = 0
                if self.detector is not None and not keep_models:
                    self.detector = None
                    self.empty_cache()
            else:
                self.close_camera()
                self.captured_frame = None
                self.capture_pending = False
        if self.camera is not None:
            self.phase, self.detail = "STREAMING", "Camera streaming"
        elif self.captured_frame is not None:
            self.phase, self.detail = "CAPTURED", "Captured frame held"
        else:
            self.phase, self.detail = "STOPPED", "Camera stopped"

    def start_stage(self, stage):
        if stage == "stream":
            if self.camera is None:
                self.phase, self.detail = "OPENING", "Opening the D435"
                self.publish_status()
                camera = open_camera()
                self.next_frame, _ = camera.__enter__()
                self.camera = camera
                if self.captured_frame is not None:
                    self.captured_frame = None
                    self.capture_pending = False
                    self.stop_from("detect")
                    self.publish_status(clear=True)
            self.phase, self.detail = "STREAMING", "Camera streaming"
        elif stage == "detect" and (self.camera is not None or self.captured_frame is not None):
            self.detecting = True
            self.detail = "Detection enabled"
        elif stage == "register" and self.detecting:
            if not self.registering:
                self.next_registration = 0.0
            self.registering = True
            self.detail = "Registration enabled"
        elif stage == "track" and self.camera is not None and self.registering and self.poses is not None:
            if not self.tracking:
                self.next_registration = time.monotonic() + TRACKING_WINDOW_S
            self.tracking = True
            self.detail = "Tracking enabled"
        else:
            self.detail = {
                "detect": "Enable Stream or Capture before Detect",
                "register": "Enable Detect before Register",
                "track": ("Enable Stream to track" if self.captured_frame is not None else
                          "Wait for a successful registration before enabling Track"),
            }[stage]

    def stop_stream(self):
        self.stop_from("stream", keep_preload=False)
        self.sample_vram()
        self.publish_status(clear=True)

    def control(self, action, argument):
        self.error = None
        if action == "apply":
            changed = argument != self.config
            same_detection = all(argument[key] == self.config[key] for key in ("model", "prompt", "labels", "score", "mesh"))
            recent_count = self.last_detected if same_detection else 0
            if changed:
                self.stop_from("register", keep_preload=False)
                if any(argument[key] != self.config[key] for key in ("model", "prompt", "loading")):
                    self.detector = None
                    self.empty_cache()
                if argument["mesh"] != self.config["mesh"]:
                    self.mesh = self.hull = None
                self.boxes, self.markers = [], []
                self.last_detected = 0
            self.config = argument
            self.detail = "Settings applied"
            if changed and self.config["loading"] == "preload":
                self.preload_models(recent_count=recent_count)
        elif action == "preload":
            if self.config["loading"] == "preload":
                self.stop_from("register", keep_preload=False)
                self.detector = None
                self.empty_cache()
                self.preload_models()
            else:
                self.detail = "Apply Preload before loading models"
        elif action == "runtime":
            mode_changed = argument["mode"] != self.config["mode"]
            if mode_changed:
                self.operation = "register"
                if self.runtime is not None:
                    self.release_runtime()
                self.clear_results()
            if argument["delivery"] != self.config["delivery"]:
                self.last_object_registration = None
            self.config.update(argument)
            self.next_registration = 0.0
            self.detail = "Pose runtime settings applied"
            if mode_changed and self.config["loading"] == "preload":
                self.preload_models()
        elif action == "reinitialize":
            if self.registering:
                self.next_registration = 0.0
                self.detail = "Reinitialization queued"
            else:
                self.detail = "Enable Register before reinitializing"
        elif action == "capture":
            self.operation = "stream"
            if argument:
                self.capture_frame()
            elif self.captured_frame is not None:
                self.stop_from("stream")
        elif argument:
            self.operation = action
            self.start_stage(action)
        elif action != "stream" or self.camera is not None:
            self.stop_from(action)
        if self.captured_frame is not None and action in {"apply", "preload", "runtime", "detect", "register", "reinitialize"}:
            self.capture_pending = True
        self.sample_vram()
        self.publish_status(clear=self.camera is None and self.captured_frame is None)

    def preload_models(self, *, recent_count=None):
        if recent_count is None:
            recent_count = self.last_detected
        if not 1 <= recent_count <= self.config["limit"]:
            recent_count = 1
        count = self.config["count"] or self.config["preload_count"] or recent_count
        self.preloading = True
        self.phase, self.detail = "LOADING", "Preloading detector"
        self.publish_status()
        try:
            self.operation = "detect"
            self.detector_for_settings(source="preload")
            self.operation = "register"
            self.detail = f"Loading registration and tracking for {count} objects"
            self.sample_vram()
            self.publish_status()
            self.ensure_runtime(count, source="preload")
        finally:
            self.preloading = False
        self.phase = ("STREAMING" if self.camera is not None else
                      "CAPTURED" if self.captured_frame is not None else "STOPPED")
        self.detail = f"Detector, registration, and tracking ready for {count} objects"

    def record_load(self, source, component, started, **metadata):
        sample = dict(ms=(time.perf_counter() - started) * 1000, **metadata)
        self.load_timings[source] = {**self.load_timings[source], component: sample}

    def detector_for_settings(self, *, source="on_demand"):
        model = self.config["model"]
        if self.detector is not None or model == "rgbd-can":
            return
        if source == "on_demand":
            self.phase, self.detail = "LOADING", "Loading detector"
            self.publish_status()
        started = time.perf_counter()
        if model == "sdetr":
            from detector import Detector

            candidate = Detector()
            if source == "preload":
                candidate.infer(np.zeros((480, 640, 3), dtype=np.uint8))
            self.detector = candidate
        else:
            from yolo_trt import YoloTensorRT

            prompt = self.config["prompt"]

            def progress(detail):
                self.phase, self.detail = "BUILDING", f'{detail} for “{prompt}”'
                self.publish_status()

            engine = prompt_engine(self.state.models[model], prompt, progress)
            if self.phase == "BUILDING":
                self.phase, self.detail = "LOADING", "Loading the YOLOE engine"
                self.publish_status()
            candidate = YoloTensorRT(engine)
            candidate.warmup()
            self.detector = candidate
        self.sync()
        self.record_load(source, "detector", started, name=model, prompt=self.config["prompt"])

    def detect_instances(self, rgb, depth, K):
        model = self.config["model"]
        if model == "rgbd-can":
            return detect_can(rgb, depth, K, 0.145)
        labels, score = self.config["labels"], self.config["score"]
        if model == "sdetr":
            return detect(self.detector, rgb, labels, score, None)
        return self.detector.infer(rgb, labels, score)

    def runtime_fits(self, count):
        return self.runtime is not None and (
            count == self.runtime_count or self.config["loading"] == "preload" and count < self.runtime_count
        )

    def ensure_runtime(self, count, *, source="on_demand"):
        if self.runtime_fits(count):
            return
        self.release_runtime()
        started = time.perf_counter()
        mode = self.config["mode"]
        configure_runtime(mode)
        paths = self.state.library.paths(self.config["mesh"])
        if self.mesh is None:
            if paths is None:
                self.mesh = load_scan(SCAN, 0.145)
            else:
                self.mesh = normalize_geometry(paths[0].read_bytes(), ".glb", None, 1)
            self.hull = self.mesh.convex_hull.vertices
            geometry = display_mesh(self.mesh)
            with self.state.condition:
                self.state.geometry = dict(geometry, id=self.config["mesh"])
        if mode == "original":
            self.runtime = Original(count, self.mesh, self.scratch)
        else:
            if paths is None:
                ply = self.scratch / "coke-can.ply"
                if not ply.exists():
                    export_ply(self.mesh, ply)
            else:
                ply = paths[1]
            self.runtime = Native(count, ply, mode, 3 if mode == "pooled-3" else 5)
            if source == "preload":
                self.runtime.prepare(1)
        if source == "preload":
            self.detail = f"Warming registration and tracking for {count} objects"
            self.publish_status()
            self.warm_pose_runtime(count, mode)
        self.sync()
        self.runtime_count = count
        self.record_load(source, "pose", started, name=mode, objects=count)
        logging.getLogger().setLevel(logging.ERROR)

    def warm_pose_runtime(self, count, mode):
        # Keep warmup independent of the camera and real scene state.
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        hull = self.hull
        lower, upper = hull.min(axis=0), hull.max(axis=0)
        size = max(upper - lower)
        center = (320, 240)
        pixels = (hull[:, :2] - (lower[:2] + upper[:2]) / 2) * (160 / size) + center
        polygon = cv2.convexHull(np.rint(pixels).astype(np.int32))
        cv2.fillConvexPoly(mask, polygon, 1)
        # A thin edge-on mesh can project to a line. Keep enough valid depth samples.
        if np.count_nonzero(mask) < 100:
            cv2.circle(mask, center, 48, 1, -1)
        depth = mask.astype(np.float32) * np.float32(np.clip(size * 4, 0.4, 2.0))
        rgb[mask != 0] = (110, 120, 130)
        K = np.array([[600, 0, center[0]], [0, 600, center[1]], [0, 0, 1]],
                     dtype=np.float64 if mode == "original" else np.float32)
        masks = [mask] * count
        self.validate_poses(self.runtime.run(rgb, depth, K, masks))
        self.validate_poses(self.runtime.run(rgb, depth, K))
        if mode.startswith("pooled"):
            # Tracking shrinks the shared buffers. Leave registration ready for the first click.
            self.validate_poses(self.runtime.run(rgb, depth, K, masks))

    def valid_count(self, count):
        if not count:
            return "No usable detections"
        if count > self.config["limit"]:
            return f"{count} detections exceed the object limit of {self.config['limit']}"
        expected = self.config["count"]
        if expected is not None and count != expected:
            return f"Found {count} objects, expected {expected}"
        return None

    def pose(self, rgb, depth, K, masks=None):
        started = time.perf_counter()
        if self.config["mode"] == "original":
            K = K.astype(np.float64)
        poses = self.runtime.run(rgb, depth, K, masks)
        self.sync()
        elapsed = (time.perf_counter() - started) * 1000
        self.validate_poses(poses)
        self.poses = poses
        return elapsed

    @staticmethod
    def validate_poses(poses):
        if not np.isfinite(poses).all() or np.any(poses[..., 2, 3] <= 0):
            raise RuntimeError("Pose output is invalid")

    def register_sequentially(self, rgb, depth, K, masks, publish_pose):
        if self.config["mode"] == "original":
            K = K.astype(np.float64)
        stream = self.runtime.register(rgb, depth, K, masks)
        elapsed = 0.0
        for _ in masks:
            started = time.perf_counter()
            pose = next(stream)
            self.sync()
            self.validate_poses(pose)
            object_ms = (time.perf_counter() - started) * 1000
            elapsed += object_ms
            publish_pose(pose, object_ms)
        return elapsed

    def frame(self):
        self.operation = "stream"
        if self.captured_frame is not None:
            rgb, depth, K, _, camera_ms = self.captured_frame
            self.capture_pending = False
            self.phase, self.detail = "CAPTURED", "Captured frame held"
        else:
            rgb, depth, K, _, camera_ms = self.next_frame()
            self.phase, self.detail = "STREAMING", "Camera streaming"
        detect_ms = pose_ms = None
        shown_poses = None
        if self.detecting:
            self.operation = "detect"
            self.detector_for_settings()
            started = time.perf_counter()
            masks, self.boxes = self.detect_instances(rgb, depth, K)
            detect_ms = (time.perf_counter() - started) * 1000
            self.last_detected = len(masks)
            self.markers = detection_centers(masks, self.boxes, depth, K)
            self.phase, self.detail = "DETECTED", f"Detected {self.last_detected} usable objects"
            issue = self.valid_count(len(masks)) if self.registering else None
            register_now = self.registering and (not self.tracking or time.monotonic() >= self.next_registration)
            if register_now and issue is None:
                self.operation = "register"
                pose_ms = self.register(rgb, depth, K, masks, detect_ms, camera_ms)
                self.last_registration = detect_ms + pose_ms
                self.last_registration_at = time.monotonic()
                self.next_registration = self.last_registration_at + TRACKING_WINDOW_S
                self.phase, self.detail = "REGISTERED", f"Registered {len(self.poses)} objects"
                shown_poses = self.poses
            elif self.tracking and self.poses is not None:
                self.operation = "track"
                pose_ms = self.pose(rgb, depth, K)
                self.phase, self.detail = "TRACKING", f"Tracking {len(self.poses)} objects"
                if issue:
                    self.detail += f" · {issue}"
                shown_poses = self.poses
            elif issue:
                self.poses = None
                self.phase, self.detail = "SEARCHING", issue

        now = time.monotonic()
        if self.camera is not None:
            if self.frame_time is not None:
                instant_fps = 1 / max(now - self.frame_time, 1e-6)
                self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps else instant_fps
            self.frame_time = now
        if pose_ms is not None or now - self.vram_time >= 0.5:
            self.sample_vram()
        self.publish_frame(rgb, K, shown_poses, detect_ms, pose_ms, camera_ms)

    def pose_opacity(self, now):
        progress = min(1.0, (now - self.pose_recovery_at) / POSE_FADE_IN_S)
        return self.pose_recovery_alpha + (1 - self.pose_recovery_alpha) * progress

    def register(self, rgb, depth, K, masks, detect_ms, camera_ms):
        count = len(masks)
        preparing = not self.runtime_fits(count)
        self.phase = "REGISTERING"
        self.detail = f"Preparing {count} pose estimators" if preparing else f"Registering {count} objects"
        previous = self.poses
        frozen = self.captured_frame is not None
        sequential = self.config["delivery"] == "sequential"
        fresh = []
        started = time.monotonic()
        finished = threading.Event()
        display_lock = threading.Lock()

        def fade_at(now):
            progress = min(1.0, (now - started) / POSE_COOLDOWN_S)
            return self.pose_opacity(now) * (1 - progress**2)

        def show_progress():
            alpha = (1 if frozen else fade_at(time.monotonic())) if previous is not None else 0
            visible = list(fresh)
            if alpha > 0 and (not fresh or len(previous) == count):
                visible.extend(previous[len(fresh):])
            display = np.array(visible) if visible else None
            self.publish_frame(rgb, K, display, detect_ms, None, camera_ms, pose_alpha=alpha,
                               fresh_count=len(fresh) if sequential else None)
            return alpha

        def fade():
            while not finished.wait(0.04):
                with display_lock:
                    if show_progress() == 0 or len(fresh) == count:
                        return

        def publish_pose(pose, object_ms):
            with display_lock:
                self.last_object_registration = object_ms
                fresh.append(pose)
                self.detail = f"Registered {len(fresh)} of {count} objects"
                show_progress()

        show_progress()
        worker = None
        if previous is not None and not frozen:
            worker = threading.Thread(target=fade)
            worker.start()
        try:
            self.ensure_runtime(count)
            if preparing:
                self.detail = f"Registering {count} objects"
                self.publish_status()
            if sequential:
                pose_ms = self.register_sequentially(rgb, depth, K, masks, publish_pose)
                self.poses = np.array(fresh)
            else:
                pose_ms = self.pose(rgb, depth, K, masks)
        finally:
            finished.set()
            if worker is not None:
                worker.join()
        completed = time.monotonic()
        self.pose_recovery_alpha = fade_at(completed) if worker is not None and not sequential else 1
        self.pose_recovery_at = completed
        return pose_ms

    def publish_frame(self, rgb, K, poses, detect_ms, pose_ms, camera_ms, *, pose_alpha=None,
                      fresh_count=None):
        initializing = self.phase == "REGISTERING"
        if pose_alpha is None:
            pose_alpha = self.pose_opacity(time.monotonic()) if poses is not None else 0
        solid_count = fresh_count or 0
        if pose_alpha == 0:
            poses = poses[:solid_count] if solid_count else None
        expected = self.config["count"] or "auto"
        visible = (fresh_count if fresh_count is not None else
                   len(poses) if poses is not None else self.last_detected)
        metrics = self.status(registered=fresh_count, mode=self.config["mode"], detector=self.config["model"].removesuffix(".plan"),
                              visible=visible, expected=expected, fps=self.fps,
                              detect_ms=detect_ms, pose_ms=pose_ms, camera_ms=camera_ms,
                              wait_to_pose_ms=camera_ms + (detect_ms or 0) + pose_ms if pose_ms is not None else None)
        with self.state.condition:
            self.state.scene = dict(intrinsics=K.tolist(), width=rgb.shape[1], height=rgb.shape[0],
                                    mesh=self.config["mesh"], tracking=self.tracking,
                                    poses=poses.tolist() if poses is not None else [],
                                    pose_alpha=pose_alpha, solid_count=solid_count,
                                    markers=[] if initializing else self.markers, state=self.phase)
        boxes = self.boxes if self.captured_frame is not None or (poses is None and not initializing) else []
        publish(self.state, rgb, K, self.hull, poses, boxes, metrics,
                pose_alpha=pose_alpha, solid_count=solid_count)

    def loop(self):
        while not self.state.stop.is_set():
            has_work = self.camera is not None or self.capture_pending
            try:
                action, config = self.state.commands.get_nowait() if has_work else self.state.commands.get(timeout=0.15)
            except Empty:
                if not has_work:
                    continue
                action = None
            try:
                if action:
                    self.control(action, config)
                else:
                    self.frame()
            except Exception as exc:
                logging.exception("Dashboard operation failed")
                self.stop_from(self.operation, keep_preload=False)
                self.sample_vram()
                self.phase, self.error = "ERROR", str(exc)
                self.capture_pending = self.captured_frame is not None
                self.publish_status(clear=self.camera is None and self.captured_frame is None)
                self.state.stop.wait(0.2)


def main():
    import torch

    torch.set_num_threads(4)
    probe = Probe()
    if probe.sample()["processes"]:
        raise RuntimeError("The selected GPU already has a compute process")
    library = os.environ.get("FP_MESH_LIBRARY") or ROOT / ".mesh-library"
    state = DashboardState(MeshLibrary(library))
    configure_runtime(state.default_config["mode"])
    server = PreviewServer(state, DashboardHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    signal.signal(signal.SIGINT, lambda *_: state.close())
    signal.signal(signal.SIGTERM, lambda *_: state.close())
    with tempfile.TemporaryDirectory(prefix="fp-dashboard-") as scratch:
        dashboard = Dashboard(state, torch.cuda.synchronize, probe, scratch, torch.cuda.empty_cache)
        try:
            dashboard.loop()
        finally:
            dashboard.stop_stream()
            state.close()
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    main()
