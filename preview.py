"""Stream a live D435 view with FoundationPose status and timing overlays."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np

from benchmark import Native, Original, RUNTIMES, SCAN, configure_runtime, make_detector
from capture import open_camera
from mesh import export_ply, load_scan
from nvml_probe import Probe

PORT = 8765
COLORS = ((83, 237, 137), (67, 202, 255), (246, 168, 94), (205, 145, 241))
PAGE = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FoundationPose live preview</title>
<style>
body{margin:0;background:#11171b;color:#edf3f5;font:16px system-ui,sans-serif}
main{max-width:1000px;margin:28px auto;padding:0 16px}
header{display:flex;align-items:center;gap:16px;justify-content:space-between;margin-bottom:12px;flex-wrap:wrap}
h1{font-size:21px;margin:0}button{background:#25353d;color:inherit;border:1px solid #4a636d;padding:8px 12px;border-radius:5px;font:inherit;cursor:pointer}
button:focus-visible{outline:2px solid #53ed89}img{display:block;width:100%;height:auto;aspect-ratio:4/3;background:#202a30}
#status{min-height:1.5em;color:#b8cbd2;font-size:14px}
</style></head>
<body><main><header><h1>FoundationPose live preview</h1><button id="reset">Re-register objects</button></header>
<img id="stream" alt="Live D435 color stream and FoundationPose overlays">
<p id="status" role="status">Loading models and opening the D435...</p></main>
<script>
const statusLine=document.getElementById('status')
const stream=document.getElementById('stream')
const reset=document.getElementById('reset')
let connected=false
async function refresh(){
  try {
    const response=await fetch('/status.json',{cache:'no-store'})
    if(!response.ok) throw new Error()
    const update=await response.json()
    statusLine.textContent=update.detail
    if(update.frame && !connected){stream.src='/stream.mjpg';connected=true}
  } catch {statusLine.textContent='Preview disconnected'}
}
reset.onclick=async()=>{
  try {
    const response=await fetch('/reset',{method:'POST',headers:{'X-Pose-Preview':'reset'}})
    if(!response.ok) throw new Error()
    await refresh()
  } catch {statusLine.textContent='Could not request registration'}
}
stream.onerror=()=>{connected=false}
refresh()
setInterval(refresh,750)
</script></body></html>"""


class PreviewState:
    def __init__(self):
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
        self.frame = 0
        self.jpeg = None
        self.status = dict(detail="Loading models and opening the D435...", frame=0)

    def publish(self, jpeg, status):
        with self.condition:
            self.frame += 1
            self.jpeg = jpeg
            self.status = dict(status, frame=self.frame)
            self.condition.notify_all()

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()


class PreviewServer(ThreadingHTTPServer):
    def __init__(self, state, handler=None):
        self.state = state
        super().__init__(("0.0.0.0", PORT), handler or PreviewHandler)


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_body(self, content, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        state = self.server.state
        try:
            if self.path == "/":
                self.send_body(PAGE, "text/html; charset=utf-8")
            elif self.path == "/status.json":
                with state.condition:
                    payload = json.dumps(state.status).encode()
                self.send_body(payload, "application/json")
            elif self.path == "/frame.jpg":
                with state.condition:
                    frame = state.jpeg
                if frame is not None:
                    self.send_body(frame, "image/jpeg")
                else:
                    self.send_error(503, "Waiting for camera frames")
            elif self.path == "/stream.mjpg":
                self.send_stream(state)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def send_stream(self, state):
        self.connection.settimeout(5)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = 0
        while not state.stop.is_set():
            with state.condition:
                state.condition.wait_for(lambda: state.frame != last or state.stop.is_set())
                if state.stop.is_set():
                    break
                last, jpeg = state.frame, state.jpeg
            if jpeg is None:
                continue
            self.wfile.write(
                f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg + b"\r\n"
            )

    def do_POST(self):
        if self.path != "/reset" or self.headers.get("X-Pose-Preview") != "reset":
            self.send_error(403)
            return
        self.server.state.reset.set()
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()


def project_outline(vertices, pose, K, shape):
    points = vertices @ pose[:3, :3].T + pose[:3, 3]
    points = points[points[:, 2] > 0.02]
    if len(points) < 3:
        return None
    projected = points @ K.T
    pixels = (projected[:, :2] / projected[:, 2:]).astype(np.float32)
    height, width = shape
    hull = cv2.convexHull(pixels).reshape(-1, 2)
    return np.rint(np.clip(hull, (-width, -height), (2 * width, 2 * height))).astype(np.int32)


def paint_poses(pose_image, K, hull_vertices, poses, offset=0):
    for i, pose in enumerate(poses, offset):
        outline = project_outline(hull_vertices, pose, K, pose_image.shape[:2])
        if outline is None:
            continue
        color = COLORS[i % len(COLORS)]
        cv2.polylines(pose_image, [outline], True, color, 2, cv2.LINE_AA)
        height, width = pose_image.shape[:2]
        x, y = outline.min(axis=0).clip((0, 0), (width - 130, height - 25))
        caption = f"Can {i + 1}  {pose[2, 3]:.2f} m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(caption, font, 0.47, 1)
        anchor = int(x) + 4, max(123, int(y) - 6)
        cv2.rectangle(pose_image, (anchor[0] - 4, anchor[1] - text_height - 4),
                      (anchor[0] + text_width + 4, anchor[1] + 4), (20, 24, 28), -1)
        cv2.putText(pose_image, caption, anchor, font, 0.47, color, 1, cv2.LINE_AA)


def publish(state, rgb, K, hull_vertices, poses, boxes, metrics, *, pose_alpha=1.0, solid_count=0):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1, score, _ = box
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
        if score is not None:
            cv2.putText(image, f"{score:.2f}", (x0 + 3, max(17, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    if poses is not None:
        if pose_alpha > 0 and len(poses) > solid_count:
            pose_image = image if pose_alpha == 1 else image.copy()
            paint_poses(pose_image, K, hull_vertices, poses[solid_count:], solid_count)
            if pose_image is not image:
                cv2.addWeighted(pose_image, pose_alpha, image, 1 - pose_alpha, 0, dst=image)
        if solid_count:
            paint_poses(image, K, hull_vertices, poses[:solid_count])

    def duration(key):
        value = metrics.get(key)
        return f"{value:.1f} ms" if value is not None else "--"

    registration = (f"Last object registration {duration('last_object_registration_ms')}"
                    if metrics.get('config', {}).get('delivery') == 'sequential'
                    else f"Last registration {duration('last_registration_ms')}")
    cv2.rectangle(image, (0, 0), (image.shape[1], 99), (20, 24, 28), -1)
    runtime = RUNTIMES.get(metrics["mode"], {}).get("label", metrics["mode"]).replace(" · ", " / ")
    lines = (
        f"{runtime} | {metrics['detector']} | {metrics['state']}",
        f"Objects {metrics['visible']}/{metrics['expected']}   "
        f"Detect {duration('detect_ms')}   Pose {duration('pose_ms')}   "
        f"Wait to pose {duration('wait_to_pose_ms')}",
        f"{registration}   "
        f"{metrics['fps']:.1f} FPS   {metrics['vram_gib']:.2f} GiB VRAM",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    header_width = cv2.getTextSize(lines[0], font, 0.56, 1)[0][0]
    header_scale = 0.56 * min(1, (image.shape[1] - 24) / max(1, header_width))
    for i, line in enumerate(lines):
        scale = header_scale if i == 0 else 0.56
        cv2.putText(image, line, (12, 25 + i * 29), font, scale, (234, 244, 246), 1,
                    cv2.LINE_AA)
    encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if encoded:
        state.publish(jpeg.tobytes(), metrics)


def run_camera(args, state, detect_objects, runtime, hull, sync, probe):
    camera_limit = None
    poses = None
    last_registration = None
    last_registered_at = 0.0
    last_frame_at = None
    frame_interval = 0.0
    fps = 0.0
    detector_name = args.engine.stem if args.engine else args.detector
    memory_gib = 0.0
    last_memory_at = 0.0
    with open_camera() as (next_frame, _):
        if args.duration:
            camera_limit = time.monotonic() + args.duration
        while not state.stop.is_set() and (camera_limit is None or time.monotonic() < camera_limit):
            if state.reset.is_set():
                state.reset.clear()
                poses = None
            rgb, depth, K, _, wait_ms = next_frame()
            if args.mode == "original":
                K = K.astype(np.float64)
            now = time.monotonic()
            if now - last_memory_at >= 0.5:
                memory_gib = sum(p["used_bytes"] for p in probe.sample()["processes"]
                                 if p["pid"] == os.getpid()) / 2**30
                last_memory_at = now
            register = args.behavior == "register" or poses is None or (
                args.redetect > 0 and now - last_registered_at >= args.redetect)
            metrics = dict(mode=args.mode, detector=detector_name, expected=args.objects,
                           last_registration_ms=last_registration, fps=fps, vram_gib=memory_gib,
                           camera_ms=wait_ms, visible=0)
            boxes = []
            masks = None
            seconds = time.perf_counter()
            if register:
                masks, boxes = detect_objects(rgb, depth, K)
                metrics.update(visible=len(boxes), detect_ms=(time.perf_counter() - seconds) * 1000)
                if masks:
                    metrics.update(state="REGISTERING", detail="Registering all target objects")
                    publish(state, rgb, K, hull, None, boxes, metrics)
                else:
                    poses = None
                    metrics.update(state="SEARCHING", detail=f"Found {len(boxes)} of {args.objects} expected objects")
            if masks or not register:
                try:
                    pose_started = time.perf_counter()
                    poses = runtime.run(rgb, depth, K, masks)
                    sync()
                    metrics["pose_ms"] = (time.perf_counter() - pose_started) * 1000
                    if not np.isfinite(poses).all() or np.any(poses[:, 2, 3] <= 0):
                        raise RuntimeError("Pose output is invalid")
                    compute_ms = metrics["pose_ms"] + metrics.get("detect_ms", 0)
                    metrics.update(wait_to_pose_ms=compute_ms + wait_ms, visible=len(poses))
                    if register:
                        last_registration = compute_ms
                        last_registered_at = time.monotonic()
                        metrics.update(state="REGISTERED", last_registration_ms=last_registration,
                                       detail=f"Registered {len(poses)} cans")
                    else:
                        metrics.update(state="TRACKING", detail=f"Tracking {len(poses)} cans")
                except RuntimeError as exc:
                    poses = None
                    metrics.update(state="RETRYING", detail=str(exc))
            done = time.monotonic()
            if last_frame_at is not None:
                interval = done - last_frame_at
                frame_interval = 0.85 * frame_interval + 0.15 * interval if frame_interval else interval
                fps = 1 / frame_interval
            last_frame_at = done
            metrics["fps"] = fps
            publish(state, rgb, K, hull, poses, boxes, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("original", "native", "pooled", "pooled-3"))
    parser.add_argument("--detector", required=True, choices=("sdetr", "rgbd-can", "yolo-trt"))
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--warm-detector", action="store_true")
    parser.add_argument("--objects", required=True, type=int)
    parser.add_argument("--height-m", required=True, type=float)
    parser.add_argument("--score", required=True, type=float)
    parser.add_argument("--labels", nargs="+", type=int, required=True)
    parser.add_argument("--behavior", default="track", choices=("track", "register"))
    parser.add_argument("--redetect", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()
    if args.objects < 1 or args.height_m <= 0:
        parser.error("provide a positive height and object count")
    if args.detector == "rgbd-can" and args.objects != 1:
        parser.error("rgbd-can supports one upright red-labeled can")
    if args.detector == "yolo-trt" and args.engine is None:
        parser.error("yolo-trt requires --engine")

    state = PreviewState()
    server = PreviewServer(state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def stop_preview(*_):
        state.close()

    signal.signal(signal.SIGINT, stop_preview)
    signal.signal(signal.SIGTERM, stop_preview)
    runtime = None
    try:
        configure_runtime(args.mode)
        import torch

        torch.set_num_threads(4)
        probe = Probe()
        if probe.sample()["processes"]:
            raise RuntimeError("The selected GPU already has a compute process")
        detector = make_detector(args)
        mesh = load_scan(SCAN, args.height_m)
        hull = mesh.convex_hull.vertices
        if args.mode == "original":
            runtime = Original(args.objects, mesh, "/tmp/foundationpose-preview")
        else:
            mesh_path = Path("/tmp/foundationpose-preview.ply")
            export_ply(mesh, mesh_path)
            runtime = Native(args.objects, mesh_path, args.mode, 3 if args.mode == "pooled-3" else 5)
        torch.cuda.synchronize()
        logging.getLogger().setLevel(logging.ERROR)
        run_camera(args, state, detector, runtime, hull, torch.cuda.synchronize, probe)
    finally:
        state.close()
        if isinstance(runtime, Native):
            runtime.close()
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    main()
