"""Compare detector hit rate and full host-to-mask latency on the D435 fixture."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from can_detector import detect_can
from nvml_probe import Probe


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) +
                           (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--detector", required=True, choices=("sdetr", "rgbd-can", "yolo-trt"))
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--labels", type=int, nargs="+", default=[0])
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--eval-box", type=float, nargs=4, required=True)
    args = parser.parse_args()
    probe = Probe()
    import os
    if probe.sample()["processes"]:
        raise RuntimeError("GPU already has a compute process")
    if args.detector == "yolo-trt":
        import torch
        from yolo_trt import YoloTensorRT

        torch.set_num_threads(4)
        detector = YoloTensorRT(args.engine)

        def infer(rgb, depth, K):
            return detector.infer(rgb, args.labels, args.score, 1)
    elif args.detector == "sdetr":
        import torch
        from benchmark import detect

        sys.path.insert(0, "/fp/runtime")
        from detector import Detector

        torch.set_num_threads(4)
        detector = Detector()

        def infer(rgb, depth, K):
            return detect(detector, rgb, args.labels, args.score, 1)
    else:
        def infer(rgb, depth, K):
            return detect_can(rgb, depth, K, 0.145)
    with np.load(args.capture) as capture:
        frames = list(zip(capture["rgb"][:-5], capture["depth"][:-5], strict=True))
        K = capture["K"]
    for _ in range(4):
        infer(*frames[0], K)
    rows = []
    for i, (rgb, depth) in enumerate(frames):
        start = time.perf_counter()
        masks, boxes = infer(rgb, depth, K)
        elapsed = (time.perf_counter() - start) * 1000
        match = max((iou(box, args.eval_box) for box in boxes), default=0.0)
        rows.append(dict(frame=i, ms=elapsed, boxes=boxes, best_box_iou=match,
                         usable=len(masks) == 1 and match >= 0.5))
    own = next((p["used_bytes"] for p in probe.sample()["processes"] if p["pid"] == os.getpid()), 0)
    stats = dict(detector=args.detector, engine=args.engine.name if args.engine else None,
                 labels=args.labels, score=args.score, eval_box=args.eval_box, n=len(rows),
                 usable=sum(row["usable"] for row in rows), process_after_trials_mib=own / 2**20,
                 p50_ms=float(np.median([row["ms"] for row in rows])),
                 p95_ms=float(np.percentile([row["ms"] for row in rows], 95)), frames=rows)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print({key: value for key, value in stats.items() if key != "frames"})


if __name__ == "__main__":
    main()
