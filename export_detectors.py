"""Export static ONNX detectors for TensorRT from official Ultralytics weights."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from ultralytics import YOLO, YOLOE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--text", nargs="+")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-box", type=float, nargs=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.text:
        model = YOLOE(str(args.weights))
        model.set_classes(args.text)
    elif args.reference:
        model = YOLOE(str(args.weights))
        prompts = dict(bboxes=np.array([args.reference_box], dtype=np.float32),
                       cls=np.array([0], dtype=np.int32))
        model.predict(str(args.reference), refer_image=str(args.reference), visual_prompts=prompts,
                      device="cpu", imgsz=640, verbose=False)
    else:
        model = YOLO(str(args.weights))
    for size in args.sizes:
        dimensions = [int(dimension) for dimension in size.split("x")]
        imgsz = dimensions[0] if len(dimensions) == 1 else dimensions
        output = model.export(format="onnx", imgsz=imgsz, batch=1, dynamic=False,
                              nms=False, simplify=False, opset=17, device="cpu")
        suffix = "-text" if args.text else ""
        target = args.weights.with_name(f"{args.weights.stem}{suffix}-{size}.onnx")
        shutil.move(output, target)


if __name__ == "__main__":
    main()
