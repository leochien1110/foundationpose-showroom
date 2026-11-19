"""Run fixed-size end-to-end YOLO26 TensorRT engines on host RGB frames."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


class YoloTensorRT:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            buffer = torch.empty(shape, device="cuda", dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype)
            self.context.set_tensor_address(name, buffer.data_ptr())
            self.buffers[name] = buffer
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input = name
                self.input_dtype = dtype
                _, _, self.height, self.width = shape
            else:
                self.outputs.append(name)
        self.detections = next(name for name in self.outputs if self.buffers[name].ndim == 3)
        self.prototype = next((name for name in self.outputs if self.buffers[name].ndim == 4), None)
        self.canvas = np.full((self.height, self.width, 3), 114, np.uint8)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def infer(self, rgb, labels, score, count=None):
        h, w = rgb.shape[:2]
        scale = min(self.width / w, self.height / h)
        resized_w, resized_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (self.width - resized_w) // 2, (self.height - resized_h) // 2
        self.canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = cv2.resize(
            rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(self.canvas.transpose(2, 0, 1)[None], dtype=self.input_dtype)
        tensor *= 1.0 / 255
        with torch.cuda.stream(self.stream):
            self.buffers[self.input].copy_(torch.from_numpy(tensor), non_blocking=True)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("YOLO TensorRT execution failed")
            raw = self.buffers[self.detections].cpu().numpy()[0]
        picked = []
        candidates = np.flatnonzero((raw[:, 4] >= score) & np.isin(raw[:, 5], labels))
        for index in candidates:
            x0, y0, x1, y1, conf, label = raw[index, :6]
            x0, x1 = np.clip(np.rint((np.array([x0, x1]) - pad_x) / scale).astype(int), 0, w)
            y0, y1 = np.clip(np.rint((np.array([y0, y1]) - pad_y) / scale).astype(int), 0, h)
            if x0 < x1 and y0 < y1:
                picked.append(((int(x0), int(y0), int(x1), int(y1), float(conf), int(label)), int(index)))
        if len(picked) > (count if count is not None else 1):
            xywh = [(x0, y0, x1 - x0, y1 - y0) for (x0, y0, x1, y1, _, _), _ in picked]
            confidence = [box[4] for box, _ in picked]
            keep = cv2.dnn.NMSBoxes(xywh, confidence, 0.0, 0.5)
            picked = [picked[int(i)] for i in keep]
        picked.sort(key=lambda item: item[0][0])
        boxes = [box for box, _ in picked]
        masks = []
        if not picked or (count is not None and len(picked) != count):
            return masks, boxes
        if self.prototype:
            mask_logits = self.decode_masks([index for _, index in picked])
        for index, (x0, y0, x1, y1, _, _) in enumerate(boxes):
            mask = np.zeros((h, w), np.uint8)
            if self.prototype:
                logits = cv2.resize(mask_logits[index], (self.width, self.height))
                unpadded = logits[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
                logits = cv2.resize(unpadded, (w, h))
                mask[y0:y1, x0:x1] = logits[y0:y1, x0:x1] > 0
                if not np.any(mask) and count is not None:
                    return [], boxes
            else:
                mask[y0:y1, x0:x1] = 1
            masks.append(mask)
        if count is None:
            valid = [(mask, box) for mask, box in zip(masks, boxes, strict=True) if np.any(mask)]
            return [mask for mask, _ in valid], [box for _, box in valid]
        return masks, boxes

    @torch.inference_mode()
    def decode_masks(self, selected_indices):
        with torch.cuda.stream(self.stream):
            proto = self.buffers[self.prototype][0].float()
            indices = torch.tensor(selected_indices, dtype=torch.int64, device="cuda")
            coeffs = self.buffers[self.detections][0, :, 6:].index_select(0, indices).float()
            mask_logits = (coeffs[:, :, None, None] * proto[None]).sum(dim=1)
            return mask_logits.cpu().numpy()

    def warmup(self):
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.infer(blank, labels=[0], score=1.1, count=1)
        if self.prototype:
            self.decode_masks([0])


def build(onnx_path, plan_path):
    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"Could not build TensorRT engine for {onnx_path}")
    plan_path.write_bytes(bytes(engine))
    print(f"Built {plan_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    build(args.onnx, args.engine)
