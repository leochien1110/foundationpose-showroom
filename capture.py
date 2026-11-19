"""Record aligned RealSense RGB-D at calibrated camera intrinsics."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time

import cv2
import numpy as np


@contextmanager
def open_camera():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(45):
            pipeline.wait_for_frames()

        def next_frame():
            started = time.perf_counter()
            frames = align.process(pipeline.wait_for_frames())
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("Missing color or depth")
            intr = color.profile.as_video_stream_profile().intrinsics
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float32)
            rgb = np.asarray(color.get_data()).copy()
            depth_m = np.asarray(depth.get_data()).astype(np.float32) * scale
            return rgb, depth_m, K, color.get_timestamp(), (time.perf_counter() - started) * 1000

        yield next_frame, device.get_info(rs.camera_info.firmware_version)
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=21)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_camera() as (next_frame, firmware):
        images, depths, intrinsics, timestamps, host_ms = zip(*(next_frame() for _ in range(args.frames)))
        np.savez_compressed(args.output, rgb=images, depth=depths, K=intrinsics[0],
                            timestamp_ms=timestamps, capture_host_ms=host_ms, firmware=firmware)
        cv2.imwrite(str(args.output.with_suffix(".png")), images[0][..., ::-1])
    print(f"Captured {len(images)} frames, K={intrinsics[0].tolist()}, firmware={firmware}")


if __name__ == "__main__":
    main()
