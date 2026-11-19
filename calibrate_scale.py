"""Estimate scan scale from a D435 depth crop on the cylindrical wall."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
from scipy.optimize import least_squares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("glb", type=Path)
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--rim-centers-v", type=float, nargs=2, metavar=("TOP", "BOTTOM"))
    args = parser.parse_args()
    x0, y0, x1, y1 = args.roi
    with np.load(args.capture) as capture:
        z = np.median(capture["depth"][:, y0:y1, x0:x1], axis=0)
        K = capture["K"]
    v, u = np.mgrid[y0:y1, x0:x1]
    median = np.median(z[z > 0])
    keep = (z > 0) & (np.abs(z - median) < 0.020)
    x = ((u - K[0, 2]) * z / K[0, 0])[keep][::5]
    y = ((v - K[1, 2]) * z / K[1, 1])[keep][::5]
    z = z[keep][::5]

    def residual(p):
        axis_x, axis_z, radius, slope_x, slope_z = p
        return np.hypot(x - (axis_x + slope_x * y), z - (axis_z + slope_z * y)) - radius

    fit = least_squares(residual, [np.median(x), median + 0.03, 0.03, 0, 0],
                        loss="soft_l1", f_scale=0.001)
    axis_x, axis_z, radius, slope_x, slope_z = fit.x
    with args.glb.open("rb") as stream:
        stream.read(12)
        size, _ = struct.unpack("<II", stream.read(8))
        gltf = json.loads(stream.read(size))
    primitive = gltf["meshes"][0]["primitives"][0]
    accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
    extents = np.subtract(accessor["max"], accessor["min"])
    diameter = float(np.mean(extents[[0, 2]]))
    scale = 2 * radius / diameter
    result = dict(roi=args.roi, glb_extents=extents.tolist(), fit_points=len(x),
                  diameter_m=2 * radius, scale=scale, estimated_height_m=scale * extents[1],
                  median_abs_radial_residual_mm=float(np.median(np.abs(fit.fun)) * 1000))
    if args.rim_centers_v:
        ratios = (np.array(args.rim_centers_v) - K[1, 2]) / K[1, 1]
        levels = ratios * axis_z / (1 - ratios * slope_z)
        result.update(rim_center_pixels_v=args.rim_centers_v,
                      rim_height_m=float(abs(np.diff(levels)[0]) * np.linalg.norm([slope_x, 1, slope_z])))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
