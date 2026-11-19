"""Generate paired latency, VRAM and projected-silhouette artifacts."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import SCAN
from mesh import load_scan

MODES = ("original", "native", "pooled", "pooled-3")
NAMES = dict(zip(MODES, ("Original FP", "FP Infer Lib", "Shared Workspace · 5 refinements",
                         "Shared Workspace · 3 refinements"), strict=True))
COLORS = ("#47637e", "#5078b3", "#15847a", "#14814c")
GIB = 2**30


def summarize(run):
    success = [row for row in run["trials"] if row["status"] == "returned"]
    cold, *warm = success if success else [None]
    row = dict(mode=run["mode"], objects=run["objects"], total=len(run["trials"]),
               returned=len(success), warm=len(warm), setup_ms=run["setup_ms"],
               cold_ms=cold["all_poses_ms"] if cold else None,
               max_process_gib=max(s["process_bytes"] for s in run["memory"]) / GIB,
               max_device_gib=max(s["device_bytes"] for s in run["memory"]) / GIB)
    retained = run.get("tracking_retained_bytes")
    row["track_retained_gib"] = retained / GIB if retained is not None else None
    for field in ("detector_ms", "registration_ms", "first_pose_ms", "all_poses_ms"):
        values = [t[field] for t in warm]
        quantiles = np.percentile(values, [50, 95]).tolist() if values else [None, None]
        row[f"{field}_p50"], row[f"{field}_p95"] = quantiles
    return row


def plot(rows, output, detector):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(rows))
    names = [NAMES[row["mode"]].replace(" · ", "\n") for row in rows]
    methods = [f"{name}\n{row['returned']}/{row['total']} returned" for name, row in zip(names, rows, strict=True)]
    colors = [COLORS[MODES.index(row["mode"])] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), layout="constrained")
    count = rows[0]["objects"]
    objects = "can" if count == 1 else "cans"
    fig.suptitle(f"RealSense D435 · {count} physical Coke {objects} · {detector}", fontsize=15)
    ax = axes[0]
    ax.set_title("Detector to all returned poses · warmed")
    for i, row in enumerate(rows):
        p50, p95 = row["all_poses_ms_p50"], row["all_poses_ms_p95"]
        if p50 is None:
            continue
        ax.bar(i, p50, width=0.52, color=colors[i])
        ax.plot([i, i], [p50, p95], color="#1e2f40")
        ax.plot(i, p95, "_", color="#1e2f40", markersize=12)
        ax.annotate(f"{p50:.1f}", (i, p95), xytext=(0, 14 if 165 <= p95 <= 200 else 6), textcoords="offset points", ha="center")
    ax.set_ylabel("Wall time (ms)")
    ax.margins(y=0.18)
    ax = axes[1]
    ax.set_title("Driver-reported process VRAM")
    for i, row in enumerate(rows):
        peak, retained = row["max_process_gib"], row["track_retained_gib"]
        ax.bar(i - 0.15, peak, width=0.30, color=colors[i], label="Sampled peak" if i == 0 else None)
        ax.annotate(f"{peak:.2f}", (i - 0.15, peak), xytext=(0, 4), textcoords="offset points", ha="center")
        if retained is not None:
            ax.bar(i + 0.15, retained, width=0.30, color=colors[i], alpha=0.38,
                   label="After tracking" if i == 0 else None)
            ax.annotate(f"{retained:.2f}", (i + 0.15, retained), xytext=(0, 4),
                        textcoords="offset points", ha="center")
    ax.set_ylabel("Allocated VRAM (GiB)")
    ax.margins(y=0.20)
    ax.legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.set_xticks(x, methods, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def compare_poses(runs, output):
    by_mode = [{t["frame"]: t for t in r["trials"] if t["status"] == "returned"} for r in runs]
    frames = sorted(set.intersection(*(set(r) for r in by_mode)))
    if not frames:
        return
    modes = [run["mode"] for run in runs]
    baseline = by_mode[modes.index("native") if "native" in modes else 0]
    comparisons = {}
    for name, trials in zip(modes, by_mode, strict=True):
        base, other = (np.array([table[i]["poses"] for i in frames]) for table in (baseline, trials))
        error = np.linalg.norm(other[..., :3, 3] - base[..., :3, 3], axis=-1) * 1000
        rel = base[..., :3, :3].swapaxes(-1, -2) @ other[..., :3, :3]
        angles = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
        angles[np.all(base[..., :3, :3] == other[..., :3, :3], axis=(-2, -1))] = 0
        comparisons[name] = dict(paired_frames=len(frames), translation_mm_median=float(np.median(error)),
                                 translation_mm_max=float(np.max(error)),
                                 rotation_deg_median=float(np.median(angles)),
                                 rotation_deg_max=float(np.max(angles)),
                                 max_elementwise_delta=float(np.abs(other - base).max()))
    (output / "paired_pose_deltas.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    index = frames[len(frames) // 2]
    with np.load(output / "d435.npz") as capture:
        rgb, K = capture["rgb"][index], capture["K"]
    mesh = load_scan(SCAN, runs[0]["height_m"])
    columns = min(2, len(runs))
    rows = (len(runs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.75 * rows),
                             layout="constrained", squeeze=False)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    fig.suptitle(f"Frame {index} · projected scan silhouette and detector box")
    for mode, trials, ax in zip(modes, by_mode, axes.flat[:len(runs)], strict=True):
        ax.imshow(rgb)
        trial = trials[index]
        for box, pose in zip(trial["boxes"], np.array(trial["poses"]), strict=True):
            x0, y0, x1, y1, score, _ = box
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#f7cf5a", linewidth=1)
            cam = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
            pos = cam[cam[:, 2] > 0.01] @ K.T
            if len(pos):
                pixels = (pos[:, :2] / pos[:, 2:]).astype(np.float32)
                hull = cv2.convexHull(pixels).reshape(-1, 2)
                ax.plot(*np.vstack([hull, hull[0]]).T, color="#47ffbc", linewidth=1.4)
            if score is not None:
                ax.text(x0, y0 - 6, f"{score:.2f}", color="#f7cf5a", fontsize=8)
        ax.set(xlim=(0, rgb.shape[1]), ylim=(rgb.shape[0], 0), title=NAMES[mode])
        ax.axis("off")
    fig.savefig(output / "pose-check.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    runs = [json.loads((args.output / f"{mode}.json").read_text()) for mode in args.modes]
    signatures = [[(row["frame"], row["boxes"]) for row in r["trials"]] for r in runs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("Detector boxes differ between implementations")
    rows = [summarize(run) for run in runs]
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    detector = {
        "rgbd-can": "RGB-D can detector",
        "sdetr": "SyntheticaDETR",
        "yolo-trt": str(runs[0].get("engine", "")).removesuffix(".plan") + " · TensorRT FP16",
    }[runs[0].get("detector", "sdetr")]
    plot(rows, args.output, detector)
    compare_poses(runs, args.output)
    print(f"Wrote comparison artifacts to {args.output}")


if __name__ == "__main__":
    main()
