"""Plot detector throughput and can localization on the RealSense fixture."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.results.glob("*.json"):
        data = json.loads(path.read_text())
        if "usable" in data:
            rows.append(dict(candidate=path.stem, **{key: data[key] for key in
                             ("usable", "n", "p50_ms", "p95_ms", "process_after_trials_mib")}))
    rows.sort(key=lambda row: (-row["usable"], row["p50_ms"]))
    with (args.results / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, (latency, hits) = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True,
                                       layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle("Can detection on 21 paired RealSense D435 frames", fontsize=15)
    indices = np.arange(len(rows))
    colors = ["#127a6d" if row["usable"] == row["n"] else "#8095ad" for row in rows]
    names = [row["candidate"].replace("yoloe-26n-seg-text", "YOLOE-26n text")
             .replace("yoloe-26n-seg", "YOLOE-26n visual").replace("yolo26n", "YOLO26n")
             .replace("rgbd-can", "RGB-D can").replace("sdetr", "SyntheticaDETR")
             for row in rows]
    latency.barh(indices, [r["p50_ms"] for r in rows], color=colors)
    for i, row in enumerate(rows):
        latency.plot([row["p50_ms"], row["p95_ms"]], [i, i], color="#1e2f40")
        latency.plot(row["p95_ms"], i, "|", color="#1e2f40", markersize=8)
        latency.annotate(f'{row["p50_ms"]:.2f}', (row["p95_ms"], i), xytext=(5, 0),
                         textcoords="offset points", va="center")
    latency.set(xlabel="Host frame to mask (ms) · median and p95", yticks=indices, yticklabels=names)
    latency.margins(x=0.19)
    hits.barh(indices, [r["usable"] for r in rows], color=colors)
    for i, row in enumerate(rows):
        hits.annotate(f'{row["usable"]}/{row["n"]}', (row["usable"], i),
                      xytext=(5, 0), textcoords="offset points", va="center")
    hits.set(xlabel="Usable detections", xlim=(0, max(row["n"] for row in rows) * 1.25))
    latency.invert_yaxis()
    for ax in (latency, hits):
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.20)
    fig.savefig(args.results / "detector-comparison.png", dpi=160)
    fig.savefig(args.results / "detector-comparison.svg")


if __name__ == "__main__":
    main()
