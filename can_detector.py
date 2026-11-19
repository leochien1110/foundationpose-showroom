"""Detect a red-labeled upright can using its color, depth and measured size."""
import cv2
import numpy as np


def detect_can(rgb, depth, K, height_m):
    hue, saturation, value = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    red = ((hue < 12) | (hue >= 167)) & (saturation > 100) & (value > 65)
    red &= (depth > 0.18) & (depth < 0.9)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    if len(stats) == 1:
        return [], []
    seed = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[seed, cv2.CC_STAT_AREA] < 100:
        return [], []
    seed_y, seed_x = np.where(labels == seed)
    front_z = float(np.median(depth[seed_y, seed_x]))
    radius = height_m * (0.33114 / 0.842159) / 2
    axis_z = front_z + radius
    width = K[0, 0] * 2 * radius / axis_z
    height = K[1, 1] * height_m / axis_z
    seed_cx, seed_cy = np.median(seed_x), np.median(seed_y)
    h, w = depth.shape
    desk_near = (depth > front_z - 0.014) & (depth < front_z + 0.020)
    y0 = max(0, int(seed_cy - 0.95 * height))
    y1 = min(h, int(seed_cy + 0.95 * height))
    x0 = max(0, int(seed_cx - 0.80 * width))
    x1 = min(w, int(seed_cx + 0.80 * width))
    window = desk_near[y0:y1, x0:x1].astype(np.uint8)
    window = cv2.morphologyEx(window, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(window)
    in_window = (seed_y >= y0) & (seed_y < y1) & (seed_x >= x0) & (seed_x < x1)
    votes = components[seed_y[in_window] - y0, seed_x[in_window] - x0]
    if not np.any(votes):
        return [], []
    component = 1 + int(np.argmax(np.bincount(votes, minlength=len(stats))[1:]))
    candidate = components == component
    filled_rows = np.flatnonzero(candidate.sum(axis=1) >= 0.35 * width)
    if not len(filled_rows) or np.count_nonzero(candidate) < 0.3 * width * height:
        return [], []
    top = int(filled_rows[0])
    wall = candidate[top + int(0.15 * height):top + int(0.6 * height)]
    _, wall_x = np.where(wall)
    if not len(wall_x):
        return [], []
    center = float(np.median(wall_x))
    x = np.arange(candidate.shape[1])
    y = np.arange(candidate.shape[0])
    candidate &= abs(x - center)[None, :] <= 0.59 * width
    candidate &= ((y >= top) & (y < top + 1.06 * height))[:, None]
    rows, cols = np.where(candidate)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = candidate
    box = (int(cols.min()) + x0, int(rows.min()) + y0, int(cols.max()) + x0 + 1,
           int(rows.max()) + y0 + 1, None, None)
    return [mask], [box]
