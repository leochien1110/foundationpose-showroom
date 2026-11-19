"""Generate a compact mesh for the dashboard without changing pose inputs."""
import numpy as np


def display_mesh(mesh, face_limit=2500):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) > face_limit:
        origin = vertices.min(axis=0)
        extent = max(np.ptp(vertices, axis=0))
        for bins in (36, 27, 21, 16, 12, 8, 6, 4):
            grid = np.floor((vertices - origin) * (bins / extent)).astype(np.int32)
            _, inverse = np.unique(grid, axis=0, return_inverse=True)
            count = np.bincount(inverse)
            reduced = np.column_stack([np.bincount(inverse, weights=vertices[:, i]) / count
                                       for i in range(3)])
            tris = inverse[faces]
            a, b, c = tris.T
            tris = tris[(a != b) & (b != c) & (a != c)]
            _, unique = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
            triangles = tris[unique]
            if 4 <= len(triangles) <= face_limit:
                vertices, faces = reduced, triangles
                break
        else:
            hull = mesh.convex_hull
            vertices, faces = hull.vertices, hull.faces
    return dict(vertices=np.round(np.asarray(vertices), 5).tolist(), faces=np.asarray(faces).tolist())


def detection_centers(masks, boxes, depth, intrinsics):
    points = []
    for mask, box in zip(masks, boxes, strict=True):
        ys, xs = np.where((mask > 0) & np.isfinite(depth) & (depth > 0))
        if not len(xs):
            points.append(None)
            continue
        z = float(np.median(depth[ys, xs]))
        x, y = float(np.median(xs)), float(np.median(ys))
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        points.append(dict(position=[(x - cx) * z / fx, (y - cy) * z / fy, z], score=box[4]))
    return points
