"""Persistent library of normalized meshes for the dashboard."""
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import uuid

import numpy as np
import trimesh
from trimesh.visual.material import SimpleMaterial

from mesh import export_ply

TYPES = {".glb", ".obj", ".ply", ".stl"}
AXES = {"X": 0, "Y": 1, "Z": 2}
MAX_BYTES = 40 * 1024 * 1024
DEFAULT_MESH = {"id": "coke-can", "name": "Coke can scan", "height_mm": 145, "source": "Built in"}


def normalize_geometry(data, suffix, height_mm, axis):
    scene = trimesh.load_scene(BytesIO(data), file_type=suffix[1:], process=False,
                               allow_remote=False)
    parts = [part for part in scene.dump() if isinstance(part, trimesh.Trimesh) and len(part.faces)]
    if not parts:
        raise ValueError("The file does not contain a triangle mesh")
    for part in parts:
        material = getattr(part.visual, "material", None)
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if part.visual.kind == "texture":
            if image is None or part.visual.uv is None:
                part.visual = trimesh.visual.ColorVisuals(mesh=part)
            else:
                part.visual.material = SimpleMaterial(image=image.convert("RGB"))
    if len(parts) == 1:
        mesh = parts[0]
    else:
        for part in parts:
            if part.visual.kind == "texture":
                part.visual = part.visual.to_color()
        mesh = trimesh.util.concatenate(parts)
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
        raise ValueError("Mesh must have finite vertices and nonzero size on all three axes")
    if height_mm is not None:
        mesh.vertices *= (height_mm / 1000) / mesh.extents[axis]
    return mesh


class MeshLibrary:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def list(self):
        items = [json.loads(path.read_text()) for path in self.folder.glob("*/mesh.json")
                 if (path.parent / "mesh.ply").is_file() and (path.parent / "mesh.glb").is_file()]
        return [dict(DEFAULT_MESH), *sorted(items, key=lambda row: (row["name"].lower(), row["id"]))]

    def paths(self, mesh_id):
        if mesh_id == DEFAULT_MESH["id"]:
            return None
        key = uuid.UUID(mesh_id).hex
        folder = self.folder / key
        if not (folder / "mesh.json").is_file():
            raise ValueError("Selected mesh is missing")
        return folder / "mesh.glb", folder / "mesh.ply"

    def upload(self, name, data, height, axis):
        name = Path(name.replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in TYPES:
            raise ValueError("Upload a GLB, OBJ, PLY, or STL file")
        if axis not in AXES:
            raise ValueError("Choose X, Y, or Z for the mesh height axis")
        height_mm = None if not height else float(height)
        if height_mm is not None and (not math.isfinite(height_mm) or height_mm <= 0):
            raise ValueError("Height must be a positive number in millimeters")
        mesh = normalize_geometry(data, suffix, height_mm, AXES[axis])
        key = uuid.uuid4().hex
        record = dict(id=key, name=name, height_mm=height_mm, source=suffix[1:].upper(),
                      triangles=len(mesh.faces), extents_mm=(mesh.extents * 1000).tolist())
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=self.folder) as temporary:
            staging = Path(temporary) / key
            staging.mkdir()
            mesh.export(staging / "mesh.glb")
            export_ply(mesh, staging / "mesh.ply")
            (staging / "mesh.json").write_text(json.dumps(record))
            staging.rename(self.folder / key)
        return record
