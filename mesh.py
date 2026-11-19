"""Convert the scan to one geometry and albedo representation for both runtimes."""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import SimpleMaterial


def load_scan(path, height_m):
    scene = trimesh.load_scene(path, process=False)
    geometries = scene.dump()
    if len(geometries) != 1:
        raise ValueError("Expected a single textured mesh")
    mesh = geometries[0]
    mesh.vertices *= height_m / mesh.extents[1]
    mesh.visual.material = SimpleMaterial(image=mesh.visual.material.baseColorTexture.convert("RGB"))
    return mesh


def export_ply(mesh, path):
    texture = path.with_suffix(".png")
    position = np.column_stack((mesh.vertices, mesh.vertex_normals)).astype("<f4")
    property_names = ("x", "y", "z", "nx", "ny", "nz")
    texture_comment = ""
    if mesh.visual.kind == "texture" and mesh.visual.uv is not None:
        mesh.visual.material.image.save(texture)
        texture_comment = f"comment TextureFile {texture.name}\n"
        vertices = np.column_stack((position, mesh.visual.uv)).astype("<f4")
        property_names += ("texture_u", "texture_v")
        colors = ""
    else:
        rgb = mesh.visual.vertex_colors[:, :3]
        vertices = np.empty(len(position), dtype=[("position_normal", "<f4", (6,)), ("rgb", "u1", (3,))])
        vertices["position_normal"] = position
        vertices["rgb"] = rgb
        colors = "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    faces = np.empty(len(mesh.faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    faces["count"] = 3
    faces["indices"] = mesh.faces
    properties = "".join(f"property float {name}\n" for name in
                         property_names) + colors
    header = (f"ply\nformat binary_little_endian 1.0\n{texture_comment}"
              f"element vertex {len(vertices)}\n{properties}"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertices.tobytes())
        stream.write(faces.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height-m", type=float, required=True)
    args = parser.parse_args()
    mesh = load_scan(args.scan, args.height_m)
    export_ply(mesh, args.output)
    print(f"Mesh extents in meters: {mesh.extents}, triangles: {len(mesh.faces)}")
