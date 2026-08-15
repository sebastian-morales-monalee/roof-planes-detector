import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


module_path = Path(__file__).resolve().parents[1] / "detect_roof_planes.py"
spec = importlib.util.spec_from_file_location("roof_detector", module_path)
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)

input_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
detector.import_glb(input_path)
data = detector.extract_world_triangles()
vertices = data["vertices"]
triangles = data["triangles"]
diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
flipped = data["normals"][:, 2] < 0
data["normals"][flipped] *= -1
candidates = np.flatnonzero(
    data["valid"] & (data["normals"][:, 2] >= math.cos(math.radians(75.0)))
)
bvh = BVHTree.FromPolygons(
    [Vector(tuple(point)) for point in vertices],
    [tuple(int(value) for value in triangle) for triangle in triangles],
    all_triangles=True,
)
for angle in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    groups = detector.group_candidate_faces(triangles, data["normals"], candidates, angle)
    planes = []
    for group in groups:
        if len(group) < 20:
            continue
        plane = detector.fit_plane(group, data)
        if plane["area"] < 0.08:
            continue
        plane["visibility"] = detector.sky_visibility(plane, data, bvh, diagonal)
        planes.append(plane)
    accepted = [
        plane
        for plane in planes
        if plane["visibility"] >= 0.25 and plane["rms"] <= 0.035
    ]
    accepted.sort(key=lambda plane: plane["area"], reverse=True)
    print(
        "ANGLE",
        angle,
        "groups", len(groups),
        "accepted", len(accepted),
        "faces", sum(len(plane["face_indices"]) for plane in accepted),
        "area", round(sum(plane["area"] for plane in accepted), 4),
        "top", [round(plane["area"], 3) for plane in accepted[:12]],
    )
