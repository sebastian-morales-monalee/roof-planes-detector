import json
import sys
from pathlib import Path

import bpy


run_directory = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
document = json.loads((run_directory / "roof_planes_multiview.json").read_text())
minimum_visibility = document["parameters"]["min_view_visibility"]

expected_colors = {
    plane["id"]: tuple(plane["display_color_rgb"])
    for plane in document["roof_planes"]
}

for view_name in document["view"]["views"]:
    expected_ids = {
        plane["id"]
        for plane in document["roof_planes"]
        if plane["visibility_by_view"][view_name] >= minimum_visibility
    }
    path = run_directory / f"roof_planes_from_{view_name}_overlay.glb"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = {
        obj.name: obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.startswith("roof_plane_")
    }
    assert set(objects) == expected_ids, (view_name, set(objects) ^ expected_ids)
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    assert len(cameras) == 1 and cameras[0].name == f"view_from_{view_name}"
    for identifier, obj in objects.items():
        actual = tuple(obj.data.materials[0].diffuse_color[:3])
        expected = expected_colors[identifier]
        assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-5
    print("VIEW_OK", view_name, len(objects), cameras[0].name)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(run_directory / "roof_planes_all_only.glb"))
all_ids = {
    obj.name
    for obj in bpy.context.scene.objects
    if obj.type == "MESH" and obj.name.startswith("roof_plane_")
}
assert all_ids == set(expected_colors)
assert sum(obj.type == "CAMERA" for obj in bpy.context.scene.objects) == 6
print("ALL_ONLY_OK", len(all_ids))
