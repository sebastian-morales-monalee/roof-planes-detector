import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


input_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output_dir = Path(sys.argv[sys.argv.index("--") + 2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(input_path))

meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
minimum = Vector(min(point[i] for point in points) for i in range(3))
maximum = Vector(max(point[i] for point in points) for i in range(3))
center = (minimum + maximum) * 0.5
dimensions = maximum - minimum
radius = max(dimensions) * 0.75

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = max(dimensions) * 1.25
bpy.context.scene.camera = camera


def point_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


light_specs = [
    ((1.2, -1.4, 1.8), 1100, 5.0),
    ((-1.5, -0.5, 0.8), 700, 4.0),
    ((0.2, 1.5, 1.3), 850, 4.0),
]
for offset, energy, size in light_specs:
    bpy.ops.object.light_add(type="AREA")
    light = bpy.context.object
    light.location = center + Vector(offset).normalized() * radius * 2.5
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    point_at(light, center)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 8
scene.render.resolution_x = 480
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.image_settings.color_mode = "RGBA"
scene.world = bpy.data.worlds.new("Analysis World")
scene.world.color = (0.035, 0.035, 0.035)
scene.view_settings.look = "AgX - Medium High Contrast"

views = {
    "isometric_1": (1.5, -1.8, 1.3),
    "isometric_2": (-1.5, 1.8, 1.3),
    "top": (0.0, 0.0, 2.5),
}

for name, direction in views.items():
    camera.location = center + Vector(direction).normalized() * radius * 3.0
    point_at(camera, center)
    scene.render.filepath = str(output_dir / f"{name}.png")
    bpy.ops.render.render(write_still=True)

print(f"RENDERS_OK={output_dir}")
