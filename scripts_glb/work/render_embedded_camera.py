import sys
from pathlib import Path

import bpy
from mathutils import Vector


input_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output_path = Path(sys.argv[sys.argv.index("--") + 2]).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(input_path))

cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
if not cameras:
    raise RuntimeError("El GLB no contiene una cámara.")
camera = cameras[0]
bpy.context.scene.camera = camera

meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
minimum = Vector(min(point[i] for point in points) for i in range(3))
maximum = Vector(max(point[i] for point in points) for i in range(3))
center = (minimum + maximum) * 0.5
radius = max(maximum - minimum) * 0.75


def point_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


for offset, energy, size in (
    ((1.2, -1.4, 1.8), 1100, 5.0),
    ((-1.5, -0.5, 0.8), 700, 4.0),
    ((0.2, 1.5, 1.3), 850, 4.0),
):
    bpy.ops.object.light_add(type="AREA")
    light = bpy.context.object
    light.location = center + Vector(offset).normalized() * radius * 2.5
    light.data.energy = energy
    light.data.size = size
    point_at(light, center)

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 8
scene.render.resolution_x = 720
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world = bpy.data.worlds.new("Analysis World")
scene.world.color = (0.035, 0.035, 0.035)
scene.view_settings.look = "AgX - Medium High Contrast"
scene.render.filepath = str(output_path)
bpy.ops.render.render(write_still=True)
print(f"RENDER_OK={output_path}")
