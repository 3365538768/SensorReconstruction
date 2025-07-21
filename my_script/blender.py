import bpy, math, os
from mathutils import Vector
import json

# —— 1. 基本渲染设置 ——
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# —— 2. Workbench Studio + Cavity 模式 ——
wb = scene.display.shading
wb.color_type    = 'TEXTURE'
wb.light         = 'STUDIO'
wb.show_shadows  = False
wb.show_cavity   = True
wb.cavity_type   = 'BOTH'
wb.cavity_ridge_factor  = 1.0
wb.cavity_valley_factor = 1.0

# —— 3. 相机准备 ——
if not scene.camera:
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
camera = scene.camera
camera.data.clip_start = 0.01
camera.data.clip_end   = 1000.0

# —— 4. 禁用所有光源阴影 ——
for light in (o for o in bpy.data.objects if o.type == 'LIGHT'):
    light.data.use_shadow = False

# —— 5. 要渲染的对象列表与对应 time 映射 ——
objects_to_capture = ["A", "B", "C", "D"]
time_map = {
    "A": 0.0,
    "B": 0.4,
    "C": 0.8,
    "D": 1.0,
}
output_root = "D:/4DGaussians/my_script/originframe"

# —— 6. 主循环：逐个对象渲染 ——
for obj_name in objects_to_capture:
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"跳过，未找到对象 {obj_name}")
        continue

    # 隐藏场景中所有网格，只显示当前
    for ob in scene.objects:
        if ob.type == 'MESH':
            ob.hide_render = True
    obj.hide_render = False

    # 计算几何中心与半径
    bbox_pts = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    min_c = Vector((min(p[i] for p in bbox_pts) for i in range(3)))
    max_c = Vector((max(p[i] for p in bbox_pts) for i in range(3)))
    center = (min_c + max_c) * 0.5
    radius = (max_c - min_c).length * 2.0

    # 创建输出目录 & 初始化 transforms
    obj_dir = os.path.join(output_root, obj_name)
    os.makedirs(obj_dir, exist_ok=True)
    transforms = {"camera_angle_x": camera.data.angle_x, "frames": []}

    # 球面视角参数
    lat_angles   = [math.radians(a) for a in [80,60,40,20,10,0,-10,-20,-40,-60,-80]]
    num_per_ring = 6

    # 取出当前对象的 time 值
    t_val = time_map.get(obj_name, 0.0)

    for lat in lat_angles:
        for k in range(num_per_ring):
            az = 2 * math.pi * k / num_per_ring
            pos = Vector((
                center.x + radius * math.cos(lat) * math.cos(az),
                center.y + radius * math.cos(lat) * math.sin(az),
                center.z + radius * math.sin(lat),
            ))
            camera.location = pos
            camera.rotation_euler = (center - pos).to_track_quat('-Z','Y').to_euler()

            # 渲染
            img_name = f"r_{len(transforms['frames']):03d}.png"
            raw = os.path.join(obj_dir, img_name).replace("\\","/")
            scene.render.filepath = raw
            bpy.ops.render.render(write_still=True)

            # 记录 frame（使用映射的 t_val）
            transforms["frames"].append({
                "file_path": f"./{obj_name}/r_{len(transforms['frames']):03d}",
                "rotation": 2 * math.pi / (num_per_ring * len(lat_angles)),
                "time":     t_val,
                "transform_matrix": [list(row) for row in camera.matrix_world]
            })

    # 写入 JSON
    with open(os.path.join(obj_dir, "transforms.json"), 'w') as f:
        json.dump(transforms, f, indent=4)

print("渲染完成，各对象的 time 已按 A–E 映射为 0,0.2,0.4,0.7,1.0。")
