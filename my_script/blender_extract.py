import bpy
import os, math, mathutils

# ========== 配置 ==========
base_path       = "/users/lshou/4DGaussians/data/bag"
model_dir       = base_path
output_dir      = base_path
NUM_VIEWS       = 10
RESOLUTION      = 800
TRANSPARENT_BG  = True
ENGINE          = 'CYCLES'
USE_GPU         = False

# ========== 场景初始化 ==========
scene = bpy.context.scene

# 删除默认立方体
bpy.ops.object.select_all(action='DESELECT')
if "Cube" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)

# 渲染引擎 & 设备
scene.render.engine = ENGINE
if ENGINE == 'CYCLES':
    scene.cycles.device = 'GPU' if USE_GPU else 'CPU'

# 分辨率 & 格式
scene.render.resolution_x       = RESOLUTION
scene.render.resolution_y       = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode  = 'RGBA'
scene.render.film_transparent         = TRANSPARENT_BG

# ========== 导入模型 ==========
imported_objs = []
for f in os.listdir(model_dir):
    if f.lower().endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=os.path.join(model_dir, f))
# 收集所有 Mesh
imported_objs = [o for o in scene.objects if o.type == 'MESH']
if not imported_objs:
    raise RuntimeError("未找到任何 OBJ 模型")

# ========== 几何中心平移 ==========
# 计算整体包围盒中心
min_c = mathutils.Vector((1e9,1e9,1e9))
max_c = mathutils.Vector((-1e9,-1e9,-1e9))
for obj in imported_objs:
    bpy.context.view_layer.update()
    for corner in obj.bound_box:
        pt = obj.matrix_world @ mathutils.Vector(corner)
        min_c.x = min(min_c.x, pt.x)
        min_c.y = min(min_c.y, pt.y)
        min_c.z = min(min_c.z, pt.z)
        max_c.x = max(max_c.x, pt.x)
        max_c.y = max(max_c.y, pt.y)
        max_c.z = max(max_c.z, pt.z)
center = (min_c + max_c) / 2.0

# Apply transforms & 平移到原点
for obj in imported_objs:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # 烘焙位置到顶点
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    # 将几何中心移到原点
    obj.location -= center
    obj.select_set(False)

# ========== 相机 & 光照 ==========
# 计算新的边界半径 & 中心
size = max_c - min_c
radius = 0.5 * math.sqrt(size.x**2 + size.y**2 + size.z**2)

cam_target = mathutils.Vector((0,0,0))
cam_radius = radius * 2.0

# 获取或创建摄像机
cam = bpy.data.objects.get("Camera")
if cam is None:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
scene.camera = cam

# 添加光源
# 方向光
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", type='SUN'))
sun.data.energy = 20
sun.rotation_euler = (math.radians(60), 0, math.radians(30))
scene.collection.objects.link(sun)
# 三盏点光
fills = [
    ( cam_target.x + cam_radius, cam_target.y + cam_radius, cam_target.z + cam_radius ),
    ( cam_target.x - cam_radius, cam_target.y + cam_radius, cam_target.z + cam_radius ),
    ( cam_target.x,               cam_target.y - cam_radius, cam_target.z + cam_radius )
]
for i,pos in enumerate(fills):
    pd = bpy.data.lights.new(f"Point{i}", type='POINT')
    pd.energy = 300
    pl = bpy.data.objects.new(f"Point{i}", object_data=pd)
    pl.location = pos
    scene.collection.objects.link(pl)
# 环境光
if scene.world and scene.world.use_nodes:
    bg = scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[1].default_value = 2.0

# ========== 生成采样点 ==========
points = []
phi = (1 + 5**0.5) / 2
for i in range(NUM_VIEWS):
    z = 1 - 2*i/float(NUM_VIEWS-1)
    theta = 2*math.pi * i/phi
    r = math.sqrt(max(0, 1-z*z))
    x,y = math.cos(theta)*r, math.sin(theta)*r
    points.append((x,y,z))

# ========== 渲染 & 划分 ==========
# 创建子目录
for sub in ["train","val","test"]:
    os.makedirs(os.path.join(output_dir,sub), exist_ok=True)

frames = {"train":[], "val":[], "test":[]}
for idx, (x,y,z) in enumerate(points):
    # 相机位置
    cam_loc = cam_target + mathutils.Vector((x,y,z))*cam_radius
    cam.location = cam_loc
    # 朝向中心
    quat = (cam_target-cam_loc).to_track_quat('-Z','Y')
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = quat
    bpy.context.view_layer.update()

    # 划分
    if idx%10==0: subset="test"
    elif idx%10==1: subset="val"
    else: subset="train"

    fn = f"r_{idx:03d}.png"
    scene.render.filepath = os.path.join(output_dir,subset,fn)
    bpy.ops.render.render(write_still=True)

    tm = [list(row) for row in cam.matrix_world]
    frames[subset].append({
        "file_path": f"./{subset}/r_{idx:03d}",
        "transform_matrix": tm
    })

# ========== 写入 JSON ==========
cam_angle = bpy.data.cameras[cam.name].angle_x
for subset in ["train","val","test"]:
    data = {"camera_angle_x": cam_angle, "frames": frames[subset]}
    with open(os.path.join(output_dir, f"transforms_{subset}.json"), "w") as f:
        import json
        json.dump(data, f, indent=4)

print("渲染和 JSON 生成完成.")

