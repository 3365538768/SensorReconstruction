import os, re, sys, subprocess, shutil

# 1. 参数检查 & 可选 skip_interp
if len(sys.argv) < 2:
    print("Usage: python auto_process.py <exp_name> [--skip_interp]", file=sys.stderr)
    sys.exit(1)
exp = sys.argv[1]
skip_interp = "--skip_interp" in sys.argv or "--skip-interp" in sys.argv

# 2. 路径设置
root     = os.getcwd()  # 假设你已 cd 到 4DGaussians
rife_dir = os.path.join(root, "ECCV2022-RIFE")
origin   = os.path.join(rife_dir, "originframe")

if not os.path.isdir(origin):
    sys.exit(f"Error: 找不到目录 {origin}")

# 3. 收集视角名（优先子文件夹，回退到文件名）
items = sorted(os.listdir(origin))
names = [d for d in items if os.path.isdir(os.path.join(origin, d))]
if not names:
    files = [f for f in items if os.path.isfile(os.path.join(origin, f))]
    names = [os.path.splitext(f)[0] for f in sorted(files)]
if not names:
    sys.exit(f"Error: {origin} 下没有可用的子文件夹或文件")

# 4. 构造 VIEWS & TIME_MAP 字符串
views_str = "[" + ",".join(f'"{n}"' for n in names) + "]"
N = len(names)
time_entries = [
    f'"{n}":{i/(N-1) if N>1 else 0.0:.6f}'
    for i, n in enumerate(names)
]
time_map_str = "{" + ",".join(time_entries) + "}"

print("→ VIEWS   =", views_str)
print("→ TIME_MAP=", time_map_str)

# 5. 替换 morepipeline.py 中的定义
mp = os.path.join(rife_dir, "morepipeline.py")
with open(mp, 'r', encoding='utf-8') as f:
    txt = f.read()
txt = re.sub(r'^\s*VIEWS\s*=.*$',     f"VIEWS = {views_str}",     txt, flags=re.M)
txt = re.sub(r'^\s*TIME_MAP\s*=.*$',  f"TIME_MAP = {time_map_str}", txt, flags=re.M)
with open(mp, 'w', encoding='utf-8') as f:
    f.write(txt)

# 6. 运行 morepipeline.py（根据 skip_interp 决定是否附加参数）
cmd = [sys.executable, mp]
if skip_interp:
    cmd.append("--skip_interp")
    print("→ skip_interp enabled, adding --skip_interp")
print("→ Running ECCV2022-RIFE/morepipeline.py …")
subprocess.run(cmd, cwd=rife_dir, check=True)

# # 7. 运行 get_together.py
rige_dir = os.path.join(root, "ECCV2022-RIFE")
gt = os.path.join(rige_dir, "get_together.py")
print("→ Running get_together.py …")
subprocess.run([sys.executable, gt],
               cwd=rige_dir, check=True)

# 8. 移动并重命名 SPLITS
src_s      = os.path.join(rige_dir, "SPLITS")
dst_parent = os.path.join(root, "data", "dnerf")
os.makedirs(dst_parent, exist_ok=True)
dst_s      = os.path.join(dst_parent, exp)
if not os.path.isdir(src_s):
    sys.exit(f"Error: 未找到生成的 SPLITS 文件夹 ({src_s})")
if os.path.exists(dst_s):
    shutil.rmtree(dst_s)
shutil.move(src_s, dst_s)

# 9. 训练
print("→ Training with train.py …")
subprocess.run([
    sys.executable,
    os.path.join(root, "train.py"),
    "-s", f"data/dnerf/{exp}",
    "--port", "6017",
    "--expname", f"dnerf/{exp}",
    "--configs", "arguments/dnerf/jumpingjacks.py"
], cwd=root, check=True)

# 10. 渲染
print("→ Rendering with render.py …")
subprocess.run([
    sys.executable,
    os.path.join(root, "render.py"),
    "--model_path", f"output/dnerf/{exp}",
    "--configs", "arguments/dnerf/jumpingjacks.py"
], cwd=root, check=True)

# 11. 导出 per-frame 3DGS
print("→ Exporting frames with export_perframe_3DGS.py …")
subprocess.run([
    sys.executable,
    os.path.join(root, "export_perframe_3DGS.py"),
    "--iteration", "100",
    "--configs", "arguments/dnerf/jumpingjacks.py",
    "--model_path", f"output/dnerf/{exp}"
], cwd=root, check=True)

# 12. 抽取移动点
print("Extracting move points …")
frames_dir = os.path.join(root, "my_script", "data", exp, "frames")
os.makedirs(frames_dir, exist_ok=True)
subprocess.run([
    sys.executable,
    "get_movepoint.py",
    "--input_dir", os.path.join(root, "output", "dnerf", exp, "gaussian_pertimestamp"),
    "--output_dir", frames_dir,
    "--percent", "0.2"
], cwd=os.path.join(root, "my_script"), check=True)

print("→ All steps completed! Waiting for sensor.csv and region.json")