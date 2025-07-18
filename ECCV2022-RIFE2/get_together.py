#!/usr/bin/env python3
import os, json, shutil

# —— 配置区 ——  
ROOT_DIR    = os.path.dirname(os.path.realpath(__file__))
FINAL_DIR   = os.path.join(ROOT_DIR, "FINAL")   # 含 000…016 子文件夹
SPLITS_DIR  = os.path.join(ROOT_DIR, "SPLITS")

# 划分规则：idx % 10 == 0 → test; ==9 → val; else → train
SPLITS      = ["test","train","val"]

# 1) 创建输出目录
for sp in SPLITS:
    os.makedirs(os.path.join(SPLITS_DIR, sp), exist_ok=True)

# 2) 读取 camera_angle_x（从第一个 transforms 文件）
time_folders = sorted(fn for fn in os.listdir(FINAL_DIR) if fn.isdigit())
first_tf = os.path.join(FINAL_DIR, f"transforms_{time_folders[0]}.json")
with open(first_tf, "r") as f:
    camera_angle_x = json.load(f)["camera_angle_x"]

# 3) 初始化各 split 的计数器与 transforms 列表
counters      = {sp: 0   for sp in SPLITS}
transforms_sp = {sp: []  for sp in SPLITS}

# 4) 遍历每个时间文件夹
for tf in time_folders:
    # 读取该时间的 transforms 文件
    tf_path = os.path.join(FINAL_DIR, f"transforms_{tf}.json")
    with open(tf_path, "r") as f:
        tf_data = json.load(f)
    # 构建：basename.png → entry
    frame_map = {
        entry["file_path"].split("/")[-1] + ".png": entry
        for entry in tf_data["frames"]
    }

    src_dir = os.path.join(FINAL_DIR, tf)
    images  = sorted(fn for fn in os.listdir(src_dir) if fn.lower().endswith(".png"))

    # 对每张图按 idx % 10 划分
    for idx, img_name in enumerate(images):
        mod = idx % 10
        if   mod == 0: split = "test"
        elif mod == 9: split = "val"
        else:           split = "train"

        # 新文件名
        cnt      = counters[split]
        new_name = f"r_{cnt:03d}.png"

        # 复制图片
        shutil.copy(
            os.path.join(src_dir, img_name),
            os.path.join(SPLITS_DIR, split, new_name)
        )
        counters[split] += 1

        # 添加 transform 条目
        e = frame_map[img_name]
        transforms_sp[split].append({
            "file_path":        f"./{split}/{new_name[:-4]}",
            "rotation":         e["rotation"],
            "time":             e["time"],
            "transform_matrix": e["transform_matrix"]
        })

# 5) 按 time 排序并写出 transforms_{split}.json
for sp in SPLITS:
    transforms_sp[sp].sort(key=lambda x: x["time"])
    out = {
        "camera_angle_x": camera_angle_x,
        "frames":         transforms_sp[sp]
    }
    with open(
        os.path.join(SPLITS_DIR, f"transforms_{sp}.json"),
        "w"
    ) as f:
        json.dump(out, f, indent=4)

print("Split complete. Results in:", SPLITS_DIR)

# 6) 删除原始 FINAL 文件夹，只保留 SPLITS
shutil.rmtree(FINAL_DIR, ignore_errors=True)
print("Removed original FINAL folder.")