import os
import shutil
import subprocess
import json

# —— 配置区 ——  
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ORIGIN_DIR = os.path.join(SCRIPT_DIR, "originframe")
RIFE_SCRIPT = os.path.join(SCRIPT_DIR, "inference_video.py")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "train_log")

VIEWS    = ["A", "B", "C", "D"]
TIME_MAP = {"A": 0.0, "B": 0.3, "C": 0.6, "D": 1.0}

EXP   = 2
SEG   = 2**EXP
N_IN  = len(VIEWS)
N_OUT = (N_IN - 1) * SEG + 1  # 65

TMP_DIR   = os.path.join(SCRIPT_DIR, "tmp_interp")
FINAL_DIR = os.path.join(SCRIPT_DIR, "FINAL")

def main():
    # —— 1. 读取原始 transforms.json ——  
    with open(os.path.join(ORIGIN_DIR, "A", "transforms.json"), "r") as f:
        tfA = json.load(f)
    CAMERA_ANGLE_X = tfA["camera_angle_x"]
    info_map = {
        e["file_path"].split("/")[-1] + ".png": {
            "rotation": e["rotation"],
            "transform_matrix": e["transform_matrix"]
        }
        for e in tfA["frames"]
    }

    # —— 2. 列出所有原始帧 ——  
    frames = sorted(
        x for x in os.listdir(os.path.join(ORIGIN_DIR, "A"))
        if x.endswith(".png")
    )

    # —— 3. 清理 & 创建目录 ——  
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    shutil.rmtree(FINAL_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 用于收集每个输出时间点的 transforms
    trans_by_time = {i: [] for i in range(N_OUT)}

    # —— 4. 主循环 ——  
    for frame in frames:
        print(f"[Processing] {frame}")
        # 创建临时子目录
        sub = os.path.join(TMP_DIR, frame[:-4])
        os.makedirs(sub, exist_ok=True)

        # 拷贝 A–D 四视角图到 sub/，重命名为 000.png … 003.png
        for idx, v in enumerate(VIEWS):
            src = os.path.join(ORIGIN_DIR, v, frame)
            dst = os.path.join(sub, f"{idx:03d}.png")
            shutil.copy(src, dst)

        # 调用 inference_video.py，每次都会重新加载模型
        cmd = [
            "python", RIFE_SCRIPT,
            "--exp", str(EXP),
            "--img", ".",
            "--model", MODEL_DIR
        ]
        subprocess.run(cmd, cwd=sub, check=True)

        # 插帧后结果 in sub/vid_out/*.png
        out_dir = os.path.join(sub, "vid_out")
        outs = sorted(fn for fn in os.listdir(out_dir) if fn.endswith(".png"))
        if len(outs) != N_OUT:
            raise RuntimeError(f"{frame}: got {len(outs)} frames, expected {N_OUT}")

        # 计算线性插值时间序列
        t_in = [TIME_MAP[v] for v in VIEWS]
        times = []
        for i in range(N_IN - 1):
            t0, t1 = t_in[i], t_in[i + 1]
            for s in range(SEG):
                times.append(t0 + (t1 - t0) * (s / SEG))
        times.append(t_in[-1])

        # 拷贝到 FINAL/ 并记录 transforms
        rot, mat = info_map[frame].values()
        for k, fn in enumerate(outs):
            tgt = os.path.join(FINAL_DIR, f"{k:03d}")
            os.makedirs(tgt, exist_ok=True)
            shutil.copy(os.path.join(out_dir, fn), os.path.join(tgt, frame))
            trans_by_time[k].append({
                "file_path":        f"./{k:03d}/{frame[:-4]}",
                "rotation":         rot,
                "time":             times[k],
                "transform_matrix": mat
            })

        # —— 中途删除本帧的临时目录 ——  
        shutil.rmtree(sub, ignore_errors=True)

    # —— 5. 写出 transforms_XXX.json ——  
    for k in range(N_OUT):
        data = {
            "camera_angle_x": CAMERA_ANGLE_X,
            "frames":         trans_by_time[k]
        }
        with open(os.path.join(FINAL_DIR, f"transforms_{k:03d}.json"), "w") as f:
            json.dump(data, f, indent=4)

    print("Done! Results saved in:", FINAL_DIR)

if __name__ == "__main__":
    main()