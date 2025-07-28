import os
import sys
import subprocess

def main():
    # 参数检查
    if len(sys.argv) != 2:
        print("Usage: python autoprocess2.py <exp_name>", file=sys.stderr)
        sys.exit(1)
    exp = sys.argv[1]

    # 根目录 & my_script 目录
    root = os.getcwd()
    script_dir = os.path.join(root, "my_script")
    train_py = os.path.join(script_dir, "train.py")

    if not os.path.isfile(train_py):
        sys.exit(f"Error: train.py not found in {script_dir}")

    # 自动探测 GPU 数
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        # 如果没有安装 torch，退回到使用 nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True
            )
            num_gpus = len(result.stdout.strip().splitlines())
        except Exception:
            num_gpus = 1

    if num_gpus < 1:
        num_gpus = 1

    # 构造命令
    data_dir   = os.path.join("data", exp)
    output_dir = os.path.join("outputs", exp)
    cmd = [
        sys.executable,
        train_py,
        "--data_dir",   data_dir,
        "--out_dir", output_dir,
        "--num_workers", str(num_gpus)
    ]

    print(f"→ Running train.py with {num_gpus} workers:")
    print("  " + " ".join(cmd))
    # 在 my_script 目录下执行
    subprocess.run(cmd, cwd=script_dir, check=True)

if __name__ == "__main__":
    main()