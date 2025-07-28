import os
import glob
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description="自动化推理 + 渲染脚本"
    )
    parser.add_argument(
        '--data_dir', '-d',
        required=True,
        help='数据目录 Z，包含 .ply/.csv/.json 文件'
    )
    parser.add_argument(
        '--model_name', '-m',
        required=True,
        help='模型文件夹名称 Y'
    )
    parser.add_argument(
        '--camera', '-c',
        required=True,
        help='camera 参数 Z'
    )
    args = parser.parse_args()

    # 根目录假设为当前运行目录 4DGaussian
    root = os.getcwd()
    data_dir = os.path.abspath(args.data_dir)
    model_name = args.model_name
    camera = args.camera

    # 1. 查找数据目录下的文件
    ply_files  = glob.glob(os.path.join(data_dir, '*.ply'))
    csv_files  = glob.glob(os.path.join(data_dir, '*.csv'))
    json_files = glob.glob(os.path.join(data_dir, '*.json'))

    if not ply_files:
        print(f"错误：在 {data_dir} 中未找到任何 .ply 文件", file=sys.stderr)
        sys.exit(1)

    init_ply = next(
        (f for f in ply_files if os.path.basename(f).lower().startswith('init')),
        ply_files[0]
    )
    print(f"[INFO] 使用 init_ply_path: {init_ply}")

    if csv_files:
        print("[INFO] CSV 文件：", csv_files)
    else:
        print(f"[WARN] 未发现 CSV 文件", file=sys.stderr)
    if json_files:
        print("[INFO] JSON 文件：", json_files)
    else:
        print(f"[WARN] 未发现 JSON 文件", file=sys.stderr)

    # 2. 推理 infer.py
    infer_py = os.path.join(root, 'my_script', 'infer.py')
    model_path = os.path.join(root, 'my_script', 'outputs', model_name, 'deform_model_final.pth')
    out_dir = os.path.join(root, 'my_script', 'inference_outputs', model_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd_infer = [
        sys.executable, infer_py,
        '--data_dir', data_dir,
        '--init_ply_path', init_ply,
        '--model_path', model_path,
        '--out_dir', out_dir
    ]
    print("[INFO] 运行 infer.py: \n  " + " ".join(cmd_infer))
    if subprocess.run(cmd_infer).returncode != 0:
        sys.exit(1)

    # 3. 渲染 custom_render.py
    custom_render_py = os.path.join(root, 'custom_render.py')
    model_path_render = os.path.join(root, 'output', 'dnerf', model_name)
    source_path = os.path.join(root, 'data', 'dnerf', model_name)
    ply_dir = os.path.join(root, 'my_script', 'inference_outputs', model_name, 'objects_world')

    cmd_render = [
        sys.executable, custom_render_py,
        '--model_path', model_path_render,
        '--source_path', source_path,
        '--ply_dir', ply_dir,
        '--camera', camera,
        '--model_name', model_name,
    ]
    print("[INFO] 运行 custom_render.py: \n  " + " ".join(cmd_render))
    if subprocess.run(cmd_render).returncode != 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
