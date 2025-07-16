import os
import glob
import argparse
import subprocess
import math

import torch
import imageio

from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import (
    ModelParams,
    PipelineParams,
    ModelHiddenParams,
    get_combined_args
)
from utils.general_utils import safe_state

def render_sequence(ply_dir, pattern, out_video, args):
    safe_state(args.quiet)

    # 1) 全部参数都在 args 里
    dataset  = args
    pipeline = args
    hyper    = args

    # 2) 初始化 GaussianModel
    gaussians = GaussianModel(dataset.sh_degree, hyper)

    # 3) 用 Scene 构造 GaussianModel 所需状态，但不直接用它的视频相机列表
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = list(scene.getVideoCameras())
    Ncams   = len(cameras)

    # 4) 找到所有 PLY
    ply_paths = sorted(glob.glob(os.path.join(args.ply_dir, pattern)))
    Np = len(ply_paths)
    if Np == 0:
        raise FileNotFoundError(f"No PLYs match {pattern} in {args.ply_dir}")

    # 5) 每个 PLY 渲染多少视角
    views = args.views if args.views is not None else Ncams
    if views > Ncams:
        print(f"Warning: requested {views} views but only {Ncams} cameras available, using {Ncams}")
        views = Ncams

    total_frames = Np * views
    print(f"Will render {Np} PLYs × {views} views = {total_frames} frames")

    # 6) 准备 device & 背景色
    device   = torch.device(dataset.data_device)
    bg_color = torch.tensor(
        [1,1,1] if dataset.white_background else [0,0,0],
        dtype=torch.float32, device=device
    )

    # 7) 如果需要，可以在此统一修改 FOV / radius_scale / scale_modifier
    #    （同之前脚本，略）

    # 8) 创建临时帧文件夹
    frames_dir = os.path.splitext(out_video)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # 9) 渲染循环：用全局 cam_idx 连续取相机
    frame_idx = 0
    cam_idx = 0
    with torch.no_grad():
        for p, ply in enumerate(ply_paths):
            print(f"\n=== PLY {p+1}/{Np}: {os.path.basename(ply)} ===")
            gaussians.load_ply(ply)

            for v in range(views):
                # 连续循环读取相机列表
                cam = cameras[cam_idx % Ncams]
                cam_idx += 1
                print(f"  frame {frame_idx+1}/{total_frames}: using camera #{(cam_idx-1)%Ncams}")

                out = render(
                    cam,
                    gaussians,
                    pipeline,
                    bg_color,
                    scaling_modifier=args.scale_modifier,
                    stage="coarse",
                    cam_type=scene.dataset_type
                )

                img = (255 * out["render"].cpu().numpy().clip(0,1)
                       ).astype("uint8").transpose(1,2,0)
                path = os.path.join(frames_dir, f"{frame_idx:05d}.png")
                imageio.imwrite(path, img)
                frame_idx += 1

    # 10) 用 ffmpeg 合成视频
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_video
    ], check=True)
    print(f"\n✅ Rendered {frame_idx} frames → {out_video}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Render PLYs via Scene + continuous cycling video_cameras"
    )
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    ModelHiddenParams(parser)
    # 加上 iteration
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="加载训练模型的迭代；不指定或 None 表示不加载（只读 PLY）"
    )

    parser.add_argument("--ply_dir", type=str, default=r'E:\notre_dame_project\data\objects_world_3',
                        help="PLY 文件夹路径")
    parser.add_argument("--pattern", type=str, default="object_*.ply")
    parser.add_argument("--views", type=int, default=10,
                        help="每个 PLY 渲染多少个视角")
    parser.add_argument("--scale_modifier", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=str, default="out.mp4")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    render_sequence(
        ply_dir   = args.ply_dir,
        pattern   = args.pattern,
        out_video = args.out,
        args      = args
    )