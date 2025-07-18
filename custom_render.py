import os
import glob
import argparse
import subprocess
import math

import torch
import imageio

from my_render import render
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

    # 1) 参数都在 args 里
    dataset  = args
    pipeline = args
    hyper    = args

    # 2) 初始化 GaussianModel
    gaussians = GaussianModel(dataset.sh_degree, hyper)

    # 3) 构造 Scene
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = list(scene.getVideoCameras())[128:129]
    Ncams   = len(cameras)

    # 4) 查找 PLY
    ply_paths = sorted(glob.glob(os.path.join(ply_dir, pattern)))
    if len(ply_paths) == 0:
        raise FileNotFoundError(f"No PLYs match {pattern} in {ply_dir}")
    Np = len(ply_paths)

    # 5) 视角数量
    views = args.views if args.views is not None else Ncams
    views = min(views, Ncams)
    total_frames = Np * views
    print(f"Will render {Np} PLYs × {views} views = {total_frames} frames")

    # 6) 设备 & 背景
    device   = torch.device(dataset.data_device)
    bg_color = torch.tensor(
        [1,1,1] if dataset.white_background else [0,0,0],
        dtype=torch.float32, device=device
    )

    # 7) 可按需调整 FOV/radius_scale/scale_modifier

    # 8) 帧目录
    frames_dir = os.path.splitext(out_video)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # 9) 渲染并保存 PNG
    frame_idx = 0
    cam_idx = 0
    with torch.no_grad():
        for p, ply in enumerate(ply_paths):
            print(f"\n=== PLY {p+1}/{Np}: {os.path.basename(ply)} ===")
            gaussians.load_ply(ply)
            for v in range(views):
                cam = cameras[cam_idx % Ncams]; cam_idx += 1
                print(f"  frame {frame_idx+1}/{total_frames}: camera #{(cam_idx-1)%Ncams}")
                out = render(
                    cam, gaussians, pipeline, bg_color,
                    scaling_modifier=args.scale_modifier,
                    stage="coarse", cam_type=scene.dataset_type
                )
                img = (255 * out["render"].cpu().numpy().clip(0,1)
                       ).astype("uint8").transpose(1,2,0)
                path = os.path.join(frames_dir, f"{frame_idx:05d}.png")
                imageio.imwrite(path, img)
                frame_idx += 1

    # 10) 用 ffmpeg 合成高质量视频
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={args.width}:{args.height}",           # 指定输出分辨率
        "-crf", str(args.ffmpeg_crf),                        # 控制质量，18~23 之间为高质量
        "-preset", args.ffmpeg_preset,                       # 预设（slow, medium, fast）
        "-b:v", args.bitrate,                                # 视频码率
        out_video
    ]
    print("Running ffmpeg:", " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"\n✅ Rendered {frame_idx} frames → {out_video}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Render PLYs via Scene + continuous cycling video_cameras"
    )
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=0,
                        help="加载模型迭代；0表示不加载（只读PLY）")
    parser.add_argument("--ply_dir", type=str, default=r'D:\4DGaussians\my_script\inference_outputs\objects_world_small',
                        help="PLY 文件夹路径")
    parser.add_argument("--pattern", type=str, default="object_*.ply")
    parser.add_argument("--views", type=int, default=10,
                        help="每个PLY渲染视角数")
    parser.add_argument("--scale_modifier", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=10)

    # 新增分辨率与编码质量参数
    parser.add_argument("--width", type=int, default=1920,
                        help="输出视频宽度")
    parser.add_argument("--height", type=int, default=1080,
                        help="输出视频高度")
    parser.add_argument("--ffmpeg_crf", type=int, default=18,
                        help="FFmpeg CRF，数值越低质量越高")
    parser.add_argument("--ffmpeg_preset", type=str, default="slow",
                        help="FFmpeg preset（如 slow, medium, fast）")
    parser.add_argument("--bitrate", type=str, default="5000k",
                        help="视频码率（如 5000k）")

    parser.add_argument("--out", type=str, default="out.mp4")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    render_sequence(
        ply_dir   = args.ply_dir,
        pattern   = args.pattern,
        out_video = args.out,
        args      = args
    )
