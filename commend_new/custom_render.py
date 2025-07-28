import os
import glob
import argparse
import subprocess
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

def render_sequence(ply_dir, pattern, args):
    safe_state(args.quiet)

    # 参数
    dataset = args
    pipeline = args
    hyper = args

    # 初始化 GaussianModel
    gaussians = GaussianModel(dataset.sh_degree, hyper)

    # 构造 Scene 并选择 camera
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras_all = list(scene.getVideoCameras())
    cams = cameras_all[args.camera: args.camera + 1]
    Ncams = len(cams)
    print(f"[INFO] 选用 camera 索引: {args.camera} ({Ncams} 个摄像头)")

    # 查找 PLY 文件
    ply_paths = sorted(glob.glob(os.path.join(ply_dir, pattern)))
    if not ply_paths:
        raise FileNotFoundError(f"No PLYs match {pattern} in {ply_dir}")
    Np = len(ply_paths)

    # 视角数量
    views = args.views if args.views is not None else Ncams
    views = min(views, Ncams)
    total_frames = Np * views
    print(f"Will render {Np} PLYs × {views} views = {total_frames} frames")

    # 设备 & 背景色
    device = torch.device(dataset.data_device)
    bg_color = torch.tensor(
        [1,1,1] if dataset.white_background else [0,0,0],
        dtype=torch.float32, device=device
    )

    # 输出根目录 video_output/Y
    model_name = args.model_name
    output_root = os.path.join(os.getcwd(), 'video_output', model_name)
    os.makedirs(output_root, exist_ok=True)
    print(f"[INFO] 视频输出目录: {output_root}")

    # 渲染并保存两个 scale_modifier 下的视频
    with torch.no_grad():
        for scale in (1.0, 1.5):
            # 帧缓存目录
            frames_dir = os.path.join(output_root, f"frames_{scale:.1f}")
            os.makedirs(frames_dir, exist_ok=True)

            frame_idx = 0
            cam_idx = 0
            for p, ply in enumerate(ply_paths):
                print(f"\n=== PLY {p+1}/{Np}: {os.path.basename(ply)} ===")
                gaussians.load_ply(ply)
                for v in range(views):
                    cam = cams[cam_idx % Ncams]; cam_idx += 1
                    print(f"  frame {frame_idx+1}/{total_frames}: camera #{(cam_idx-1)%Ncams}")
                    out = render(
                        cam, gaussians, pipeline, bg_color,
                        scaling_modifier=scale,
                        stage="coarse", cam_type=scene.dataset_type
                    )
                    img = (
                        255 * out["render"].cpu().numpy().clip(0,1)
                    ).astype("uint8").transpose(1,2,0)
                    path = os.path.join(frames_dir, f"{frame_idx:05d}.png")
                    imageio.imwrite(path, img)
                    frame_idx += 1

            # 合成视频
            out_video = os.path.join(output_root, f"video_{scale:.1f}.mp4")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(args.fps),
                "-i", os.path.join(frames_dir, "%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-vf", f"scale={args.width}:{args.height}",
                "-crf", str(args.ffmpeg_crf),
                "-preset", args.ffmpeg_preset,
                "-b:v", args.bitrate,
                out_video
            ]
            print("[INFO] Running ffmpeg:", " ".join(ffmpeg_cmd))
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"\n✅ Rendered {frame_idx} frames → {out_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render PLYs via Scene + continuous cycling video_cameras"
    )
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    ModelHiddenParams(parser)
    parser.add_argument(
        "--iteration", type=int, default=0,
        help="加载模型迭代；0表示不加载（只读PLY）"
    )
    parser.add_argument(
        "--ply_dir", type=str,
        default=r'/users/lshou/4DGaussians/my_script/inference_outputs/bend/objects_world',
        help="PLY 文件夹路径"
    )
    parser.add_argument("--pattern", type=str, default="object_*.ply")
    parser.add_argument("--views", type=int, default=10,
                        help="每个PLY渲染视角数")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=1920,
                        help="输出视频宽度")
    parser.add_argument("--height", type=int, default=1080,
                        help="输出视频高度")
    parser.add_argument("--ffmpeg_crf", type=int, default=18,
                        help="FFmpeg CRF，数值越低质量越高")
    parser.add_argument("--ffmpeg_preset", type=str, default="slow",
                        help="FFmpeg preset（如 slow, medium, fast）")
    parser.add_argument("--bitrate", type=str, default="5000k",
                        help="视频码率（如 5000k")
    # 新增参数
    parser.add_argument("--model_name", "-p", required=True,
                        help="模型文件夹名称 Y 用于输出路径")
    parser.add_argument("--camera", "-c", required=True, type=int,
                        help="视频渲染时使用的 camera 索引")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print(">>> using source_path =", args.source_path)
    render_sequence(
        ply_dir=args.ply_dir,
        pattern=args.pattern,
        args=args
    )
