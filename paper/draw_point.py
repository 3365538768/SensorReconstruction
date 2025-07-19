import open3d as o3d
import argparse
import time

def display_and_save_ply(ply_path, image_path):
    # 加载点云
    pcd = o3d.io.read_point_cloud(ply_path)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Viewer')

    # 添加点云到窗口
    vis.add_geometry(pcd)

    # 运行几帧刷新画面（必要）
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    # 截图保存
    vis.capture_screen_image(image_path)
    print(f"保存截图到: {image_path}")

    # 保持窗口打开直到用户关闭
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="显示并保存PLY点云截图")
    parser.add_argument("--ply", type=str, required=True, help="输入的PLY文件路径")
    parser.add_argument("--out", type=str, default="screenshot.png", help="输出图片路径")
    args = parser.parse_args()

    display_and_save_ply(args.ply, args.out)