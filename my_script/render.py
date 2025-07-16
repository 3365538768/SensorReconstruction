#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_gauss_plys_headless.py

Headless batch renderer for a folder of Gaussian‐sphere PLYs.
输出 PNG 文件，VS Code 可以直接打开预览。
"""

import os, glob
import numpy as np
import imageio
import open3d as o3d

# --- 配置 ---
PLY_DIR    = "/users/lshou/4DGaussians/my_script/inference_output_full_run/point_cloud"  # 输入 PLY 目录
OUT_DIR    = os.path.join(PLY_DIR, "renders")         # 输出 PNG 目录
os.makedirs(OUT_DIR, exist_ok=True)

# 渲染分辨率
WIDTH, HEIGHT = 1024, 768

# 视角列表：(azimuth, elevation, 半径倍率)
VIEWS = [
    (   0,  30, 2.5),
    (  90,  30, 2.5),
    ( 180,  30, 2.5),
    ( 270,  30, 2.5),
]

def render_one(mesh: o3d.geometry.TriangleMesh, base_name: str):
    # 材质：使用 defaultLit
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    # Offscreen 渲染器
    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    renderer.scene.set_background([1,1,1,1])  # 白色背景
    renderer.scene.add_geometry("mesh", mesh, mat)

    # 计算包围盒中心和半径
    bbox   = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = np.linalg.norm(bbox.get_extent()) / 2.0

    for az, el, mult in VIEWS:
        # 球面坐标计算相机位置
        az_r = np.deg2rad(az)
        el_r = np.deg2rad(el)
        dist = radius * mult
        eye = center + dist * np.array([
            np.cos(el_r)*np.cos(az_r),
            np.cos(el_r)*np.sin(az_r),
            np.sin(el_r)
        ])
        up = [0,0,1]

        # 配置相机并渲染
        renderer.scene.camera.look_at(center, eye, up)
        img = renderer.render_to_image()
        arr = np.asarray(img)  # 转 numpy H×W×4 (RGBA)

        # 保存为 PNG
        fname = f"{base_name}_az{az:03d}_el{el:03d}.png"
        imageio.imwrite(os.path.join(OUT_DIR, fname), arr)
        print("Saved", fname)

    # 清理并让渲染器析构
    renderer.scene.clear_geometry()

def main():
    ply_files = sorted(glob.glob(os.path.join(PLY_DIR, "frame_*.ply")))
    if not ply_files:
        print("No PLYs found in", PLY_DIR)
        return

    for ply_path in ply_files:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        print("Rendering", base)
        # 读取网格；如果 PLY 只是点云，可改为 read_point_cloud
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        render_one(mesh, base)

    print("All done. PNGs are in", OUT_DIR)

if __name__ == "__main__":
    main()
