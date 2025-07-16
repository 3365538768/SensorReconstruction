import open3d as o3d
import numpy as np
import json
import os

# --- 配置路径 (请根据您的实际情况修改) ---
BASE_DIR = "/users/lshou/4DGaussians"
PLY_PATH = os.path.join(BASE_DIR, "my_script/test", "point_cloud.ply")
BOX_JSON_PATH = os.path.join(BASE_DIR, "my_script/test", "region.json")

def visualize_point_cloud_with_bbox(ply_path, json_path):
    """
    加载PLY点云和JSON中的包围盒数据，并使用Open3D进行可视化。
    """
    print(f"正在加载点云文件: {ply_path}")
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_points():
            print(f"警告: 点云文件 {ply_path} 中没有点。")
            return
    except Exception as e:
        print(f"加载点云文件失败: {e}")
        return

    print(f"正在加载包围盒文件: {json_path}")
    bbox_data = None
    try:
        with open(json_path, 'r') as f:
            bbox_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 包围盒文件 {json_path} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 包围盒文件 {json_path} 不是有效的JSON格式。")
        return

    bbox_min = bbox_data.get("bbox", [[None,None,None],[None,None,None]])[0]
    bbox_max = bbox_data.get("bbox", [[None,None,None],[None,None,None]])[1]

    if None in bbox_min or None in bbox_max:
        print("错误: JSON文件中未找到有效的 'bbox' 数据。请确保JSON包含 'bbox': [[xmin,ymin,zmin],[xmax,ymax,zmax]]。")
        return

    # 将列表转换为NumPy数组
    bbox_min = np.array(bbox_min, dtype=np.float64)
    bbox_max = np.array(bbox_max, dtype=np.float64)

    # 创建Open3D的AxisAlignedBoundingBox对象
    # 这是一个轴对齐的包围盒，通常用于表示AABB
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
    aabb.color = (1, 0, 0)  # 设置包围盒颜色为红色

    # 如果您的region.json中包含法线信息，并且您想可视化一个OrientedBoundingBox
    # 您可能需要根据法线和中心点计算旋转矩阵和尺寸
    # 这里我们只使用简单的AABB，因为它直接对应xmin,ymin,zmin,xmax,ymax,zmax

    # 可视化点云和包围盒
    print("正在打开可视化窗口...")
    o3d.visualization.draw_geometries([pcd, aabb])
    print("可视化完成。")

if __name__ == "__main__":
    visualize_point_cloud_with_bbox(PLY_PATH, BOX_JSON_PATH)