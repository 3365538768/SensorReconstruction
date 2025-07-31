#!/usr/bin/env python3
"""
3D柱状图演示脚本
用于展示show_heatmap程序的3D可视化效果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def generate_demo_data():
    """生成演示用的传感器数据"""
    ROWS, COLS = 10, 10
    
    # 创建一个中心高周围低的热力分布
    center_x, center_y = ROWS // 2, COLS // 2
    data = np.zeros((ROWS, COLS))
    
    for r in range(ROWS):
        for c in range(COLS):
            # 距离中心的距离
            dist = np.sqrt((r - center_x)**2 + (c - center_y)**2)
            # 创建中心辐射模式
            value = np.exp(-dist * 0.3) + np.random.normal(0, 0.1)
            data[r, c] = max(0, min(1, value))  # 限制在0-1范围
    
    return data

def create_3d_heatmap_demo():
    """创建3D柱状图演示"""
    # 生成演示数据
    data = generate_demo_data()
    ROWS, COLS = data.shape
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建坐标网格
    x = np.arange(COLS)
    y = np.arange(ROWS)
    X, Y = np.meshgrid(x, y)
    
    # 扁平化坐标
    xpos = X.flatten()
    ypos = Y.flatten()
    zpos = np.zeros_like(xpos)
    
    # 柱子参数
    dx = dy = 0.8
    dz = data.flatten()
    
    # 颜色映射
    colors = plt.cm.viridis(dz)
    
    # 绘制3D柱状图
    bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置坐标轴
    ax.set_xlabel('传感器列 (X)', fontsize=12)
    ax.set_ylabel('传感器行 (Y)', fontsize=12) 
    ax.set_zlabel('归一化数值', fontsize=12)
    ax.set_title('传感器3D热力图演示', fontsize=14, fontweight='bold')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, COLS-0.5)
    ax.set_ylim(-0.5, ROWS-0.5)
    ax.set_zlim(0, 1)
    
    # 添加颜色条
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm_obj = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20, label='归一化数值')
    
    # 添加统计信息文本
    stats_text = f'统计信息:\n最小值: {np.min(data):.3f}\n最大值: {np.max(data):.3f}\n平均值: {np.mean(data):.3f}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('sensor_3d_heatmap_demo.png', dpi=200, bbox_inches='tight')
    print("✅ 3D演示图已保存到: sensor_3d_heatmap_demo.png")
    
    # 显示数据统计
    print(f"\n📊 演示数据统计:")
    print(f"   数据形状: {data.shape}")
    print(f"   最小值: {np.min(data):.3f}")
    print(f"   最大值: {np.max(data):.3f}")
    print(f"   平均值: {np.mean(data):.3f}")
    print(f"   柱子数量: {len(dz)}")
    
    return fig, ax, data

def create_comparison_demo():
    """创建2D vs 3D对比演示"""
    data = generate_demo_data()
    
    # 创建包含两个子图的图形
    fig = plt.figure(figsize=(16, 6))
    
    # 2D热力图
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(data, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('原始2D热力图', fontsize=14, fontweight='bold')
    ax1.set_xlabel('传感器列')
    ax1.set_ylabel('传感器行')
    fig.colorbar(im, ax=ax1, label='归一化数值')
    
    # 3D柱状图
    ax2 = fig.add_subplot(122, projection='3d')
    
    ROWS, COLS = data.shape
    x = np.arange(COLS)
    y = np.arange(ROWS)
    X, Y = np.meshgrid(x, y)
    
    xpos = X.flatten()
    ypos = Y.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.8
    dz = data.flatten()
    colors = plt.cm.viridis(dz)
    
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, 
             color=colors, alpha=0.8, edgecolor='black', linewidth=0.3)
    
    ax2.set_title('新3D柱状图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('传感器列')
    ax2.set_ylabel('传感器行')
    ax2.set_zlabel('归一化数值')
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('2d_vs_3d_comparison.png', dpi=200, bbox_inches='tight')
    print("✅ 2D vs 3D对比图已保存到: 2d_vs_3d_comparison.png")

if __name__ == "__main__":
    print("🎨 开始生成3D热力图演示...")
    
    # 生成3D演示
    fig, ax, data = create_3d_heatmap_demo()
    
    # 生成对比演示
    create_comparison_demo()
    
    print("\n🎯 演示完成！")
    print("   可查看生成的PNG文件了解3D效果")
    print("   在实际GUI环境中，这些图形会实时动态更新")