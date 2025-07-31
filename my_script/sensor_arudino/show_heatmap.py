import serial
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import csv
import os
import platform
import sys
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration & Environment ---
def detect_serial_port():
    """自动检测可用的串口"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    
    # 根据操作系统优先选择合适的串口
    if platform.system() == "Windows":
        # Windows: 查找COM端口
        for port in ports:
            if "COM" in port.device:
                return port.device
        return 'COM5'  # 默认值
    else:
        # Linux/macOS: 查找USB或ACM设备
        for port in ports:
            if any(keyword in port.device.lower() for keyword in ['usb', 'acm', 'tty']):
                return port.device
        # 如果没找到，返回常见的Linux串口
        return '/dev/ttyUSB0'

def check_display_environment():
    """检查是否有可用的显示环境"""
    if platform.system() != "Windows":
        display = os.environ.get('DISPLAY')
        if not display:
            return False
    return True

# --- Serial & Array Config ---
SERIAL_PORT = detect_serial_port()
BAUD_RATE = 115200
ROWS, COLS = 10, 10
MIN_ADC, MAX_ADC = 0, 4095

# --- Global Config ---
SIMULATION_MODE = False  # 是否使用模拟数据
GUI_MODE = True  # 是否使用GUI模式

# 全局状态
current_raw   = np.zeros((ROWS, COLS), dtype=int)
saved_frames  = []   # 每个元素是 [entry_index, v00, …, v99]
running       = True

# --- 数据生成（真实串口或模拟） ---
def generate_simulation_data():
    """生成模拟传感器数据"""
    # 创建一个动态变化的热力图模式
    t = time.time()
    center_x, center_y = ROWS // 2, COLS // 2
    
    data = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            # 距离中心的距离
            dist = np.sqrt((r - center_x)**2 + (c - center_y)**2)
            # 时间相关的波动
            wave = np.sin(t * 0.5 + dist * 0.3) * 0.5 + 0.5
            # 随机噪声
            noise = np.random.normal(0, 0.1)
            # 合成值，映射到ADC范围
            value = (wave + noise) * (MAX_ADC - MIN_ADC) + MIN_ADC
            data[r, c] = int(np.clip(value, MIN_ADC, MAX_ADC))
    
    return data

def serial_reader():
    """串口读取线程（支持模拟模式）"""
    global current_raw, running, SIMULATION_MODE
    
    ser = None
    if not SIMULATION_MODE:
        try:
            print(f"尝试连接串口: {SERIAL_PORT}")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"✅ 串口连接成功: {SERIAL_PORT}")
        except Exception as e:
            print(f"⚠️  串口连接失败: {e}")
            print("🔄 自动切换到模拟模式")
            SIMULATION_MODE = True

    print(f"📊 数据源: {'模拟模式' if SIMULATION_MODE else '串口模式'}")

    while running:
        try:
            if SIMULATION_MODE:
                # 模拟模式：生成模拟数据
                current_raw[:] = generate_simulation_data()
                time.sleep(0.1)  # 模拟数据更新频率
            else:
                # 真实串口模式
                received = []
                in_frame = False
                
                # 读取一帧
                while running:
                    line = ser.readline().decode('utf-8', 'ignore').strip()
                    if line == 'START_FRAME':
                        received, in_frame = [], True
                        continue
                    if line == 'END_FRAME':
                        in_frame = False
                        break
                    if in_frame:
                        parts = line.split(',')
                        if len(parts) == COLS:
                            try:
                                received.append([int(x) for x in parts])
                            except:
                                pass
                
                if len(received) == ROWS:
                    current_raw[:] = np.array(received, dtype=int)
        except Exception as e:
            print(f"❌ 数据读取错误: {e}")
            if not SIMULATION_MODE:
                print("🔄 尝试切换到模拟模式")
                SIMULATION_MODE = True
                if ser:
                    ser.close()
                    ser = None
            else:
                time.sleep(1)  # 模拟模式出错时等待

    if ser:
        ser.close()
        print("🔌 串口已关闭")

# --- GUI 与绘图 ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Sensor 3D Heatmap UI")
        
        # 配置grid权重，让界面能够自适应调整
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)

        # 创建3D matplotlib图形
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 将canvas放在左侧
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 右侧控制面板
        control_frame = tk.Frame(root, width=150)
        control_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        control_frame.grid_propagate(False)  # 固定控制面板宽度

        # 添加标题标签
        title_label = tk.Label(control_frame, text="控制面板", font=("Arial", 12, "bold"))
        title_label.pack(pady=(10, 20))

        # Save Frame 按钮
        btn_save = tk.Button(control_frame, text="保存当前帧", command=self.save_frame, 
                           width=15, height=2, bg="#4CAF50", fg="white")
        btn_save.pack(pady=10)
        
        # Export CSV 按钮
        btn_exp = tk.Button(control_frame, text="导出CSV", command=self.export_csv,
                          width=15, height=2, bg="#2196F3", fg="white")
        btn_exp.pack(pady=10)

        # 状态信息标签
        self.status_label = tk.Label(control_frame, text="状态: 运行中", 
                                   font=("Arial", 10), wraplength=140)
        self.status_label.pack(pady=(20, 10))

        # 数据统计标签
        self.stats_label = tk.Label(control_frame, text="", 
                                  font=("Arial", 9), wraplength=140, justify="left")
        self.stats_label.pack(pady=10)

        # 初始化3D图形
        self.setup_3d_plot()
        self.update_plot()

    def setup_3d_plot(self):
        """初始化3D图形设置"""
        # 创建坐标网格
        x = np.arange(COLS)
        y = np.arange(ROWS)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 扁平化坐标用于柱状图
        self.xpos = self.X.flatten()
        self.ypos = self.Y.flatten()
        self.zpos = np.zeros_like(self.xpos)
        
        # 柱子的宽度和深度
        self.dx = np.ones_like(self.xpos) * 0.8
        self.dy = np.ones_like(self.ypos) * 0.8
        
        # 设置坐标轴标签和标题
        self.ax.set_xlabel('传感器列 (X)')
        self.ax.set_ylabel('传感器行 (Y)')
        self.ax.set_zlabel('归一化数值')
        self.ax.set_title('实时传感器3D热力图')
        
        # 设置固定的视角以获得更好的3D效果
        self.ax.view_init(elev=30, azim=45)
        
        # 设置坐标轴范围
        self.ax.set_xlim(-0.5, COLS-0.5)
        self.ax.set_ylim(-0.5, ROWS-0.5)
        self.ax.set_zlim(0, 1)

    def update_plot(self):
        """更新3D柱状图显示"""
        # 归一化并更新
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        
        # 清除之前的图形
        self.ax.clear()
        
        # 重新设置3D图形
        self.setup_3d_plot()
        
        # 扁平化数据用于柱状图高度
        dz = norm.flatten()
        
        # 根据数值设置颜色
        # 使用viridis色彩映射
        colors = plt.cm.viridis(dz)
        
        # 绘制3D柱状图
        bars = self.ax.bar3d(self.xpos, self.ypos, self.zpos, 
                           self.dx, self.dy, dz, 
                           color=colors, alpha=0.8, edgecolor='black', linewidth=0.1)
        
        # 添加颜色条
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        
        # 创建一个映射对象用于颜色条
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm_obj = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm_obj)
        sm.set_array([])
        self.colorbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.8, aspect=20, label='归一化数值')
        
        # 更新统计信息
        min_val = np.min(norm)
        max_val = np.max(norm)
        avg_val = np.mean(norm)
        
        stats_text = f"统计信息:\n最小值: {min_val:.3f}\n最大值: {max_val:.3f}\n平均值: {avg_val:.3f}\n已保存: {len(saved_frames)}帧"
        self.stats_label.config(text=stats_text)
        
        # 更新状态
        data_source = "模拟数据" if SIMULATION_MODE else "串口数据"
        self.status_label.config(text=f"状态: 运行中\n数据源: {data_source}")
        
        # 刷新画布
        self.canvas.draw()
        
        # 继续更新
        self.root.after(100, self.update_plot)

    def save_frame(self):
        # 用已保存条目数作为 index
        eid = len(saved_frames)
        # 扁平化并归一后保留两位小数
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        flat = np.round(norm.flatten(), 2).tolist()
        saved_frames.append([eid] + flat)
        messagebox.showinfo("Saved", f"Entry {eid} saved (total: {len(saved_frames)})")

    def export_csv(self):
        if not saved_frames:
            messagebox.showwarning("No Data", "没有保存任何条目！")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")])
        if not path:
            return
        header = ['frame'] + [f'v{r}{c}' for r in range(ROWS) for c in range(COLS)]
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(saved_frames)
            messagebox.showinfo("Exported",
                f"Exported {len(saved_frames)} entries to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"写入 CSV 时出错：{e}")

# --- 命令行模式支持 ---
def run_console_mode():
    """命令行模式运行（无GUI）"""
    print("🖥️  运行在命令行模式")
    print("按 Ctrl+C 停止程序")
    
    try:
        frame_count = 0
        while running:
            time.sleep(1)
            frame_count += 1
            
            # 显示当前数据统计
            if frame_count % 10 == 0:  # 每10秒显示一次
                norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
                print(f"帧 {frame_count}: 平均值={np.mean(norm):.3f}, "
                      f"最小值={np.min(norm):.3f}, 最大值={np.max(norm):.3f}")
                
                # 显示简化的热力图（ASCII）
                print("📊 当前热力图(归一化, 0.0-1.0):")
                for r in range(ROWS):
                    row_str = ""
                    for c in range(COLS):
                        val = norm[r, c]
                        if val < 0.2:
                            char = "░"
                        elif val < 0.4:
                            char = "▒"
                        elif val < 0.6:
                            char = "▓"
                        elif val < 0.8:
                            char = "█"
                        else:
                            char = "█"
                        row_str += char
                    print(f"  {row_str}")
                print()
                
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")

def parse_arguments():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="传感器热力图显示程序")
    parser.add_argument('--simulation', '-s', action='store_true',
                       help='使用模拟数据模式')
    parser.add_argument('--console', '-c', action='store_true',
                       help='使用命令行模式（无GUI）')
    parser.add_argument('--port', '-p', type=str,
                       help='指定串口（如 COM5 或 /dev/ttyUSB0）')
    return parser.parse_args()

# --- 主程序 ---
def main():
    """主程序入口"""
    global SIMULATION_MODE, SERIAL_PORT, GUI_MODE, running
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 应用配置
    if args.simulation:
        SIMULATION_MODE = True
        print("🎭 强制启用模拟模式")
    
    if args.port:
        SERIAL_PORT = args.port
        print(f"🔌 使用指定串口: {SERIAL_PORT}")
    
    # 检查显示环境
    has_display = check_display_environment()
    if args.console or not has_display:
        GUI_MODE = False
        if not has_display:
            print("⚠️  未检测到图形显示环境，自动切换到命令行模式")
        else:
            print("🖥️  用户选择命令行模式")

    # 启动数据读取线程
    print(f"🚀 启动传感器热力图程序")
    print(f"   数据源: {'模拟数据' if SIMULATION_MODE else f'串口 {SERIAL_PORT}'}")
    print(f"   界面模式: {'GUI' if GUI_MODE else '命令行'}")
    
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    try:
        if GUI_MODE:
            # GUI模式
            try:
                # 设置matplotlib后端，避免显示问题
                if not has_display:
                    import matplotlib
                    matplotlib.use('Agg')  # 无显示后端
                
                root = tk.Tk()
                app = App(root)
                root.mainloop()
            except Exception as e:
                print(f"❌ GUI启动失败: {e}")
                print("🔄 切换到命令行模式")
                run_console_mode()
        else:
            # 命令行模式
            run_console_mode()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    finally:
        running = False
        print("🔚 程序结束")

if __name__ == '__main__':
    main()