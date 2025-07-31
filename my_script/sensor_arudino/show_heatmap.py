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
        root.title("Sensor Heatmap UI")

        # matplotlib 图
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)

        self.im = self.ax.imshow(np.zeros((ROWS,COLS)), cmap='viridis', vmin=0, vmax=1)
        self.ax.set_title("Real-time Heatmap")
        self.ax.set_xticks(np.arange(COLS)); self.ax.set_yticks(np.arange(ROWS))
        self.fig.colorbar(self.im, ax=self.ax, label='Normalized')

        # 文本标签
        self.texts = []
        for r in range(ROWS):
            row = []
            for c in range(COLS):
                txt = self.ax.text(c, r, '', ha='center', va='center', fontsize=8)
                row.append(txt)
            self.texts.append(row)

        # Save Frame 按钮
        btn_save = tk.Button(root, text="Save Frame", command=self.save_frame)
        btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        # Export CSV 按钮
        btn_exp  = tk.Button(root, text="Export CSV", command=self.export_csv)
        btn_exp.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        self.update_plot()

    def update_plot(self):
        # 归一化并更新
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        self.im.set_data(norm)
        for r in range(ROWS):
            for c in range(COLS):
                v = norm[r,c]
                self.texts[r][c].set_text(f"{v:.2f}")
                self.texts[r][c].set_color('white' if v<0.5 else 'black')
        self.canvas.draw()
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