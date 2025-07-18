import serial
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import csv
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Serial & Array Config ---
SERIAL_PORT   = 'COM5'
BAUD_RATE     = 115200
ROWS, COLS    = 10, 10
MIN_ADC, MAX_ADC = 0, 4095

# 全局状态
current_raw   = np.zeros((ROWS, COLS), dtype=int)
saved_frames  = []   # 每个元素是 [entry_index, v00, …, v99]
running       = True

# --- 串口读取线程 ---
def serial_reader():
    global current_raw, running
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"无法打开串口: {e}")
        return

    while running:
        received = []; in_frame = False
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
    ser.close()

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

# --- 主程序 ---
if __name__ == '__main__':
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    root = tk.Tk()
    app  = App(root)
    try:
        root.mainloop()
    finally:
        running = False