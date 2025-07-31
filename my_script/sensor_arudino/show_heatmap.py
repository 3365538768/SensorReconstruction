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
SIMULATION_MODE = False  # Whether to use simulation data
GUI_MODE = True  # Whether to use GUI mode

# Global state
current_raw   = np.zeros((ROWS, COLS), dtype=int)
saved_frames  = []   # Each element is [entry_index, v00, …, v99]
running       = True

# --- Data Generation (Real Serial or Simulation) ---
def generate_simulation_data():
    """Generate simulated sensor data"""
    # Create a dynamic changing pressure pattern
    t = time.time()
    center_x, center_y = ROWS // 2, COLS // 2
    
    data = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            # Distance from center
            dist = np.sqrt((r - center_x)**2 + (c - center_y)**2)
            # Time-related fluctuation
            wave = np.sin(t * 0.5 + dist * 0.3) * 0.5 + 0.5
            # Random noise
            noise = np.random.normal(0, 0.1)
            # Composite value, mapped to ADC range
            value = (wave + noise) * (MAX_ADC - MIN_ADC) + MIN_ADC
            data[r, c] = int(np.clip(value, MIN_ADC, MAX_ADC))
    
    return data

def serial_reader():
    """Serial reading thread (supports simulation mode)"""
    global current_raw, running, SIMULATION_MODE
    
    ser = None
    if not SIMULATION_MODE:
        try:
            print(f"Attempting to connect to serial port: {SERIAL_PORT}")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"✅ Serial port connected successfully: {SERIAL_PORT}")
        except Exception as e:
            print(f"⚠️  Serial port connection failed: {e}")
            print("🔄 Automatically switching to simulation mode")
            SIMULATION_MODE = True

    print(f"📊 Data source: {'Simulation Mode' if SIMULATION_MODE else 'Serial Mode'}")

    while running:
        try:
            if SIMULATION_MODE:
                # Simulation mode: generate simulation data
                current_raw[:] = generate_simulation_data()
                time.sleep(0.1)  # Simulation data update frequency
            else:
                # Real serial port mode
                received = []
                in_frame = False
                
                # Read one frame
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
            print(f"❌ Data reading error: {e}")
            if not SIMULATION_MODE:
                print("🔄 Trying to switch to simulation mode")
                SIMULATION_MODE = True
                if ser:
                    ser.close()
                    ser = None
            else:
                time.sleep(1)  # Wait when simulation mode encounters error

    if ser:
        ser.close()
        print("🔌 Serial port closed")

# --- GUI and Plotting ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Sensor 3D Pressure Monitor")
        
        # Configure grid weights for responsive layout
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)

        # Create 3D matplotlib figure
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Place canvas on the left side
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Right side control panel
        control_frame = tk.Frame(root, width=150)
        control_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        control_frame.grid_propagate(False)  # Fixed control panel width

        # Add title label
        title_label = tk.Label(control_frame, text="Control Panel", font=("Arial", 12, "bold"))
        title_label.pack(pady=(10, 20))

        # Save Frame button
        btn_save = tk.Button(control_frame, text="Save Current Frame", command=self.save_frame, 
                           width=15, height=2, bg="#4CAF50", fg="white")
        btn_save.pack(pady=10)
        
        # Export CSV button
        btn_exp = tk.Button(control_frame, text="Export CSV", command=self.export_csv,
                          width=15, height=2, bg="#2196F3", fg="white")
        btn_exp.pack(pady=10)

        # Status information label
        self.status_label = tk.Label(control_frame, text="Status: Running", 
                                   font=("Arial", 10), wraplength=140)
        self.status_label.pack(pady=(20, 10))

        # Data statistics label
        self.stats_label = tk.Label(control_frame, text="", 
                                  font=("Arial", 9), wraplength=140, justify="left")
        self.stats_label.pack(pady=10)

        # Initialize 3D plot
        self.setup_3d_plot()
        self.update_plot()

    def setup_3d_plot(self):
        """Initialize 3D plot settings"""
        # Create coordinate grid
        x = np.arange(COLS)
        y = np.arange(ROWS)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Flattened coordinates for bar chart
        self.xpos = self.X.flatten()
        self.ypos = self.Y.flatten()
        self.zpos = np.zeros_like(self.xpos)
        
        # Width and depth of bars
        self.dx = np.ones_like(self.xpos) * 0.8
        self.dy = np.ones_like(self.ypos) * 0.8
        
        # Set axis labels and title
        self.ax.set_xlabel('Sensor Column (X)')
        self.ax.set_ylabel('Sensor Row (Y)')
        self.ax.set_zlabel('Normalized Pressure')
        self.ax.set_title('Real-time Sensor 3D Pressure Map')
        
        # Set fixed view angle for better 3D effect
        self.ax.view_init(elev=30, azim=45)
        
        # Set axis ranges
        self.ax.set_xlim(-0.5, COLS-0.5)
        self.ax.set_ylim(-0.5, ROWS-0.5)
        self.ax.set_zlim(0, 1)

    def update_plot(self):
        """Update 3D bar chart display"""
        # Normalize and update
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        
        # Clear previous graphics
        self.ax.clear()
        
        # Re-setup 3D plot
        self.setup_3d_plot()
        
        # Flatten data for bar chart heights
        dz = norm.flatten()
        
        # Set colors based on values
        # Use viridis colormap
        colors = plt.cm.viridis(dz)
        
        # Draw 3D bar chart
        bars = self.ax.bar3d(self.xpos, self.ypos, self.zpos, 
                           self.dx, self.dy, dz, 
                           color=colors, alpha=0.8, edgecolor='black', linewidth=0.1)
        
        # Add colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        
        # Create a mapping object for colorbar
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm_obj = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm_obj)
        sm.set_array([])
        self.colorbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.8, aspect=20, label='Normalized Pressure')
        
        # Update statistics
        min_val = np.min(norm)
        max_val = np.max(norm)
        avg_val = np.mean(norm)
        
        stats_text = f"Statistics:\nMin: {min_val:.3f}\nMax: {max_val:.3f}\nAvg: {avg_val:.3f}\nSaved: {len(saved_frames)} frames"
        self.stats_label.config(text=stats_text)
        
        # Update status
        data_source = "Simulation" if SIMULATION_MODE else "Serial Port"
        self.status_label.config(text=f"Status: Running\nData Source: {data_source}")
        
        # Refresh canvas
        self.canvas.draw()
        
        # Continue updating
        self.root.after(100, self.update_plot)

    def save_frame(self):
        # Use number of saved entries as index
        eid = len(saved_frames)
        # Flatten and normalize, keep two decimal places
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        flat = np.round(norm.flatten(), 2).tolist()
        saved_frames.append([eid] + flat)
        messagebox.showinfo("Frame Saved", f"Entry {eid} saved successfully (Total: {len(saved_frames)} frames)")

    def export_csv(self):
        if not saved_frames:
            messagebox.showwarning("No Data", "No frames saved! Please save some frames first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV Files","*.csv")])
        if not path:
            return
        header = ['frame'] + [f'sensor_{r}_{c}' for r in range(ROWS) for c in range(COLS)]
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(saved_frames)
            messagebox.showinfo("Export Complete",
                f"Successfully exported {len(saved_frames)} frames to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error writing CSV file: {e}")

# --- Console Mode Support ---
def run_console_mode():
    """Console mode operation (No GUI)"""
    print("🖥️  Running in console mode")
    print("Press Ctrl+C to stop the program")
    
    try:
        frame_count = 0
        while running:
            time.sleep(1)
            frame_count += 1
            
            # Display current data statistics
            if frame_count % 10 == 0:  # Display every 10 seconds
                norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
                print(f"Frame {frame_count}: Average={np.mean(norm):.3f}, "
                      f"Min={np.min(norm):.3f}, Max={np.max(norm):.3f}")
                
                # Display simplified heatmap (ASCII)
                print("📊 Current pressure map (normalized, 0.0-1.0):")
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
        print("\n👋 User interrupted, program exiting")

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Sensor 3D Pressure Monitor")
    parser.add_argument('--simulation', '-s', action='store_true',
                       help='Use simulation data mode')
    parser.add_argument('--console', '-c', action='store_true',
                       help='Use console mode (no GUI)')
    parser.add_argument('--port', '-p', type=str,
                       help='Specify serial port (e.g. COM5 or /dev/ttyUSB0)')
    return parser.parse_args()

# --- Main Program ---
def main():
    """Main program entry"""
    global SIMULATION_MODE, SERIAL_PORT, GUI_MODE, running
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply configuration
    if args.simulation:
        SIMULATION_MODE = True
        print("🎭 Simulation mode enabled")
    
    if args.port:
        SERIAL_PORT = args.port
        print(f"🔌 Using specified serial port: {SERIAL_PORT}")
    
    # Check display environment
    has_display = check_display_environment()
    if args.console or not has_display:
        GUI_MODE = False
        if not has_display:
            print("⚠️  No display environment detected, switching to console mode")
        else:
            print("🖥️  User selected console mode")

    # Start data reading thread
    print(f"🚀 Starting Sensor 3D Pressure Monitor")
    print(f"   Data Source: {'Simulation' if SIMULATION_MODE else f'Serial Port {SERIAL_PORT}'}")
    print(f"   Interface Mode: {'GUI' if GUI_MODE else 'Console'}")
    
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    try:
        if GUI_MODE:
            # GUI mode
            try:
                # Set matplotlib backend to avoid display issues
                if not has_display:
                    import matplotlib
                    matplotlib.use('Agg')  # No display backend
                
                root = tk.Tk()
                app = App(root)
                root.mainloop()
            except Exception as e:
                print(f"❌ GUI startup failed: {e}")
                print("🔄 Switching to console mode")
                run_console_mode()
        else:
            # Console mode
            run_console_mode()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    finally:
        running = False
        print("🔚 Program ended")

if __name__ == '__main__':
    main()