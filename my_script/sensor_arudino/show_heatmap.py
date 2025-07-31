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
        root.grid_columnconfigure(0, weight=3)  # Give more weight to plot area
        root.grid_columnconfigure(1, weight=1)  # Less weight to control panel

        # Create 3D matplotlib figure with fixed size
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set tight layout to prevent overlap
        self.fig.tight_layout()
        
        # Place canvas on the left side
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Right side control panel
        control_frame = tk.Frame(root, width=200)
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
                                   font=("Arial", 10), wraplength=180)
        self.status_label.pack(pady=(20, 10))

        # Data statistics label
        self.stats_label = tk.Label(control_frame, text="", 
                                  font=("Arial", 9), wraplength=180, justify="left")
        self.stats_label.pack(pady=10)

        # Add mode switch button
        self.mode_btn = tk.Button(control_frame, text="Switch to 2D Mode", command=self.toggle_display_mode,
                           width=15, height=1, bg="#FF9800", fg="white")
        self.mode_btn.pack(pady=5)
        
        # Initialize display mode
        self.display_mode_3d = True
        self.bars = None
        self.colorbar = None
        self.heatmap_im = None
        
        # Initialize plot
        self.setup_plot()
        self.last_update_time = 0
        self.update_plot()

    def toggle_display_mode(self):
        """Toggle between 2D and 3D display modes"""
        self.display_mode_3d = not self.display_mode_3d
        
        # Reset plot objects
        self.bars = None
        self.heatmap_im = None
        self.colorbar = None
        
        # Setup new plot mode
        self.setup_plot()
        
        # Update button text
        mode_text = "Switch to 2D Mode" if self.display_mode_3d else "Switch to 3D Mode"
        self.mode_btn.config(text=mode_text)
        
        # Force canvas update
        self.canvas.draw()

    def setup_plot(self):
        """Setup plot based on current display mode"""
        if self.display_mode_3d:
            self.setup_3d_plot()
        else:
            self.setup_2d_plot()

    def setup_2d_plot(self):
        """Initialize 2D heatmap settings"""
        # Clear and recreate 2D axis
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        # Create 2D heatmap
        initial_data = np.zeros((ROWS, COLS))
        self.heatmap_im = self.ax.imshow(initial_data, 
                                       cmap='viridis', vmin=0, vmax=1, 
                                       interpolation='nearest')
        
        # Set labels and title
        self.ax.set_xlabel('Sensor Column (X)', fontsize=10)
        self.ax.set_ylabel('Sensor Row (Y)', fontsize=10)
        self.ax.set_title('Real-time Sensor 2D Pressure Map', fontsize=12)
        
        # Set ticks
        self.ax.set_xticks(np.arange(COLS))
        self.ax.set_yticks(np.arange(ROWS))
        
        # Add colorbar
        self.colorbar = self.fig.colorbar(self.heatmap_im, ax=self.ax, 
                                        shrink=0.8, aspect=20, label='Normalized Pressure')
        
        # Adjust layout
        self.fig.tight_layout()

    def setup_3d_plot(self):
        """Initialize 3D plot settings"""
        # Clear and recreate 3D axis
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
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
        self.ax.set_xlabel('Sensor Column (X)', fontsize=10)
        self.ax.set_ylabel('Sensor Row (Y)', fontsize=10)
        self.ax.set_zlabel('Normalized Pressure', fontsize=10)
        self.ax.set_title('Real-time Sensor 3D Pressure Map', fontsize=12, pad=20)
        
        # Set fixed view angle for better 3D effect
        self.ax.view_init(elev=30, azim=45)
        
        # Set axis ranges
        self.ax.set_xlim(-0.5, COLS-0.5)
        self.ax.set_ylim(-0.5, ROWS-0.5)
        self.ax.set_zlim(0, 1)
        
        # Initialize empty bars collection
        self.bars = None
        
        # Adjust layout
        self.fig.tight_layout()

    def update_plot(self):
        """Update display based on current mode"""
        if self.display_mode_3d:
            self.update_3d_plot()
        else:
            self.update_2d_plot()

    def update_2d_plot(self):
        """Update 2D heatmap display"""
        import time
        current_time = time.time()
        
        # Limit update frequency
        if current_time - self.last_update_time < 0.2:
            self.root.after(100, self.update_plot)
            return
        
        self.last_update_time = current_time
        
        # Normalize data
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        
        # Update heatmap
        if self.heatmap_im is not None:
            self.heatmap_im.set_data(norm)
        
        # Update statistics
        self.update_statistics(norm)
        
        # Refresh canvas
        try:
            self.canvas.draw_idle()
        except:
            pass
        
        # Continue updating
        self.root.after(200, self.update_plot)

    def update_3d_plot(self):
        """Update 3D bar chart display with optimized rendering"""
        import time
        current_time = time.time()
        
        # Limit update frequency to reduce rendering issues
        if current_time - self.last_update_time < 0.2:  # Update every 200ms
            self.root.after(100, self.update_plot)
            return
        
        self.last_update_time = current_time
        
        # Normalize and update
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        
        # Flatten data for bar chart heights
        dz = norm.flatten()
        
        # Only clear and redraw if this is the first update or data has significantly changed
        if self.bars is None:
            # Initial setup - only do this once
            # Set colors based on values
            colors = plt.cm.viridis(dz)
            
            # Draw 3D bar chart
            self.bars = self.ax.bar3d(self.xpos, self.ypos, self.zpos, 
                               self.dx, self.dy, dz, 
                               color=colors, alpha=0.8, edgecolor='black', linewidth=0.1)
            
            # Add colorbar only once
            if self.colorbar is None:
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                norm_obj = mcolors.Normalize(vmin=0, vmax=1)
                sm = cm.ScalarMappable(cmap='viridis', norm=norm_obj)
                sm.set_array([])
                self.colorbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.8, aspect=20, label='Normalized Pressure')
        else:
            # Efficient update - only clear bars and redraw them
            # Remove old bars (bar3d returns a Poly3DCollection object)
            if self.bars is not None:
                self.bars.remove()
            
            # Set new colors
            colors = plt.cm.viridis(dz)
            
            # Draw new bars
            self.bars = self.ax.bar3d(self.xpos, self.ypos, self.zpos, 
                               self.dx, self.dy, dz, 
                               color=colors, alpha=0.8, edgecolor='black', linewidth=0.1)
        
        # Update statistics
        self.update_statistics(norm)
        
        # Refresh canvas less frequently
        try:
            self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
        except:
            pass  # Ignore drawing errors
        
        # Continue updating
        self.root.after(200, self.update_plot)  # Slower update rate

    def update_statistics(self, norm):
        """Update statistics display"""
        min_val = np.min(norm)
        max_val = np.max(norm)
        avg_val = np.mean(norm)
        
        mode_text = "3D" if self.display_mode_3d else "2D"
        stats_text = f"Mode: {mode_text}\nMin: {min_val:.3f}\nMax: {max_val:.3f}\nAvg: {avg_val:.3f}\nSaved: {len(saved_frames)} frames"
        self.stats_label.config(text=stats_text)
        
        # Update status
        data_source = "Simulation" if SIMULATION_MODE else "Serial Port"
        self.status_label.config(text=f"Status: Running\nData Source: {data_source}")

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