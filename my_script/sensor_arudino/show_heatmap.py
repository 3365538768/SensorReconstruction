import serial
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import threading
import csv
import os
import platform
import sys
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

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
        
        # Set window size optimized for small screens
        window_width = 1100
        window_height = 700
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Adjust window size if screen is too small
        if screen_width < 1300:
            window_width = min(1000, screen_width - 100)
        if screen_height < 900:
            window_height = min(650, screen_height - 100)
            
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.minsize(800, 500)  # Minimum size for usability
        
        # Configure root window for responsive layout
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Create resizable PanedWindow for main layout
        self.main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Left frame for plot
        plot_frame = tk.Frame(self.main_paned)
        
        # Create 3D matplotlib figure optimized for small screens
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set tight layout to prevent overlap
        self.fig.tight_layout(pad=1.0)
        
        # Place canvas in plot frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right side control panel with enhanced design
        control_frame = tk.Frame(self.main_paned, width=280, bg="#ffffff", relief=tk.RAISED, bd=3)
        
        # Add frames to PanedWindow (plot area gets more initial space)
        self.main_paned.add(plot_frame, weight=3)
        self.main_paned.add(control_frame, weight=1)

        # Add title label with improved styling
        title_label = tk.Label(control_frame, text="⚙️ Control Panel", 
                             font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#2c3e50")
        title_label.pack(pady=(20, 25))

        # Buttons container with better spacing
        btn_frame = tk.Frame(control_frame, bg="#ffffff")
        btn_frame.pack(pady=10)
        
        # Save Frame button with enhanced styling
        btn_save = tk.Button(btn_frame, text="💾 Save Frame", command=self.save_frame, 
                           width=16, height=2, bg="#27ae60", fg="black", 
                           font=("Helvetica", 12, "bold"), relief=tk.RAISED, bd=4,
                           activebackground="#2ecc71", activeforeground="black",
                           cursor="hand2")
        btn_save.pack(pady=10)
        
        # Export CSV button with enhanced styling
        btn_exp = tk.Button(btn_frame, text="📊 Export CSV", command=self.export_csv,
                          width=16, height=2, bg="#3498db", fg="black",
                          font=("Helvetica", 12, "bold"), relief=tk.RAISED, bd=4,
                          activebackground="#5dade2", activeforeground="black",
                          cursor="hand2")
        btn_exp.pack(pady=10)

        # Mode switch button with enhanced styling
        self.mode_btn = tk.Button(btn_frame, text="🔄 2D Mode", command=self.toggle_display_mode,
                           width=16, height=2, bg="#e67e22", fg="black",
                           font=("Helvetica", 12, "bold"), relief=tk.RAISED, bd=4,
                           activebackground="#f39c12", activeforeground="black",
                           cursor="hand2")
        self.mode_btn.pack(pady=10)

        # Separator line
        separator = tk.Frame(control_frame, height=3, bg="#bdc3c7")
        separator.pack(fill=tk.X, padx=25, pady=(15, 20))

        # Status information with enhanced styling
        status_frame = tk.Frame(control_frame, bg="#ecf0f1", relief=tk.GROOVE, bd=2)
        status_frame.pack(pady=8, padx=15, fill=tk.X)
        
        status_title = tk.Label(status_frame, text="📡 System Status", 
                              font=("Helvetica", 13, "bold"), bg="#ecf0f1", fg="#2c3e50")
        status_title.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        self.status_label = tk.Label(status_frame, text="Status: Running", 
                                   font=("Helvetica", 11), bg="#ecf0f1", fg="#34495e",
                                   wraplength=200, justify=tk.LEFT)
        self.status_label.pack(anchor=tk.W, padx=10, pady=(0, 8))

        # Data statistics with enhanced styling
        stats_frame = tk.Frame(control_frame, bg="#ecf0f1", relief=tk.GROOVE, bd=2)
        stats_frame.pack(pady=8, padx=15, fill=tk.X)
        
        stats_title = tk.Label(stats_frame, text="📈 Data Statistics", 
                             font=("Helvetica", 13, "bold"), bg="#ecf0f1", fg="#2c3e50")
        stats_title.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        self.stats_label = tk.Label(stats_frame, text="", 
                                  font=("Helvetica", 11), bg="#ecf0f1", fg="#34495e",
                                  wraplength=200, justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W, padx=10, pady=(0, 8))
        
        # View angle controls (only for 3D mode)
        view_frame = tk.Frame(control_frame, bg="#ecf0f1", relief=tk.GROOVE, bd=2)
        view_frame.pack(pady=8, padx=15, fill=tk.X)
        
        view_title = tk.Label(view_frame, text="👁️ View Controls", 
                            font=("Helvetica", 13, "bold"), bg="#ecf0f1", fg="#2c3e50")
        view_title.pack(anchor=tk.W, padx=10, pady=(8, 5))
        
        # View angle buttons
        view_btn_frame = tk.Frame(view_frame, bg="#ecf0f1")
        view_btn_frame.pack(pady=(5, 10))
        
        btn_top = tk.Button(view_btn_frame, text="⬆️", command=lambda: self.adjust_view("top"),
                          width=4, height=1, bg="#8e44ad", fg="black", font=("Helvetica", 11, "bold"),
                          relief=tk.RAISED, bd=3, cursor="hand2")
        btn_top.grid(row=0, column=1, padx=3, pady=3)
        
        btn_left = tk.Button(view_btn_frame, text="⬅️", command=lambda: self.adjust_view("left"),
                           width=4, height=1, bg="#8e44ad", fg="black", font=("Helvetica", 11, "bold"),
                           relief=tk.RAISED, bd=3, cursor="hand2")
        btn_left.grid(row=1, column=0, padx=3, pady=3)
        
        btn_reset = tk.Button(view_btn_frame, text="🎯", command=lambda: self.adjust_view("reset"),
                            width=4, height=1, bg="#7f8c8d", fg="black", font=("Helvetica", 11, "bold"),
                            relief=tk.RAISED, bd=3, cursor="hand2")
        btn_reset.grid(row=1, column=1, padx=3, pady=3)
        
        btn_right = tk.Button(view_btn_frame, text="➡️", command=lambda: self.adjust_view("right"),
                            width=4, height=1, bg="#8e44ad", fg="black", font=("Helvetica", 11, "bold"),
                            relief=tk.RAISED, bd=3, cursor="hand2")
        btn_right.grid(row=1, column=2, padx=3, pady=3)
        
        btn_bottom = tk.Button(view_btn_frame, text="⬇️", command=lambda: self.adjust_view("bottom"),
                             width=4, height=1, bg="#8e44ad", fg="black", font=("Helvetica", 11, "bold"),
                             relief=tk.RAISED, bd=3, cursor="hand2")
        btn_bottom.grid(row=2, column=1, padx=3, pady=3)
        
        # Initialize display mode
        self.display_mode_3d = True
        self.bars = None
        self.colorbar = None
        self.heatmap_im = None
        
        # Initialize plot
        self.current_elev = 35  # Store current elevation angle
        self.current_azim = 135  # Store current azimuth angle
        self.setup_plot()
        self.last_update_time = 0
        self.update_plot()

    def toggle_display_mode(self):
        """Toggle between 2D and 3D display modes"""
        self.display_mode_3d = not self.display_mode_3d
        
        # Clear existing text annotations
        for text in getattr(self, '_text_annotations', []):
            text.remove()
        for text in getattr(self, '_text_annotations_3d', []):
            text.remove()
        self._text_annotations = []
        self._text_annotations_3d = []
        
        # Reset plot objects
        self.bars = None
        self.heatmap_im = None
        self.colorbar = None
        
        # Setup new plot mode
        self.setup_plot()
        
        # Update button text with emoji
        mode_text = "🔄 2D Mode" if self.display_mode_3d else "🔄 3D Mode"
        self.mode_btn.config(text=mode_text)
        
        # Force canvas update
        self.canvas.draw()

    def adjust_view(self, direction):
        """Adjust 3D view angle"""
        if not self.display_mode_3d:
            return  # Only works in 3D mode
            
        if direction == "left":
            self.current_azim -= 15
        elif direction == "right":
            self.current_azim += 15
        elif direction == "top":
            self.current_elev += 10
        elif direction == "bottom":
            self.current_elev -= 10
        elif direction == "reset":
            self.current_elev = 35
            self.current_azim = 135
            
        # Constrain angles to reasonable ranges
        self.current_elev = max(5, min(85, self.current_elev))
        self.current_azim = self.current_azim % 360
        
        # Update the view
        if hasattr(self, 'ax') and self.ax:
            self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
            self.canvas.draw_idle()

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
        
        # Set optimized view angle for better visibility of individual bars
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)  # Use stored angles
        
        # Set axis ranges with better spacing
        self.ax.set_xlim(-0.5, COLS-0.5)
        self.ax.set_ylim(-0.5, ROWS-0.5)
        self.ax.set_zlim(0, 1.1)  # Slightly higher to show tall bars better
        
        # Improve grid and background for better visibility
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        # Make pane edges more subtle
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)
        
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
        
        # Normalize data (for saving)
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        
        # Create display data (1 - norm for higher values with more pressure)
        display_norm = 1 - norm
        
        # Flip Y-axis to match physical sensor orientation
        display_flipped = np.flipud(display_norm)
        
        # Update heatmap
        if self.heatmap_im is not None:
            self.heatmap_im.set_data(display_flipped)
        
        # Add value labels on 2D heatmap
        self.add_2d_value_labels(display_flipped)
        
        # Update statistics (use original norm for consistency)
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
        
        # Normalize data (for saving)
        norm = (current_raw - MIN_ADC) / (MAX_ADC - MIN_ADC)
        norm = np.clip(norm, 0, 1)
        
        # Create display data (1 - norm for higher values with more pressure)
        display_norm = 1 - norm
        
        # Flip Y-axis to match physical sensor orientation
        display_flipped = np.flipud(display_norm)
        
        # Flatten data for bar chart heights
        dz = display_flipped.flatten()
        
        # Only clear and redraw if this is the first update or data has significantly changed
        if self.bars is None:
            # Initial setup - only do this once
            # Set colors based on values
            colors = plt.cm.viridis(dz)
            
            # Draw 3D bar chart with enhanced visual effects
            self.bars = self.ax.bar3d(self.xpos, self.ypos, self.zpos, 
                               self.dx, self.dy, dz, 
                               color=colors, alpha=0.9, 
                               edgecolor='darkgray', linewidth=0.5,
                               shade=True)
            
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
            
            # Draw new bars with enhanced visual effects
            self.bars = self.ax.bar3d(self.xpos, self.ypos, self.zpos, 
                               self.dx, self.dy, dz, 
                               color=colors, alpha=0.9, 
                               edgecolor='darkgray', linewidth=0.5,
                               shade=True)
        
        # Add value labels on 3D bars
        self.add_3d_value_labels(display_flipped, dz)
        
        # Update statistics (use original norm for consistency)
        self.update_statistics(norm)
        
        # Refresh canvas less frequently
        try:
            self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
        except:
            pass  # Ignore drawing errors
        
        # Continue updating
        self.root.after(200, self.update_plot)  # Slower update rate

    def add_2d_value_labels(self, display_data):
        """Add value labels to 2D heatmap"""
        # Clear existing text annotations
        for text in getattr(self, '_text_annotations', []):
            text.remove()
        self._text_annotations = []
        
        # Add text annotations with values
        rows, cols = display_data.shape
        for i in range(rows):
            for j in range(cols):
                value = display_data[i, j]
                # Only show values above a threshold to avoid clutter
                if value > 0.1:  # Show values > 0.1
                    text = self.ax.text(j, i, f'{value:.2f}', 
                                      ha='center', va='center', 
                                      color='black', fontsize=8, 
                                      fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.1', 
                                              facecolor='white', alpha=0.7))
                    self._text_annotations.append(text)

    def add_3d_value_labels(self, display_data, dz):
        """Add value labels to 3D bar chart"""
        # Clear existing text annotations
        for text in getattr(self, '_text_annotations_3d', []):
            text.remove()
        self._text_annotations_3d = []
        
        # Add text annotations on top of bars
        rows, cols = display_data.shape
        for i in range(rows):
            for j in range(cols):
                value = display_data[i, j]
                # Only show values above a threshold to avoid clutter
                if value > 0.1:  # Show values > 0.1
                    # Calculate position on top of the bar
                    x_pos = j
                    y_pos = i  
                    z_pos = value + 0.05  # Slightly above the bar
                    
                    text = self.ax.text(x_pos, y_pos, z_pos, f'{value:.2f}', 
                                      ha='center', va='center', 
                                      color='black', fontsize=7, 
                                      fontweight='bold')
                    self._text_annotations_3d.append(text)

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