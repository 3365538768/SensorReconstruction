import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

def read_ply_vertices(filepath):
    """Parse PLY file header and return x,y,z vertex data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    vertex_count = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        if line.strip() == 'end_header':
            header_end = i + 1
            break
    data = np.loadtxt(lines[header_end:header_end + vertex_count])
    return data[:, :3]

# --- Load frames ---
ply_dir = r"E:\notre_dame_project\code\show_cage\cages_pred"
ply_paths = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
if not ply_paths:
    raise FileNotFoundError(f"No PLY files found in {ply_dir}")

frames = [read_ply_vertices(p) for p in ply_paths]

# --- Initialize plot ---
fig = plt.figure(figsize=(10, 10), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

# Compute global limits for consistent view
all_pts = np.vstack(frames)
mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])

# Remove axes for cleaner focus
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.grid(False)

# Smaller, consistent marker
scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], frames[0][:, 2], s=2, c='blue')

def update(frame_idx):
    pts = frames[frame_idx]
    scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
    return scat,

anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)

# --- Save animation ---
output_path = os.path.join(ply_dir, "cage_nodes_only.mp4")
writer = FFMpegWriter(fps=6)
anim.save(output_path, writer=writer,dpi=300)
print(f"Saved node-focused animation to: {output_path}")
