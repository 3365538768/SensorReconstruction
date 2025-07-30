import os
import glob
import numpy as np
from plyfile import PlyData, PlyElement

def load_ply_points(path):
    """Load (N,3) numpy array of points from a PLY file."""
    ply = PlyData.read(path)
    x = ply['vertex']['x']
    y = ply['vertex']['y']
    z = ply['vertex']['z']
    pts = np.vstack((x, y, z)).T
    return pts.astype(np.float32)

def write_ply_points(path, pts):
    """Write (M,3) numpy array of points to a PLY file."""
    verts = np.core.records.fromarrays(
        pts.T, names='x,y,z', formats='f4,f4,f4'
    )
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)

def extract_top_dynamic_points(
    input_dir: str,
    output_dir: str,
    top_percent: float
):
    """
    For a folder of PLY frames (same topology & ordering), compute
    per-point total motion magnitude across frames, select the top
    `top_percent` of points, and write those points (same indices)
    for each frame to the output folder.
    
    Args:
        input_dir:   Path to folder containing '*.ply' frames.
        output_dir:  Path to folder to write filtered frames.
        top_percent: Fraction in (0,1] of most dynamic points to keep.
    """
    # 1. Collect frame file paths
    ply_paths = sorted(glob.glob(os.path.join(input_dir, '*.ply')))
    if not ply_paths:
        raise ValueError(f"No PLY files found in {input_dir}")
    
    # 2. Load all frames into a list of (N,3) arrays
    frames = [load_ply_points(p) for p in ply_paths]
    num_frames = len(frames)
    
    # 2.1. Check point counts and handle dynamic point changes
    point_counts = [frame.shape[0] for frame in frames]
    min_points = min(point_counts)
    max_points = max(point_counts)
    
    print(f"Loaded {num_frames} frames")
    print(f"Point count range: {min_points} - {max_points}")
    
    if min_points != max_points:
        print(f"⚠️  Warning: Point counts vary across frames!")
        print(f"   This is normal for 4DGaussians due to densification during training")
        print(f"   Truncating all frames to {min_points} points for consistency")
        
        # Truncate all frames to the minimum point count
        frames = [frame[:min_points] for frame in frames]
        N = min_points
    else:
        N = frames[0].shape[0]
        print(f"All frames have consistent {N} points")
    
    # 3. Stack into (F, N, 3) array
    data = np.stack(frames, axis=0)  # shape = (F, N, 3)
    
    # 4. Compute dynamic magnitude per point:
    #    sum of frame-to-frame L2 displacements
    disp = np.linalg.norm(np.diff(data, axis=0), axis=2)  # shape = (F-1, N)
    motion = disp.sum(axis=0)                              # shape = (N,)
    
    # 5. Select top k indices
    k = max(1, int(np.ceil(N * top_percent)))
    top_idx = np.argsort(motion)[-k:]                     # indices of top k dynamic points
    top_idx.sort()
    print(f"Selecting top {top_percent*100:.1f}% dynamic points: {k} points")
    
    # 6. Prepare output folder
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. For each frame, write only the selected points
    for i, path in enumerate(ply_paths):
        pts = frames[i][top_idx]  # shape = (k,3)
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        write_ply_points(out_path, pts)
    
    print(f"Done! Filtered frames written to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract top-n% dynamic points from a sequence of PLY frames"
    )
    parser.add_argument(
        "--input_dir", type=str, default='my_script/data/experiment1',
        help="Directory containing input PLY frames"
    )
    parser.add_argument(
        "--output_dir", type=str, default='data/experiment1/frames',
        help="Directory to write filtered PLYs"
    )
    parser.add_argument(
        "--percent", type=float, default=0.2,
        help="Fraction of top dynamic points to keep (e.g. 0.1 for top 10%%)"
    )
    args = parser.parse_args()
    
    extract_top_dynamic_points(
        args.input_dir,
        args.output_dir,
        args.percent
    )