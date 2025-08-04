import os
import glob
import numpy as np
from plyfile import PlyData, PlyElement

def load_ply_vertex_array(path):
    """Load full structured vertex array from a PLY file."""
    ply = PlyData.read(path)
    verts = ply['vertex'].data  # numpy structured array with all properties
    return verts, ply

def write_filtered_ply(path, original_ply: PlyData, filtered_vertices):
    """
    Write a new PLY keeping the same header/format but replacing
    the vertex element with filtered_vertices (a structured array).
    """
    # Describe new vertex element with same name and dtype
    el = PlyElement.describe(filtered_vertices, 'vertex')
    # Preserve other non-vertex elements if any
    other_elements = [e for e in original_ply.elements if e.name != 'vertex']
    # Build new PlyData; keep format/version same as original
    new_ply = PlyData([el] + other_elements,
                      text=original_ply.text,
                      byte_order=original_ply.byte_order)
    new_ply.write(path)

def extract_top_dynamic_points(
    input_dir: str,
    output_dir: str,
    top_percent: float
):
    """
    For a folder of PLY frames (same topology & ordering), compute
    per-point total motion magnitude using x,y,z, select top `top_percent`
    dynamic points, and write filtered PLYs with all original properties.
    """
    ply_paths = sorted(glob.glob(os.path.join(input_dir, '*.ply')))
    if not ply_paths:
        raise ValueError(f"No PLY files found in {input_dir}")

    # Load all frames: keep structured arrays and also extract xyz for motion
    vertex_structs = []
    xyz_frames = []
    ply_objs = []
    for p in ply_paths:
        verts, ply_obj = load_ply_vertex_array(p)
        ply_objs.append(ply_obj)
        vertex_structs.append(verts)
        # Extract x,y,z as float arrays for motion computation
        x = verts['x'].astype(np.float64)
        y = verts['y'].astype(np.float64)
        z = verts['z'].astype(np.float64)
        xyz = np.vstack((x, y, z)).T  # shape (N,3)
        xyz_frames.append(xyz)

    num_frames = len(xyz_frames)
    point_counts = [f.shape[0] for f in xyz_frames]
    min_points = min(point_counts)
    max_points = max(point_counts)

    print(f"Loaded {num_frames} frames")
    print(f"Point count range: {min_points} - {max_points}")

    if min_points != max_points:
        print("⚠️  Warning: Point counts vary across frames; truncating to minimum for consistency")
        # Truncate both structured and xyz arrays
        xyz_frames = [frame[:min_points] for frame in xyz_frames]
        vertex_structs = [vs[:min_points] for vs in vertex_structs]
        N = min_points
    else:
        N = point_counts[0]
        print(f"All frames have consistent {N} points")

    # Stack into (F, N, 3)
    data = np.stack(xyz_frames, axis=0)  # shape = (F, N, 3)

    # Motion: sum of frame-to-frame L2 displacements
    disp = np.linalg.norm(np.diff(data, axis=0), axis=2)  # (F-1, N)
    motion = disp.sum(axis=0)  # (N,)

    k = max(1, int(np.ceil(N * top_percent)))
    top_idx = np.argsort(motion)[-k:]
    top_idx.sort()
    print(f"Selecting top {top_percent*100:.1f}% dynamic points: {k} points")

    os.makedirs(output_dir, exist_ok=True)

    # Write filtered PLY for each frame, keeping all original properties
    for i, path in enumerate(ply_paths):
        verts_full = vertex_structs[i]
        filtered = verts_full[top_idx]  # structured array slice
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        write_filtered_ply(out_path, ply_objs[i], filtered)

    print(f"Done! Filtered frames written to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract top-n% dynamic points from a sequence of PLY frames, preserving all vertex properties"
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
        "--percent", type=float, default=0.8,
        help="Fraction of top dynamic points to keep (e.g. 0.1 for top 10%%)"
    )
    args = parser.parse_args()

    extract_top_dynamic_points(
        args.input_dir,
        args.output_dir,
        args.percent
    )