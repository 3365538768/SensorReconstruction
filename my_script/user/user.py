#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dash UI · Supports RangeSliders & Manual Input + Real-time Normal Arrow + Highlighted Region + Axis Reference
Dependencies: dash dash-bootstrap-components plotly plyfile numpy torch pandas
"""

import os, io, json, base64
import numpy as np
import torch
from plyfile import PlyData, PlyElement

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go

# ─────────── Paths ───────────
UPLOAD_DIR, EXPORT_DIR = "uploads", "pred_out"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
REGION_JSON = "region.json"
MODEL_CKPT = "model_final.pth"  # Placeholder for your actual model checkpoint


# ─────────── DummyModel ───────────
# This is a dummy model for demonstration. Replace with your actual model if needed.
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Example: A simple linear layer that takes 3D coords + 128 sensor features
        # and outputs 9 values (e.g., 3 for delta_xyz, 3 for delta_normal, 3 for delta_scale)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3 + 128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 9)
        )

    def forward(self, coords, sens):
        # coords: [B, N, 3], sens: [B, 128]
        B, N, _ = coords.shape
        s = sens.unsqueeze(1).expand(-1, N, -1)  # Expand sensor features to match N points
        x = torch.cat([coords, s], -1)  # Concatenate coords and sensor features
        return self.fc(x)


device = "cpu"  # Use "cuda" if GPU is available and model is trained on GPU
model = DummyModel().to(device)
# Load your actual model weights if available
if os.path.exists(MODEL_CKPT):
    try:
        model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
        print(f"Loaded model from {MODEL_CKPT}")
    except Exception as e:
        print(f"Error loading model from {MODEL_CKPT}: {e}. Using dummy model.")
model.eval()  # Set model to evaluation mode

# ─────────── Global Variables ───────────
global_xyz = None  # Stores the XYZ coordinates of the uploaded PLY

# ─────────── Dash App ───────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Deformation Prediction UI"  # Set browser tab title

# 3D graph component
graph = dcc.Graph(
    id="ply-graph",
    config={"displayModeBar": False, "responsive": True},  # Disable modebar, enable responsiveness
    style={"height": "65vh", "border": "1px solid #dee2e6", "borderRadius": "0.25rem"}
    # Added subtle border and border-radius
)

# RangeSliders & numeric inputs for Bounding Box
x_range = dcc.RangeSlider(id="x-range", min=0, max=1, value=[0, 1], allowCross=False,
                          tooltip={"placement": "bottom", "always_visible": True})
y_range = dcc.RangeSlider(id="y-range", min=0, max=1, value=[0, 1], allowCross=False,
                          tooltip={"placement": "bottom", "always_visible": True})
z_range = dcc.RangeSlider(id="z-range", min=0, max=1, value=[0, 1], allowCross=False,
                          tooltip={"placement": "bottom", "always_visible": True})

num_inputs = dbc.Row([
    dbc.Col(dbc.Input(id="xmin-input", type="number", placeholder="X Min", className="form-control-sm"), width="auto"),
    dbc.Col(dbc.Input(id="xmax-input", type="number", placeholder="X Max", className="form-control-sm"), width="auto"),
    dbc.Col(dbc.Input(id="ymin-input", type="number", placeholder="Y Min", className="form-control-sm"), width="auto"),
    dbc.Col(dbc.Input(id="ymax-input", type="number", placeholder="Y Max", className="form-control-sm"), width="auto"),
    dbc.Col(dbc.Input(id="zmin-input", type="number", placeholder="Z Min", className="form-control-sm"), width="auto"),
    dbc.Col(dbc.Input(id="zmax-input", type="number", placeholder="Z Max", className="form-control-sm"), width="auto"),
], className="g-2 flex-wrap justify-content-center")  # Added justify-content-center for better alignment

# Sliders for Normal Vector Orientation
theta = dcc.Slider(
    0, np.pi, 0.01, value=0, id="theta",
    updatemode="drag",
    marks={0: {"label": "0°"}, np.pi / 2: {"label": "90° (π/2)"}, np.pi: {"label": "180° (π)"}},
    tooltip={"placement": "bottom", "always_visible": True}
)
phi = dcc.Slider(
    0, 2 * np.pi, 0.01, value=0, id="phi",
    updatemode="drag",
    marks={0: {"label": "0°"}, np.pi: {"label": "180° (π)"}, 2 * np.pi: {"label": "360° (2π)"}},
    tooltip={"placement": "bottom", "always_visible": True}
)

# Upload and Save/Predict Buttons
upload_btn = dcc.Upload(
    id="upload",
    children=dbc.Button("Upload Initial PLY", color="primary", className="fs-5 w-100"),
    multiple=False
)
save_btn = dbc.Button("Save & Predict", id="predict", color="success", className="fs-5 w-100")

# Status Log Textarea
status_box = dbc.Textarea(
    id="status", disabled=True,
    style={"height": "10rem", "fontSize": "0.9rem", "fontFamily": "monospace", "backgroundColor": "#f8f9fa",
           "color": "#343a40"}
)

# AABB and Normal JSON Output Textarea
bbox_normal_output = dcc.Textarea(
    id="bbox-normal-output",
    style={"width": "100%", "height": "250px", "fontFamily": "monospace", "fontSize": "0.85rem",
           "backgroundColor": "#e9ecef", "color": "#495057"},
    readOnly=True
)

# App Layout
app.layout = dbc.Container([
    html.H3("Deformation Prediction UI", className="my-3 text-center fs-2"),

    dbc.Row([
        dbc.Col(upload_btn, md=6, className="mb-3"),
        dbc.Col(save_btn, md=6, className="mb-3"),
    ], className="g-3"),  # Use g-3 for consistent gutter

    dbc.Row([
        dbc.Col(graph, md=8, className="mb-3"),  # Graph takes 8 columns
        dbc.Col([  # Controls take 4 columns
            dbc.Card([
                dbc.CardHeader(html.H5("Bounding Box Coordinates", className="mb-0 fs-5")),
                dbc.CardBody([
                    num_inputs,  # Numeric inputs row
                    html.Div([
                        html.H6("X Range", className="fs-6 mt-3 mb-1"), x_range,
                        html.H6("Y Range", className="fs-6 mt-3 mb-1"), y_range,
                        html.H6("Z Range", className="fs-6 mt-3 mb-1"), z_range,
                    ], className="mt-2"),
                ])
            ], className="mb-3 shadow-sm"),  # Added shadow-sm for subtle depth

            dbc.Card([
                dbc.CardHeader(html.H5("Normal Vector Orientation (θ/φ)", className="mb-0 fs-5")),
                dbc.CardBody([
                    html.H6("Theta (θ): Inclination from Z-axis (0 to π)", className="fs-6 mb-1"), theta,
                    html.H6("Phi (φ): Azimuth in XY-plane (0 to 2π)", className="fs-6 mt-3 mb-1"), phi,
                ])
            ], className="mb-3 shadow-sm"),

            dbc.Card([
                dbc.CardHeader(html.H5("Current Bounding Box & Normal (JSON)", className="mb-0 fs-5")),
                dbc.CardBody(bbox_normal_output)
            ], className="mb-3 shadow-sm"),

        ], md=4, className="d-flex flex-column"),  # Use flex-column to make cards fill height if needed
    ]),

    dbc.Card([
        dbc.CardHeader(html.H5("Status Log", className="mb-0 fs-5")),
        dbc.CardBody(status_box)
    ], className="mb-3 shadow-sm")

], fluid=True,
    style={"fontSize": "1.2rem", "backgroundColor": "#f8f9fa", "minHeight": "100vh"})  # Added light background


# ─────────── 1. Upload and Initialize Graph ───────────
@app.callback(
    Output("ply-graph", "figure"),
    Output("x-range", "min"), Output("x-range", "max"), Output("x-range", "value"),
    Output("xmin-input", "value"), Output("xmax-input", "value"),
    Output("y-range", "min"), Output("y-range", "max"), Output("y-range", "value"),
    Output("ymin-input", "value"), Output("ymax-input", "value"),
    Output("z-range", "min"), Output("z-range", "max"), Output("z-range", "value"),
    Output("zmin-input", "value"), Output("zmax-input", "value"),
    Input("upload", "contents"),
    prevent_initial_call=True
)
def on_upload(contents):
    global global_xyz
    if not contents:
        raise dash.exceptions.PreventUpdate

    # Decode PLY from base64
    content_type, content_string = contents.split(",", 1)
    data = base64.b64decode(content_string)

    # Save uploaded PLY to disk (optional, but useful for debugging/reuse)
    path = os.path.join(UPLOAD_DIR, "init.ply")
    with open(path, "wb") as f:
        f.write(data)

    # Read PLY data using plyfile
    ply = PlyData.read(io.BytesIO(data))
    v = ply['vertex'].data
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    global_xyz = xyz  # Store for later use in callbacks

    # Calculate initial bounds for sliders and inputs
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()

    # Create initial Plotly figure with point cloud
    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers", marker=dict(size=2, color="blue", opacity=0.5),
        name="Point Cloud"
    ))
    fig.update_layout(scene=dict(aspectmode="data"), title="Point Cloud Preview", showlegend=True)

    # Add axes reference lines (X, Y, Z)
    Lax = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.5  # Length for axis lines
    fig.add_trace(go.Scatter3d(
        x=[0, Lax], y=[0, 0], z=[0, 0], mode="lines",
        line=dict(color="red", width=5), name="X-axis"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, Lax], z=[0, 0], mode="lines",
        line=dict(color="green", width=5), name="Y-axis"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, Lax], mode="lines",
        line=dict(color="blue", width=5), name="Z-axis"
    ))

    # Add a placeholder for the normal arrow (will be updated by another callback)
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0], u=[0], v=[0], w=[0],  # Initial dummy arrow
        colorscale=[[0, "magenta"], [1, "magenta"]],
        sizemode="absolute", showscale=False,
        anchor="tail", name="normal_arrow"
    ))

    # Return figure, and set slider/input defaults based on loaded PLY bounds
    return (
        fig,
        xmin, xmax, [xmin, xmax],  # X-range slider min, max, value
        xmin, xmax,  # X-min/max input values
        ymin, ymax, [ymin, ymax],  # Y-range slider min, max, value
        ymin, ymax,  # Y-min/max input values
        zmin, zmax, [zmin, zmax],  # Z-range slider min, max, value
        zmin, zmax  # Z-min/max input values
    )


# ─────────── 2. Update Bounding Box & Highlight ───────────
@app.callback(
    Output("ply-graph", "figure", allow_duplicate=True),
    Input("x-range", "value"), Input("xmin-input", "value"), Input("xmax-input", "value"),
    Input("y-range", "value"), Input("ymin-input", "value"), Input("ymax-input", "value"),
    Input("z-range", "value"), Input("zmin-input", "value"), Input("zmax-input", "value"),
    State("ply-graph", "figure"),
    prevent_initial_call=True
)
def update_region(xr, xmin_i, xmax_i,
                  yr, ymin_i, ymax_i,
                  zr, zmin_i, zmax_i,
                  fig):
    if global_xyz is None:  # Ensure PLY data is loaded
        raise dash.exceptions.PreventUpdate

    # Determine current bounding box values, prioritizing manual input if available
    xmin = xmin_i if xmin_i is not None else xr[0]
    xmax = xmax_i if xmax_i is not None else xr[1]
    ymin = ymin_i if ymin_i is not None else yr[0]
    ymax = ymax_i if ymax_i is not None else yr[1]
    zmin = zmin_i if zmin_i is not None else zr[0]
    zmax = zmax_i if zmax_i is not None else zr[1]

    # Create a mask to identify points within the selected bounding box
    mask = (
            (global_xyz[:, 0] >= xmin) & (global_xyz[:, 0] <= xmax) &
            (global_xyz[:, 1] >= ymin) & (global_xyz[:, 1] <= ymax) &
            (global_xyz[:, 2] >= zmin) & (global_xyz[:, 2] <= zmax)
    )
    pts_in = global_xyz[mask]  # Points inside the bounding box

    # Define bounding box corners and edges for visualization
    corners = np.array([[i, j, k] for i in [xmin, xmax]
                        for j in [ymin, ymax]
                        for k in [zmin, zmax]])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5),
             (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

    # Rebuild figure data, removing old bbox lines and highlight traces
    new_data = []
    for d in fig["data"]:
        if d.get("name") in ("bbox_line", "highlight"):
            continue
        new_data.append(d)

    # Add new bounding box lines
    for i, j in edges:
        p1, p2 = corners[i], corners[j]
        new_data.append(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode="lines", line=dict(color="red", width=4),
            name="bbox_line", showlegend=False  # Hide from legend
        ))

    # Add highlight for points inside the bounding box
    if pts_in.shape[0] > 0:  # Only add if there are points to highlight
        new_data.append(go.Scatter3d(
            x=pts_in[:, 0], y=pts_in[:, 1], z=pts_in[:, 2],
            mode="markers",
            marker=dict(size=4, color="green", opacity=1.0),
            name="highlight", showlegend=False  # Hide from legend
        ))

    fig["data"] = new_data
    return fig


# ─────────── 3. Update Normal Arrow ───────────
@app.callback(
    Output("ply-graph", "figure", allow_duplicate=True),
    Input("theta", "value"), Input("phi", "value"),
    State("ply-graph", "figure"),
    prevent_initial_call=True
)
def update_arrow(th, ph, fig):
    # Retrieve current bbox center from the existing bbox_line traces
    # This ensures the arrow originates from the center of the currently displayed bbox
    xs, ys, zs = [], [], []
    for tr in fig["data"]:
        if tr.get("name") == "bbox_line":
            xs.extend(tr["x"]);
            ys.extend(tr["y"]);
            zs.extend(tr["z"])

    if not xs:  # If no bbox lines exist (e.g., before upload), prevent update
        raise dash.exceptions.PreventUpdate

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    cx, cy, cz = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)  # Bbox center

    # Calculate normal vector components from spherical coordinates (theta, phi)
    nx = np.sin(th) * np.cos(ph)
    ny = np.sin(th) * np.sin(ph)
    nz = np.cos(th)

    # Determine arrow length based on bbox diagonal for proper scaling
    diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    L = 0.3 * diag  # Arrow length is 30% of bbox diagonal

    ux, uy, uz = nx * L, ny * L, nz * L  # Arrow vector components

    # Remove old normal arrow trace and add new one
    new_data = [d for d in fig["data"] if d.get("name") != "normal_arrow"]
    new_data.append(go.Cone(
        x=[cx], y=[cy], z=[cz],
        u=[ux], v=[uy], w=[uz],
        sizemode="absolute", sizeref=L * 0.25,  # sizeref controls cone head size
        colorscale=[[0, "magenta"], [1, "magenta"]],
        showscale=False, anchor="tail", name="normal_arrow", showlegend=False  # Hide from legend
    ))
    fig["data"] = new_data
    return fig


# ─────────── 4. Display Current Bounding Box & Normal as JSON ───────────
@app.callback(
    Output("bbox-normal-output", "value"),
    Input("x-range", "value"), Input("xmin-input", "value"), Input("xmax-input", "value"),
    Input("y-range", "value"), Input("ymin-input", "value"), Input("ymax-input", "value"),
    Input("z-range", "value"), Input("zmin-input", "value"), Input("zmax-input", "value"),
    Input("theta", "value"), Input("phi", "value"),
    prevent_initial_call=False  # Allow initial call to populate on load
)
def update_bbox_normal_json(xr, xmin_i, xmax_i,
                            yr, ymin_i, ymax_i,
                            zr, zmin_i, zmax_i,
                            th, ph):
    # Determine current bbox values, prioritizing manual input if available
    xmin = xmin_i if xmin_i is not None else xr[0]
    xmax = xmax_i if xmax_i is not None else xr[1]
    ymin = ymin_i if ymin_i is not None else yr[0]
    ymax = ymax_i if ymax_i is not None else yr[1]
    zmin = zmin_i if zmin_i is not None else zr[0]
    zmax = zmax_i if zmax_i is not None else zr[1]

    # Calculate normal vector components from spherical coordinates (theta, phi)
    nx = np.sin(th) * np.cos(ph)
    ny = np.sin(th) * np.sin(ph)
    nz = np.cos(th)

    # Construct the dictionary matching the requested JSON format
    output_dict = {
        "bbox": [
            [xmin, ymin, zmin],
            [xmax, ymax, zmax]
        ],
        "normal": [nx, ny, nz]
    }

    # Return as a formatted JSON string with 2-space indentation
    return json.dumps(output_dict, indent=2)


# ─────────── 5. Save BBox & Normal to JSON and Run Prediction ───────────
@app.callback(
    Output("status", "value"),
    Input("predict", "n_clicks"),
    # States from all inputs that define bbox and normal
    State("x-range", "value"), State("xmin-input", "value"), State("xmax-input", "value"),
    State("y-range", "value"), State("ymin-input", "value"), State("ymax-input", "value"),
    State("z-range", "value"), State("zmin-input", "value"), State("zmax-input", "value"),
    State("theta", "value"), State("phi", "value"),
    prevent_initial_call=True
)
def do_predict(_, xr, xmin_i, xmax_i, yr, ymin_i, ymax_i, zr, zmin_i, zmax_i, th, ph):
    # Determine final bbox values (same logic as in update_region and update_bbox_normal_json)
    xmin = xmin_i if xmin_i is not None else xr[0]
    xmax = xmax_i if xmax_i is not None else xr[1]
    ymin = ymin_i if ymin_i is not None else yr[0]
    ymax = ymax_i if ymax_i is not None else yr[1]
    zmin = zmin_i if zmin_i is not None else zr[0]
    zmax = zmax_i if zmax_i is not None else zr[1]

    bbox = [[xmin, ymin, zmin], [xmax, ymax, zmax]]  # Form the bbox list of lists

    # Calculate normal vector from theta and phi
    normal = [np.sin(th) * np.cos(ph),
              np.sin(th) * np.sin(ph),
              np.cos(th)]

    # Save bbox and normal to region.json
    json.dump({"bbox": bbox, "normal": normal},
              open(REGION_JSON, "w"), indent=2)

    # Try to run the prediction function
    try:
        # Pass the path to the uploaded PLY, and the determined bbox/normal
        out_ply = run_predict(os.path.join(UPLOAD_DIR, "init.ply"), bbox, normal)
        return f"✅ Exported predicted PLY to: {out_ply}"
    except FileNotFoundError:
        return "❌ Failed: Initial PLY file not found. Please upload a PLY first."
    except Exception as e:
        return f"❌ Prediction failed: {e}"


# ─────────── Prediction Function (remains largely the same) ───────────
# This function is called by the Dash app to perform the prediction logic.
def run_predict(ply_path, bbox, normal):
    # Read the initial PLY data
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    # Extract XYZ, Normals (nx,ny,nz), and Scales (scale_0,1,2)
    # Ensure these properties exist in your PLY files or adjust accordingly
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    # Check if normals exist, otherwise create dummy ones (important for model input)
    if 'nx' in v.dtype.names:
        nrm = np.vstack([v['nx'], v['ny'], v['nz']]).T.astype(np.float32)
    else:
        print("Warning: Normals (nx,ny,nz) not found in PLY. Using dummy normals.")
        nrm = np.zeros_like(xyz, dtype=np.float32)  # Placeholder if no normals
    # Check if scales exist, otherwise create dummy ones
    if 'scale_0' in v.dtype.names:
        scl = np.vstack([v['scale_0'], v['scale_1'], v['scale_2']]).T.astype(np.float32)
    else:
        print("Warning: Scales (scale_0,1,2) not found in PLY. Using dummy scales.")
        scl = np.ones_like(xyz, dtype=np.float32)  # Placeholder if no scales

    # Calculate center and scale of the selected bounding box
    ctr = (np.array(bbox[0]) + np.array(bbox[1])) / 2
    scale = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))

    # Calculate rotation matrix R to align the provided normal to Z-axis (similar to your training code)
    n = np.array(normal) / np.linalg.norm(normal)  # Normalize input normal
    z = np.array([0, 0, 1])
    v_ = np.cross(n, z);
    c = n.dot(z)
    vx = np.array([[0, -v_[2], v_[1]], [v_[2], 0, -v_[0]], [-v_[1], v_[0], 0]])
    R = np.eye(3) + vx + vx @ vx / (1 + c + 1e-8)  # Rotation matrix

    # Identify points within the selected bounding box
    mask = np.all((xyz >= bbox[0]) & (xyz <= bbox[1]), axis=1)
    idx = np.where(mask)[0]  # Indices of points inside the bbox

    # Transform selected points to the normalized local coordinate system (centered, scaled, rotated)
    xyz_c = ((xyz[idx] - ctr) / scale) @ R.T

    # --- Model Prediction ---
    # Create a dummy sensor feature (your actual model might need a real sensor input here)
    # The DummyModel expects a [B, 128] sensor input.
    # For a single prediction, B=1.
    sens_feat = torch.zeros(1, 128)  # Replace with actual sensor data if available

    # Run the model to predict delta values (delta_xyz, delta_normal, delta_scale)
    # Model input: [1, num_points_in_bbox, 3] for xyz_c, [1, 128] for sens_feat
    # Model output: [1, num_points_in_bbox, 9] (delta_xyz, delta_normal, delta_scale)
    with torch.no_grad():  # Ensure no gradients are computed during inference
        delta = model(torch.from_numpy(xyz_c)[None].float(),
                      sens_feat)[0].detach().cpu().numpy()  # Get numpy array, remove batch dim

    # Apply predicted deltas to positions, normals, and scales
    xyz_c_deformed = xyz_c + delta[:, :3]  # Apply delta_xyz
    nrm_c_deformed = (nrm[idx] @ R.T) + delta[:, 3:6]  # Apply delta_normal to transformed normals
    scl_deformed = scl[idx] + delta[:, 6:9]  # Apply delta_scale

    # Transform deformed points and normals back to original world coordinate system
    xyz_deformed_world = xyz_c_deformed @ R * scale + ctr
    nrm_deformed_world = nrm_c_deformed @ R

    # Update the original full point cloud arrays with deformed values
    xyz[idx] = xyz_deformed_world
    nrm[idx] = nrm_deformed_world
    scl[idx] = scl_deformed  # Update scales for affected points

    # Prepare data for PLY export
    out = np.empty(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4')
    ])
    out['x'], out['y'], out['z'] = xyz.T
    out['nx'], out['ny'], out['nz'] = nrm.T
    out['scale_0'], out['scale_1'], out['scale_2'] = scl.T

    # Save the modified PLY file
    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_path = os.path.join(EXPORT_DIR, "frame_001_pred.ply")  # Fixed output name for simplicity
    PlyData([PlyElement.describe(out, 'vertex')]).write(out_path)

    return out_path


# ─────────── Run server ───────────
if __name__ == "__main__":
    # app.run_server has been replaced by app.run in newer Dash versions
    app.run(debug=True, port=8050)