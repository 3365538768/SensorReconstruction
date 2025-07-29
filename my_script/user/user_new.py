#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Dash UI · Modern Architecture with Unified State Management
Fixes: Callback conflicts, bidirectional sync, state management issues
Dependencies: dash dash-bootstrap-components plotly plyfile numpy torch pandas
"""

import os
import io
import json
import base64
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from copy import deepcopy

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx, callback, clientside_callback
import plotly.graph_objs as go


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration and Constants
# ═══════════════════════════════════════════════════════════════════════════════

UPLOAD_DIR = "uploads"
EXPORT_DIR = "pred_out"
REGION_JSON = "region.json"
MODEL_CKPT = "model_final.pth"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes for State Management
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PointCloudData:
    """Point cloud data container"""
    xyz: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
    loaded: bool = False


@dataclass
class BoundingBoxState:
    """Bounding box state container"""
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    zmin: float = 0.0
    zmax: float = 1.0


@dataclass
class NormalVectorState:
    """Normal vector state container"""
    theta: float = 0.0  # Inclination from Z-axis (0 to π)
    phi: float = 0.0    # Azimuth in XY-plane (0 to 2π)
    
    @property
    def vector(self) -> List[float]:
        """Calculate normal vector components"""
        nx = np.sin(self.theta) * np.cos(self.phi)
        ny = np.sin(self.theta) * np.sin(self.phi)
        nz = np.cos(self.theta)
        return [nx, ny, nz]


@dataclass
class AppState:
    """Unified application state"""
    pointcloud: PointCloudData
    bbox: BoundingBoxState
    normal: NormalVectorState
    status_log: List[str]
    last_update: float
    
    def add_status(self, message: str):
        """Add status message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")
        if len(self.status_log) > 20:  # Keep last 20 messages
            self.status_log = self.status_log[-20:]
        self.last_update = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# Global State Manager
# ═══════════════════════════════════════════════════════════════════════════════

class StateManager:
    """Thread-safe global state manager"""
    
    def __init__(self):
        self._state = AppState(
            pointcloud=PointCloudData(),
            bbox=BoundingBoxState(),
            normal=NormalVectorState(),
            status_log=[],
            last_update=time.time()
        )
        self._lock = threading.Lock()
    
    def get_state(self) -> AppState:
        """Get current state (deep copy for thread safety)"""
        with self._lock:
            return deepcopy(self._state)
    
    def update_pointcloud(self, xyz: np.ndarray, normals=None, scales=None):
        """Update point cloud data"""
        with self._lock:
            bounds = {
                'x': (xyz[:, 0].min(), xyz[:, 0].max()),
                'y': (xyz[:, 1].min(), xyz[:, 1].max()),
                'z': (xyz[:, 2].min(), xyz[:, 2].max())
            }
            self._state.pointcloud = PointCloudData(
                xyz=xyz, normals=normals, scales=scales,
                bounds=bounds, loaded=True
            )
            # Update bbox to point cloud bounds
            self._state.bbox = BoundingBoxState(
                xmin=bounds['x'][0], xmax=bounds['x'][1],
                ymin=bounds['y'][0], ymax=bounds['y'][1],
                zmin=bounds['z'][0], zmax=bounds['z'][1]
            )
            self._state.add_status(f"Point cloud loaded: {len(xyz)} points")
    
    def update_bbox(self, **kwargs):
        """Update bounding box parameters"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state.bbox, key):
                    setattr(self._state.bbox, key, float(value))
            self._state.add_status("Bounding box updated")
    
    def update_normal(self, theta=None, phi=None):
        """Update normal vector parameters"""
        with self._lock:
            if theta is not None:
                self._state.normal.theta = float(theta)
            if phi is not None:
                self._state.normal.phi = float(phi)
            self._state.add_status("Normal vector updated")
    
    def get_bbox_json(self) -> str:
        """Get current bbox and normal as JSON string"""
        state = self.get_state()
        bbox = state.bbox
        normal = state.normal.vector
        
        output_dict = {
            "bbox": [
                [bbox.xmin, bbox.ymin, bbox.zmin],
                [bbox.xmax, bbox.ymax, bbox.zmax]
            ],
            "normal": normal
        }
        return json.dumps(output_dict, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Component Factory
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentFactory:
    """Factory for creating reusable UI components"""
    
    @staticmethod
    def create_range_slider(id_prefix: str, label: str, min_val=0, max_val=1) -> html.Div:
        """Create a range slider with synchronized inputs"""
        return html.Div([
            html.H6(f"{label} Range", className="fs-6 mt-3 mb-1"),
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id=f"{id_prefix}-min-input",
                        type="number",
                        placeholder=f"{label} Min",
                        className="form-control-sm mb-2",
                        size="sm"
                    )
                ], width=6),
                dbc.Col([
                    dbc.Input(
                        id=f"{id_prefix}-max-input",
                        type="number", 
                        placeholder=f"{label} Max",
                        className="form-control-sm mb-2",
                        size="sm"
                    )
                ], width=6),
            ]),
            dcc.RangeSlider(
                id=f"{id_prefix}-range-slider",
                min=min_val,
                max=max_val,
                value=[min_val, max_val],
                allowCross=False,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag"
            )
        ])
    
    @staticmethod
    def create_angle_slider(id_name: str, label: str, min_val: float, 
                          max_val: float, step: float = 0.01) -> html.Div:
        """Create an angle slider with proper labeling"""
        marks = {}
        if max_val == np.pi:  # Theta slider
            marks = {
                0: {"label": "0°"},
                np.pi/2: {"label": "90°"},
                np.pi: {"label": "180°"}
            }
        elif max_val == 2*np.pi:  # Phi slider
            marks = {
                0: {"label": "0°"},
                np.pi: {"label": "180°"},
                2*np.pi: {"label": "360°"}
            }
        
        return html.Div([
            html.H6(label, className="fs-6 mb-1"),
            dcc.Slider(
                id=id_name,
                min=min_val,
                max=max_val,
                step=step,
                value=0,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag"
            )
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization Engine
# ═══════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """Handles 3D visualization and figure updates"""
    
    @staticmethod
    def create_empty_figure() -> go.Figure:
        """Create empty placeholder figure"""
        fig = go.Figure()
        fig.update_layout(
            scene=dict(aspectmode="data"),
            title="Upload a PLY file to begin",
            showlegend=True,
            height=600
        )
        return fig
    
    @staticmethod
    def create_pointcloud_figure(xyz: np.ndarray) -> go.Figure:
        """Create initial point cloud figure with axes"""
        fig = go.Figure()
        
        # Add point cloud
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.5),
            name="Point Cloud",
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
        ))
        
        # Add coordinate axes
        VisualizationEngine._add_coordinate_axes(fig, xyz)
        
        fig.update_layout(
            scene=dict(aspectmode="data"),
            title="Point Cloud Visualization",
            showlegend=True,
            height=600
        )
        
        return fig
    
    @staticmethod
    def _add_coordinate_axes(fig: go.Figure, xyz: np.ndarray):
        """Add coordinate axes to figure"""
        xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
        ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
        zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()
        
        # Axis length
        axis_length = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.3
        
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode="lines",
            line=dict(color="red", width=5),
            name="X-axis",
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode="lines",
            line=dict(color="green", width=5),
            name="Y-axis",
            showlegend=False,
            hoverinfo="skip"
        ))
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode="lines",
            line=dict(color="blue", width=5),
            name="Z-axis",
            showlegend=False,
            hoverinfo="skip"
        ))
    
    @staticmethod
    def update_figure_with_selection(fig: go.Figure, state: AppState) -> go.Figure:
        """Update figure with bounding box and normal arrow"""
        if not state.pointcloud.loaded:
            return fig
        
        # Remove old selection visualizations
        fig.data = [trace for trace in fig.data 
                   if trace.name not in ["bbox_line", "highlight", "normal_arrow"]]
        
        # Add bounding box
        VisualizationEngine._add_bounding_box(fig, state.bbox)
        
        # Add highlighted points
        VisualizationEngine._add_highlighted_points(fig, state)
        
        # Add normal arrow
        VisualizationEngine._add_normal_arrow(fig, state)
        
        return fig
    
    @staticmethod
    def _add_bounding_box(fig: go.Figure, bbox: BoundingBoxState):
        """Add bounding box wireframe to figure"""
        # Define box corners
        corners = np.array([
            [bbox.xmin, bbox.ymin, bbox.zmin],  # 0
            [bbox.xmax, bbox.ymin, bbox.zmin],  # 1
            [bbox.xmin, bbox.ymax, bbox.zmin],  # 2
            [bbox.xmax, bbox.ymax, bbox.zmin],  # 3
            [bbox.xmin, bbox.ymin, bbox.zmax],  # 4
            [bbox.xmax, bbox.ymin, bbox.zmax],  # 5
            [bbox.xmin, bbox.ymax, bbox.zmax],  # 6
            [bbox.xmax, bbox.ymax, bbox.zmax],  # 7
        ])
        
        # Define edges
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Add edges as line traces
        for i, (start, end) in enumerate(edges):
            p1, p2 = corners[start], corners[end]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(color="red", width=4),
                name="bbox_line",
                showlegend=False,
                hoverinfo="skip"
            ))
    
    @staticmethod
    def _add_highlighted_points(fig: go.Figure, state: AppState):
        """Add highlighted points within bounding box"""
        xyz = state.pointcloud.xyz
        bbox = state.bbox
        
        # Create mask for points within bbox
        mask = (
            (xyz[:, 0] >= bbox.xmin) & (xyz[:, 0] <= bbox.xmax) &
            (xyz[:, 1] >= bbox.ymin) & (xyz[:, 1] <= bbox.ymax) &
            (xyz[:, 2] >= bbox.zmin) & (xyz[:, 2] <= bbox.zmax)
        )
        
        points_inside = xyz[mask]
        
        if len(points_inside) > 0:
            fig.add_trace(go.Scatter3d(
                x=points_inside[:, 0],
                y=points_inside[:, 1],
                z=points_inside[:, 2],
                mode="markers",
                marker=dict(size=4, color="green", opacity=1.0),
                name="highlight",
                showlegend=False,
                hovertemplate="Selected Point<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
            ))
    
    @staticmethod
    def _add_normal_arrow(fig: go.Figure, state: AppState):
        """Add normal vector arrow at bbox center"""
        bbox = state.bbox
        normal = state.normal
        
        # Calculate bbox center
        center = np.array([
            (bbox.xmin + bbox.xmax) / 2,
            (bbox.ymin + bbox.ymax) / 2,
            (bbox.zmin + bbox.zmax) / 2
        ])
        
        # Calculate arrow length (30% of bbox diagonal)
        diag = np.linalg.norm([
            bbox.xmax - bbox.xmin,
            bbox.ymax - bbox.ymin,
            bbox.zmax - bbox.zmin
        ])
        arrow_length = 0.3 * diag
        
        # Calculate arrow vector
        normal_vec = np.array(normal.vector)
        arrow_vec = normal_vec * arrow_length
        
        # Add cone (arrow)
        fig.add_trace(go.Cone(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            u=[arrow_vec[0]],
            v=[arrow_vec[1]],
            w=[arrow_vec[2]],
            sizemode="absolute",
            sizeref=arrow_length * 0.25,
            colorscale=[[0, "magenta"], [1, "magenta"]],
            showscale=False,
            anchor="tail",
            name="normal_arrow",
            showlegend=False,
            hovertemplate="Normal Vector<br>Direction: [%{u:.3f}, %{v:.3f}, %{w:.3f}]<extra></extra>"
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Model and Prediction
# ═══════════════════════════════════════════════════════════════════════════════

class DummyModel(torch.nn.Module):
    """Dummy model for demonstration (same as original)"""
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3 + 128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 9)
        )
    
    def forward(self, coords, sens):
        B, N, _ = coords.shape
        s = sens.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([coords, s], -1)
        return self.fc(x)


# Load model
device = "cpu"
model = DummyModel().to(device)
if os.path.exists(MODEL_CKPT):
    try:
        model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
        print(f"Loaded model from {MODEL_CKPT}")
    except Exception as e:
        print(f"Error loading model: {e}")
model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# Initialize Global State
# ═══════════════════════════════════════════════════════════════════════════════

state_manager = StateManager()


# ═══════════════════════════════════════════════════════════════════════════════
# Dash App Setup
# ═══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Enhanced Deformation Prediction UI"

# Main 3D graph
main_graph = dcc.Graph(
    id="main-graph",
    figure=VisualizationEngine.create_empty_figure(),
    config={"displayModeBar": False, "responsive": True},
    style={"height": "65vh", "border": "1px solid #dee2e6", "borderRadius": "0.25rem"}
)

# Upload component
upload_component = dcc.Upload(
    id="file-upload",
    children=dbc.Button(
        "Upload Initial PLY", 
        color="primary", 
        className="fs-5 w-100",
        id="upload-button"
    ),
    multiple=False
)

# Action buttons
action_buttons = dbc.Row([
    dbc.Col([
        dbc.Button(
            "Reset View", 
            id="reset-button", 
            color="secondary", 
            className="w-100 mb-2"
        )
    ], width=12),
    dbc.Col([
        dbc.Button(
            "Save & Predict", 
            id="predict-button", 
            color="success", 
            className="fs-5 w-100"
        )
    ], width=12),
])

# Status and JSON output
status_output = dbc.Textarea(
    id="status-output",
    disabled=True,
    style={
        "height": "8rem",
        "fontSize": "0.9rem",
        "fontFamily": "monospace",
        "backgroundColor": "#f8f9fa"
    }
)

json_output = dcc.Textarea(
    id="json-output",
    readOnly=True,
    style={
        "width": "100%",
        "height": "250px",
        "fontFamily": "monospace",
        "fontSize": "0.85rem",
        "backgroundColor": "#e9ecef"
    }
)

# App Layout
app.layout = dbc.Container([
    # Header
    html.H3("Enhanced Deformation Prediction UI", className="my-3 text-center fs-2"),
    
    # Hidden div for state storage
    html.Div(id="state-store", style={"display": "none"}),
    
    # Main content
    dbc.Row([
        # Left column - 3D visualization
        dbc.Col([
            upload_component,
            html.Hr(),
            main_graph
        ], md=8, className="mb-3"),
        
        # Right column - Controls
        dbc.Col([
            # Bounding box controls
            dbc.Card([
                dbc.CardHeader(html.H5("Bounding Box Selection", className="mb-0")),
                dbc.CardBody([
                    ComponentFactory.create_range_slider("x", "X"),
                    ComponentFactory.create_range_slider("y", "Y"),
                    ComponentFactory.create_range_slider("z", "Z"),
                ])
            ], className="mb-3"),
            
            # Normal vector controls  
            dbc.Card([
                dbc.CardHeader(html.H5("Normal Vector Orientation", className="mb-0")),
                dbc.CardBody([
                    ComponentFactory.create_angle_slider(
                        "theta-slider", 
                        "Theta (θ): Inclination from Z-axis", 
                        0, np.pi
                    ),
                    ComponentFactory.create_angle_slider(
                        "phi-slider",
                        "Phi (φ): Azimuth in XY-plane", 
                        0, 2*np.pi
                    ),
                ])
            ], className="mb-3"),
            
            # Action buttons
            dbc.Card([
                dbc.CardHeader(html.H5("Actions", className="mb-0")),
                dbc.CardBody(action_buttons)
            ], className="mb-3"),
            
            # JSON output
            dbc.Card([
                dbc.CardHeader(html.H5("Configuration (JSON)", className="mb-0")),
                dbc.CardBody(json_output)
            ], className="mb-3"),
            
        ], md=4),
    ]),
    
    # Status log
    dbc.Card([
        dbc.CardHeader(html.H5("Status Log", className="mb-0")),
        dbc.CardBody(status_output)
    ], className="mb-3")
    
], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})


# ═══════════════════════════════════════════════════════════════════════════════
# Callback Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# File upload callback
@callback(
    Output("main-graph", "figure"),
    Output("x-range-slider", "min"), Output("x-range-slider", "max"), Output("x-range-slider", "value"),
    Output("x-min-input", "value"), Output("x-max-input", "value"),
    Output("y-range-slider", "min"), Output("y-range-slider", "max"), Output("y-range-slider", "value"),
    Output("y-min-input", "value"), Output("y-max-input", "value"),
    Output("z-range-slider", "min"), Output("z-range-slider", "max"), Output("z-range-slider", "value"),
    Output("z-min-input", "value"), Output("z-max-input", "value"),
    Input("file-upload", "contents"),
    prevent_initial_call=True
)
def handle_file_upload(contents):
    """Handle PLY file upload and initialize UI components"""
    if not contents:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Decode and save file
        content_type, content_string = contents.split(",", 1)
        data = base64.b64decode(content_string)
        
        upload_path = os.path.join(UPLOAD_DIR, "init.ply")
        with open(upload_path, "wb") as f:
            f.write(data)
        
        # Read PLY data
        ply = PlyData.read(io.BytesIO(data))
        v = ply['vertex'].data
        xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
        
        # Extract normals and scales if available
        normals = None
        if all(key in v.dtype.names for key in ['nx', 'ny', 'nz']):
            normals = np.vstack([v['nx'], v['ny'], v['nz']]).T.astype(np.float32)
        
        scales = None
        if all(key in v.dtype.names for key in ['scale_0', 'scale_1', 'scale_2']):
            scales = np.vstack([v['scale_0'], v['scale_1'], v['scale_2']]).T.astype(np.float32)
        
        # Update global state
        state_manager.update_pointcloud(xyz, normals, scales)
        
        # Get updated state
        state = state_manager.get_state()
        bounds = state.pointcloud.bounds
        
        # Create initial figure
        fig = VisualizationEngine.create_pointcloud_figure(xyz)
        fig = VisualizationEngine.update_figure_with_selection(fig, state)
        
        return (
            fig,
            # X controls
            bounds['x'][0], bounds['x'][1], [bounds['x'][0], bounds['x'][1]],
            bounds['x'][0], bounds['x'][1],
            # Y controls  
            bounds['y'][0], bounds['y'][1], [bounds['y'][0], bounds['y'][1]],
            bounds['y'][0], bounds['y'][1],
            # Z controls
            bounds['z'][0], bounds['z'][1], [bounds['z'][0], bounds['z'][1]],
            bounds['z'][0], bounds['z'][1]
        )
        
    except Exception as e:
        state_manager.get_state().add_status(f"Upload failed: {str(e)}")
        raise dash.exceptions.PreventUpdate


# Unified update callback for bounding box changes
@callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("x-range-slider", "value", allow_duplicate=True),
    Output("x-min-input", "value", allow_duplicate=True),
    Output("x-max-input", "value", allow_duplicate=True),
    Output("y-range-slider", "value", allow_duplicate=True),
    Output("y-min-input", "value", allow_duplicate=True),
    Output("y-max-input", "value", allow_duplicate=True),
    Output("z-range-slider", "value", allow_duplicate=True),
    Output("z-min-input", "value", allow_duplicate=True),
    Output("z-max-input", "value", allow_duplicate=True),
    [
        Input("x-range-slider", "value"),
        Input("x-min-input", "value"), Input("x-max-input", "value"),
        Input("y-range-slider", "value"), 
        Input("y-min-input", "value"), Input("y-max-input", "value"),
        Input("z-range-slider", "value"),
        Input("z-min-input", "value"), Input("z-max-input", "value")
    ],
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def update_bounding_box(x_range, x_min, x_max, y_range, y_min, y_max, 
                       z_range, z_min, z_max, current_figure):
    """Unified callback for bounding box updates with bidirectional sync"""
    state = state_manager.get_state()
    if not state.pointcloud.loaded:
        raise dash.exceptions.PreventUpdate
    
    # Determine which input triggered the callback
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else None
    
    # Initialize values from current state
    bbox_updates = {}
    
    # Handle different input types and sync
    if trigger and "x-range-slider" in trigger:
        bbox_updates.update(xmin=x_range[0], xmax=x_range[1])
        sync_x_min, sync_x_max = x_range[0], x_range[1]
    elif trigger and ("x-min-input" in trigger or "x-max-input" in trigger):
        x_min_val = x_min if x_min is not None else state.bbox.xmin
        x_max_val = x_max if x_max is not None else state.bbox.xmax
        bbox_updates.update(xmin=x_min_val, xmax=x_max_val)
        sync_x_min, sync_x_max = x_min_val, x_max_val
    else:
        sync_x_min, sync_x_max = state.bbox.xmin, state.bbox.xmax
    
    if trigger and "y-range-slider" in trigger:
        bbox_updates.update(ymin=y_range[0], ymax=y_range[1])
        sync_y_min, sync_y_max = y_range[0], y_range[1]
    elif trigger and ("y-min-input" in trigger or "y-max-input" in trigger):
        y_min_val = y_min if y_min is not None else state.bbox.ymin
        y_max_val = y_max if y_max is not None else state.bbox.ymax
        bbox_updates.update(ymin=y_min_val, ymax=y_max_val)
        sync_y_min, sync_y_max = y_min_val, y_max_val
    else:
        sync_y_min, sync_y_max = state.bbox.ymin, state.bbox.ymax
    
    if trigger and "z-range-slider" in trigger:
        bbox_updates.update(zmin=z_range[0], zmax=z_range[1])
        sync_z_min, sync_z_max = z_range[0], z_range[1]
    elif trigger and ("z-min-input" in trigger or "z-max-input" in trigger):
        z_min_val = z_min if z_min is not None else state.bbox.zmin
        z_max_val = z_max if z_max is not None else state.bbox.zmax
        bbox_updates.update(zmin=z_min_val, zmax=z_max_val)
        sync_z_min, sync_z_max = z_min_val, z_max_val
    else:
        sync_z_min, sync_z_max = state.bbox.zmin, state.bbox.zmax
    
    # Update state if needed
    if bbox_updates:
        state_manager.update_bbox(**bbox_updates)
        state = state_manager.get_state()
    
    # Update figure
    fig = VisualizationEngine.update_figure_with_selection(current_figure, state)
    
    return (
        fig,
        [sync_x_min, sync_x_max], sync_x_min, sync_x_max,
        [sync_y_min, sync_y_max], sync_y_min, sync_y_max,
        [sync_z_min, sync_z_max], sync_z_min, sync_z_max
    )


# Normal vector update callback
@callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Input("theta-slider", "value"),
    Input("phi-slider", "value"),
    State("main-graph", "figure"),
    prevent_initial_call=True
)
def update_normal_vector(theta, phi, current_figure):
    """Update normal vector visualization"""
    state = state_manager.get_state()
    if not state.pointcloud.loaded:
        raise dash.exceptions.PreventUpdate
    
    # Update normal vector in state
    state_manager.update_normal(theta=theta, phi=phi)
    updated_state = state_manager.get_state()
    
    # Update figure
    fig = VisualizationEngine.update_figure_with_selection(current_figure, updated_state)
    
    return fig


# JSON output update callback
@callback(
    Output("json-output", "value"),
    [
        Input("x-range-slider", "value"), Input("x-min-input", "value"), Input("x-max-input", "value"),
        Input("y-range-slider", "value"), Input("y-min-input", "value"), Input("y-max-input", "value"),
        Input("z-range-slider", "value"), Input("z-min-input", "value"), Input("z-max-input", "value"),
        Input("theta-slider", "value"), Input("phi-slider", "value")
    ],
    prevent_initial_call=True
)
def update_json_output(*args):
    """Update JSON configuration output"""
    return state_manager.get_bbox_json()


# Status log update callback
@callback(
    Output("status-output", "value"),
    Input("main-graph", "figure"),
    prevent_initial_call=True
)
def update_status_log(figure):
    """Update status log display"""
    state = state_manager.get_state()
    return "\n".join(state.status_log[-10:])  # Show last 10 messages


# Prediction callback
@callback(
    Output("status-output", "value", allow_duplicate=True),
    Input("predict-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_prediction(n_clicks):
    """Handle prediction execution"""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    state = state_manager.get_state()
    if not state.pointcloud.loaded:
        return "❌ No point cloud loaded. Please upload a PLY file first."
    
    try:
        # Save current configuration
        with open(REGION_JSON, "w") as f:
            f.write(state_manager.get_bbox_json())
        
        # Run prediction (using existing run_predict function)
        bbox = [
            [state.bbox.xmin, state.bbox.ymin, state.bbox.zmin],
            [state.bbox.xmax, state.bbox.ymax, state.bbox.zmax]
        ]
        normal = state.normal.vector
        
        result_path = run_predict(
            os.path.join(UPLOAD_DIR, "init.ply"),
            bbox,
            normal
        )
        
        state_manager.get_state().add_status(f"✅ Prediction completed: {result_path}")
        return f"✅ Prediction completed successfully!\nOutput saved to: {result_path}"
        
    except Exception as e:
        error_msg = f"❌ Prediction failed: {str(e)}"
        state_manager.get_state().add_status(error_msg)
        return error_msg


# Reset callback
@callback(
    Output("main-graph", "figure", allow_duplicate=True),
    Output("theta-slider", "value"),
    Output("phi-slider", "value"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_view(n_clicks):
    """Reset view to initial state"""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    
    state = state_manager.get_state()
    if not state.pointcloud.loaded:
        raise dash.exceptions.PreventUpdate
    
    # Reset bounding box to full point cloud bounds
    bounds = state.pointcloud.bounds
    state_manager.update_bbox(
        xmin=bounds['x'][0], xmax=bounds['x'][1],
        ymin=bounds['y'][0], ymax=bounds['y'][1],
        zmin=bounds['z'][0], zmax=bounds['z'][1]
    )
    
    # Reset normal vector
    state_manager.update_normal(theta=0, phi=0)
    
    # Get updated state and recreate figure
    updated_state = state_manager.get_state()
    fig = VisualizationEngine.create_pointcloud_figure(updated_state.pointcloud.xyz)
    fig = VisualizationEngine.update_figure_with_selection(fig, updated_state)
    
    return fig, 0, 0


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction Function (Enhanced version of original)
# ═══════════════════════════════════════════════════════════════════════════════

def run_predict(ply_path: str, bbox: List[List[float]], normal: List[float]) -> str:
    """Enhanced prediction function with better error handling"""
    try:
        # Read PLY data
        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        
        # Extract coordinates
        xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
        
        # Extract normals with fallback
        if all(key in v.dtype.names for key in ['nx', 'ny', 'nz']):
            nrm = np.vstack([v['nx'], v['ny'], v['nz']]).T.astype(np.float32)
        else:
            nrm = np.zeros_like(xyz, dtype=np.float32)
            state_manager.get_state().add_status("⚠️ No normals found, using zeros")
        
        # Extract scales with fallback
        if all(key in v.dtype.names for key in ['scale_0', 'scale_1', 'scale_2']):
            scl = np.vstack([v['scale_0'], v['scale_1'], v['scale_2']]).T.astype(np.float32)
        else:
            scl = np.ones_like(xyz, dtype=np.float32)
            state_manager.get_state().add_status("⚠️ No scales found, using ones")
        
        # Calculate transformation parameters
        ctr = (np.array(bbox[0]) + np.array(bbox[1])) / 2
        scale = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
        
        # Calculate rotation matrix
        n = np.array(normal) / np.linalg.norm(normal)
        z = np.array([0, 0, 1])
        v_ = np.cross(n, z)
        c = n.dot(z)
        vx = np.array([[0, -v_[2], v_[1]], [v_[2], 0, -v_[0]], [-v_[1], v_[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c + 1e-8)
        
        # Find points in bounding box
        mask = np.all((xyz >= bbox[0]) & (xyz <= bbox[1]), axis=1)
        idx = np.where(mask)[0]
        
        if len(idx) == 0:
            raise ValueError("No points found within the specified bounding box")
        
        state_manager.get_state().add_status(f"Processing {len(idx)} points in selection")
        
        # Transform points
        xyz_c = ((xyz[idx] - ctr) / scale) @ R.T
        
        # Model prediction
        sens_feat = torch.zeros(1, 128)
        
        with torch.no_grad():
            delta = model(
                torch.from_numpy(xyz_c)[None].float(),
                sens_feat
            )[0].detach().cpu().numpy()
        
        # Apply deformations
        xyz_c_deformed = xyz_c + delta[:, :3]
        nrm_c_deformed = (nrm[idx] @ R.T) + delta[:, 3:6]
        scl_deformed = scl[idx] + delta[:, 6:9]
        
        # Transform back to world coordinates
        xyz_deformed_world = xyz_c_deformed @ R * scale + ctr
        nrm_deformed_world = nrm_c_deformed @ R
        
        # Update arrays
        xyz[idx] = xyz_deformed_world
        nrm[idx] = nrm_deformed_world
        scl[idx] = scl_deformed
        
        # Prepare output data
        out = np.empty(len(xyz), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4')
        ])
        
        out['x'], out['y'], out['z'] = xyz.T
        out['nx'], out['ny'], out['nz'] = nrm.T
        out['scale_0'], out['scale_1'], out['scale_2'] = scl.T
        
        # Save result
        os.makedirs(EXPORT_DIR, exist_ok=True)
        out_path = os.path.join(EXPORT_DIR, "frame_001_pred.ply")
        PlyData([PlyElement.describe(out, 'vertex')]).write(out_path)
        
        return out_path
        
    except Exception as e:
        state_manager.get_state().add_status(f"❌ Prediction error: {str(e)}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# Run Server
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Starting Enhanced Deformation Prediction UI...")
    print("📊 Features: Unified state management, bidirectional sync, enhanced error handling")
    print("🌐 Access at: http://localhost:8051")
    app.run(debug=True, port=8051) 