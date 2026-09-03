"""
app.py — Sainsbury's WiFi CSI Sensing Dashboard
================================================
Streamlit application visualising the entrance basket/trolley detection pipeline.

Run:
    cd csi_sensing
    streamlit run app.py

The app works immediately with synthetic data.  When real ESP32/RPi data is
available, switch the dataset selector in the sidebar to "entrance".
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.synthetic import SyntheticCSIDataset
from src.models import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens  (dark-mode validated palette, Slot 1-4)
# ─────────────────────────────────────────────────────────────────────────────
SURFACE      = "#1a1a19"
PAGE_BG      = "#0d0d0d"
TEXT_PRIMARY = "#ffffff"
TEXT_MUTED   = "#898781"
GRIDLINE     = "#2c2c2a"

# Categorical slots (dark, adjacent-form validated)
C_EMPTY   = "#3987e5"   # slot 1  blue
C_PERSON  = "#d95926"   # slot 2  orange
C_BASKET  = "#199e70"   # slot 3  aqua
C_TROLLEY = "#c98500"   # slot 4  yellow

CLASS_COLORS  = [C_EMPTY, C_PERSON, C_BASKET, C_TROLLEY]
CLASS_LABELS  = ["Empty", "Person only", "Person + basket", "Person + trolley"]
CLASS_ICONS   = ["⬛", "🚶", "🛒🧺", "🛒"]

# Status
STATUS_GOOD     = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
STATUS_WARN     = "#fab219"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sainsbury's CSI Sensing",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* Global dark canvas */
  html, body, [data-testid="stAppViewContainer"] {{
      background: {PAGE_BG};
      color: {TEXT_PRIMARY};
  }}
  [data-testid="stSidebar"] {{ background: {SURFACE}; }}
  [data-testid="stHeader"]  {{ background: {PAGE_BG}; }}

  /* Stat tile cards */
  .kpi-card {{
      background: {SURFACE};
      border: 1px solid {GRIDLINE};
      border-radius: 10px;
      padding: 18px 22px 14px;
      text-align: center;
  }}
  .kpi-value {{
      font-size: 2.2rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      margin: 0;
      line-height: 1.1;
  }}
  .kpi-label {{
      font-size: 0.78rem;
      color: {TEXT_MUTED};
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }}
  .kpi-delta {{
      font-size: 0.85rem;
      margin-top: 6px;
  }}

  /* Prediction badge */
  .pred-badge {{
      display: inline-block;
      padding: 10px 22px;
      border-radius: 8px;
      font-size: 1.4rem;
      font-weight: 700;
      letter-spacing: 0.02em;
  }}

  /* Section headers */
  .section-header {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: {TEXT_MUTED};
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid {GRIDLINE};
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — load model & config
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_checkpoint(ckpt_path: str, config: dict):
    """Load a trained model from disk.  Returns None if not found."""
    path = Path(ckpt_path)
    if not path.exists():
        return None
    try:
        model = build_model(config)
        state = torch.load(str(path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner="Generating synthetic CSI samples…")
def get_synthetic_dataset(config: dict):
    """Build the synthetic dataset once and cache it."""
    return SyntheticCSIDataset(config=config, split="test")


@st.cache_data(show_spinner=False)
def load_results_log() -> list[dict]:
    """Load all result JSONs from logs/."""
    results = []
    for p in sorted(Path("logs").glob("results_*.json")):
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3D Entrance Scene
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _box_mesh(x0, x1, y0, y1, z0, z1, color, opacity=1.0, name=""):
    """Return a go.Mesh3d that draws a solid cuboid."""
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]
    i  = [0, 0, 0, 4, 4, 2]
    j  = [1, 2, 4, 5, 6, 3]
    k  = [2, 3, 5, 6, 7, 7]
    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=i, j=j, k=k,
        color=color, opacity=opacity,
        flatshading=True,
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _scene_annotation_color(inference_fired, correct, zone_alpha_val):
    if inference_fired:
        return STATUS_GOOD if correct else STATUS_CRITICAL
    if zone_alpha_val > 0.05:
        return STATUS_WARN
    return TEXT_MUTED

def _scene_annotation(inference_fired, correct, pred_cls, true_cls, confidence, zone_alpha_val):
    pc = pred_cls % len(CLASS_LABELS)
    if inference_fired:
        verdict = "✓ CORRECT" if correct else "✗ INCORRECT"
        return (
            f"<b>{verdict}</b><br>"
            f"{CLASS_ICONS[pc]}  {CLASS_LABELS[pc]}<br>"
            f"<span style='font-size:11px;color:{TEXT_MUTED}'>"
            f"Conf: {confidence[pc]:.1%}</span>"
        )
    if zone_alpha_val > 0.05:
        pct = int(zone_alpha_val * 100)
        return (
            f"<b>⏳ Approaching…</b><br>"
            f"Signal build-up: {pct}%<br>"
            f"<span style='font-size:11px;color:{TEXT_MUTED}'>"
            f"Model fires at 30%</span>"
        )
    return (
        f"<b>● Idle</b><br>Detection zone clear<br>"
        f"<span style='font-size:11px;color:{TEXT_MUTED}'>Drag slider →</span>"
    )

def build_entrance_scene(
    person_y: float,
    true_cls: int,
    pred_cls: int,
    confidence: np.ndarray,
    csi_row: np.ndarray,
    zone_alpha_val: float = 0.0,
    inference_fired: bool = False,
) -> go.Figure:
    """
    3D store-entrance visualisation.

    Coordinate system (right-hand, Z up):
      X : left / right across the entrance corridor  (-4 → +4 m)
      Y : front / back — person walks 0 → 12 m       (0 = outside, 12 = inside)
      Z : vertical                                    (0 = floor)

    Objects:
      • Store floor
      • Left & right barrier pillars (the turnstile posts)
      • WiFi TX node  (left wall, mid-height)
      • WiFi RX node  (right wall, mid-height)
      • Signal arcs   (TX → detection zone → RX)
      • Person/object (moves along Y, shaped by true class)
      • Detection zone (semi-transparent box between the APs)
    """
    fig = go.Figure()

    correct    = (pred_cls == true_cls) if inference_fired else None
    # Zone outline: green=correct, red=incorrect, amber=person approaching, grey=idle
    if inference_fired:
        zone_color = STATUS_GOOD if correct else STATUS_CRITICAL
    elif zone_alpha_val > 0.05:
        zone_color = STATUS_WARN   # amber — person entering, model not yet triggered
    else:
        zone_color = GRIDLINE      # grey — idle, nobody nearby
    pred_color  = CLASS_COLORS[pred_cls % len(CLASS_COLORS)]

    # ── Floor ────────────────────────────────────────────────────────────────
    fx = np.linspace(-4, 4, 20)
    fy = np.linspace(0, 12, 20)
    FX, FY = np.meshgrid(fx, fy)
    FZ = np.zeros_like(FX)
    fig.add_trace(go.Surface(
        x=FX, y=FY, z=FZ,
        colorscale=[[0, "#1e2433"], [1, "#252b3b"]],
        showscale=False, opacity=0.95,
        hoverinfo="skip", name="Floor",
        contours=dict(x=dict(show=True, color=GRIDLINE, width=1),
                      y=dict(show=True, color=GRIDLINE, width=1)),
    ))

    # ── Entrance barrier pillars ──────────────────────────────────────────────
    for bx in [-4, 4]:
        fig.add_trace(_box_mesh(bx - 0.25, bx + 0.25, 4.5, 7.5, 0, 1.2,
                                color="#2d3748", opacity=0.9))
        fig.add_trace(_box_mesh(bx - 0.12, bx + 0.12, 5.5, 6.5, 1.2, 1.6,
                                color="#4a5568", opacity=0.9))   # top cap

    # ── Detection zone (transparent volume between APs) ───────────────────────
    fig.add_trace(_box_mesh(-3.5, 3.5, 5.0, 7.0, 0, 2.5,
                            color=zone_color, opacity=0.08, name="zone"))
    # Detection zone border lines
    for ys, ye in [(5, 7), (5, 7)]:
        for xs, xe in [(-3.5, 3.5)]:
            for zv in [0, 2.5]:
                fig.add_trace(go.Scatter3d(
                    x=[xs, xe], y=[ys, ye], z=[zv, zv],
                    mode="lines",
                    line=dict(color=zone_color, width=2),
                    hoverinfo="skip", showlegend=False,
                ))

    # ── WiFi AP nodes ─────────────────────────────────────────────────────────
    tx_pos = dict(x=[-3.8], y=[6.0], z=[1.4])
    rx_pos = dict(x=[ 3.8], y=[6.0], z=[1.4])
    ap_signal_color = CLASS_COLORS[pred_cls]

    for pos, label in [(tx_pos, "WiFi TX"), (rx_pos, "WiFi RX")]:
        # Glow halo (larger, transparent)
        fig.add_trace(go.Scatter3d(
            **pos, mode="markers",
            marker=dict(size=22, color=ap_signal_color, opacity=0.15,
                        symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ))
        # Core node
        fig.add_trace(go.Scatter3d(
            **pos, mode="markers+text",
            marker=dict(size=12, color=ap_signal_color, opacity=0.95,
                        symbol="circle",
                        line=dict(color=TEXT_PRIMARY, width=1)),
            text=[label], textposition="top center",
            textfont=dict(color=TEXT_MUTED, size=10),
            name=label, hoverinfo="skip",
        ))

    # ── Signal arcs (TX → person zone → RX) ──────────────────────────────────
    # We draw a curved arc from TX through the detection zone to RX.
    # Opacity varies by whether the person is in the detection zone.
    in_zone = 5.0 <= person_y <= 7.0
    sig_opacity = 0.8 if in_zone else 0.3
    perturb_level = float(np.mean(np.abs(csi_row - csi_row.mean())))
    perturb_norm  = min(perturb_level / 3.0, 1.0)  # 0..1

    # Signal color: blue (low perturbation) → orange (high perturbation)
    # Simple linear interpolation between palette slot 1 and slot 2
    def lerp_color(t):
        """t in [0,1]: 0=blue, 1=orange"""
        r = int(0x39 + t * (0xd9 - 0x39))
        g = int(0x87 + t * (0x59 - 0x87))
        b = int(0xe5 + t * (0x26 - 0xe5))
        return f"rgb({r},{g},{b})"

    sig_color = lerp_color(perturb_norm)

    # Arc from TX to RX — parabola through y=6.0 apex
    t_arc = np.linspace(0, 1, 30)
    arc_x = -3.8 + 7.6 * t_arc
    arc_z = 1.4 + 0.4 * np.sin(np.pi * t_arc)   # gentle arc over zone
    arc_y = np.full(30, 6.0)

    fig.add_trace(go.Scatter3d(
        x=arc_x, y=arc_y, z=arc_z,
        mode="lines",
        line=dict(color=sig_color, width=4 if in_zone else 2),
        opacity=sig_opacity,
        name="Signal path", hoverinfo="skip", showlegend=False,
    ))

    # Secondary fainter arcs (show multipath)
    for dy_offset, z_offset in [(-0.5, 0.1), (0.5, 0.2), (0, 0.6)]:
        arc_z2 = 1.4 + (0.4 + z_offset) * np.sin(np.pi * t_arc)
        arc_y2 = arc_y + dy_offset
        fig.add_trace(go.Scatter3d(
            x=arc_x, y=arc_y2, z=arc_z2,
            mode="lines",
            line=dict(color=sig_color, width=1),
            opacity=sig_opacity * 0.4,
            hoverinfo="skip", showlegend=False,
        ))

    # ── Person / object ───────────────────────────────────────────────────────
    person_color = CLASS_COLORS[true_cls]
    px_pos = 0.0
    py_pos = float(person_y)

    # Body (cylinder approximated as a tall marker)
    body_z = [0.1, 0.5, 0.9, 1.3, 1.6]   # stack of markers = body
    body_sizes = [14, 16, 14, 10, 8]
    for bz, bs in zip(body_z, body_sizes):
        fig.add_trace(go.Scatter3d(
            x=[px_pos], y=[py_pos], z=[bz],
            mode="markers",
            marker=dict(size=bs, color=person_color, opacity=0.9,
                        symbol="circle",
                        line=dict(color=hex_to_rgba(person_color, 0.3), width=2)),
            hoverinfo="skip", showlegend=False,
        ))
    # Head
    fig.add_trace(go.Scatter3d(
        x=[px_pos], y=[py_pos], z=[1.85],
        mode="markers",
        marker=dict(size=12, color=person_color, opacity=0.95,
                    symbol="circle",
                    line=dict(color=TEXT_PRIMARY, width=1)),
        hoverinfo="skip", showlegend=False,
    ))

    # Object carried (basket = small box at hip; trolley = larger box in front)
    if true_cls == 2:    # basket
        fig.add_trace(_box_mesh(0.2, 0.7, py_pos - 0.2, py_pos + 0.2,
                                0.7, 1.0, color=C_BASKET, opacity=0.7))
    elif true_cls == 3:  # trolley
        fig.add_trace(_box_mesh(-0.4, 0.8, py_pos + 0.3, py_pos + 1.1,
                                0, 0.9, color=C_TROLLEY, opacity=0.6))
        # Trolley handle
        fig.add_trace(go.Scatter3d(
            x=[-0.4, 0.8], y=[py_pos + 0.7, py_pos + 0.7], z=[0.85, 0.85],
            mode="lines", line=dict(color=C_TROLLEY, width=5),
            hoverinfo="skip", showlegend=False,
        ))

    # Shadow / footprint on floor
    theta = np.linspace(0, 2 * np.pi, 24)
    r_foot = 0.55 if true_cls == 3 else 0.3
    fig.add_trace(go.Scatter3d(
        x=px_pos + r_foot * np.cos(theta),
        y=py_pos + (0.5 if true_cls == 3 else 0) + r_foot * np.sin(theta),
        z=np.zeros(24),
        mode="lines",
        line=dict(color=person_color, width=2),
        opacity=0.4,
        hoverinfo="skip", showlegend=False,
    ))

    # "WiFi hit" ray from person to TX and RX (only when in zone)
    if in_zone:
        for rx_node, ry_node, rz_node in [(-3.8, 6.0, 1.4), (3.8, 6.0, 1.4)]:
            fig.add_trace(go.Scatter3d(
                x=[px_pos, rx_node],
                y=[py_pos, ry_node],
                z=[1.4, rz_node],
                mode="lines",
                line=dict(color=sig_color, width=2, dash="dot"),
                opacity=0.7,
                hoverinfo="skip", showlegend=False,
            ))

    # ── Layout & camera ───────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=PAGE_BG,
        scene=dict(
            bgcolor=SURFACE,
            xaxis=dict(
                range=[-5, 5], title="", showgrid=True, gridcolor=GRIDLINE,
                showticklabels=False, zeroline=False,
                backgroundcolor=SURFACE,
            ),
            yaxis=dict(
                range=[0, 12], title="", showgrid=True, gridcolor=GRIDLINE,
                showticklabels=False, zeroline=False,
                backgroundcolor=SURFACE,
            ),
            zaxis=dict(
                range=[0, 3.5], title="", showgrid=False,
                showticklabels=False, zeroline=False,
                backgroundcolor=SURFACE,
            ),
            camera=dict(
                eye=dict(x=-1.6, y=-1.8, z=1.1),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.6, z=0.5),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        height=520,
        annotations=[
            dict(
                text=_scene_annotation(
                    inference_fired, correct, pred_cls, true_cls,
                    confidence, zone_alpha_val,
                ),
                x=0.02, y=0.97,
                xref="paper", yref="paper",
                align="left", showarrow=False,
                font=dict(color=_scene_annotation_color(inference_fired, correct, zone_alpha_val), size=13),
                bgcolor=hex_to_rgba(_scene_annotation_color(inference_fired, correct, zone_alpha_val), 0.12),
                bordercolor=_scene_annotation_color(inference_fired, correct, zone_alpha_val),
                borderwidth=1, borderpad=8,
            )
        ],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSI Heatmap (time × subcarrier)
# ─────────────────────────────────────────────────────────────────────────────
def build_csi_heatmap(csi_2d: np.ndarray, title: str = "CSI Amplitude") -> go.Figure:
    """
    csi_2d: (time_steps, num_subcarriers)
    Sequential blue ramp from the validated palette.
    """
    # Normalise for display
    vmin, vmax = np.percentile(csi_2d, [2, 98])

    fig = go.Figure(go.Heatmap(
        z=csi_2d,
        colorscale=[
            [0.00, "#0d366b"],  # blue-700
            [0.20, "#104281"],  # blue-650
            [0.40, "#1c5cab"],  # blue-550
            [0.60, "#2a78d6"],  # blue-450
            [0.80, "#5598e7"],  # blue-350
            [1.00, "#cde2fb"],  # blue-100
        ],
        zmin=vmin, zmax=vmax,
        colorbar=dict(
            thickness=10, len=0.9,
            tickfont=dict(color=TEXT_MUTED, size=10),
            outlinewidth=0,
        ),
        hovertemplate="Subcarrier: %{x}<br>Frame: %{y}<br>Amplitude: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY, family="system-ui"),
        title=dict(text=title, font=dict(size=12, color=TEXT_MUTED), x=0),
        xaxis=dict(
            title="Subcarrier index", title_font=dict(size=10, color=TEXT_MUTED),
            gridcolor=GRIDLINE, linecolor=GRIDLINE, zeroline=False,
            tickfont=dict(size=9, color=TEXT_MUTED),
        ),
        yaxis=dict(
            title="Time frame", title_font=dict(size=10, color=TEXT_MUTED),
            gridcolor=GRIDLINE, linecolor=GRIDLINE, zeroline=False,
            tickfont=dict(size=9, color=TEXT_MUTED),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=230,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Confidence bar chart
# ─────────────────────────────────────────────────────────────────────────────
def build_confidence_chart(confidence: np.ndarray, pred_cls: int, true_cls: int) -> go.Figure:
    """Horizontal bar chart of per-class softmax confidence.  Categorical palette."""
    fig = go.Figure()

    for i, (label, color, conf) in enumerate(
        zip(CLASS_LABELS, CLASS_COLORS, confidence)
    ):
        is_pred = (i == pred_cls)
        is_true = (i == true_cls)
        bar_color = color
        marker_line = dict(color=TEXT_PRIMARY, width=2) if is_pred else dict(color=color, width=0)

        fig.add_trace(go.Bar(
            y=[label],
            x=[conf],
            orientation="h",
            marker=dict(color=bar_color, opacity=0.9 if is_pred else 0.45,
                        line=marker_line),
            name=label,
            showlegend=False,
            text=[f"{conf:.1%}"],
            textposition="outside" if conf > 0.15 else "outside",
            textfont=dict(color=TEXT_PRIMARY if is_pred else TEXT_MUTED, size=11),
            hovertemplate=f"{label}: {conf:.2%}<extra></extra>",
        ))

    # True label indicator
    if true_cls is not None:
        fig.add_annotation(
            x=1.02, y=CLASS_LABELS[true_cls],
            text="◀ ground truth",
            showarrow=False,
            font=dict(color=TEXT_MUTED, size=9),
            xref="x", yref="y",
            xanchor="left",
        )

    fig.update_layout(
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=SURFACE,
        barmode="overlay",
        font=dict(color=TEXT_PRIMARY, family="system-ui"),
        xaxis=dict(
            range=[0, 1.25], showgrid=True, gridcolor=GRIDLINE,
            tickformat=".0%", tickfont=dict(size=9, color=TEXT_MUTED),
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor=GRIDLINE, linecolor=GRIDLINE,
            tickfont=dict(size=10, color=TEXT_PRIMARY),
        ),
        margin=dict(l=0, r=80, t=0, b=10),
        height=170,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────
def build_training_curves(log_path: str) -> go.Figure | None:
    """Load training log CSV (if it exists) and plot train/val accuracy curves."""
    # We reconstruct from the results JSONs — simple approximation
    results = load_results_log()
    if not results:
        return None

    fig = go.Figure()
    for res in results[-2:]:   # show last 2 runs
        label = f"{res.get('model','?').upper()} on {res.get('dataset','?')}"
        acc = res.get("test_accuracy", None)
        if acc is None:
            continue
        fig.add_trace(go.Bar(
            x=[label], y=[acc],
            marker_color=C_EMPTY if "cnn" in label.lower() else C_BASKET,
            text=[f"{acc:.1%}"],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_PRIMARY),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY),
        yaxis=dict(
            range=[0, 1.1], tickformat=".0%",
            gridcolor=GRIDLINE, zeroline=False,
            tickfont=dict(size=9, color=TEXT_MUTED),
        ),
        xaxis=dict(tickfont=dict(size=10, color=TEXT_PRIMARY), linecolor=GRIDLINE),
        margin=dict(l=10, r=10, t=0, b=10),
        height=170,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_confusion_matrix(model, dataset, class_names: list[str]) -> go.Figure | None:
    """Run model over full dataset, build and return confusion heatmap."""
    if model is None:
        return None
    try:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                # CNN expects (B, 1, T, S); LSTM/Transformer want (B, T, S)
                try:
                    logits = model(xb)
                except Exception:
                    logits = model(xb.squeeze(1))
                all_preds.extend(logits.argmax(dim=1).numpy())
                all_true.extend(yb.numpy())

        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(all_true, all_preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig = go.Figure(go.Heatmap(
            z=cm_pct,
            x=class_names,
            y=class_names,
            colorscale=[
                [0.0, SURFACE],
                [0.5, "#1c5cab"],
                [1.0, "#cde2fb"],
            ],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                thickness=10, tickformat=".0%",
                tickfont=dict(color=TEXT_MUTED, size=9),
                outlinewidth=0,
            ),
            text=[[f"{v:.0%}" for v in row] for row in cm_pct],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Rate: %{z:.1%}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=PAGE_BG,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT_PRIMARY, size=10),
            xaxis=dict(
                title="Predicted", tickfont=dict(size=9, color=TEXT_MUTED),
                linecolor=GRIDLINE,
            ),
            yaxis=dict(
                title="True", tickfont=dict(size=9, color=TEXT_MUTED),
                autorange="reversed", linecolor=GRIDLINE,
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 from result JSON
# ─────────────────────────────────────────────────────────────────────────────
def build_per_class_f1(result: dict) -> go.Figure | None:
    per_class = result.get("per_class", {})
    if not per_class:
        return None
    labels = list(per_class.keys())
    f1s    = [per_class[k]["f1"] for k in labels]
    colors = [C_EMPTY, C_PERSON, C_BASKET, C_TROLLEY][:len(labels)]
    if len(colors) < len(labels):
        colors = [C_EMPTY] * len(labels)

    fig = go.Figure(go.Bar(
        x=labels,
        y=f1s,
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color=GRIDLINE, width=0),
        ),
        text=[f"{v:.2f}" for v in f1s],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="%{x}: F1 = %{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY),
        yaxis=dict(
            range=[0, 1.15], gridcolor=GRIDLINE, zeroline=False,
            tickfont=dict(size=9, color=TEXT_MUTED),
        ),
        xaxis=dict(
            tickfont=dict(size=9, color=TEXT_MUTED), linecolor=GRIDLINE,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=210,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI tile HTML helper
# ─────────────────────────────────────────────────────────────────────────────
def kpi(value: str, label: str, delta: str = "", delta_good: bool = True) -> str:
    delta_color = STATUS_GOOD if delta_good else STATUS_CRITICAL
    delta_html = (
        f'<div class="kpi-delta" style="color:{delta_color}">{delta}</div>'
        if delta else ""
    )
    return f"""
    <div class="kpi-card">
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 CSI Sensing Controls")
    st.divider()

    st.markdown("**Model checkpoint**")
    ckpt_options = {
        "CNN → Synthetic (fast)": "checkpoints/best_cnn_synthetic.pt",
        "ResNet → UT-HAR (real CSI)": "checkpoints/best_resnet_uthar.pt",
        "CNN → UT-HAR": "checkpoints/best_cnn_uthar.pt",
        "No model (random)": None,
    }
    ckpt_label = st.selectbox("", list(ckpt_options.keys()), index=0, label_visibility="collapsed")
    ckpt_path  = ckpt_options[ckpt_label]

    st.divider()

    # ── Controls adapt to which model is selected ─────────────────────────────
    # We need to know is_uthar_model here, but model loading happens below.
    # Read ckpt_path to infer it early (same logic as below).
    _is_uthar_sidebar = "uthar" in (ckpt_path or "")

    if not _is_uthar_sidebar:
        # ── Synthetic mode: entrance walkthrough slider ───────────────────────
        st.markdown("**🚶 Simulate entrance walk-through**")
        person_y = st.slider(
            "Person position (m from entrance)", 0.0, 11.5, 6.0, step=0.5,
            help="Drag to walk the person through the gate. "
                 "Model fires when signal perturbation exceeds 30% (≈5.5–6.5 m).",
        )
        uthar_sample_idx = 0   # unused in this mode

        st.divider()
        st.markdown("**True object class**")
        true_cls = st.radio(
            "", CLASS_LABELS, index=3,
            help="What the person is carrying — compared against model prediction.",
            label_visibility="collapsed",
        )
        true_cls_idx = CLASS_LABELS.index(true_cls)

        st.divider()
        sample_seed = st.number_input("Random sample seed", 0, 9999, 42, step=1)
        if st.button("🎲  New sample", use_container_width=True):
            st.session_state["seed"] = int(np.random.randint(0, 9999))

        st.divider()
        st.markdown(
            '<div style="font-size:0.72rem;color:#898781">'
            "Zone: 5–7 m · Green = correct · Red = incorrect"
            "</div>", unsafe_allow_html=True,
        )

    else:
        # ── UT-HAR mode: step through real test samples ───────────────────────
        st.markdown("**🗂 Step through real UT-HAR test samples**")
        uthar_sample_idx = st.slider(
            "Test sample index", 0, 499, 0, step=1,
            help="Each position is a real WiFi CSI recording from the UT-HAR dataset. "
                 "Drag to see how the ResNet classifies different samples.",
        )
        person_y     = 6.0     # always in-zone for UT-HAR display
        true_cls_idx = 0       # overridden by actual UT-HAR label below
        true_cls     = CLASS_LABELS[0]
        sample_seed  = 42

        st.divider()
        st.markdown(
            '<div style="font-size:0.72rem;color:#898781">'
            "Each sample is real 802.11n CSI from the Intel 5300 NIC.<br>"
            "7 classes: lie down · fall · pick up · run · sit · stand · walk<br>"
            "ResNet test accuracy: <b>84.2%</b>"
            "</div>", unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load model & config
# ─────────────────────────────────────────────────────────────────────────────
seed = st.session_state.get("seed", int(sample_seed))

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Determine which dataset the chosen checkpoint was trained on ───────────────
# This is the source of truth: inference is ONLY meaningful when the model
# sees data from its own training distribution.
is_uthar_model   = "uthar" in (ckpt_path or "")
is_synth_model   = not is_uthar_model and ckpt_path is not None
model_arch       = "resnet" if "resnet" in (ckpt_path or "") else "cnn"
model_n_classes  = 7 if is_uthar_model else 4
model_class_names = (
    ["lie_down", "fall", "pick_up", "run", "sit_down", "stand_up", "walk"]
    if is_uthar_model else CLASS_LABELS
)

cfg_for_model = {**cfg, "model": {**cfg["model"],
    "num_classes": model_n_classes,
    "architecture": model_arch,
}}
model = load_checkpoint(ckpt_path, cfg_for_model) if ckpt_path else None

# ── Synthetic dataset — always used for the 3D SIMULATION visual ──────────────
cfg_synth = {**cfg, "data": {**cfg["data"], "dataset": "synthetic",
    "synthetic": {**cfg["data"]["synthetic"], "num_classes": 4}}}
ds_synth = get_synthetic_dataset(cfg_synth)

rng = np.random.default_rng(seed)

# ── Position-dependent CSI: the key fix that makes the slider meaningful ──────
# We blend two samples:
#   alpha = 0  →  background (empty zone, low perturbation)
#   alpha = 1  →  full class-specific signal (person+object in detection zone)
# The blend is driven by how deep into the detection zone the person is.

ZONE_CENTER, ZONE_HALF = 6.0, 1.5   # metres

def zone_alpha(y: float) -> float:
    """Smooth 0→1→0 ramp as person traverses the detection zone (5–7 m)."""
    t = max(0.0, 1.0 - abs(y - ZONE_CENTER) / ZONE_HALF)
    return t * t * (3.0 - 2.0 * t)   # smoothstep

alpha = zone_alpha(person_y)

# Background sample (empty class = flat, low-amplitude)
empty_candidates = [i for i, (_, y) in enumerate(ds_synth) if int(y) == 0]
bg_idx   = empty_candidates[int(rng.integers(0, len(empty_candidates)))]
bg_x, _  = ds_synth[bg_idx]
bg_csi   = bg_x.numpy()[0]   # (250, 90)

# Class-specific sample (person+object in full perturbation)
class_candidates = [i for i, (_, y) in enumerate(ds_synth) if int(y) == true_cls_idx]
if not class_candidates:
    class_candidates = list(range(len(ds_synth)))
cl_idx      = class_candidates[int(rng.integers(0, len(class_candidates)))]
class_x, _  = ds_synth[cl_idx]
class_csi   = class_x.numpy()[0]   # (250, 90)

# Blended CSI — changes continuously as the slider moves
csi_np   = (1.0 - alpha) * bg_csi + alpha * class_csi   # (250, 90)
sample_x = torch.from_numpy(csi_np[np.newaxis, :, :].astype(np.float32))  # (1, 250, 90)

# ── UT-HAR dataset — loaded on demand for model-matched evaluation ─────────────
@st.cache_resource(show_spinner="Loading UT-HAR test split…")
def get_uthar_test(config):
    try:
        from src.datasets.uthar import UTHARDataset
        return UTHARDataset(config=config, split="test")
    except FileNotFoundError:
        return None

ds_uthar = get_uthar_test(cfg) if is_uthar_model else None

# ── Choose which dataset backs the confusion matrix / F1 panels ───────────────
eval_ds          = ds_uthar if is_uthar_model else ds_synth
eval_class_names = model_class_names

# ── Run inference — gated on both model validity AND zone entry ────────────────
# Only fire the model when:
#   (a) a synthetic-trained model is loaded (correct distribution), AND
#   (b) the person has entered the detection zone (alpha > 0.3)
# Outside the zone the display shows "Waiting…" — exactly what a real
# deployment would do: inference is not triggered by background noise.

IN_ZONE_THRESHOLD = 0.3   # alpha must exceed this to trigger model

if model is not None and is_synth_model:
    if alpha >= IN_ZONE_THRESHOLD:
        # ✓ Person in zone — run inference on position-blended CSI
        with torch.no_grad():
            logits = model(sample_x.unsqueeze(0))
            probs  = torch.softmax(logits, dim=1).numpy()[0]
        pred_cls        = int(np.argmax(probs))
        confidence      = probs
        inference_valid = True
    else:
        # Person outside zone — model is idle
        confidence      = np.full(4, 0.25)
        pred_cls        = -1   # sentinel: no prediction yet
        inference_valid = False

elif model is not None and is_uthar_model:
    # Use the sample index driven by the sidebar slider (not a random pick)
    uthar_idx = min(uthar_sample_idx, len(eval_ds) - 1) if eval_ds else 0
    if eval_ds:
        ux, uy = eval_ds[uthar_idx]
        with torch.no_grad():
            logits = model(ux.unsqueeze(0))
            probs  = torch.softmax(logits, dim=1).numpy()[0]
        pred_cls_uthar   = int(np.argmax(probs))
        confidence_uthar = probs
        true_cls_uthar   = int(uy)
        # Override the CSI heatmap with the REAL UT-HAR sample's data
        csi_np = ux.numpy()[0]   # shape (250, 90) — real measured CSI
    # Map UT-HAR 7-class label → 4-class shape for the 3D scene silhouette
    # lie→empty, fall→person, pickup→basket, run→person,
    # sitdown→empty, standup→person, walk→person
    _uthar_to_4 = [0, 1, 2, 1, 0, 1, 1]
    true_cls_idx    = _uthar_to_4[true_cls_uthar] if eval_ds else 0
    confidence      = np.full(4, 0.25)
    pred_cls        = true_cls_idx
    inference_valid = False

else:
    confidence      = np.full(4, 0.25)
    pred_cls        = -1
    inference_valid = False

correct = (pred_cls == true_cls_idx) if (inference_valid and pred_cls >= 0) else None


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px'>📡 Sainsbury's WiFi CSI Entrance Monitor</h1>"
    "<p style='color:#898781;margin-top:0;font-size:0.9rem'>"
    "WiFi Channel State Information — real-time basket &amp; trolley detection without cameras"
    "</p>",
    unsafe_allow_html=True,
)

# ── KPI row ───────────────────────────────────────────────────────────────────
results_list = load_results_log()
best_acc = max((r.get("test_accuracy", 0) for r in results_list), default=None)
best_model_label = "—"
if results_list:
    best_run = max(results_list, key=lambda r: r.get("test_accuracy", 0))
    best_model_label = f"{best_run.get('model','?').upper()} / {best_run.get('dataset','?')}"

in_zone = 5.0 <= person_y <= 7.0
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(kpi(
        f"{best_acc:.1%}" if best_acc else "—",
        "Best test accuracy",
        f"↑ {best_model_label}",
        delta_good=True,
    ), unsafe_allow_html=True)

with k2:
    st.markdown(kpi(
        "4",
        "Detection classes",
        "Empty · Person · Basket · Trolley",
        delta_good=True,
    ), unsafe_allow_html=True)

with k3:
    st.markdown(kpi(
        "✓ In zone" if in_zone else "– Outside",
        "Current position",
        f"{person_y:.1f} m from gate",
        delta_good=in_zone,
    ), unsafe_allow_html=True)

with k4:
    if inference_valid and pred_cls >= 0:
        st.markdown(kpi(
            CLASS_ICONS[pred_cls] + "  " + CLASS_LABELS[pred_cls],
            "Model prediction (live)",
            f"{'✓ Correct' if correct else '✗ Incorrect'} — conf {confidence[pred_cls]:.1%}",
            delta_good=correct,
        ), unsafe_allow_html=True)
    elif is_synth_model and pred_cls == -1:
        # Person not yet in zone
        pct = f"{alpha:.0%}"
        st.markdown(kpi(
            "⏳  Waiting…",
            "Model idle — person outside zone",
            f"Signal perturbation: {pct} of peak",
            delta_good=False,
        ), unsafe_allow_html=True)
    elif is_uthar_model and eval_ds:
        st.markdown(kpi(
            model_class_names[pred_cls_uthar],
            "UT-HAR prediction (real CSI)",
            f"True: {model_class_names[true_cls_uthar]} — conf {confidence_uthar[pred_cls_uthar]:.1%}",
            delta_good=(pred_cls_uthar == true_cls_uthar),
        ), unsafe_allow_html=True)
    else:
        st.markdown(kpi("—", "Model prediction", "No model loaded"), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main layout: 3D scene  |  CSI heatmap + confidence
# ─────────────────────────────────────────────────────────────────────────────
col_scene, col_right = st.columns([3, 2], gap="medium")

with col_scene:
    st.markdown('<div class="section-header">3D ENTRANCE SIMULATION</div>', unsafe_allow_html=True)

    # ── Context banner — honest about what the scene is showing ──────────────
    if inference_valid:
        st.success(
            f"**Live inference** — CNN trained on synthetic 4-class data. "
            f"Prediction: **{CLASS_LABELS[pred_cls]}** ({confidence[pred_cls]:.1%} confidence). "
            f"Green/red zone outline = correct/incorrect.",
            icon="✅",
        )
    elif is_uthar_model:
        st.info(
            "**Simulation only** — The 3D scene shows the Sainsbury's entrance scenario "
            "using synthetic data. The UT-HAR model was trained on *real* CSI data with "
            "7 different activity classes — running it on synthetic data would produce "
            "meaningless outputs. Its real performance is shown in the bottom panels below.",
            icon="ℹ️",
        )
    else:
        st.warning("No model loaded — scene is illustrative only.", icon="⚠️")

    scene_pred = pred_cls if (inference_valid and pred_cls >= 0) else true_cls_idx
    scene_conf = confidence if inference_valid else np.full(4, 0.25)

    st.plotly_chart(
        build_entrance_scene(
            person_y, true_cls_idx, scene_pred, scene_conf, csi_np[125],
            # Pass alpha so the scene can colour the zone correctly when idle
            zone_alpha_val=alpha,
            inference_fired=(inference_valid and pred_cls >= 0),
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        "Drag to rotate · Scroll to zoom · "
        + ("Green outline = correct prediction · " if inference_valid else "Outline = detection zone · ")
        + "Signal arc: blue (low perturbation) → orange (high)"
    )

with col_right:
    # CSI heatmap — updates continuously as the slider moves
    st.markdown('<div class="section-header">CSI SIGNAL HEATMAP — LIVE</div>',
                unsafe_allow_html=True)

    # Perturbation meter: shows signal build-up as person enters zone
    bar_color  = STATUS_GOOD if (inference_valid and pred_cls >= 0) else (
                 STATUS_WARN if alpha > 0.05 else GRIDLINE)
    bar_pct    = int(alpha * 100)
    idle_label = "MODEL FIRING" if (inference_valid and pred_cls >= 0) else (
                 f"APPROACHING — {bar_pct}%" if alpha > 0.05 else "IDLE")
    st.markdown(
        f'<div style="margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.7rem;color:{TEXT_MUTED};margin-bottom:3px">'
        f'<span>SIGNAL PERTURBATION</span><span style="color:{bar_color}">{idle_label}</span></div>'
        f'<div style="background:{GRIDLINE};border-radius:3px;height:6px;width:100%">'
        f'<div style="background:{bar_color};border-radius:3px;height:6px;'
        f'width:{bar_pct}%;transition:width 0.2s"></div></div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_csi_heatmap(csi_np, title=""),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown('<div class="section-header" style="margin-top:10px">MODEL CONFIDENCE</div>',
                unsafe_allow_html=True)

    if inference_valid:
        # ✓ Synthetic model on synthetic data — honest confidence
        st.plotly_chart(
            build_confidence_chart(confidence, pred_cls, true_cls_idx),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    elif is_uthar_model and eval_ds:
        # Show the UT-HAR model's real confidence on a UT-HAR sample
        st.caption(
            f"Showing UT-HAR model confidence on **real CSI data** (test split). "
            f"True class: **{model_class_names[true_cls_uthar]}**."
        )
        # Build confidence chart with UT-HAR class labels
        fig_uthar = go.Figure()
        for i, (lbl, conf) in enumerate(zip(model_class_names, confidence_uthar)):
            is_pred = (i == pred_cls_uthar)
            is_true = (i == true_cls_uthar)
            fig_uthar.add_trace(go.Bar(
                y=[lbl], x=[conf], orientation="h",
                marker=dict(
                    color=CLASS_COLORS[i % 4],
                    opacity=0.9 if is_pred else 0.4,
                    line=dict(color=TEXT_PRIMARY, width=2) if is_pred else dict(width=0),
                ),
                text=[f"{conf:.1%}"], textposition="outside",
                textfont=dict(color=TEXT_PRIMARY if is_pred else TEXT_MUTED, size=10),
                showlegend=False,
                hovertemplate=f"{lbl}: {conf:.2%}<extra></extra>",
            ))
        fig_uthar.update_layout(
            paper_bgcolor=PAGE_BG, plot_bgcolor=SURFACE,
            xaxis=dict(range=[0, 1.25], tickformat=".0%",
                       gridcolor=GRIDLINE, tickfont=dict(size=9, color=TEXT_MUTED),
                       zeroline=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=9, color=TEXT_PRIMARY),
                       gridcolor=GRIDLINE),
            margin=dict(l=0, r=80, t=0, b=10), height=210,
            font=dict(color=TEXT_PRIMARY),
        )
        st.plotly_chart(fig_uthar, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Load a model checkpoint to see prediction confidence.")

# ─────────────────────────────────────────────────────────────────────────────
# Bottom row: per-class F1  |  confusion matrix  |  run comparison
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
col_f1, col_cm, col_runs = st.columns([2, 2, 1], gap="medium")

with col_f1:
    st.markdown('<div class="section-header">PER-CLASS F1 (LAST RUN)</div>', unsafe_allow_html=True)
    if results_list:
        last = results_list[-1]
        f1_fig = build_per_class_f1(last)
        if f1_fig:
            st.plotly_chart(f1_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No per-class data in last result.")
    else:
        st.caption("No training results found yet — run `python train.py` first.")

with col_cm:
    st.markdown('<div class="section-header">CONFUSION MATRIX</div>', unsafe_allow_html=True)
    if eval_ds is not None:
        st.caption(
            f"Model: **{model_arch.upper()}** evaluated on "
            f"**{'UT-HAR real CSI' if is_uthar_model else 'synthetic'} test split** "
            f"({len(eval_ds)} samples, {model_n_classes} classes)"
        )
    cm_fig = build_confusion_matrix(model, eval_ds, eval_class_names)
    if cm_fig:
        st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption(
            "Load a model checkpoint in the sidebar to see the confusion matrix. "
            + ("UT-HAR data not found — run `python data/download.py` first."
               if is_uthar_model and eval_ds is None else "")
        )

with col_runs:
    st.markdown('<div class="section-header">RUNS</div>', unsafe_allow_html=True)
    if results_list:
        run_df = pd.DataFrame([
            {
                "Model": r.get("model", "?").upper(),
                "Data": r.get("dataset", "?"),
                "Acc": f"{r.get('test_accuracy', 0):.1%}",
                "F1": f"{r.get('macro_f1', 0):.2f}",
            }
            for r in results_list
        ])
        st.dataframe(
            run_df,
            use_container_width=True,
            hide_index=True,
            height=220,
        )
    else:
        st.caption("No runs yet.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="font-size:0.72rem;color:#898781;text-align:center">'
    "Sainsbury's WiFi CSI Sensing PoC — "
    "Model: <b>CSICNN / CSIResNet</b> (PyTorch) — "
    "Data: <b>Synthetic (4-class) · UT-HAR (real CSI, 7-class)</b> — "
    "Hardware: ESP32 · Raspberry Pi 4 + Nexmon CSI (arriving soon)"
    "</div>",
    unsafe_allow_html=True,
)
