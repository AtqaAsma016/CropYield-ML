"""
utils.py
Shared utilities: path resolution, metric formatting, plot helpers,
and tooltip/help text constants for the GUI.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


# ──────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────

def project_root() -> str:
    """Return absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def results_path(filename: str) -> str:
    return os.path.join(project_root(), 'results', filename)


def models_path(filename: str) -> str:
    return os.path.join(project_root(), 'models', filename)


# ──────────────────────────────────────────────
# Metric formatting
# ──────────────────────────────────────────────

def fmt_metrics_dt(m: dict) -> str:
    return (f"Accuracy  : {m['accuracy']:.4f}\n"
            f"Precision : {m['precision']:.4f}\n"
            f"Recall    : {m['recall']:.4f}\n"
            f"F1-Score  : {m['f1']:.4f}")


def fmt_metrics_km(m: dict) -> str:
    return (f"Clusters       : {m['n_clusters']}\n"
            f"Silhouette Score: {m['silhouette']:.4f}")


def fmt_metrics_lr(m: dict) -> str:
    return (f"RMSE : {m['RMSE']:.4f} t/ha\n"
            f"MAE  : {m['MAE']:.4f} t/ha\n"
            f"R²   : {m['R2']:.4f}")


# ──────────────────────────────────────────────
# Tkinter plot embedding
# ──────────────────────────────────────────────

def embed_figure(fig: plt.Figure, parent: tk.Widget) -> FigureCanvasTkAgg:
    """Embed a matplotlib Figure into a Tkinter widget."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


# ──────────────────────────────────────────────
# Feature input validation
# ──────────────────────────────────────────────

FEATURE_RANGES = {
    'N':           (0,   200,  "Nitrogen content in soil (kg/ha)"),
    'P':           (0,   150,  "Phosphorus content in soil (kg/ha)"),
    'K':           (0,   210,  "Potassium content in soil (kg/ha)"),
    'temperature': (0,   50,   "Ambient temperature (°C)"),
    'humidity':    (0,   100,  "Relative humidity (%)"),
    'ph':          (0,   14,   "Soil pH value"),
    'rainfall':    (0,   400,  "Annual rainfall (mm)"),
}


def validate_inputs(values: dict) -> list:
    """Return list of error strings; empty list means all valid."""
    errors = []
    for feat, val in values.items():
        lo, hi, _ = FEATURE_RANGES[feat]
        if val < lo or val > hi:
            errors.append(f"{feat}: {val:.2f} outside valid range [{lo}, {hi}]")
    return errors


# ──────────────────────────────────────────────
# Inline bar chart (feature importance for GUI)
# ──────────────────────────────────────────────

PALETTE = ['#2d6a4f','#40916c','#52b788','#74c69d',
           '#95d5b2','#b7e4c7','#d8f3dc','#1b4332']

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def make_importance_fig(importances: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(FEATURE_COLS))]
    bars = ax.barh(FEATURE_COLS, importances, color=colors,
                   edgecolor='#1b4332', linewidth=0.6)
    ax.set_xlabel('Gini Importance', fontsize=8)
    ax.set_title('Feature Importance', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    for bar, val in zip(bars, importances):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=6)
    fig.tight_layout(pad=0.4)
    return fig


def make_cluster_fig(X_scaled, km_bundle) -> plt.Figure:
    from sklearn.decomposition import PCA
    from matplotlib.colors import ListedColormap
    pca    = km_bundle['pca']
    km     = km_bundle['model']
    X2     = pca.transform(X_scaled)
    labels = km.predict(X_scaled)
    n_cl   = km.n_clusters
    fig, ax = plt.subplots(figsize=(5, 2.8))
    cmap = ListedColormap(PALETTE[:n_cl])
    ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=cmap,
               alpha=0.55, s=8, linewidths=0)
    c2d = pca.transform(km.cluster_centers_)
    ax.scatter(c2d[:, 0], c2d[:, 1], c='#d62828', marker='X',
               s=80, zorder=5, edgecolors='white', linewidths=0.5)
    ax.set_title('Soil Cluster Distribution', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_xlabel('PC-1', fontsize=7)
    ax.set_ylabel('PC-2', fontsize=7)
    fig.tight_layout(pad=0.4)
    return fig


def make_residual_fig(lr, X_test_scaled, y_test) -> plt.Figure:
    y_pred    = lr.predict(X_test_scaled)
    residuals = y_test - y_pred
    fig, ax   = plt.subplots(figsize=(5, 2.8))
    ax.scatter(y_pred, residuals, alpha=0.45, s=8, color=PALETTE[1], edgecolors='none')
    ax.axhline(0, color='#d62828', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Fitted Values', fontsize=8)
    ax.set_ylabel('Residuals', fontsize=8)
    ax.set_title('Residual Plot', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    return fig
