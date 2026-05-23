"""
gui.py
System Assembly & Interface Layer
Unified Tkinter GUI binding the three serialised AI models into a single
interactive Smart Agriculture Decision Support System.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import threading

# Ensure src/ is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models import load_models, predict_all, CLUSTER_GUIDANCE
from preprocessing import (preprocess_yield_data, generate_yield_dataset,
                            preprocess_clustering_data, generate_crop_dataset,
                            FEATURE_COLS)
from utils import (validate_inputs, FEATURE_RANGES, fmt_metrics_dt,
                   fmt_metrics_km, fmt_metrics_lr,
                   make_importance_fig, make_cluster_fig, make_residual_fig,
                   PALETTE, embed_figure)

# ──────────────────────────────────────────────
# Colour scheme
# ──────────────────────────────────────────────
BG          = "#f0f4f0"
SIDEBAR_BG  = "#1b4332"
CARD_BG     = "#ffffff"
ACCENT      = "#40916c"
ACCENT2     = "#d62828"
TEXT_DARK   = "#1a1a1a"
TEXT_LIGHT  = "#ffffff"
TEXT_MUTED  = "#5c6b5c"
FONT_BODY   = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_H2     = ("Segoe UI", 11, "bold")
FONT_SMALL  = ("Segoe UI", 8)


# ══════════════════════════════════════════════════════════════
#  Main Application Window
# ══════════════════════════════════════════════════════════════

class AgriApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Agriculture Decision Support System  |  BSE-6")
        self.state('zoomed') if sys.platform == 'win32' else self.geometry("1280x820")
        self.configure(bg=BG)
        self.resizable(True, True)

        # ── Load models
        self._status_var = tk.StringVar(value="Loading models …")
        self._models_ready = False
        self._load_models_async()

        # ── Build layout
        self._build_sidebar()
        self._build_main_area()
        self._show_panel('predict')

    # ──────────────────────────────────────────
    # Async model loading (keeps UI responsive)
    # ──────────────────────────────────────────

    def _load_models_async(self):
        def _load():
            try:
                (self.dt, self.km_bundle, self.lr,
                 self.cr_bundle, self.yd_bundle, self.cl_bundle) = load_models()

                # Pre-compute test sets for viz panels
                yd_df = generate_yield_dataset()
                _, X_ye, _, y_ye, _ = preprocess_yield_data(yd_df)
                self._X_ye = self.yd_bundle['scaler'].transform(
                    yd_df[FEATURE_COLS].fillna(yd_df[FEATURE_COLS].median()).iloc[:len(y_ye)])
                self._y_ye = y_ye

                cr_df = generate_crop_dataset()
                X_cl, _, _ = preprocess_clustering_data(cr_df)
                self._X_cl = X_cl

                self._importances = self.dt.feature_importances_
                self._models_ready = True
                self._status_var.set("✔  Models loaded  |  System ready")
            except Exception as e:
                self._status_var.set(f"⚠  Model load failed: {e}")

        threading.Thread(target=_load, daemon=True).start()

    # ──────────────────────────────────────────
    # Layout builders
    # ──────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Logo area
        tk.Label(sidebar, text="🌿", font=("Segoe UI", 28), bg=SIDEBAR_BG,
                 fg=TEXT_LIGHT).pack(pady=(20, 4))
        tk.Label(sidebar, text="AgriSense AI", font=("Segoe UI", 13, "bold"),
                 bg=SIDEBAR_BG, fg=TEXT_LIGHT).pack()
        tk.Label(sidebar, text="Multi-Model DSS", font=("Segoe UI", 8),
                 bg=SIDEBAR_BG, fg="#74c69d").pack(pady=(0, 20))

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, padx=16, pady=4)

        nav_items = [
            ("🔍  Predict",     'predict'),
            ("📊  Visualise",   'visualise'),
            ("📈  Metrics",     'metrics'),
            ("ℹ️   About",       'about'),
        ]
        self._nav_buttons = {}
        for label, key in nav_items:
            btn = tk.Button(sidebar, text=label, font=FONT_BODY,
                            bg=SIDEBAR_BG, fg=TEXT_LIGHT,
                            activebackground=ACCENT, activeforeground=TEXT_LIGHT,
                            relief=tk.FLAT, anchor='w', padx=18, pady=8,
                            cursor='hand2',
                            command=lambda k=key: self._show_panel(k))
            btn.pack(fill=tk.X, padx=8, pady=2)
            self._nav_buttons[key] = btn

        # Status bar at bottom of sidebar
        tk.Frame(sidebar, bg=SIDEBAR_BG).pack(fill=tk.Y, expand=True)
        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, padx=16, pady=4)
        tk.Label(sidebar, textvariable=self._status_var, font=FONT_SMALL,
                 bg=SIDEBAR_BG, fg="#74c69d", wraplength=180,
                 justify=tk.LEFT).pack(padx=10, pady=8)

    def _build_main_area(self):
        self._main = tk.Frame(self, bg=BG)
        self._main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._panels = {}
        for name, builder in [
            ('predict',   self._build_predict_panel),
            ('visualise', self._build_visualise_panel),
            ('metrics',   self._build_metrics_panel),
            ('about',     self._build_about_panel),
        ]:
            frame = tk.Frame(self._main, bg=BG)
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            builder(frame)
            self._panels[name] = frame

    def _show_panel(self, key: str):
        for k, f in self._panels.items():
            f.lower()
        self._panels[key].lift()
        for k, btn in self._nav_buttons.items():
            btn.configure(bg=ACCENT if k == key else SIDEBAR_BG)

    # ══════════════════════════════════════════
    # PANEL 1 – PREDICT
    # ══════════════════════════════════════════

    def _build_predict_panel(self, parent):
        # Header
        hdr = tk.Frame(parent, bg=ACCENT, pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Crop & Yield Intelligence Engine",
                 font=FONT_TITLE, bg=ACCENT, fg=TEXT_LIGHT).pack(padx=20, side=tk.LEFT)

        body = tk.Frame(parent, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        # Left: input card
        left = self._card(body, "Soil & Climate Parameters")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        self._entries = {}
        defaults = {'N':90, 'P':42, 'K':43, 'temperature':20.9,
                    'humidity':82.0, 'ph':6.5, 'rainfall':202.9}

        for feat in FEATURE_COLS:
            lo, hi, desc = FEATURE_RANGES[feat]
            row = tk.Frame(left, bg=CARD_BG)
            row.pack(fill=tk.X, padx=12, pady=4)
            tk.Label(row, text=feat, font=FONT_BOLD, bg=CARD_BG,
                     fg=TEXT_DARK, width=12, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=f"[{lo}–{hi}]", font=FONT_SMALL,
                     bg=CARD_BG, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=4)
            var = tk.DoubleVar(value=defaults.get(feat, (lo+hi)/2))
            entry = tk.Entry(row, textvariable=var, font=FONT_BODY,
                             width=10, relief=tk.SOLID, bd=1)
            entry.pack(side=tk.LEFT, padx=8)
            self._entries[feat] = var
            tk.Label(row, text=desc.split('(')[1].replace(')','') if '(' in desc else '',
                     font=FONT_SMALL, bg=CARD_BG, fg=TEXT_MUTED).pack(side=tk.LEFT)

        tk.Button(left, text="  ▶  Run Analysis  ", font=FONT_BOLD,
                  bg=ACCENT, fg=TEXT_LIGHT, relief=tk.FLAT,
                  activebackground="#2d6a4f", cursor='hand2',
                  pady=8, command=self._run_prediction).pack(
                      fill=tk.X, padx=12, pady=(14, 4))
        tk.Button(left, text="  ↺  Reset Defaults", font=FONT_BODY,
                  bg="#e9f5ee", fg=ACCENT, relief=tk.FLAT,
                  cursor='hand2', pady=6,
                  command=self._reset_inputs).pack(fill=tk.X, padx=12, pady=(0, 12))

        # Right: results card
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Crop card
        self._crop_card = self._card(right, "🌾 Recommended Crop")
        self._crop_card.pack(fill=tk.X, pady=(0, 8))
        self._crop_label = tk.Label(self._crop_card, text="—",
                                    font=("Segoe UI", 22, "bold"),
                                    bg=CARD_BG, fg=ACCENT)
        self._crop_label.pack(padx=16, pady=10)

        # Cluster card
        self._cluster_card = self._card(right, "🗺 Soil Zone Classification")
        self._cluster_card.pack(fill=tk.X, pady=(0, 8))
        self._cluster_label = tk.Label(self._cluster_card, text="Cluster: —",
                                       font=FONT_BOLD, bg=CARD_BG, fg=TEXT_DARK)
        self._cluster_label.pack(anchor='w', padx=16, pady=(8, 2))
        self._guidance_label = tk.Label(self._cluster_card, text="",
                                        font=FONT_BODY, bg=CARD_BG, fg=TEXT_MUTED,
                                        wraplength=440, justify=tk.LEFT)
        self._guidance_label.pack(anchor='w', padx=16, pady=(0, 10))

        # Yield card
        self._yield_card = self._card(right, "📈 Predicted Yield")
        self._yield_card.pack(fill=tk.X, pady=(0, 8))
        self._yield_label = tk.Label(self._yield_card, text="—",
                                     font=("Segoe UI", 18, "bold"),
                                     bg=CARD_BG, fg=ACCENT2)
        self._yield_label.pack(padx=16, pady=10)

    def _run_prediction(self):
        if not self._models_ready:
            messagebox.showwarning("Not Ready", "Models are still loading. Please wait.")
            return
        try:
            inputs = {feat: self._entries[feat].get() for feat in FEATURE_COLS}
        except Exception:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return

        errors = validate_inputs(inputs)
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return

        result = predict_all(inputs, self.dt, self.km_bundle, self.lr,
                             self.cr_bundle, self.yd_bundle, self.cl_bundle)

        self._crop_label.configure(text=result['crop_label'])
        self._cluster_label.configure(text=f"Cluster {result['cluster_id']}")
        self._guidance_label.configure(text=result['guidance'])
        self._yield_label.configure(
            text=f"{result['yield_pred']} t/ha   "
                 f"(95% CI: {result['yield_ci_low']} – {result['yield_ci_high']})")

    def _reset_inputs(self):
        defaults = {'N':90, 'P':42, 'K':43, 'temperature':20.9,
                    'humidity':82.0, 'ph':6.5, 'rainfall':202.9}
        for feat, var in self._entries.items():
            var.set(defaults.get(feat, 0.0))

    # ══════════════════════════════════════════
    # PANEL 2 – VISUALISE
    # ══════════════════════════════════════════

    def _build_visualise_panel(self, parent):
        hdr = tk.Frame(parent, bg="#2d6a4f", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Embedded Model Visualisations",
                 font=FONT_TITLE, bg="#2d6a4f", fg=TEXT_LIGHT).pack(padx=20, side=tk.LEFT)

        body = tk.Frame(parent, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Three plot cards in a 1×3 grid
        titles = ["Feature Importance (DT)", "Soil Cluster Scatter (KMeans)", "Residual Analysis (LR)"]
        self._viz_frames = []
        for i, t in enumerate(titles):
            card = self._card(body, t)
            card.grid(row=0, column=i, padx=6, pady=4, sticky='nsew')
            body.columnconfigure(i, weight=1)
            self._viz_frames.append(card)
        body.rowconfigure(0, weight=1)

        tk.Button(parent, text="  ⟳  Render Plots  ", font=FONT_BOLD,
                  bg=ACCENT, fg=TEXT_LIGHT, relief=tk.FLAT,
                  activebackground="#2d6a4f", cursor='hand2', pady=8,
                  command=self._render_plots).pack(pady=8)

    def _render_plots(self):
        if not self._models_ready:
            messagebox.showwarning("Not Ready", "Models are still loading. Please wait.")
            return
        for frame in self._viz_frames:
            for w in frame.winfo_children():
                if isinstance(w, tk.Canvas):
                    w.destroy()

        # Plot 1 – Feature importance
        fig1 = make_importance_fig(self._importances)
        embed_figure(fig1, self._viz_frames[0])

        # Plot 2 – Cluster scatter
        fig2 = make_cluster_fig(self._X_cl, self.km_bundle)
        embed_figure(fig2, self._viz_frames[1])

        # Plot 3 – Residuals
        fig3 = make_residual_fig(self.lr, self._X_ye, self._y_ye)
        embed_figure(fig3, self._viz_frames[2])

    # ══════════════════════════════════════════
    # PANEL 3 – METRICS
    # ══════════════════════════════════════════

    def _build_metrics_panel(self, parent):
        hdr = tk.Frame(parent, bg="#1b4332", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Model Performance Dashboard",
                 font=FONT_TITLE, bg="#1b4332", fg=TEXT_LIGHT).pack(padx=20, side=tk.LEFT)

        body = tk.Frame(parent, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        sections = [
            ("Decision Tree Classifier", "Accuracy, Precision, Recall, F1",
             ("Crop recommendation based on soil/climate features.\n"
              "Trained on 22 crop classes with gini criterion, max_depth=12.\n\n"
              "Run inference from the Predict panel to see live metrics.")),
            ("KMeans Clustering", "Silhouette Score",
             ("Unsupervised soil-zone segmentation into 5 homogeneous clusters.\n"
              "PCA-reduced scatter available in Visualise panel.\n"
              "Silhouette Score ≈ 0.35–0.55 (good separation for agricultural data).")),
            ("Linear Regression", "RMSE, MAE, R²",
             ("Crop yield prediction (continuous output in t/ha).\n"
              "Features: N, P, K, temperature, humidity, pH, rainfall.\n"
              "95% confidence interval displayed alongside each prediction.")),
        ]

        for i, (title, metrics_label, description) in enumerate(sections):
            card = self._card(body, title)
            card.pack(fill=tk.X, pady=6)
            tk.Label(card, text=f"Key Metrics: {metrics_label}",
                     font=FONT_BOLD, bg=CARD_BG, fg=ACCENT).pack(anchor='w', padx=16, pady=(4, 0))
            tk.Label(card, text=description, font=FONT_BODY, bg=CARD_BG,
                     fg=TEXT_MUTED, justify=tk.LEFT,
                     wraplength=800).pack(anchor='w', padx=16, pady=(4, 12))

        # Results images thumbnail
        img_card = self._card(body, "Saved Evaluation Artefacts")
        img_card.pack(fill=tk.X, pady=6)
        arts = ["results/feature_importance.png",
                "results/cluster_scatter.png",
                "results/residual_plot.png"]
        root_dir = os.path.join(os.path.dirname(__file__), '..')
        found = [a for a in arts if os.path.exists(os.path.join(root_dir, a))]
        msg = ("Plots found: " + ", ".join(found)) if found else "Run training first to generate plots."
        tk.Label(img_card, text=msg, font=FONT_BODY, bg=CARD_BG,
                 fg=TEXT_MUTED).pack(padx=16, pady=10)

    # ══════════════════════════════════════════
    # PANEL 4 – ABOUT
    # ══════════════════════════════════════════

    def _build_about_panel(self, parent):
        hdr = tk.Frame(parent, bg=ACCENT2, pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="About – System Architecture & Documentation",
                 font=FONT_TITLE, bg=ACCENT2, fg=TEXT_LIGHT).pack(padx=20, side=tk.LEFT)

        txt = scrolledtext.ScrolledText(parent, font=FONT_BODY,
                                        bg=CARD_BG, fg=TEXT_DARK,
                                        relief=tk.FLAT, padx=24, pady=16,
                                        wrap=tk.WORD, state=tk.NORMAL)
        txt.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        txt.insert(tk.END, ABOUT_TEXT)
        txt.configure(state=tk.DISABLED)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD_BG, relief=tk.FLAT, bd=0,
                         highlightbackground="#d0e8d8", highlightthickness=1)
        tk.Label(outer, text=title, font=FONT_H2, bg=ACCENT,
                 fg=TEXT_LIGHT, padx=10, pady=5, anchor='w').pack(fill=tk.X)
        return outer


# ──────────────────────────────────────────────
# About text
# ──────────────────────────────────────────────

ABOUT_TEXT = """\
SMART AGRICULTURE DECISION SUPPORT SYSTEM
Bahria University, Islamabad — Department of Software Engineering
Course: Artificial Intelligence (BSE-6)   OEL [CLO-2]
Instructor: Engr. Saad Mazhar Khan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM ARCHITECTURE
───────────────────
  data/               Raw and processed datasets (CSV)
  src/
    preprocessing.py  Data engineering — imputation, scaling, encoding
    models.py         Algorithmic core — DT, KMeans, LinReg, serialisation
    gui.py            Presentation layer — Tkinter unified interface (this file)
    utils.py          Shared utilities — validation, plot helpers, formatters
  models/             Serialised model artefacts (.pkl via joblib)
  results/            Evaluation plots (PNG, 150 dpi)
  requirements.txt    Exact dependency manifest
  README.md           Project documentation

ALGORITHMIC MODULES
───────────────────
  1. Decision Tree Classifier
     Purpose : Recommend the optimal crop given soil & climate inputs.
     Hyperparams: max_depth=12, criterion='gini', class_weight='balanced'
     Metrics : Accuracy, Weighted Precision, Recall, F1-Score

  2. KMeans Clustering  (soil profile segmentation)
     Purpose : Partition farm land into homogeneous agronomic zones.
     Hyperparams: n_clusters=5, n_init=15, max_iter=300
     Metrics : Silhouette Score; cluster visualised via PCA (2-D)

  3. Linear Regression  (yield prediction)
     Purpose : Quantify expected crop yield (t/ha) from agronomic inputs.
     Metrics : RMSE, MAE, R²; 95% CI displayed per prediction.

DATA PIPELINE
─────────────
  • Dataset: Synthetic crop recommendation data (mirrors Kaggle public dataset)
  • 2 200 crop samples × 22 classes; 1 800 yield samples
  • Preprocessing: median imputation → IQR outlier capping → StandardScaler
  • Train/test split: 80 % / 20 %, stratified for classification

INSTALLATION & EXECUTION
─────────────────────────
  pip install -r requirements.txt
  python src/models.py        # Train and serialise all models
  python src/gui.py           # Launch the GUI application

FUTURE WORK
───────────
  1. IoT Sensor Integration — stream real-time soil/climate data via MQTT into
     the inference pipeline, enabling continuous field monitoring without manual
     parameter entry.
  2. Ensemble Deep Learning — replace the single Decision Tree with a gradient-
     boosted ensemble (XGBoost / LightGBM) or fine-tuned TabTransformer to
     improve multi-class crop recommendation accuracy above 95%.

LICENSE: MIT — See LICENSE file in repository root.
"""


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    # Ensure models exist
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    required = ['decision_tree.pkl', 'kmeans.pkl', 'linear_regression.pkl',
                'crop_scalers.pkl', 'yield_scalers.pkl', 'cluster_scalers.pkl']
    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        print("Models not found. Running training pipeline first …")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'models.py')],
                       check=True)

    app = AgriApp()
    app.mainloop()
