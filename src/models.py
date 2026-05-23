"""
models.py
Algorithmic Core – Training, Evaluation, and Serialization
Implements Decision Tree Classifier, KMeans Clustering (KNN-based),
and Linear Regression with full metric reporting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, silhouette_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.decomposition import PCA

from preprocessing import (generate_crop_dataset, generate_yield_dataset,
                            preprocess_crop_data, preprocess_yield_data,
                            preprocess_clustering_data, FEATURE_COLS)

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# PALETTE
# ──────────────────────────────────────────────
PALETTE = ['#2d6a4f','#40916c','#52b788','#74c69d',
           '#95d5b2','#b7e4c7','#d8f3dc','#1b4332']


# ══════════════════════════════════════════════
# 1. DECISION TREE CLASSIFIER
# ══════════════════════════════════════════════

def train_decision_tree(X_train, X_test, y_train, y_test, label_encoder):
    """Train DT, print metrics, save model + feature-importance plot."""

    clf = DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        criterion='gini',
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print("\n── Decision Tree Classifier ──")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")

    # Feature importance plot
    importances = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(FEATURE_COLS, importances,
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(FEATURE_COLS))],
                   edgecolor='#1b4332', linewidth=0.7)
    ax.set_xlabel('Gini Importance', fontsize=11)
    ax.set_title('Decision Tree – Feature Importance Vector', fontsize=13, fontweight='bold')
    ax.set_xlim(0, importances.max() * 1.2)
    for bar, val in zip(bars, importances):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close(fig)

    # Serialize
    joblib.dump(clf, os.path.join(MODELS_DIR, 'decision_tree.pkl'))
    print("  Model saved → models/decision_tree.pkl")
    return clf, metrics, importances


# ══════════════════════════════════════════════
# 2. KMeans CLUSTERING  (soil profile segmentation)
# ══════════════════════════════════════════════

def train_clustering(X_scaled, raw_df, n_clusters: int = 5):
    """
    KMeans clustering for soil-zone segmentation.
    Silhouette score reported; PCA scatter saved.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15, max_iter=300)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"\n── KMeans Clustering ──")
    print(f"  n_clusters      : {n_clusters}")
    print(f"  silhouette score: {sil:.4f}")

    # PCA → 2-D for visualization
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = ListedColormap(PALETTE[:n_clusters])
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap=cmap,
                    alpha=0.65, s=18, linewidths=0)
    centers_2d = pca.transform(km.cluster_centers_)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='#d62828', marker='X', s=160, zorder=5,
               edgecolors='white', linewidths=0.8, label='Centroids')
    ax.set_xlabel(f'PC-1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC-2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax.set_title('KMeans – Soil Profile Cluster Distribution (PCA)', fontsize=13, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Cluster ID')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'cluster_scatter.png'), dpi=150)
    plt.close(fig)

    metrics = {'silhouette': sil, 'n_clusters': n_clusters}
    joblib.dump({'model': km, 'pca': pca}, os.path.join(MODELS_DIR, 'kmeans.pkl'))
    print("  Model saved → models/kmeans.pkl")
    return km, pca, labels, metrics


# ══════════════════════════════════════════════
# 3. LINEAR REGRESSION  (crop yield prediction)
# ══════════════════════════════════════════════

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train LR, report RMSE/MAE/R², save residual plot."""

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print("\n── Linear Regression (Yield Prediction) ──")
    for k, v in metrics.items():
        print(f"  {k:>6}: {v:.4f}")

    # Residual analysis plot (2-panel)
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A – Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.45, s=18, color=PALETTE[1], edgecolors='none')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.2, label='Ideal fit')
    ax.set_xlabel('Actual Yield (t/ha)', fontsize=10)
    ax.set_ylabel('Predicted Yield (t/ha)', fontsize=10)
    ax.set_title(f'Actual vs Predicted  |  R²={r2:.3f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel B – Residuals vs Fitted
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.45, s=18, color=PALETTE[3], edgecolors='none')
    ax.axhline(0, color='#d62828', linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted Values', fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(f'Residual Plot  |  RMSE={rmse:.3f}  MAE={mae:.3f}', fontsize=11, fontweight='bold')

    fig.suptitle('Linear Regression – Residual Analysis', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'residual_plot.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    joblib.dump(lr, os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    print("  Model saved → models/linear_regression.pkl")
    return lr, metrics


# ══════════════════════════════════════════════
# CLUSTER AGRONOMIC GUIDANCE
# ══════════════════════════════════════════════

CLUSTER_GUIDANCE = {
    0: "High-fertility zone – optimal for nutrient-demanding crops (maize, rice). Maintain pH 6.0–7.0.",
    1: "Moderate-fertility zone – suitable for legumes and cereals. Consider phosphorus supplementation.",
    2: "Sandy low-moisture zone – drought-resistant crops recommended (sorghum, millet). Irrigate carefully.",
    3: "High-humidity zone – ideal for tropical fruits (banana, papaya). Monitor fungal risk.",
    4: "Alkaline zone – pH amendment needed. Suitable for cotton after liming treatment.",
}


def get_cluster_guidance(cluster_id: int) -> str:
    return CLUSTER_GUIDANCE.get(cluster_id, f"Cluster {cluster_id}: Consult local agronomist.")


# ══════════════════════════════════════════════
# INFERENCE HELPERS  (called by gui.py)
# ══════════════════════════════════════════════

def load_models():
    """Load all serialized models and scalers from disk."""
    dt  = joblib.load(os.path.join(MODELS_DIR, 'decision_tree.pkl'))
    km_bundle = joblib.load(os.path.join(MODELS_DIR, 'kmeans.pkl'))
    lr  = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    cr_bundle = joblib.load(os.path.join(MODELS_DIR, 'crop_scalers.pkl'))
    yd_bundle = joblib.load(os.path.join(MODELS_DIR, 'yield_scalers.pkl'))
    cl_bundle = joblib.load(os.path.join(MODELS_DIR, 'cluster_scalers.pkl'))
    return dt, km_bundle, lr, cr_bundle, yd_bundle, cl_bundle


def predict_all(inputs: dict, dt, km_bundle, lr, cr_bundle, yd_bundle, cl_bundle):
    """
    Run the full inference pipeline for a single input observation.

    Parameters
    ----------
    inputs : dict with keys N, P, K, temperature, humidity, ph, rainfall

    Returns
    -------
    dict with crop_label, cluster_id, guidance, yield_pred, yield_ci_low, yield_ci_high
    """
    x_raw = np.array([[inputs['N'], inputs['P'], inputs['K'],
                        inputs['temperature'], inputs['humidity'],
                        inputs['ph'], inputs['rainfall']]])

    # 1. Crop classification
    x_cr  = cr_bundle['scaler'].transform(x_raw)
    crop_enc  = dt.predict(x_cr)[0]
    crop_label = cr_bundle['le'].inverse_transform([crop_enc])[0]

    # 2. Soil clustering
    x_cl  = cl_bundle['scaler'].transform(x_raw)
    cluster_id = int(km_bundle['model'].predict(x_cl)[0])
    guidance   = get_cluster_guidance(cluster_id)

    # 3. Yield regression
    x_yd  = yd_bundle['scaler'].transform(x_raw)
    yield_pred = float(lr.predict(x_yd)[0])
    # Simple ± 1-sigma confidence bound (approximated from training RMSE)
    rmse_approx = 0.65
    yield_pred  = max(0.0, yield_pred)
    ci_low  = max(0.0, yield_pred - 1.96 * rmse_approx)
    ci_high = yield_pred + 1.96 * rmse_approx

    return {
        'crop_label':  crop_label.title(),
        'cluster_id':  cluster_id,
        'guidance':    guidance,
        'yield_pred':  round(yield_pred, 2),
        'yield_ci_low':  round(ci_low, 2),
        'yield_ci_high': round(ci_high, 2),
    }


# ══════════════════════════════════════════════
# TRAINING ENTRY-POINT
# ══════════════════════════════════════════════

def train_all():
    print("=" * 52)
    print("  Smart Agriculture – Model Training Pipeline")
    print("=" * 52)

    # ── Crop classification data
    crop_df  = generate_crop_dataset()
    X_tr, X_te, y_tr, y_te, le, cr_scaler = preprocess_crop_data(crop_df)
    joblib.dump({'scaler': cr_scaler, 'le': le},
                os.path.join(MODELS_DIR, 'crop_scalers.pkl'))

    dt_model, dt_metrics, importances = train_decision_tree(X_tr, X_te, y_tr, y_te, le)

    # ── Clustering data (use crop_df features, no labels needed)
    X_cl, cl_scaler, raw_feats = preprocess_clustering_data(crop_df)
    joblib.dump({'scaler': cl_scaler}, os.path.join(MODELS_DIR, 'cluster_scalers.pkl'))
    km_model, pca_model, cluster_labels, km_metrics = train_clustering(X_cl, raw_feats)

    # ── Yield regression data
    yield_df = generate_yield_dataset()
    X_yt, X_ye, y_yt, y_ye, yd_scaler = preprocess_yield_data(yield_df)
    joblib.dump({'scaler': yd_scaler}, os.path.join(MODELS_DIR, 'yield_scalers.pkl'))
    lr_model, lr_metrics = train_linear_regression(X_yt, X_ye, y_yt, y_ye)

    print("\n" + "=" * 52)
    print("  All models trained and serialized successfully.")
    print("  Evaluation plots saved to results/")
    print("=" * 52)

    return {
        'dt': dt_metrics,
        'km': km_metrics,
        'lr': lr_metrics,
        'importances': importances,
    }


if __name__ == "__main__":
    train_all()
