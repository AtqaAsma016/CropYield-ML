#  Smart Agriculture Decision Support System

---

## System Architecture

```
agri_system/
├── data/
│   ├── crop_data.csv          # 2 200-sample synthetic crop dataset (22 classes)
│   └── yield_data.csv         # 1 800-sample yield regression dataset
│
├── src/
│   ├── preprocessing.py       # Data engineering layer
│   ├── models.py              # Algorithmic core + training entry-point
│   ├── gui.py                 # Tkinter unified GUI (presentation layer)
│   └── utils.py               # Shared utilities & plot helpers
│
├── models/
│   ├── decision_tree.pkl      # Serialised DT classifier
│   ├── kmeans.pkl             # Serialised KMeans + PCA bundle
│   ├── linear_regression.pkl  # Serialised LR model
│   ├── crop_scalers.pkl       # StandardScaler + LabelEncoder for DT
│   ├── yield_scalers.pkl      # StandardScaler for LR
│   └── cluster_scalers.pkl    # StandardScaler for KMeans
│
├── results/
│   ├── feature_importance.png # DT Gini importance bar chart
│   ├── cluster_scatter.png    # KMeans PCA scatter plot
│   └── residual_plot.png      # LR residual analysis (2-panel)
│
├── requirements.txt
├── LICENSE
└── README.md
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   USER INPUT  (Tkinter GUI)                     │
│     N · P · K · Temperature · Humidity · pH · Rainfall          │
└────────────────────┬────────────────────────────────────────────┘
                     │  raw feature vector
          ┌──────────▼──────────────────────────────────┐
          │         PREPROCESSING LAYER                  │
          │  Imputation → IQR Capping → StandardScaler   │
          └──────┬──────────────┬──────────────┬─────────┘
                 │              │              │
    ┌────────────▼──┐  ┌────────▼──────┐  ┌───▼────────────┐
    │  Decision Tree│  │  KMeans (k=5) │  │Linear Regression│
    │  Classifier   │  │  Clustering   │  │ Yield Predictor │
    └────────────┬──┘  └────────┬──────┘  └───┬────────────┘
                 │              │              │
    Crop Label   │   Cluster ID + Guidance     │  Yield (t/ha) + CI
                 └──────────────▼──────────────┘
                        INTEGRATED OUTPUT
                     (rendered in GUI panels)
```

---

## Installation & Execution

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/agri-dss.git
cd agri-dss

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (generates data, trains, serialises, saves plots)
python src/models.py

# 4. Launch the GUI
python src/gui.py
```

---

## Algorithmic Modules

### 1. Decision Tree Classifier
- **Purpose**: Recommend the optimal crop based on soil macro-nutrients (N, P, K) and climate parameters (temperature, humidity, pH, rainfall).
- **Configuration**: `max_depth=12`, `criterion='gini'`, `class_weight='balanced'`, `min_samples_leaf=5`
- **Training**: 80/20 stratified train/test split on 2 200 samples across 22 crop classes.

### 2. KMeans Clustering *(Soil Profile Segmentation)*
- **Purpose**: Partition farm zones into 5 homogeneous soil profiles for targeted agronomic guidance.
- **Configuration**: `n_clusters=5`, `n_init=15`, `max_iter=300`
- **Visualisation**: PCA projection to 2D for scatter plot; cluster centroids marked.

### 3. Linear Regression *(Crop Yield Prediction)*
- **Purpose**: Quantify expected yield (tonnes/hectare) from agronomic input features.
- **Output**: Point estimate + 95% confidence interval (±1.96 × RMSE).

---

## Quantitative Performance Summary

| Model | Key Metric | Value |
|-------|-----------|-------|
| Decision Tree Classifier | Accuracy | **97.27 %** |
| Decision Tree Classifier | Weighted F1-Score | **0.9727** |
| KMeans Clustering | Silhouette Score | **0.2875** |
| Linear Regression | R² | **0.8695** |
| Linear Regression | RMSE | **0.6589 t/ha** |
| Linear Regression | MAE | **0.5301 t/ha** |

---

## Dataset

A synthetic dataset was generated to approximate the publicly available
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
(Kaggle, 2020) and yield datasets from the UCI ML Repository.

**Features (all models share the same 7-feature input vector):**

| Feature | Unit | Range |
|---------|------|-------|
| N | kg/ha | 0–200 |
| P | kg/ha | 0–150 |
| K | kg/ha | 0–210 |
| temperature | °C | 0–50 |
| humidity | % | 0–100 |
| ph | — | 0–14 |
| rainfall | mm | 0–400 |

**Preprocessing pipeline:**
1. Median imputation for ~2% synthetic missing values
2. IQR-based outlier capping (1.5× IQR rule)
3. Label encoding for crop categories (DT only)
4. StandardScaler normalisation (zero mean, unit variance)

---

## GUI Overview

The Tkinter interface provides four panels:

| Panel | Description |
|-------|-------------|
| **Predict** | Enter soil/climate parameters → receive crop recommendation, soil cluster + guidance, and predicted yield with CI |
| **Visualise** | Three embedded matplotlib plots rendered inside the GUI frame |
| **Metrics** | Summary of model configurations and evaluation metrics |
| **About** | System architecture, installation guide, and future work |

---

## Future Work

1. **IoT Sensor Integration** — Connect real-time soil and microclimate sensors via MQTT broker into the inference pipeline, enabling autonomous continuous monitoring of farm zones without manual parameter entry. Sensor fusion with GPS coordinates would allow spatial yield mapping.

2. **Ensemble Deep Learning Upgrade** — Replace the single Decision Tree with a gradient-boosted ensemble (XGBoost / LightGBM) or a fine-tuned TabTransformer architecture to push crop recommendation accuracy beyond 95% on field-collected data with high class imbalance and missing sensor readings.

---


