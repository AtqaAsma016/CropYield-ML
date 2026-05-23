"""
preprocessing.py
Data Engineering Layer for Smart Agriculture Decision Support System
Handles dataset loading, cleaning, feature engineering, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Synthetic Dataset Generator
# (mirrors real Crop Recommendation datasets from Kaggle/UCI)
# ──────────────────────────────────────────────

def generate_crop_dataset(n_samples: int = 2200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic agricultural dataset approximating the
    Crop Recommendation Dataset (Kaggle, 2020).

    Features
    --------
    N, P, K   : soil macro-nutrients (kg/ha)
    temperature: ambient temperature (°C)
    humidity   : relative humidity (%)
    ph         : soil pH
    rainfall   : annual rainfall (mm)

    Target
    ------
    label : crop category (22 classes)
    """
    rng = np.random.default_rng(random_state)

    CROP_PROFILES = {
        'rice':        dict(N=(80,120), P=(40,60),  K=(40,60),  T=(20,27), H=(80,90), pH=(5.5,6.5), R=(200,300)),
        'maize':       dict(N=(60,100), P=(50,70),  K=(50,70),  T=(18,27), H=(55,75), pH=(5.5,7.0), R=(60,110)),
        'chickpea':    dict(N=(40,60),  P=(65,85),  K=(75,95),  T=(18,24), H=(15,25), pH=(6.0,8.0), R=(80,120)),
        'kidneybeans': dict(N=(20,40),  P=(55,75),  K=(15,30),  T=(18,24), H=(18,22), pH=(5.5,7.0), R=(100,150)),
        'pigeonpeas':  dict(N=(15,30),  P=(60,80),  K=(15,30),  T=(25,35), H=(70,90), pH=(5.0,7.0), R=(150,200)),
        'mothbeans':   dict(N=(15,25),  P=(45,65),  K=(75,95),  T=(28,38), H=(45,65), pH=(3.5,6.5), R=(30,60)),
        'mungbean':    dict(N=(20,40),  P=(40,60),  K=(15,30),  T=(25,35), H=(80,90), pH=(6.2,7.2), R=(45,75)),
        'blackgram':   dict(N=(35,55),  P=(65,85),  K=(25,45),  T=(25,35), H=(60,80), pH=(5.5,7.5), R=(65,105)),
        'lentil':      dict(N=(15,30),  P=(65,85),  K=(15,30),  T=(15,25), H=(65,75), pH=(6.0,8.0), R=(20,40)),
        'pomegranate': dict(N=(15,25),  P=(15,25),  K=(195,215),T=(18,24), H=(90,95), pH=(6.5,8.0), R=(100,120)),
        'banana':      dict(N=(95,115), P=(75,95),  K=(48,62),  T=(25,33), H=(75,90), pH=(5.5,6.5), R=(100,130)),
        'mango':       dict(N=(15,25),  P=(15,25),  K=(28,42),  T=(27,37), H=(48,57), pH=(4.5,7.0), R=(90,110)),
        'grapes':      dict(N=(20,30),  P=(120,140),K=(195,215),T=(8,16),  H=(80,85), pH=(5.5,7.0), R=(65,75)),
        'watermelon':  dict(N=(95,115), P=(10,20),  K=(48,52),  T=(25,35), H=(85,95), pH=(6.0,7.0), R=(40,55)),
        'muskmelon':   dict(N=(95,115), P=(10,20),  K=(48,52),  T=(28,38), H=(90,95), pH=(6.0,7.0), R=(20,30)),
        'apple':       dict(N=(20,30),  P=(130,150),K=(195,215),T=(21,25), H=(92,96), pH=(5.5,7.5), R=(110,130)),
        'orange':      dict(N=(15,25),  P=(15,25),  K=(8,18),   T=(10,14), H=(90,95), pH=(6.0,8.0), R=(100,120)),
        'papaya':      dict(N=(45,55),  P=(15,25),  K=(40,55),  T=(33,37), H=(90,95), pH=(6.5,7.5), R=(140,160)),
        'coconut':     dict(N=(14,22),  P=(8,18),   K=(28,42),  T=(25,35), H=(93,98), pH=(5.0,8.0), R=(150,200)),
        'cotton':      dict(N=(115,125),P=(45,55),  K=(40,50),  T=(23,30), H=(78,88), pH=(6.0,8.0), R=(60,90)),
        'jute':        dict(N=(60,90),  P=(40,60),  K=(38,48),  T=(25,36), H=(70,90), pH=(6.0,8.0), R=(160,200)),
        'coffee':      dict(N=(95,105), P=(27,33),  K=(28,32),  T=(23,28), H=(58,68), pH=(6.0,7.0), R=(150,200)),
    }

    rows = []
    per_crop = n_samples // len(CROP_PROFILES)
    for crop, p in CROP_PROFILES.items():
        for _ in range(per_crop):
            rows.append({
                'N':           rng.uniform(*p['N']),
                'P':           rng.uniform(*p['P']),
                'K':           rng.uniform(*p['K']),
                'temperature': rng.uniform(*p['T']),
                'humidity':    rng.uniform(*p['H']),
                'ph':          rng.uniform(*p['pH']),
                'rainfall':    rng.uniform(*p['R']),
                'label':       crop,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Inject ~2 % noise / missing values to simulate real-world conditions
    for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
        mask = rng.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df


def generate_yield_dataset(n_samples: int = 1800, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic yield dataset for linear regression.
    Yield (tons/ha) is a noisy linear function of agronomic inputs.
    """
    rng = np.random.default_rng(random_state)
    N   = rng.uniform(20, 140, n_samples)
    P   = rng.uniform(10, 145, n_samples)
    K   = rng.uniform(5,  205, n_samples)
    T   = rng.uniform(8,  38,  n_samples)
    H   = rng.uniform(15, 98,  n_samples)
    pH  = rng.uniform(3.5,8.5, n_samples)
    R   = rng.uniform(20, 300, n_samples)

    yield_val = (
        0.018 * N +
        0.012 * P +
        0.008 * K +
        0.15  * T +
        0.04  * H -
        0.30  * np.abs(pH - 6.5) +
        0.007 * R +
        rng.normal(0, 0.6, n_samples)
    )
    yield_val = np.clip(yield_val, 0.5, 12.0)

    return pd.DataFrame({'N':N,'P':P,'K':K,'temperature':T,
                         'humidity':H,'ph':pH,'rainfall':R,'yield':yield_val})


# ──────────────────────────────────────────────
# Preprocessing Pipeline
# ──────────────────────────────────────────────

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def preprocess_crop_data(df: pd.DataFrame):
    """
    Returns X_train, X_test, y_train, y_test, label_encoder, scaler
    """
    df = df.copy()

    # 1. Missing-value imputation (median – robust to outliers)
    for col in FEATURE_COLS:
        df[col].fillna(df[col].median(), inplace=True)

    # 2. Outlier treatment via IQR capping
    for col in FEATURE_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

    # 3. Encode target
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])

    # 4. Feature scaling (StandardScaler – zero mean, unit variance)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y = df['label_enc'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, le, scaler


def preprocess_yield_data(df: pd.DataFrame):
    """
    Returns X_train, X_test, y_train, y_test, scaler
    """
    df = df.copy()
    for col in FEATURE_COLS:
        df[col].fillna(df[col].median(), inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y = df['yield'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


def preprocess_clustering_data(df: pd.DataFrame):
    """
    Returns scaled feature matrix and scaler for KNN/clustering use.
    """
    df = df.copy()
    for col in FEATURE_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Ensure no NaNs remain
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    return X, scaler, df[FEATURE_COLS]


if __name__ == "__main__":
    print("Generating datasets …")
    crop_df  = generate_crop_dataset()
    yield_df = generate_yield_dataset()
    os.makedirs("../data", exist_ok=True)
    crop_df.to_csv("../data/crop_data.csv",  index=False)
    yield_df.to_csv("../data/yield_data.csv", index=False)
    print(f"Crop dataset : {crop_df.shape}")
    print(f"Yield dataset: {yield_df.shape}")
    print("Datasets saved to data/")
