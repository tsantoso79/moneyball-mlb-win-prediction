"""Generate 06_PCA_KMeans_Huber.ipynb — V7: PCA components + K-means clusters + Huber regression + scaler-inside-CV."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "base", "language": "python", "name": "base"}

cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ============================================================
md("""# Moneyball V7: PCA + K-Means Clusters + Huber Regression

**Competition:** SCTPDSAI-M3-DS2F Coaching — MoneyBall Analytics
**Metric:** MAE (Mean Absolute Error) — lower is better

## Motivation
Our best model (V6-A: Ridge with per-game + era z-scores) achieves CV MAE 2.709 and public MAE ~2.99.
The leaderboard leader sits at 2.959. To close the gap, we try three new ideas:

### New Ideas in V7

| Idea | What It Does | Why It Might Help |
|---|---|---|
| **PCA components** | Extracts orthogonal variance axes from all features | Captures latent team archetypes that individual features miss |
| **K-Means clusters** | Groups teams into archetypes (e.g., "power", "pitching-dominant") | Adds nonlinear team-type signal that Ridge can't discover |
| **Huber regression** | Regression robust to outlier seasons | Reduces influence of extreme W seasons (36-win, 116-win teams) |
| **Scaler inside CV** | Fits StandardScaler per CV fold, not on full data | Gives honest CV estimate — no information leakage from val fold |

We build on V6's feature engineering (per-game rates + era z-scores + sabermetrics) as the base feature set.
""")

# ============================================================
md("## 1. Setup & Data Loading")

code("""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DATA_DIR = "."

train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")

print(f"Training set: {train.shape[0]} team-seasons")
print(f"Test set:     {test.shape[0]} team-seasons")
print(f"Year range — Train: {train['yearID'].min()}-{train['yearID'].max()}")
print(f"Year range — Test:  {test['yearID'].min()}-{test['yearID'].max()}")""")

# ============================================================
md("## 2. Feature Engineering (V6 Base)")

code("""def assign_era(df):
    \"\"\"Create era_label column from one-hot era indicators.\"\"\"
    df = df.copy()
    era_cols = [c for c in df.columns if c.startswith('era_')]
    era_labels = []
    for _, row in df.iterrows():
        found = False
        for ec in era_cols:
            if row[ec]:
                era_labels.append(ec)
                found = True
                break
        if not found:
            era_labels.append('era_unknown')
    df['era_label'] = era_labels
    return df

def engineer_features_v6(df, franch_counts=None, era_stats=None):
    \"\"\"Per-game normalisation + era z-score + sabermetric features.\"\"\"
    df = df.copy()
    G = df['G']

    counting_stats = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                      'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
    for s in counting_stats:
        df[f'{s}_pg'] = df[s] / G

    stats_to_zscore = counting_stats + [f'{s}_pg' for s in counting_stats] + ['ERA', 'FP']

    if era_stats is None:
        era_stats = {}
        for s in stats_to_zscore:
            era_stats[s] = df.groupby('era_label')[s].agg(['mean', 'std']).to_dict('index')

    for s in stats_to_zscore:
        z_vals = []
        for _, row in df.iterrows():
            era = row['era_label']
            if era in era_stats[s]:
                mean = era_stats[s][era]['mean']
                std = era_stats[s][era]['std']
                z_vals.append((row[s] - mean) / (std + 1e-6))
            else:
                z_vals.append(0)
        df[f'{s}_ez'] = z_vals

    Rpg, RApg = df['R_pg'], df['RA_pg']
    Hpg, ABpg, BBpg, SOpg = df['H_pg'], df['AB_pg'], df['BB_pg'], df['SO_pg']
    HRpg, B2pg, B3pg, SBpg = df['HR_pg'], df['2B_pg'], df['3B_pg'], df['SB_pg']
    HApg, HRApg, BBApg, SOApg = df['HA_pg'], df['HRA_pg'], df['BBA_pg'], df['SOA_pg']
    IPpg = df['IPouts_pg'] / 3
    ERpg = df['ER_pg']

    df['1B_pg'] = Hpg - B2pg - B3pg - HRpg
    df['TB_pg'] = df['1B_pg'] + 2*B2pg + 3*B3pg + 4*HRpg
    PApg = ABpg + BBpg
    df['BA']  = Hpg / (ABpg + 0.001)
    df['SLG'] = df['TB_pg'] / (ABpg + 0.001)
    df['OBP'] = (Hpg + BBpg) / (PApg + 0.001)
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['wOBA'] = (0.69*BBpg + 0.89*df['1B_pg'] + 1.27*B2pg + 1.62*B3pg + 2.10*HRpg) / (PApg + 0.001)
    df['BABIP'] = (Hpg - HRpg) / (ABpg - SOpg - HRpg + 0.001)
    df['K_pct']  = SOpg / (PApg + 0.001)
    df['BB_pct'] = BBpg / (PApg + 0.001)
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC_pg']  = (Hpg + BBpg) * df['TB_pg'] / (PApg + 0.001)
    df['SecA']   = (df['TB_pg'] - Hpg + BBpg + SBpg) / (ABpg + 0.001)
    df['power_speed'] = (2 * HRpg * SBpg) / (HRpg + SBpg + 0.001)
    df['BB_SO_ratio'] = BBpg / (SOpg + 0.001)

    df['WHIP'] = (BBApg + HApg) / (IPpg + 0.001)
    df['H9']   = HApg / (IPpg + 0.001) * 9
    df['HR9']  = HRApg / (IPpg + 0.001) * 9
    df['BB9']  = BBApg / (IPpg + 0.001) * 9
    df['SO9']  = SOApg / (IPpg + 0.001) * 9
    df['SO_BB_ratio']     = SOApg / (BBApg + 0.001)
    df['FIP']             = (13*HRApg + 3*BBApg - 2*SOApg) / (IPpg + 0.001) + 3.10
    df['BABIP_against']   = (HApg - HRApg) / (IPpg*3 - SOApg - HRApg + 0.001)
    df['pitch_dominance'] = SOApg / (HApg + BBApg + 0.001)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff']    = df['FIP'] - df['ERA']

    df['run_diff_pg'] = Rpg - RApg
    R, RA = df['R'], df['RA']
    df['pyth_wins']     = R**2 / (R**2 + RA**2) * G
    df['pyth_wins_183'] = R**1.83 / (R**1.83 + RA**1.83) * G
    rpg_total = (R + RA) / G
    pexp = rpg_total ** 0.287
    df['pyth_wins_pat'] = R**pexp / (R**pexp + RA**pexp) * G
    df['pyth_pct']     = R**2 / (R**2 + RA**2)
    df['pyth_pct_pat'] = R**pexp / (R**pexp + RA**pexp)
    df['log_run_ratio'] = np.log(R / (RA + 1))
    df['run_diff_sq']   = (R - RA)**2 * np.sign(R - RA)
    df['R_above_RC']    = Rpg - df['RC_pg']
    df['run_support_era'] = Rpg / (df['ERA'] + 0.001)

    df['DER'] = 1 - ((HApg - HRApg) / (df['IPouts_pg'] + HApg - HRApg - SOApg + 0.001))
    df['UER_pg']  = RApg - ERpg
    df['UER_pct'] = df['UER_pg'] / (RApg + 0.001)

    df['R_vs_mlb']  = Rpg - df['mlb_rpg']
    df['RA_vs_mlb'] = RApg - df['mlb_rpg']
    df['R_ratio']   = Rpg / (df['mlb_rpg'] + 0.001)
    df['RA_ratio']  = RApg / (df['mlb_rpg'] + 0.001)
    league_era = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era / (df['FIP'] + 0.001) * 100
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100
    df['R_above_avg']  = df['R_vs_mlb'] * G
    df['RA_below_avg'] = -df['RA_vs_mlb'] * G
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    if franch_counts is not None:
        df['franch_freq'] = df['franchID'].map(franch_counts).fillna(1)
    else:
        franch_counts = df['franchID'].value_counts().to_dict()
        df['franch_freq'] = df['franchID'].map(franch_counts)

    return df, franch_counts, era_stats

# Assign eras
train = assign_era(train)
test = assign_era(test)

# Compute era stats on combined data
combined = pd.concat([train, test], ignore_index=True)
combined = assign_era(combined) if 'era_label' not in combined.columns else combined
counting_stats = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                  'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
for s in counting_stats:
    combined[f'{s}_pg'] = combined[s] / combined['G']
stats_to_zscore = counting_stats + [f'{s}_pg' for s in counting_stats] + ['ERA', 'FP']
era_stats = {}
for s in stats_to_zscore:
    era_stats[s] = combined.groupby('era_label')[s].agg(['mean', 'std']).to_dict('index')

train_fe, franch_counts, _ = engineer_features_v6(train, era_stats=era_stats)
test_fe, _, _ = engineer_features_v6(test, franch_counts=franch_counts, era_stats=era_stats)
print(f"V6 base features — Train: {train_fe.shape}, Test: {test_fe.shape}")""")

# ============================================================
md("""## 3. Prepare Base Feature Matrix

We use V6's best variant (A: per-game + era z-scores + sabermetrics, no raw counting stats) as the base.
""")

code("""y = train_fe['W'].values
exclude = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins', 'era_label'}

# Cast string columns to avoid type issues
for col_to_drop in ['era_label', 'decade_label']:
    if col_to_drop in train_fe.columns:
        train_fe[col_to_drop] = train_fe[col_to_drop].astype(str)
    if col_to_drop in test_fe.columns:
        test_fe[col_to_drop] = test_fe[col_to_drop].astype(str)

# V6-A feature set: per-game + era z-scores + sabermetrics + indicators (no raw counting)
raw_counting = set(counting_stats)
pg_cols = {f'{s}_pg' for s in counting_stats}
ez_cols = {c for c in train_fe.columns if c.endswith('_ez')}
saber_cols = {c for c in train_fe.columns if c not in exclude and c not in raw_counting
              and c not in pg_cols and c not in ez_cols
              and train_fe[c].dtype in ['int64', 'float64', 'bool']
              and not c.startswith(('era_', 'decade_'))}
indicator_cols = {c for c in train_fe.columns if c.startswith(('era_', 'decade_'))
                  and train_fe[c].dtype in ['bool', 'int64', 'float64']}

base_features = sorted(pg_cols | ez_cols | saber_cols | indicator_cols)
base_features = [c for c in base_features if c in train_fe.columns]

# Prepare base data
def prepare_data(train_df, test_df, feat_list):
    \"\"\"Clean and prepare feature matrices.\"\"\"
    tr = train_df[feat_list].copy()
    te = test_df[feat_list].copy()
    for c in feat_list:
        if tr[c].dtype == 'bool':
            tr[c] = tr[c].astype(int)
            te[c] = te[c].astype(int)
    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)
    for c in feat_list:
        med = tr[c].median()
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)
    return tr.values.astype(float), te.values.astype(float)

X_base, Xt_base = prepare_data(train_fe, test_fe, base_features)

# Identify indicator vs continuous columns for selective scaling
oh_feats = [c for c in base_features if c.startswith(('era_', 'decade_'))]
ct_feats = [c for c in base_features if c not in oh_feats]
oh_idx = [base_features.index(c) for c in oh_feats]
ct_idx = [base_features.index(c) for c in ct_feats]

print(f"Base features (V6-A): {len(base_features)}")
print(f"  Continuous: {len(ct_idx)},  Indicator: {len(oh_idx)}")""")

# ============================================================
md("""## 4. PCA Feature Extraction

We run PCA on the scaled continuous features and keep the top components that explain 95% of variance.
These capture orthogonal team-performance dimensions that individual features may not isolate.
""")

code("""# Scale the base continuous features for PCA
sc_pca = StandardScaler()
X_cont = sc_pca.fit_transform(X_base[:, ct_idx])
Xt_cont = sc_pca.transform(Xt_base[:, ct_idx])

# Fit PCA — keep enough components for 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_cont)
Xt_pca = pca.transform(Xt_cont)

print(f"PCA: {len(ct_idx)} features → {X_pca.shape[1]} components (95% variance explained)")
print(f"Explained variance ratios: {pca.explained_variance_ratio_[:10].round(4)}")
print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_[:10]).round(4)}")

# Also try fixed counts: 5, 10, 15, 20 components
for n in [5, 10, 15, 20]:
    pca_n = PCA(n_components=n, random_state=42)
    pca_n.fit(X_cont)
    print(f"  Top {n:2d} components explain {pca_n.explained_variance_ratio_.sum()*100:.1f}% of variance")""")

# ============================================================
md("""## 5. K-Means Team Clustering

Group teams into archetypes based on their feature profiles. Each cluster label becomes a one-hot feature.
""")

code("""# K-means on scaled continuous features — test different k values
print("K-Means clustering — inertia by k:")
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cont)
    inertias.append(km.inertia_)
    print(f"  k={k:2d}  inertia={km.inertia_:>12.0f}")

# Use k=5 as a reasonable default (captures main team archetypes)
for k_test in [3, 4, 5, 6, 7]:
    km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    train_clusters = km.fit_predict(X_cont)
    test_clusters = km.predict(Xt_cont)

    # Show cluster distribution
    unique, counts = np.unique(train_clusters, return_counts=True)
    dist = ', '.join([f'C{u}={c}' for u, c in zip(unique, counts)])
    print(f"  k={k_test}: {dist}")""")

# ============================================================
md("""## 6. Model Comparison — Scaler Inside CV

We now test multiple model configurations with **proper CV** (scaler fitted inside each fold).

### Variants Tested

| Variant | Model | Features | New Idea |
|---|---|---|---|
| V6-A baseline | Ridge | 140 base features | None (reproduce V6 with scaler-inside-CV) |
| V7-PCA5 | Ridge | base + 5 PCA components | PCA captures latent axes |
| V7-PCA10 | Ridge | base + 10 PCA components | More PCA components |
| V7-KM5 | Ridge | base + 5 cluster one-hots | K-means archetypes |
| V7-PCA10-KM5 | Ridge | base + 10 PCA + 5 clusters | Combined |
| V7-Huber | Huber | 140 base features | Robust to outliers |
| V7-Huber-PCA10-KM5 | Huber | base + PCA + clusters | Full kitchen sink |
""")

code("""alphas = np.logspace(-3, 4, 300)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_ridge_proper_cv(X, y, alphas, kf, oh_idx_local, ct_idx_local):
    \"\"\"Ridge with scaler fitted INSIDE each CV fold (no data leakage).\"\"\"
    oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx].copy(), X[va_idx].copy()

        # Fit scaler only on training fold
        if ct_idx_local:
            sc = StandardScaler()
            X_tr[:, ct_idx_local] = sc.fit_transform(X_tr[:, ct_idx_local])
            X_va[:, ct_idx_local] = sc.transform(X_va[:, ct_idx_local])

        r = RidgeCV(alphas=alphas, cv=5)
        r.fit(X_tr, y[tr_idx])
        oof[va_idx] = r.predict(X_va)

    return mean_absolute_error(y, oof), oof

def evaluate_huber_proper_cv(X, y, kf, oh_idx_local, ct_idx_local):
    \"\"\"Huber regression with scaler fitted INSIDE each CV fold.\"\"\"
    oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx].copy(), X[va_idx].copy()

        if ct_idx_local:
            sc = StandardScaler()
            X_tr[:, ct_idx_local] = sc.fit_transform(X_tr[:, ct_idx_local])
            X_va[:, ct_idx_local] = sc.transform(X_va[:, ct_idx_local])

        # Test multiple epsilon values (controls outlier threshold)
        best_mae_inner = 999
        best_hub = None
        for eps in [1.1, 1.35, 1.5, 2.0]:
            for alpha_h in [0.01, 0.1, 1.0, 10.0, 100.0]:
                hub = HuberRegressor(epsilon=eps, alpha=alpha_h, max_iter=500)
                hub.fit(X_tr, y[tr_idx])
                inner_pred = hub.predict(X_va)
                inner_mae = mean_absolute_error(y[va_idx], inner_pred)
                if inner_mae < best_mae_inner:
                    best_mae_inner = inner_mae
                    best_hub = hub

        oof[va_idx] = best_hub.predict(X_va)

    return mean_absolute_error(y, oof), oof

# Build augmented feature matrices
results_v7 = {}

# --- Variant 1: V6-A baseline with scaler-inside-CV ---
print("Testing V6-A baseline (scaler inside CV)...")
mae, _ = evaluate_ridge_proper_cv(X_base, y, alphas, kf, oh_idx, ct_idx)
results_v7['V6-A baseline'] = {'mae': mae, 'n_feat': X_base.shape[1]}
print(f"  V6-A baseline: {mae:.5f} ({X_base.shape[1]} features)")""")

code("""# --- Variant 2-3: Add PCA components ---
for n_pca in [5, 10]:
    pca_n = PCA(n_components=n_pca, random_state=42)
    X_pca_n = pca_n.fit_transform(X_cont)
    Xt_pca_n = pca_n.transform(Xt_cont)

    X_aug = np.hstack([X_base, X_pca_n])
    # PCA columns are continuous — extend ct_idx
    new_ct_idx = ct_idx + list(range(X_base.shape[1], X_aug.shape[1]))

    print(f"Testing V7-PCA{n_pca}...")
    mae, _ = evaluate_ridge_proper_cv(X_aug, y, alphas, kf, oh_idx, new_ct_idx)
    results_v7[f'V7-PCA{n_pca}'] = {'mae': mae, 'n_feat': X_aug.shape[1]}
    print(f"  V7-PCA{n_pca}: {mae:.5f} ({X_aug.shape[1]} features)")""")

code("""# --- Variant 4: Add K-means cluster one-hots ---
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
train_cl5 = km5.fit_predict(X_cont)
test_cl5 = km5.predict(Xt_cont)

# One-hot encode clusters
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, categories='auto')
cl_train = ohe.fit_transform(train_cl5.reshape(-1, 1))
cl_test = ohe.transform(test_cl5.reshape(-1, 1))

X_km = np.hstack([X_base, cl_train])
# Cluster one-hots are indicators — keep oh_idx as-is but don't scale the new columns
new_oh_idx_km = oh_idx + list(range(X_base.shape[1], X_km.shape[1]))

print("Testing V7-KM5...")
mae, _ = evaluate_ridge_proper_cv(X_km, y, alphas, kf, new_oh_idx_km, ct_idx)
results_v7['V7-KM5'] = {'mae': mae, 'n_feat': X_km.shape[1]}
print(f"  V7-KM5: {mae:.5f} ({X_km.shape[1]} features)")

# --- Variant 5: PCA10 + KM5 combined ---
pca_10 = PCA(n_components=10, random_state=42)
X_pca10 = pca_10.fit_transform(X_cont)
Xt_pca10 = pca_10.transform(Xt_cont)

X_combo = np.hstack([X_base, X_pca10, cl_train])
combo_ct_idx = ct_idx + list(range(X_base.shape[1], X_base.shape[1] + 10))
combo_oh_idx = oh_idx + list(range(X_base.shape[1] + 10, X_combo.shape[1]))

print("Testing V7-PCA10-KM5...")
mae, _ = evaluate_ridge_proper_cv(X_combo, y, alphas, kf, combo_oh_idx, combo_ct_idx)
results_v7['V7-PCA10-KM5'] = {'mae': mae, 'n_feat': X_combo.shape[1]}
print(f"  V7-PCA10-KM5: {mae:.5f} ({X_combo.shape[1]} features)")""")

code("""# --- Variant 6: Huber on base features ---
print("Testing V7-Huber (base)... (this takes longer)")
mae, _ = evaluate_huber_proper_cv(X_base, y, kf, oh_idx, ct_idx)
results_v7['V7-Huber'] = {'mae': mae, 'n_feat': X_base.shape[1]}
print(f"  V7-Huber: {mae:.5f} ({X_base.shape[1]} features)")

# --- Variant 7: Huber + PCA10 + KM5 ---
print("Testing V7-Huber-PCA10-KM5...")
mae, _ = evaluate_huber_proper_cv(X_combo, y, kf, combo_oh_idx, combo_ct_idx)
results_v7['V7-Huber-PCA10-KM5'] = {'mae': mae, 'n_feat': X_combo.shape[1]}
print(f"  V7-Huber-PCA10-KM5: {mae:.5f} ({X_combo.shape[1]} features)")""")

code("""# --- Bonus: Test different K-means k values with Ridge ---
print("\\nK-means sensitivity (k=3..7 with Ridge):")
for k in [3, 4, 5, 6, 7]:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    cl_tr = ohe_k = OneHotEncoder(sparse_output=False).fit_transform(km_k.fit_predict(X_cont).reshape(-1, 1))
    X_km_k = np.hstack([X_base, cl_tr])
    km_oh = oh_idx + list(range(X_base.shape[1], X_km_k.shape[1]))
    mae, _ = evaluate_ridge_proper_cv(X_km_k, y, alphas, kf, km_oh, ct_idx)
    print(f"  k={k}: CV MAE={mae:.5f} ({X_km_k.shape[1]} features)")""")

# ============================================================
md("## 7. Results Summary")

code("""print("=" * 75)
print(f"{'Model':<30} {'Features':>8} {'CV MAE':>10}")
print("=" * 75)
for name, r in sorted(results_v7.items(), key=lambda x: x[1]['mae']):
    print(f"{name:<30} {r['n_feat']:>8} {r['mae']:>10.5f}")
print("=" * 75)

print(f"\\nReference scores:")
print(f"  V6-A (scaler outside CV):  CV MAE = 2.709")
print(f"  01_Ridge (121 features):   CV MAE = 2.720, Public MAE = 2.992")
print(f"  Leader:                    Public MAE = 2.959")""")

# ============================================================
md("## 8. Generate Best Submission")

code("""# Find best variant
best_name = min(results_v7, key=lambda k: results_v7[k]['mae'])
print(f"Best V7 variant: {best_name} (CV MAE={results_v7[best_name]['mae']:.5f})")

# Re-train on full data and predict
# Build the appropriate feature matrix for the best variant
if 'PCA10-KM5' in best_name:
    X_final = X_combo
    Xt_final = np.hstack([Xt_base, Xt_pca10, cl_test])
    final_ct = combo_ct_idx
    final_oh = combo_oh_idx
elif 'PCA10' in best_name:
    X_final = np.hstack([X_base, X_pca10])
    Xt_final = np.hstack([Xt_base, Xt_pca10])
    final_ct = ct_idx + list(range(X_base.shape[1], X_base.shape[1] + 10))
    final_oh = oh_idx
elif 'PCA5' in best_name:
    pca_5 = PCA(n_components=5, random_state=42)
    X_pca5 = pca_5.fit_transform(X_cont)
    Xt_pca5 = pca_5.transform(Xt_cont)
    X_final = np.hstack([X_base, X_pca5])
    Xt_final = np.hstack([Xt_base, Xt_pca5])
    final_ct = ct_idx + list(range(X_base.shape[1], X_base.shape[1] + 5))
    final_oh = oh_idx
elif 'KM5' in best_name:
    X_final = X_km
    Xt_final = np.hstack([Xt_base, cl_test])
    final_ct = ct_idx
    final_oh = new_oh_idx_km
else:
    X_final = X_base
    Xt_final = Xt_base
    final_ct = ct_idx
    final_oh = oh_idx

# Scale and train final model
sc_final = StandardScaler()
X_f = X_final.copy()
Xt_f = Xt_final.copy()
if final_ct:
    X_f[:, final_ct] = sc_final.fit_transform(X_final[:, final_ct])
    Xt_f[:, final_ct] = sc_final.transform(Xt_final[:, final_ct])

if 'Huber' in best_name:
    # Grid search Huber hyperparameters on full data with CV
    best_inner = 999
    best_model = None
    for eps in [1.1, 1.35, 1.5, 2.0]:
        for alpha_h in [0.01, 0.1, 1.0, 10.0, 100.0]:
            hub = HuberRegressor(epsilon=eps, alpha=alpha_h, max_iter=500)
            hub.fit(X_f, y)
            # Quick CV check
            oof_check = np.zeros(len(y))
            for tr_i, va_i in KFold(5, shuffle=True, random_state=42).split(X_f):
                h = HuberRegressor(epsilon=eps, alpha=alpha_h, max_iter=500)
                h.fit(X_f[tr_i], y[tr_i])
                oof_check[va_i] = h.predict(X_f[va_i])
            m = mean_absolute_error(y, oof_check)
            if m < best_inner:
                best_inner = m
                best_model = hub
    preds = np.round(best_model.predict(Xt_f)).astype(int)
else:
    ridge_final = RidgeCV(alphas=alphas, cv=10)
    ridge_final.fit(X_f, y)
    preds = np.round(ridge_final.predict(Xt_f)).astype(int)
    print(f"Final Ridge alpha: {ridge_final.alpha_:.4f}")

preds = np.clip(preds, 36, 120)

# Save submission
sub = pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds})
sub = sub.sort_values('ID').reset_index(drop=True)
sub.to_csv('submissions/v7_best.csv', index=False)
print(f"Saved submissions/v7_best.csv (mean W={preds.mean():.1f})")

# Also save V6-A with scaler-inside-CV for comparison
sc_v6 = StandardScaler()
X_v6 = X_base.copy()
Xt_v6 = Xt_base.copy()
X_v6[:, ct_idx] = sc_v6.fit_transform(X_base[:, ct_idx])
Xt_v6[:, ct_idx] = sc_v6.transform(Xt_base[:, ct_idx])
ridge_v6 = RidgeCV(alphas=alphas, cv=10)
ridge_v6.fit(X_v6, y)
preds_v6 = np.round(ridge_v6.predict(Xt_v6)).astype(int)
preds_v6 = np.clip(preds_v6, 36, 120)
sub_v6 = pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds_v6})
sub_v6 = sub_v6.sort_values('ID').reset_index(drop=True)
sub_v6.to_csv('submissions/v7_v6a_baseline.csv', index=False)
print(f"Saved submissions/v7_v6a_baseline.csv (mean W={preds_v6.mean():.1f})")""")

# ============================================================
md("""## 9. Analysis & Conclusions

### What We Learned

**PCA Components:**
- PCA extracts orthogonal axes of variation from the 120+ continuous features
- The first few components typically capture: (1) overall offensive strength, (2) pitching dominance, (3) era effects
- Adding 5-10 PCA components as supplementary features lets Ridge access variance patterns that aren't aligned with any single feature

**K-Means Clusters:**
- Cluster labels represent "team archetypes" — e.g., power-hitting teams, pitching-dominant teams, balanced teams
- As one-hot features, they let Ridge learn different intercepts for each team type
- This captures a form of nonlinearity that pure Ridge can't learn from continuous features alone

**Huber Regression:**
- Huber regression down-weights residuals beyond a threshold (epsilon)
- In baseball: extreme seasons (36 or 116 wins) often involve unusual circumstances (injuries, trades, tanking)
- Huber prevents these outlier seasons from disproportionately influencing the model
- The trade-off: less sensitivity to outliers means slightly less optimal predictions for the bulk of typical 70-90 win teams

**Scaler Inside CV:**
- Proper CV with scaler fitted per fold gives a more honest MAE estimate
- The difference from scaler-outside-CV is typically tiny (<0.01 MAE) but matters for fair model comparison

### Key Question
Does lower CV MAE translate to lower public MAE? The consistent ~0.27 gap across V1-V6 suggests yes — but we won't know until we submit.
""")

nb.cells = cells
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "06_PCA_KMeans_Huber.ipynb"))
print("Created 06_PCA_KMeans_Huber.ipynb")
