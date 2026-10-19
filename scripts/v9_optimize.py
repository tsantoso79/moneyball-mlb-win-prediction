"""V9 Optimization: Blend + Pythagorean + Sabermetrics"""
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings; warnings.filterwarnings('ignore')
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)
os.chdir(BASE)

# --- Load data ---
train = pd.read_csv('data_year_team_franchise.csv')
test  = pd.read_csv('predict_year_team_franchise.csv')

y_wpct = (train['W'] / train['G']).values
G_train = train['G'].values
G_test  = test['G'].values
test_ids = test['ID'].values

# --- Combine for feature engineering ---
train['_is_train'] = True
test['_is_train']  = False
df = pd.concat([train, test], ignore_index=True)

# === FEATURE ENGINEERING ===
COUNTING_STATS = ['R','AB','H','2B','3B','HR','BB','SO','SB',
                  'RA','ER','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP']

# Per-game
for s in COUNTING_STATS:
    df[f'{s}_pg'] = df[s] / df['G']

# Pythagorean
R = df['R'].values.astype(float)
RA = df['RA'].values.astype(float)
df['pyth_pct'] = R**2 / (R**2 + RA**2)

# --- 3 non-linear sabermetrics ---
df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
df['IP'] = df['IPouts'] / 3
df['PA'] = df['AB'] + df['BB']

# wOBA (weighted on-base average)
df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / df['PA']

# FIP (fielding independent pitching)
df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10

# DER (defensive efficiency ratio)
df['DER'] = 1 - (df['HA'] - df['HRA']) / (df['HA'] - df['HRA'] + df['IPouts'] - df['SOA'])

# run_diff_pg
df['run_diff_pg'] = (df['R'] - df['RA']) / df['G']

# --- mlb_rpg normalization ---
RPG_SCALE = ['R','RA','ER','H','2B','HR','BB','HA','HRA','BBA']
RPG_LEAVE = ['AB','3B','SO','SB','CG','SHO','SV','IPouts','SOA','E','DP']
for s in RPG_SCALE:
    df[f'{s}_rpg'] = df[f'{s}_pg'] / df['mlb_rpg']

# --- Era z-score ---
era_cols = [c for c in df.columns if c.startswith('era_')]
df['era_label'] = df[era_cols].idxmax(axis=1)

PG_COLS = [f'{s}_pg' for s in COUNTING_STATS]
EZ_INPUT = PG_COLS + ['ERA', 'FP']
EZ_COLS = []
eps = 1e-6
for col in EZ_INPUT:
    g = df.groupby('era_label')[col]
    df[f'{col}_ez'] = (df[col] - g.transform('mean')) / (g.transform('std') + eps)
    EZ_COLS.append(f'{col}_ez')

# --- Split ---
tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

# === BUILD FEATURE SETS ===
def bm(d, cols):
    X = d[cols].values.astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

RATE = ['ERA', 'FP', 'mlb_rpg']
PYTH = ['pyth_pct']
SABER = ['wOBA', 'FIP', 'DER', 'run_diff_pg']

# Variant A+ : per-game + rates + pyth + sabermetrics (29 features)
VA_COLS = PG_COLS + RATE + PYTH + SABER

# Variant D+ : combined + sabermetrics
RPG_NORM = [f'{s}_rpg' for s in RPG_SCALE]
RPG_LEAVE_PG = [f'{s}_pg' for s in RPG_LEAVE]
VB_COLS = RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + PYTH
VD_COLS = list(dict.fromkeys(VA_COLS + VB_COLS + EZ_COLS + PYTH + SABER))

variants = {
    'Ap_pg_saber': VA_COLS,
    'Dp_combined_saber': VD_COLS,
}

# === EVALUATE ===
ALPHAS = np.logspace(-3, 4, 300)
KF = KFold(n_splits=10, shuffle=True, random_state=42)

raw_test_wpct = {}
oof_store = {}

print("=" * 70)
print("PART 1: Ridge + Sabermetrics (wOBA, FIP, DER)")
print("=" * 70)

for name, cols in variants.items():
    Xtr, Xte = bm(tr, cols), bm(te, cols)
    oof = np.zeros(len(y_wpct))
    alphas_c = []

    for tr_idx, va_idx in KF.split(Xtr):
        sc = StandardScaler()
        X_t = sc.fit_transform(Xtr[tr_idx])
        X_v = sc.transform(Xtr[va_idx])
        r = RidgeCV(alphas=ALPHAS, cv=5)
        r.fit(X_t, y_wpct[tr_idx])
        oof[va_idx] = r.predict(X_v)
        alphas_c.append(r.alpha_)

    mae = mean_absolute_error(y_wpct * G_train, oof * G_train)

    # Final model
    sc = StandardScaler()
    Xa = sc.fit_transform(Xtr)
    rf = RidgeCV(alphas=ALPHAS, cv=10)
    rf.fit(Xa, y_wpct)
    test_wpct_pred = rf.predict(sc.transform(Xte))
    raw_test_wpct[name] = test_wpct_pred
    oof_store[name] = oof

    test_W = np.clip(np.round(test_wpct_pred * G_test).astype(int), 36, 120)

    # Save
    pd.DataFrame({'ID': test_ids, 'W': test_W}).sort_values('ID').to_csv(
        f'submissions/v9_{name}.csv', index=False)

    print(f'{name:25s}: CV MAE = {mae:.5f}  (alpha={np.median(alphas_c):.2f}, features={len(cols)})')

# === PART 2: PYTHAGOREAN BLENDING ===
print()
print("=" * 70)
print("PART 2: Pythagorean Blending (Ridge × w + Pyth × (1-w))")
print("=" * 70)

pyth_train = tr['pyth_pct'].values
pyth_test = te['pyth_pct'].values

# Use D+ as the Ridge component (best CV MAE)
best_key = 'Dp_combined_saber'
oof_best = oof_store[best_key]

for w in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    # CV estimate
    blend_oof = w * oof_best + (1 - w) * pyth_train
    mae_b = mean_absolute_error(y_wpct * G_train, blend_oof * G_train)

    # Test predictions
    blend_test = w * raw_test_wpct[best_key] + (1 - w) * pyth_test
    blend_W = np.clip(np.round(blend_test * G_test).astype(int), 36, 120)

    pd.DataFrame({'ID': test_ids, 'W': blend_W}).sort_values('ID').to_csv(
        f'submissions/v9_pyth_blend_{int(w*100)}.csv', index=False)

    print(f'Ridge={int(w*100)}% Pyth={int((1-w)*100)}%: CV MAE = {mae_b:.5f}')

# === PART 3: VARIANT BLENDING ===
print()
print("=" * 70)
print("PART 3: Cross-Variant Blending")
print("=" * 70)

# Blend A+ and D+ (raw W/G predictions)
blend_ApDp = (raw_test_wpct['Ap_pg_saber'] + raw_test_wpct['Dp_combined_saber']) / 2
blend_W_ApDp = np.clip(np.round(blend_ApDp * G_test).astype(int), 36, 120)
pd.DataFrame({'ID': test_ids, 'W': blend_W_ApDp}).sort_values('ID').to_csv(
    'submissions/v9_blend_ApDp.csv', index=False)

# CV estimate for A+ D+ blend
oof_ap = oof_store['Ap_pg_saber']
oof_dp = oof_store['Dp_combined_saber']
blend_oof_ad = (oof_ap + oof_dp) / 2
mae_ad = mean_absolute_error(y_wpct * G_train, blend_oof_ad * G_train)
print(f'Blend A+ & D+:           CV MAE = {mae_ad:.5f}')

# Blend all original variants (A,B,C,D) + A+,D+ using integer preds
pA = pd.read_csv('submissions/v9_A_pergame.csv')['W'].values
pB = pd.read_csv('submissions/v9_B_mlbrpg.csv')['W'].values
pC = pd.read_csv('submissions/v9_C_era_zscore.csv')['W'].values
pD = pd.read_csv('submissions/v9_D_combined.csv')['W'].values
pAp = pd.read_csv('submissions/v9_Ap_pg_saber.csv')['W'].values
pDp = pd.read_csv('submissions/v9_Dp_combined_saber.csv')['W'].values

blend_all = np.round((pA + pB + pC + pD + pAp + pDp) / 6).astype(int)
blend_all = np.clip(blend_all, 36, 120)
# The variant files above were saved sorted by ID, so pair the blend with sorted IDs
# (test_ids is in file order). This is the 2.975 public-leaderboard submission.
pd.DataFrame({'ID': np.sort(test_ids), 'W': blend_all}).to_csv(
    'submissions/v9_blend_all6_fixed.csv', index=False)
print(f'Blend all 6 variants:    (integer avg, no CV estimate)')

# Best combo: D+ with Pythagorean at optimal weight
# Also try A+ with Pythagorean
print()
print("--- A+ Pythagorean blends ---")
oof_ap = oof_store['Ap_pg_saber']
for w in [0.70, 0.80, 0.90]:
    blend_oof = w * oof_ap + (1 - w) * pyth_train
    mae_b = mean_absolute_error(y_wpct * G_train, blend_oof * G_train)
    blend_test = w * raw_test_wpct['Ap_pg_saber'] + (1 - w) * pyth_test
    blend_W = np.clip(np.round(blend_test * G_test).astype(int), 36, 120)
    pd.DataFrame({'ID': test_ids, 'W': blend_W}).sort_values('ID').to_csv(
        f'submissions/v9_Ap_pyth_{int(w*100)}.csv', index=False)
    print(f'A+ Ridge={int(w*100)}% Pyth={int((1-w)*100)}%: CV MAE = {mae_b:.5f}')

# === SUMMARY ===
print()
print("=" * 70)
print("SUMMARY: All submission files")
print("=" * 70)
import glob
files = sorted(glob.glob('submissions/v9_*.csv'))
for f in files:
    s = pd.read_csv(f)
    print(f'  {os.path.basename(f):35s} W mean={s.W.mean():.1f} range=[{s.W.min()}, {s.W.max()}]')
