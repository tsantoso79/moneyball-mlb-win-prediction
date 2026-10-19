"""V12: Separate model for era_1, blend for era_2-8"""
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings; warnings.filterwarnings('ignore')
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)
os.chdir(BASE)

train = pd.read_csv('data_year_team_franchise.csv')
test  = pd.read_csv('predict_year_team_franchise.csv')

y_wpct = (train['W'] / train['G']).values
G_train = train['G'].values
G_test  = test['G'].values
test_ids = test['ID'].values
actual_W = y_wpct * G_train

train['_is_train'] = True
test['_is_train']  = False
df = pd.concat([train, test], ignore_index=True)

# Feature engineering (same as v11)
COUNTING_STATS = ['R','AB','H','2B','3B','HR','BB','SO','SB',
                  'RA','ER','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP']

for s in COUNTING_STATS:
    df[f'{s}_pg'] = df[s] / df['G']

R = df['R'].values.astype(float)
RA = df['RA'].values.astype(float)
df['pyth_pct'] = R**2 / (R**2 + RA**2)

df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
df['PA'] = df['AB'] + df['BB']
df['IP'] = df['IPouts'] / 3
df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / df['PA']
df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
df['DER'] = 1 - (df['HA'] - df['HRA']) / (df['HA'] - df['HRA'] + df['IPouts'] - df['SOA'])
df['run_diff_pg'] = (df['R'] - df['RA']) / df['G']

era_cols = [c for c in df.columns if c.startswith('era_')]
df['era_label'] = df[era_cols].idxmax(axis=1)

PG_COLS = [f'{s}_pg' for s in COUNTING_STATS]
RATE = ['ERA', 'FP', 'mlb_rpg']
PYTH = ['pyth_pct']
SABER = ['wOBA', 'FIP', 'DER', 'run_diff_pg']

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

# Split into era_1 and era_2+
tr_e1 = tr[tr['era_label'] == 'era_1'].copy()
tr_rest = tr[tr['era_label'] != 'era_1'].copy()
te_e1 = te[te['era_label'] == 'era_1'].copy()
te_rest = te[te['era_label'] != 'era_1'].copy()

idx_e1_train = tr['era_label'] == 'era_1'
idx_e1_test = te['era_label'] == 'era_1'

print(f'Era 1: {len(tr_e1)} train, {len(te_e1)} test')
print(f'Era 2-8: {len(tr_rest)} train, {len(te_rest)} test')
print()

def bm(d, cols):
    X = d[cols].values.astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

ALPHAS = np.logspace(-3, 4, 300)

# ============================================================
# APPROACH 1: Pythagorean-weighted for era_1
# pyth_pct has r=0.96 with W/G in era_1 — strongest single predictor
# ============================================================
print('=== APPROACH 1: Pythagorean weighting for era_1 ===')

y_e1 = (tr_e1['W'] / tr_e1['G']).values
G_e1 = tr_e1['G'].values
pyth_e1 = tr_e1['pyth_pct'].values

# Pure Pythagorean MAE for era_1
pyth_pred_W = pyth_e1 * G_e1
mae_pyth_pure = mean_absolute_error(y_e1 * G_e1, pyth_pred_W)
print(f'Pure Pythagorean (era_1): MAE = {mae_pyth_pure:.3f}')

# ============================================================
# APPROACH 2: Minimal-feature Ridge for era_1
# Only 5-8 most relevant features to avoid overfitting on 174 rows
# ============================================================
print()
print('=== APPROACH 2: Minimal-feature Ridge for era_1 ===')

# Dead-ball era relevant features: scoring was low, few HR
# Key: run differential, pythagorean, basic rates
MINIMAL_COLS = ['pyth_pct', 'run_diff_pg', 'R_pg', 'RA_pg', 'H_pg', 'BB_pg',
                'HA_pg', 'BBA_pg', 'E_pg', 'FP', 'ERA']

for n_feat in [3, 5, 8, 11]:
    cols = MINIMAL_COLS[:n_feat]
    X_e1 = bm(tr_e1, cols)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold for small data
    oof_e1 = np.zeros(len(y_e1))
    for tr_idx, va_idx in kf.split(X_e1):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=3)
        r.fit(sc.fit_transform(X_e1[tr_idx]), y_e1[tr_idx])
        oof_e1[va_idx] = r.predict(sc.transform(X_e1[va_idx]))
    mae_min = mean_absolute_error(y_e1 * G_e1, oof_e1 * G_e1)
    print(f'  {n_feat} features: CV MAE = {mae_min:.3f} (features: {cols})')

# ============================================================
# APPROACH 3: Ridge on ALL data but with era_1 indicator interaction
# Train on all 1812 rows but add era_1 interaction terms
# ============================================================
print()
print('=== APPROACH 3: Full data + era_1 interactions ===')

ALL_COLS = PG_COLS + RATE + PYTH + SABER
df['is_era1'] = (df['era_label'] == 'era_1').astype(float)

# Add interaction: is_era1 * each feature
INTERACT_COLS = []
for col in ['pyth_pct', 'run_diff_pg', 'R_pg', 'RA_pg', 'ERA', 'FP']:
    icol = f'{col}_x_era1'
    df[icol] = df[col] * df['is_era1']
    INTERACT_COLS.append(icol)

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

V3_COLS = ALL_COLS + INTERACT_COLS
X_all = bm(tr, V3_COLS)
y_all = (tr['W'] / tr['G']).values

kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_v3 = np.zeros(len(y_all))
for tr_idx, va_idx in kf.split(X_all):
    sc = StandardScaler()
    r = RidgeCV(alphas=ALPHAS, cv=5)
    r.fit(sc.fit_transform(X_all[tr_idx]), y_all[tr_idx])
    oof_v3[va_idx] = r.predict(sc.transform(X_all[va_idx]))

mae_v3_all = mean_absolute_error(y_all * G_train, oof_v3 * G_train)
mae_v3_e1 = mean_absolute_error(y_all[idx_e1_train.values] * G_train[idx_e1_train.values],
                                 oof_v3[idx_e1_train.values] * G_train[idx_e1_train.values])
mae_v3_rest = mean_absolute_error(y_all[~idx_e1_train.values] * G_train[~idx_e1_train.values],
                                   oof_v3[~idx_e1_train.values] * G_train[~idx_e1_train.values])
print(f'Full+interactions: Overall CV MAE = {mae_v3_all:.5f}')
print(f'  Era_1 CV MAE = {mae_v3_e1:.3f}')
print(f'  Era_2-8 CV MAE = {mae_v3_rest:.3f}')

# ============================================================
# APPROACH 4: Blend + Pythagorean override for era_1 only
# Use our 8-blend for era_2-8, blend with Pythagorean for era_1
# ============================================================
print()
print('=== APPROACH 4: 8-blend + Pythagorean blend for era_1 ===')

# First, get our 8-blend OOF predictions (recompute)
RPG_SCALE = ['R','RA','ER','H','2B','HR','BB','HA','HRA','BBA']
RPG_LEAVE = ['AB','3B','SO','SB','CG','SHO','SV','IPouts','SOA','E','DP']
for s in RPG_SCALE:
    df[f'{s}_rpg'] = df[f'{s}_pg'] / df['mlb_rpg']

EZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('era_label')[col]
    df[f'{col}_ez'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-6)
    EZ_COLS.append(f'{col}_ez')

YZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('yearID')[col]
    year_std = g.transform('std')
    safe_std = year_std.where(year_std > 0.001, other=1.0)
    df[f'{col}_yz'] = (df[col] - g.transform('mean')) / (safe_std + 1e-6)
    YZ_COLS.append(f'{col}_yz')

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

RPG_NORM = [f'{s}_rpg' for s in RPG_SCALE]
RPG_LEAVE_PG = [f'{s}_pg' for s in RPG_LEAVE]

variant_cols = {
    'A':  PG_COLS + RATE + PYTH,
    'B':  RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + PYTH,
    'C':  EZ_COLS + PYTH,
    'D':  list(dict.fromkeys(PG_COLS + RATE + PYTH + RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + EZ_COLS)),
    'Ap': PG_COLS + RATE + PYTH + SABER,
    'Dp': list(dict.fromkeys(PG_COLS + RATE + PYTH + SABER + RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + EZ_COLS)),
    'Fy': YZ_COLS + PYTH,
    'Gy': list(dict.fromkeys(PG_COLS + RATE + PYTH + SABER + YZ_COLS)),
}

oof_all = {}
test_wpct_all = {}
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)

for name, cols in variant_cols.items():
    Xtr, Xte = bm(tr, cols), bm(te, cols)
    oof = np.zeros(len(y_all))
    for tr_idx, va_idx in kf10.split(Xtr):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=5)
        r.fit(sc.fit_transform(Xtr[tr_idx]), y_all[tr_idx])
        oof[va_idx] = r.predict(sc.transform(Xtr[va_idx]))
    oof_all[name] = oof

    sc = StandardScaler()
    rf = RidgeCV(alphas=ALPHAS, cv=10)
    rf.fit(sc.fit_transform(Xtr), y_all)
    test_wpct_all[name] = rf.predict(sc.transform(Xte))

# 8-blend OOF
blend8_oof = np.mean([oof_all[k] for k in variant_cols.keys()], axis=0)
blend8_test = np.mean([test_wpct_all[k] for k in variant_cols.keys()], axis=0)

pyth_train_all = tr['pyth_pct'].values
pyth_test_all = te['pyth_pct'].values
e1_mask_train = idx_e1_train.values
e1_mask_test = idx_e1_test.values

# Test different Pythagorean weights for era_1
print('Pyth weight for era_1 only (blend keeps era_2-8):')
for w_pyth in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
    hybrid_oof = blend8_oof.copy()
    hybrid_oof[e1_mask_train] = (1 - w_pyth) * blend8_oof[e1_mask_train] + w_pyth * pyth_train_all[e1_mask_train]

    mae_total = mean_absolute_error(y_all * G_train, hybrid_oof * G_train)
    mae_e1 = mean_absolute_error(y_all[e1_mask_train] * G_train[e1_mask_train],
                                  hybrid_oof[e1_mask_train] * G_train[e1_mask_train])
    mae_rest = mean_absolute_error(y_all[~e1_mask_train] * G_train[~e1_mask_train],
                                    hybrid_oof[~e1_mask_train] * G_train[~e1_mask_train])
    print(f'  Pyth={int(w_pyth*100):>2d}%: total={mae_total:.5f}, era1={mae_e1:.3f}, era2-8={mae_rest:.3f}')

    # Generate test predictions
    hybrid_test = blend8_test.copy()
    hybrid_test[e1_mask_test] = (1 - w_pyth) * blend8_test[e1_mask_test] + w_pyth * pyth_test_all[e1_mask_test]
    test_W = np.clip(np.round(hybrid_test * G_test).astype(int), 36, 120)
    pd.DataFrame({'ID': test_ids, 'W': test_W}).sort_values('ID').to_csv(
        f'submissions/v12_era1_pyth{int(w_pyth*100):02d}.csv', index=False)

# ============================================================
# APPROACH 5: Separate Ridge for era_1 trained only on era_1 data
# ============================================================
print()
print('=== APPROACH 5: Separate Ridge trained only on era_1 ===')

for cols_name, cols in [
    ('minimal_5', ['pyth_pct', 'run_diff_pg', 'R_pg', 'RA_pg', 'ERA']),
    ('minimal_8', ['pyth_pct', 'run_diff_pg', 'R_pg', 'RA_pg', 'H_pg', 'BB_pg', 'ERA', 'FP']),
    ('full_25', PG_COLS + RATE + PYTH),
    ('full_saber', PG_COLS + RATE + PYTH + SABER),
]:
    X_e1_tr = bm(tr_e1, cols)
    y_e1_tr = (tr_e1['W'] / tr_e1['G']).values
    G_e1_tr = tr_e1['G'].values

    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_e1 = np.zeros(len(y_e1_tr))
    for tr_idx, va_idx in kf5.split(X_e1_tr):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=3)
        r.fit(sc.fit_transform(X_e1_tr[tr_idx]), y_e1_tr[tr_idx])
        oof_e1[va_idx] = r.predict(sc.transform(X_e1_tr[va_idx]))

    mae_e1_sep = mean_absolute_error(y_e1_tr * G_e1_tr, oof_e1 * G_e1_tr)

    # Build hybrid: separate model for era_1, 8-blend for rest
    sc = StandardScaler()
    rf = RidgeCV(alphas=ALPHAS, cv=5)
    rf.fit(sc.fit_transform(X_e1_tr), y_e1_tr)
    X_e1_te = bm(te_e1, cols)
    e1_test_pred = rf.predict(sc.transform(X_e1_te))

    hybrid_test = blend8_test.copy()
    hybrid_test[e1_mask_test] = e1_test_pred
    test_W = np.clip(np.round(hybrid_test * G_test).astype(int), 36, 120)
    pd.DataFrame({'ID': test_ids, 'W': test_W}).sort_values('ID').to_csv(
        f'submissions/v12_era1_sep_{cols_name}.csv', index=False)

    print(f'  {cols_name:15s}: era_1 CV MAE = {mae_e1_sep:.3f} (vs blend era_1: 3.557)')

# ============================================================
print()
print('=== REFERENCE ===')
mae_blend8_e1 = mean_absolute_error(y_all[e1_mask_train] * G_train[e1_mask_train],
                                     blend8_oof[e1_mask_train] * G_train[e1_mask_train])
mae_blend8_all = mean_absolute_error(y_all * G_train, blend8_oof * G_train)
print(f'8-blend overall: CV MAE = {mae_blend8_all:.5f}')
print(f'8-blend era_1:   CV MAE = {mae_blend8_e1:.3f}')
print(f'8-blend era_2-8: CV MAE = {mean_absolute_error(y_all[~e1_mask_train]*G_train[~e1_mask_train], blend8_oof[~e1_mask_train]*G_train[~e1_mask_train]):.3f}')
