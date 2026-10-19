"""V13: Stacked meta-learner + franchise prior + smart rounding
Goal: close the gap from 2.975 to 2.909 (leader)
"""
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV, LinearRegression
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
actual_W = train['W'].values

train['_is_train'] = True
test['_is_train']  = False
df = pd.concat([train, test], ignore_index=True)

# === Feature engineering (same as v9/v11) ===
COUNTING_STATS = ['R','AB','H','2B','3B','HR','BB','SO','SB',
                  'RA','ER','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP']
RPG_SCALE = ['R','RA','ER','H','2B','HR','BB','HA','HRA','BBA']
RPG_LEAVE = ['AB','3B','SO','SB','CG','SHO','SV','IPouts','SOA','E','DP']

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

for s in RPG_SCALE:
    df[f'{s}_rpg'] = df[f'{s}_pg'] / df['mlb_rpg']

era_cols = [c for c in df.columns if c.startswith('era_')]
df['era_label'] = df[era_cols].idxmax(axis=1)

PG_COLS = [f'{s}_pg' for s in COUNTING_STATS]
EZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('era_label')[col]
    df[f'{col}_ez'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-6)
    EZ_COLS.append(f'{col}_ez')

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

RATE = ['ERA', 'FP', 'mlb_rpg']
PYTH = ['pyth_pct']
SABER = ['wOBA', 'FIP', 'DER', 'run_diff_pg']
RPG_NORM = [f'{s}_rpg' for s in RPG_SCALE]
RPG_LEAVE_PG = [f'{s}_pg' for s in RPG_LEAVE]

# The 6 variants that gave public 2.975
variant_cols = {
    'A':  PG_COLS + RATE + PYTH,
    'B':  RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + PYTH,
    'C':  EZ_COLS + PYTH,
    'D':  list(dict.fromkeys(PG_COLS + RATE + PYTH + RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + EZ_COLS)),
    'Ap': PG_COLS + RATE + PYTH + SABER,
    'Dp': list(dict.fromkeys(PG_COLS + RATE + PYTH + SABER + RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + EZ_COLS)),
}

def bm(d, cols):
    X = d[cols].values.astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

ALPHAS = np.logspace(-3, 4, 300)
KF = KFold(n_splits=10, shuffle=True, random_state=42)

# ============================================================
# STEP 1: Generate OOF predictions for all 6 variants
# ============================================================
print('=== STEP 1: Base model OOF predictions ===')
oof_all = {}
test_wpct_all = {}

for name, cols in variant_cols.items():
    Xtr, Xte = bm(tr, cols), bm(te, cols)
    oof = np.zeros(len(y_wpct))
    for tr_idx, va_idx in KF.split(Xtr):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=5)
        r.fit(sc.fit_transform(Xtr[tr_idx]), y_wpct[tr_idx])
        oof[va_idx] = r.predict(sc.transform(Xtr[va_idx]))
    oof_all[name] = oof

    sc = StandardScaler()
    rf = RidgeCV(alphas=ALPHAS, cv=10)
    rf.fit(sc.fit_transform(Xtr), y_wpct)
    test_wpct_all[name] = rf.predict(sc.transform(Xte))

    mae = mean_absolute_error(actual_W, oof * G_train)
    print(f'  {name:5s}: CV MAE = {mae:.5f}')

# Equal-weight baseline (our current 2.975 public)
oof_equal = np.mean([oof_all[k] for k in variant_cols], axis=0)
test_equal = np.mean([test_wpct_all[k] for k in variant_cols], axis=0)
mae_equal = mean_absolute_error(actual_W, oof_equal * G_train)
print(f'\n  Equal-weight 6-blend: CV MAE = {mae_equal:.5f} (public: 2.975)')

# ============================================================
# STEP 2: Stacked meta-learner
# ============================================================
print('\n=== STEP 2: Stacked meta-learner ===')

# Build OOF matrix: each column is a variant's OOF prediction
oof_matrix = np.column_stack([oof_all[k] for k in variant_cols])
test_matrix = np.column_stack([test_wpct_all[k] for k in variant_cols])
variant_names = list(variant_cols.keys())

# 2a: Ridge stacker (regularized optimal weights)
oof_stacked_ridge = np.zeros(len(y_wpct))
test_stacked_preds = []

for tr_idx, va_idx in KF.split(oof_matrix):
    sc = StandardScaler()
    meta = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=5)
    meta.fit(sc.fit_transform(oof_matrix[tr_idx]), y_wpct[tr_idx])
    oof_stacked_ridge[va_idx] = meta.predict(sc.transform(oof_matrix[va_idx]))
    test_stacked_preds.append(meta.predict(sc.transform(test_matrix)))

test_stacked_ridge = np.mean(test_stacked_preds, axis=0)
mae_stacked_ridge = mean_absolute_error(actual_W, oof_stacked_ridge * G_train)
print(f'  Ridge stacker:  CV MAE = {mae_stacked_ridge:.5f}')

# 2b: OLS stacker (unregularized — learn exact weights)
oof_stacked_ols = np.zeros(len(y_wpct))
test_ols_preds = []

for tr_idx, va_idx in KF.split(oof_matrix):
    meta = LinearRegression()
    meta.fit(oof_matrix[tr_idx], y_wpct[tr_idx])
    oof_stacked_ols[va_idx] = meta.predict(oof_matrix[va_idx])
    test_ols_preds.append(meta.predict(test_matrix))

test_stacked_ols = np.mean(test_ols_preds, axis=0)
mae_stacked_ols = mean_absolute_error(actual_W, oof_stacked_ols * G_train)
print(f'  OLS stacker:    CV MAE = {mae_stacked_ols:.5f}')

# Show learned weights
meta_final = LinearRegression()
meta_final.fit(oof_matrix, y_wpct)
print(f'  OLS weights: {dict(zip(variant_names, [f"{w:.3f}" for w in meta_final.coef_]))}')
print(f'  OLS intercept: {meta_final.intercept_:.4f}')

# 2c: Stacker with pyth_pct as additional feature
oof_matrix_plus = np.column_stack([oof_matrix, tr['pyth_pct'].values])
test_matrix_plus = np.column_stack([test_matrix, te['pyth_pct'].values])

oof_stacked_pyth = np.zeros(len(y_wpct))
test_pyth_preds = []
for tr_idx, va_idx in KF.split(oof_matrix_plus):
    sc = StandardScaler()
    meta = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=5)
    meta.fit(sc.fit_transform(oof_matrix_plus[tr_idx]), y_wpct[tr_idx])
    oof_stacked_pyth[va_idx] = meta.predict(sc.transform(oof_matrix_plus[va_idx]))
    test_pyth_preds.append(meta.predict(sc.transform(test_matrix_plus)))

test_stacked_pyth = np.mean(test_pyth_preds, axis=0)
mae_stacked_pyth = mean_absolute_error(actual_W, oof_stacked_pyth * G_train)
print(f'  Ridge+pyth:     CV MAE = {mae_stacked_pyth:.5f}')

# ============================================================
# STEP 3: Franchise prior
# ============================================================
print('\n=== STEP 3: Franchise prior ===')

# Compute expanding-window franchise historical win rate (no leakage)
# For each franchise, compute cumulative mean W/G from all prior train years
train_only = df[df['_is_train']].copy()
train_only['wpct'] = train_only['W'] / train_only['G']

# Group by franchise+year, get mean wpct per year
fran_year = train_only.groupby(['franchID', 'yearID'])['wpct'].mean().reset_index()
fran_year = fran_year.sort_values(['franchID', 'yearID'])

# Expanding mean shifted by 1 (exclude current year)
fran_year['hist_wpct'] = fran_year.groupby('franchID')['wpct'].transform(
    lambda x: x.expanding().mean().shift(1))

# Merge back: for each (franchID, yearID), get the historical wpct
# For test rows from years not in fran_year, we'll use the latest available
hist_lookup = fran_year.set_index(['franchID', 'yearID'])['hist_wpct'].to_dict()

# Also build "latest available" per franchise for test rows from unseen years
fran_latest = fran_year.groupby('franchID').apply(
    lambda g: g.sort_values('yearID')['wpct'].expanding().mean().iloc[-1]
).to_dict()

def get_hist_wpct(row):
    key = (row['franchID'], row['yearID'])
    if key in hist_lookup:
        v = hist_lookup[key]
        return v if not np.isnan(v) else 0.500
    # Year not in train — use franchise's overall historical mean
    return fran_latest.get(row['franchID'], 0.500)

df['franchise_hist_wpct'] = df.apply(get_hist_wpct, axis=1)

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

print(f'  Franchise hist wpct: train NaN = {tr["franchise_hist_wpct"].isna().sum()}, '
      f'test NaN = {te["franchise_hist_wpct"].isna().sum()}')
print(f'  Train range: [{tr["franchise_hist_wpct"].min():.3f}, {tr["franchise_hist_wpct"].max():.3f}]')

# Fill NaN with 0.500 (global mean)
tr['franchise_hist_wpct'] = tr['franchise_hist_wpct'].fillna(0.500)
te['franchise_hist_wpct'] = te['franchise_hist_wpct'].fillna(0.500)

# 3a: Add franchise prior to stacker
oof_matrix_fran = np.column_stack([oof_matrix, tr['franchise_hist_wpct'].values])
test_matrix_fran = np.column_stack([test_matrix, te['franchise_hist_wpct'].values])

oof_stacked_fran = np.zeros(len(y_wpct))
test_fran_preds = []
for tr_idx, va_idx in KF.split(oof_matrix_fran):
    sc = StandardScaler()
    meta = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=5)
    meta.fit(sc.fit_transform(oof_matrix_fran[tr_idx]), y_wpct[tr_idx])
    oof_stacked_fran[va_idx] = meta.predict(sc.transform(oof_matrix_fran[va_idx]))
    test_fran_preds.append(meta.predict(sc.transform(test_matrix_fran)))

test_stacked_fran = np.mean(test_fran_preds, axis=0)
mae_stacked_fran = mean_absolute_error(actual_W, oof_stacked_fran * G_train)
print(f'  Stacker + franchise: CV MAE = {mae_stacked_fran:.5f}')

# 3b: Add franchise prior + pyth to stacker
oof_matrix_full = np.column_stack([oof_matrix, tr['franchise_hist_wpct'].values, tr['pyth_pct'].values])
test_matrix_full = np.column_stack([test_matrix, te['franchise_hist_wpct'].values, te['pyth_pct'].values])

oof_stacked_full = np.zeros(len(y_wpct))
test_full_preds = []
for tr_idx, va_idx in KF.split(oof_matrix_full):
    sc = StandardScaler()
    meta = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=5)
    meta.fit(sc.fit_transform(oof_matrix_full[tr_idx]), y_wpct[tr_idx])
    oof_stacked_full[va_idx] = meta.predict(sc.transform(oof_matrix_full[va_idx]))
    test_full_preds.append(meta.predict(sc.transform(test_matrix_full)))

test_stacked_full = np.mean(test_full_preds, axis=0)
mae_stacked_full = mean_absolute_error(actual_W, oof_stacked_full * G_train)
print(f'  Stacker + franchise + pyth: CV MAE = {mae_stacked_full:.5f}')

# 3c: Add franchise prior as feature to each base variant, then re-blend
print('\n  --- Franchise as base feature ---')
oof_fran_base = {}
test_fran_base = {}
for name, cols in variant_cols.items():
    cols_f = cols + ['franchise_hist_wpct']
    Xtr, Xte = bm(tr, cols_f), bm(te, cols_f)
    oof = np.zeros(len(y_wpct))
    for tr_idx, va_idx in KF.split(Xtr):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=5)
        r.fit(sc.fit_transform(Xtr[tr_idx]), y_wpct[tr_idx])
        oof[va_idx] = r.predict(sc.transform(Xtr[va_idx]))
    oof_fran_base[name] = oof
    sc = StandardScaler()
    rf = RidgeCV(alphas=ALPHAS, cv=10)
    rf.fit(sc.fit_transform(Xtr), y_wpct)
    test_fran_base[name] = rf.predict(sc.transform(Xte))
    mae = mean_absolute_error(actual_W, oof * G_train)
    print(f'    {name}+fran: CV MAE = {mae:.5f}')

oof_fran_blend = np.mean([oof_fran_base[k] for k in variant_cols], axis=0)
test_fran_blend = np.mean([test_fran_base[k] for k in variant_cols], axis=0)
mae_fran_blend = mean_absolute_error(actual_W, oof_fran_blend * G_train)
print(f'  Fran-base 6-blend: CV MAE = {mae_fran_blend:.5f}')

# ============================================================
# STEP 4: Smart rounding
# ============================================================
print('\n=== STEP 4: Smart rounding ===')

# Current method: standard round
def standard_submit(wpct, G, label):
    W = np.clip(np.round(wpct * G).astype(int), 36, 120)
    mae = mean_absolute_error(actual_W, np.clip(np.round(oof_equal * G_train).astype(int), 36, 120))
    return W

# Analyze rounding bias from equal-weight blend OOF
pred_W_raw = oof_equal * G_train
pred_W_rounded = np.round(pred_W_raw)
errors = actual_W - pred_W_rounded

# For teams with predicted W > 85: model underpredicts, so ceil might help
# For teams with predicted W < 70: model overpredicts, so floor might help
print(f'  Rounding residuals (rounded vs actual):')
for lo, hi in [(36, 60), (61, 70), (71, 80), (81, 90), (91, 100), (101, 120)]:
    mask = (pred_W_rounded >= lo) & (pred_W_rounded <= hi)
    if mask.sum() > 0:
        bias = errors[mask].mean()
        print(f'    Pred {lo:>3d}-{hi:>3d}: n={mask.sum():>4d}, mean error={bias:+.3f}')

# Try biased rounding
for shift_name, shift_func in [
    ('no shift', lambda w: w),
    ('shift +0.3 if pred>85', lambda w: np.where(w > 85, w + 0.3, w)),
    ('shift -0.3 if pred<65', lambda w: np.where(w < 65, w - 0.3, w)),
    ('shift ±0.3 at extremes', lambda w: np.where(w > 85, w + 0.3, np.where(w < 65, w - 0.3, w))),
    ('shift ±0.5 at extremes', lambda w: np.where(w > 85, w + 0.5, np.where(w < 65, w - 0.5, w))),
    ('linear shrink 2%', lambda w: 81 + 0.98 * (w - 81)),
    ('linear shrink 3%', lambda w: 81 + 0.97 * (w - 81)),
    ('linear shrink 5%', lambda w: 81 + 0.95 * (w - 81)),
]:
    pred_adj = shift_func(oof_equal * G_train)
    W_adj = np.clip(np.round(pred_adj).astype(int), 36, 120)
    mae_adj = mean_absolute_error(actual_W, W_adj)
    print(f'  {shift_name:30s}: CV MAE = {mae_adj:.5f}')

# ============================================================
# STEP 5: Combine best approaches
# ============================================================
print('\n=== STEP 5: Combined submissions ===')

submissions = {}

# 5a: Equal-weight 6-blend (baseline, already scored 2.975)
sub_equal = np.clip(np.round(test_equal * G_test).astype(int), 36, 120)
submissions['v13_equal_baseline'] = sub_equal

# 5b: Ridge stacker on 6 variants
sub_stacked = np.clip(np.round(test_stacked_ridge * G_test).astype(int), 36, 120)
submissions['v13_stacked_ridge'] = sub_stacked

# 5c: OLS stacker
sub_ols = np.clip(np.round(test_stacked_ols * G_test).astype(int), 36, 120)
submissions['v13_stacked_ols'] = sub_ols

# 5d: Stacker + pyth
sub_pyth = np.clip(np.round(test_stacked_pyth * G_test).astype(int), 36, 120)
submissions['v13_stacked_pyth'] = sub_pyth

# 5e: Stacker + franchise
sub_fran = np.clip(np.round(test_stacked_fran * G_test).astype(int), 36, 120)
submissions['v13_stacked_fran'] = sub_fran

# 5f: Stacker + franchise + pyth
sub_full = np.clip(np.round(test_stacked_full * G_test).astype(int), 36, 120)
submissions['v13_stacked_full'] = sub_full

# 5g: Franchise-base 6-blend
sub_fran_blend = np.clip(np.round(test_fran_blend * G_test).astype(int), 36, 120)
submissions['v13_fran_blend'] = sub_fran_blend

# 5h: Best stacker + smart rounding (apply best shift from Step 4)
# Use the best stacker's test predictions with shrinkage
for shrink, label in [(0.02, '2pct'), (0.03, '3pct')]:
    for src_name, src_test in [('equal', test_equal), ('ridge', test_stacked_ridge)]:
        pred_raw = src_test * G_test
        pred_adj = 81 + (1 - shrink) * (pred_raw - 81)
        W_adj = np.clip(np.round(pred_adj).astype(int), 36, 120)
        submissions[f'v13_{src_name}_shrink{label}'] = W_adj

# Save all submissions
print(f'\n{"Submission":<35s} {"W mean":>7s} {"W min":>6s} {"W max":>6s} {"Diff vs baseline":>16s}')
print('-' * 75)
for name, W in submissions.items():
    pd.DataFrame({'ID': test_ids, 'W': W}).sort_values('ID').to_csv(
        f'submissions/{name}.csv', index=False)
    diff = np.mean(np.abs(W - sub_equal))
    print(f'{name:<35s} {W.mean():>7.1f} {W.min():>6d} {W.max():>6d} {diff:>16.2f}')

# ============================================================
# SUMMARY TABLE
# ============================================================
print('\n=== CV MAE SUMMARY ===')
print(f'{"Method":<35s} {"CV MAE":>8s} {"vs baseline":>12s}')
print('-' * 58)
results = [
    ('Equal-weight 6-blend (baseline)', mae_equal, 0),
    ('Ridge stacker', mae_stacked_ridge, mae_stacked_ridge - mae_equal),
    ('OLS stacker', mae_stacked_ols, mae_stacked_ols - mae_equal),
    ('Stacker + pyth', mae_stacked_pyth, mae_stacked_pyth - mae_equal),
    ('Stacker + franchise', mae_stacked_fran, mae_stacked_fran - mae_equal),
    ('Stacker + franchise + pyth', mae_stacked_full, mae_stacked_full - mae_equal),
    ('Franchise-base 6-blend', mae_fran_blend, mae_fran_blend - mae_equal),
]
for name, mae, delta in sorted(results, key=lambda x: x[1]):
    sign = '+' if delta > 0 else ''
    print(f'{name:<35s} {mae:>8.5f} {sign}{delta:>11.5f}')

print('\nWARNING: Lower CV MAE has consistently produced WORSE public scores.')
print('The baseline (CV 2.745, public 2.975) may still be optimal.')
print('Submit stacker variants cautiously — they may overfit more despite lower CV.')
