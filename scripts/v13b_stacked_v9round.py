"""V13b: Ridge stacker with v9-style integer-averaging rounding"""
import pandas as pd, numpy as np, os
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
train = pd.read_csv('data.csv')
test  = pd.read_csv('predict.csv')
#train = pd.read_csv('data_year_team_franchise.csv')
#test  = pd.read_csv('predict_year_team_franchise.csv')
y_wpct = (train['W'] / train['G']).values
G_train = train['G'].values
G_test  = test['G'].values
test_ids = test['ID'].values
actual_W = train['W'].values

train['_is_train'] = True; test['_is_train'] = False
df = pd.concat([train, test], ignore_index=True)

# Feature engineering (same as v9)
COUNTING_STATS = ['R','AB','H','2B','3B','HR','BB','SO','SB',
                  'RA','ER','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP']
RPG_SCALE = ['R','RA','ER','H','2B','HR','BB','HA','HRA','BBA']
RPG_LEAVE = ['AB','3B','SO','SB','CG','SHO','SV','IPouts','SOA','E','DP']
for s in COUNTING_STATS: df[f'{s}_pg'] = df[s] / df['G']
R = df['R'].values.astype(float); RA = df['RA'].values.astype(float)
df['pyth_pct'] = R**2 / (R**2 + RA**2)
df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
df['PA'] = df['AB'] + df['BB']; df['IP'] = df['IPouts'] / 3
df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / df['PA']
df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
df['DER'] = 1 - (df['HA'] - df['HRA']) / (df['HA'] - df['HRA'] + df['IPouts'] - df['SOA'])
df['run_diff_pg'] = (df['R'] - df['RA']) / df['G']
for s in RPG_SCALE: df[f'{s}_rpg'] = df[f'{s}_pg'] / df['mlb_rpg']
era_cols = [c for c in df.columns if c.startswith('era_')]
df['era_label'] = df[era_cols].idxmax(axis=1)
PG_COLS = [f'{s}_pg' for s in COUNTING_STATS]
EZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('era_label')[col]
    df[f'{col}_ez'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-6)
    EZ_COLS.append(f'{col}_ez')

tr = df[df['_is_train']].copy(); te = df[~df['_is_train']].copy()
RATE = ['ERA', 'FP', 'mlb_rpg']; PYTH = ['pyth_pct']
SABER = ['wOBA', 'FIP', 'DER', 'run_diff_pg']
RPG_NORM = [f'{s}_rpg' for s in RPG_SCALE]
RPG_LEAVE_PG = [f'{s}_pg' for s in RPG_LEAVE]

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

# Step 1: Generate OOF and test predictions for all 6 variants
print('=== Step 1: Base variant predictions ===')
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
    print(f'  {name}: CV MAE = {mae:.5f}')

# Step 2: Train Ridge stacker on OOF, get 10 fold-level test predictions
print('\n=== Step 2: Ridge stacker (10 fold predictions) ===')
oof_matrix = np.column_stack([oof_all[k] for k in variant_cols])
test_matrix = np.column_stack([test_wpct_all[k] for k in variant_cols])

# Each CV fold trains a stacker and produces test predictions
# We round each fold's test prediction to integer W, then average (v9 style)
test_stacked_per_fold = []
oof_stacked = np.zeros(len(y_wpct))

for tr_idx, va_idx in KF.split(oof_matrix):
    sc = StandardScaler()
    m = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=5)
    m.fit(sc.fit_transform(oof_matrix[tr_idx]), y_wpct[tr_idx])
    oof_stacked[va_idx] = m.predict(sc.transform(oof_matrix[va_idx]))

    # Test prediction from this fold's stacker
    fold_wpct = m.predict(sc.transform(test_matrix))
    # v9 rounding: convert to integer W immediately
    fold_W = np.clip(np.round(fold_wpct * G_test).astype(int), 36, 120)
    test_stacked_per_fold.append(fold_W)

# v9 method: average the 10 integer predictions, then round
stacked_int_avg = np.mean(test_stacked_per_fold, axis=0)
stacked_v9_W = np.clip(np.round(stacked_int_avg).astype(int), 36, 120)

mae_stacked = mean_absolute_error(actual_W, oof_stacked * G_train)
print(f'  Stacker CV MAE: {mae_stacked:.5f}')

# Step 3: Also try v9-style integer-averaging of 6 base variants (reference)
print('\n=== Step 3: v9 reference (equal-weight integer avg) ===')
test_W_per_variant = {}
for name in variant_cols:
    raw_W = test_wpct_all[name] * G_test
    test_W_per_variant[name] = np.clip(np.round(raw_W).astype(int), 36, 120)

v9_int_avg = np.mean([test_W_per_variant[k] for k in variant_cols], axis=0)
v9_W = np.clip(np.round(v9_int_avg).astype(int), 36, 120)

# Step 4: Compare
print('\n=== Comparison ===')
v9_ref = pd.read_csv('submissions/v9_blend_all6_fixed.csv').sort_values('ID')['W'].values

for label, W_arr in [
    ('v9 integer-avg (2.975 public)', v9_W),
    ('stacked Ridge + v9 rounding', stacked_v9_W),
]:
    sub = pd.DataFrame({'ID': test_ids, 'W': W_arr}).sort_values('ID')
    diff = np.abs(sub['W'].values - v9_ref)
    match = (diff == 0).sum()
    print(f'  {label}')
    print(f'    W mean={W_arr.mean():.1f}, range=[{W_arr.min()}, {W_arr.max()}]')
    print(f'    vs v9_blend_all6_fixed: {match}/453 exact, mean diff={diff.mean():.3f}, max diff={diff.max()}')

# Save
pd.DataFrame({'ID': test_ids, 'W': stacked_v9_W}).sort_values('ID').to_csv(
    'submissions/v13_stacked_ridge_v9round.csv', index=False)
print('\nSaved: v13_stacked_ridge_v9round.csv')
