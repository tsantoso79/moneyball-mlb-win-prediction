"""V11: Year-level z-score variants + expanded blend"""
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
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

# Era z-score
era_cols = [c for c in df.columns if c.startswith('era_')]
df['era_label'] = df[era_cols].idxmax(axis=1)
PG_COLS = [f'{s}_pg' for s in COUNTING_STATS]
EZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('era_label')[col]
    df[f'{col}_ez'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-6)
    EZ_COLS.append(f'{col}_ez')

# === NEW: Year-level z-score ===
YZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('yearID')[col]
    year_std = g.transform('std')
    safe_std = year_std.where(year_std > 0.001, other=1.0)
    df[f'{col}_yz'] = (df[col] - g.transform('mean')) / (safe_std + 1e-6)
    YZ_COLS.append(f'{col}_yz')

print(f'Year z-scored features: {len(YZ_COLS)}')
print(f'Teams per year: {df.groupby("yearID").size().min()} - {df.groupby("yearID").size().max()}')

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()

RATE = ['ERA', 'FP', 'mlb_rpg']
PYTH = ['pyth_pct']
SABER = ['wOBA', 'FIP', 'DER', 'run_diff_pg']
RPG_NORM = [f'{s}_rpg' for s in RPG_SCALE]
RPG_LEAVE_PG = [f'{s}_pg' for s in RPG_LEAVE]

def bm(d, cols):
    X = d[cols].values.astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

ALPHAS = np.logspace(-3, 4, 300)
KF = KFold(n_splits=10, shuffle=True, random_state=42)

# All variant feature sets
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

print()
for name, cols in variant_cols.items():
    Xtr, Xte = bm(tr, cols), bm(te, cols)
    oof = np.zeros(len(y_wpct))
    alphas_c = []
    for tr_idx, va_idx in KF.split(Xtr):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=5)
        r.fit(sc.fit_transform(Xtr[tr_idx]), y_wpct[tr_idx])
        oof[va_idx] = r.predict(sc.transform(Xtr[va_idx]))
        alphas_c.append(r.alpha_)
    oof_all[name] = oof
    mae = mean_absolute_error(actual_W, oof * G_train)

    sc = StandardScaler()
    rf = RidgeCV(alphas=ALPHAS, cv=10)
    rf.fit(sc.fit_transform(Xtr), y_wpct)
    test_wpct_all[name] = rf.predict(sc.transform(Xte))

    print(f'{name:5s}: CV MAE = {mae:.5f} (alpha={np.median(alphas_c):.2f}, feat={len(cols)})')

# === BLEND EXPERIMENTS ===
print()
print('=== BLEND EXPERIMENTS ===')

orig6 = ['A', 'B', 'C', 'D', 'Ap', 'Dp']
oof_6 = np.mean([oof_all[k] for k in orig6], axis=0)
mae_6 = mean_absolute_error(actual_W, oof_6 * G_train)
print(f'Original 6 blend:     CV MAE = {mae_6:.5f} (public: 2.975)')

for bname, keys in [
    ('blend7_Fy', orig6 + ['Fy']),
    ('blend7_Gy', orig6 + ['Gy']),
    ('blend8_FyGy', orig6 + ['Fy', 'Gy']),
]:
    oof_b = np.mean([oof_all[k] for k in keys], axis=0)
    mae_b = mean_absolute_error(actual_W, oof_b * G_train)
    print(f'{bname:20s}: CV MAE = {mae_b:.5f}')

    # Test predictions
    test_wpct = np.mean([test_wpct_all[k] for k in keys], axis=0)
    test_W = np.clip(np.round(test_wpct * G_test).astype(int), 36, 120)
    pd.DataFrame({'ID': test_ids, 'W': test_W}).sort_values('ID').to_csv(
        f'submissions/v11_{bname}.csv', index=False)

# Diversity check
print()
print('=== DIVERSITY CHECK ===')
corr_fy = np.corrcoef(oof_all['Fy'] * G_train, oof_6 * G_train)[0, 1]
corr_gy = np.corrcoef(oof_all['Gy'] * G_train, oof_6 * G_train)[0, 1]
corr_c  = np.corrcoef(oof_all['C'] * G_train, oof_6 * G_train)[0, 1]
print(f'Fy (year z-score) vs 6-blend: {corr_fy:.4f}')
print(f'Gy (pg+year+saber) vs 6-blend: {corr_gy:.4f}')
print(f'C  (era z-score)  vs 6-blend: {corr_c:.4f}  (reference)')

print()
print('Saved submission files:')
for f in ['v11_blend7_Fy.csv', 'v11_blend7_Gy.csv', 'v11_blend8_FyGy.csv']:
    s = pd.read_csv(f'submissions/{f}')
    print(f'  {f}: W mean={s.W.mean():.1f}, range=[{s.W.min()}, {s.W.max()}]')
