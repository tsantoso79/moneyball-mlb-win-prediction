"""V14: XGBoost + LightGBM tree models, standalone and blended with Ridge
Conservative hyperparameters for 1812-row dataset.
Uses v9-style integer-averaging rounding.
"""
import pandas as pd, numpy as np, os
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

train = pd.read_csv('data_year_team_franchise.csv')
test  = pd.read_csv('predict_year_team_franchise.csv')
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

PG_COLS = [f'{s}_pg' for s in COUNTING_STATS]
RATE = ['ERA', 'FP', 'mlb_rpg']
PYTH = ['pyth_pct']
SABER = ['wOBA', 'FIP', 'DER', 'run_diff_pg']

# Feature set for trees: per-game stats + rate stats + sabermetrics
# Trees don't need normalization variants — they handle scale naturally
TREE_FEATURES = PG_COLS + RATE + PYTH + SABER

def bm(d, cols):
    X = d[cols].values.astype(float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

tr = df[df['_is_train']].copy()
te = df[~df['_is_train']].copy()
X_tr = bm(tr, TREE_FEATURES)
X_te = bm(te, TREE_FEATURES)

KF = KFold(n_splits=10, shuffle=True, random_state=42)

# ============================================================
# XGBoost configurations (conservative for small data)
# ============================================================
print('=== XGBoost ===')
xgb_configs = {
    'xgb_conservative': {
        'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
        'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'random_state': 42,
    },
    'xgb_moderate': {
        'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
        'min_child_weight': 8, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 0.5, 'reg_lambda': 3.0, 'random_state': 42,
    },
    'xgb_deep': {
        'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 800,
        'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'random_state': 42,
    },
    'xgb_linear': {
        'booster': 'gblinear', 'learning_rate': 0.1, 'n_estimators': 300,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'random_state': 42,
    },
}

xgb_oof = {}
xgb_test = {}
for name, params in xgb_configs.items():
    oof = np.zeros(len(y_wpct))
    test_preds = []
    for tr_idx, va_idx in KF.split(X_tr):
        m = xgb.XGBRegressor(**params, verbosity=0)
        m.fit(X_tr[tr_idx], y_wpct[tr_idx],
              eval_set=[(X_tr[va_idx], y_wpct[va_idx])],
              verbose=False)
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_preds.append(np.clip(np.round(m.predict(X_te) * G_test).astype(int), 36, 120))

    xgb_oof[name] = oof
    xgb_test[name] = test_preds  # list of 10 integer arrays (for v9-style averaging)
    mae = mean_absolute_error(actual_W, oof * G_train)
    print(f'  {name:25s}: CV MAE = {mae:.5f}')

# ============================================================
# LightGBM configurations
# ============================================================
print('\n=== LightGBM ===')
lgb_configs = {
    'lgb_conservative': {
        'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
        'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'num_leaves': 8, 'random_state': 42,
        'verbosity': -1,
    },
    'lgb_moderate': {
        'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
        'min_child_samples': 15, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 0.5, 'reg_lambda': 3.0, 'num_leaves': 12, 'random_state': 42,
        'verbosity': -1,
    },
    'lgb_deep': {
        'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 800,
        'min_child_samples': 10, 'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'num_leaves': 16, 'random_state': 42,
        'verbosity': -1,
    },
}

lgb_oof = {}
lgb_test = {}
for name, params in lgb_configs.items():
    oof = np.zeros(len(y_wpct))
    test_preds = []
    for tr_idx, va_idx in KF.split(X_tr):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y_wpct[tr_idx],
              eval_set=[(X_tr[va_idx], y_wpct[va_idx])])
        oof[va_idx] = m.predict(X_tr[va_idx])
        test_preds.append(np.clip(np.round(m.predict(X_te) * G_test).astype(int), 36, 120))

    lgb_oof[name] = oof
    lgb_test[name] = test_preds
    mae = mean_absolute_error(actual_W, oof * G_train)
    print(f'  {name:25s}: CV MAE = {mae:.5f}')

# ============================================================
# Ridge 6-blend reference (for blending comparison)
# ============================================================
print('\n=== Ridge 6-blend reference ===')
era_cols = [c for c in df.columns if c.startswith('era_')]
df['era_label'] = df[era_cols].idxmax(axis=1)
EZ_COLS = []
for col in PG_COLS + ['ERA', 'FP']:
    g = df.groupby('era_label')[col]
    df[f'{col}_ez'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-6)
    EZ_COLS.append(f'{col}_ez')

RPG_NORM = [f'{s}_rpg' for s in RPG_SCALE]
RPG_LEAVE = ['AB','3B','SO','SB','CG','SHO','SV','IPouts','SOA','E','DP']
RPG_LEAVE_PG = [f'{s}_pg' for s in RPG_LEAVE]

tr = df[df['_is_train']].copy(); te = df[~df['_is_train']].copy()
ALPHAS = np.logspace(-3, 4, 300)

ridge_variants = {
    'A':  PG_COLS + RATE + PYTH,
    'B':  RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + PYTH,
    'C':  EZ_COLS + PYTH,
    'D':  list(dict.fromkeys(PG_COLS + RATE + PYTH + RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + EZ_COLS)),
    'Ap': PG_COLS + RATE + PYTH + SABER,
    'Dp': list(dict.fromkeys(PG_COLS + RATE + PYTH + SABER + RPG_NORM + RPG_LEAVE_PG + ['ERA', 'FP'] + EZ_COLS)),
}

ridge_oof = {}
ridge_test = {}
for name, cols in ridge_variants.items():
    Xtr, Xte_r = bm(tr, cols), bm(te, cols)
    oof = np.zeros(len(y_wpct))
    test_preds = []
    for tr_idx, va_idx in KF.split(Xtr):
        sc = StandardScaler()
        r = RidgeCV(alphas=ALPHAS, cv=5)
        r.fit(sc.fit_transform(Xtr[tr_idx]), y_wpct[tr_idx])
        oof[va_idx] = r.predict(sc.transform(Xtr[va_idx]))
        # v9-style: round each fold's prediction to integer
        sc2 = StandardScaler()
        r2 = RidgeCV(alphas=ALPHAS, cv=5)
        r2.fit(sc2.fit_transform(Xtr[tr_idx]), y_wpct[tr_idx])
        test_preds.append(np.clip(np.round(r2.predict(sc2.transform(Xte_r)) * G_test).astype(int), 36, 120))
    ridge_oof[name] = oof
    ridge_test[name] = test_preds

# v9 baseline: average integer predictions across variants and folds
# Actually v9 trains on ALL data for test predictions (not per-fold)
# Let's do it correctly: train on all, predict test, round to int, average across variants
ridge_test_int = {}
for name, cols in ridge_variants.items():
    Xtr, Xte_r = bm(tr, cols), bm(te, cols)
    sc = StandardScaler()
    r = RidgeCV(alphas=ALPHAS, cv=10)
    r.fit(sc.fit_transform(Xtr), y_wpct)
    ridge_test_int[name] = np.clip(np.round(r.predict(sc.transform(Xte_r)) * G_test).astype(int), 36, 120)

ridge_blend_W = np.clip(np.round(
    np.mean([ridge_test_int[k] for k in ridge_variants], axis=0)
).astype(int), 36, 120)

ridge_blend_oof = np.mean([ridge_oof[k] for k in ridge_variants], axis=0)
mae_ridge = mean_absolute_error(actual_W, ridge_blend_oof * G_train)
print(f'  Ridge 6-blend: CV MAE = {mae_ridge:.5f} (public: 2.975)')

# ============================================================
# Cross-type blending: Ridge + best tree
# ============================================================
print('\n=== Cross-type blending (Ridge 6-blend + tree) ===')

# Find best XGB and LGB by CV
all_tree_oof = {**xgb_oof, **lgb_oof}
all_tree_test = {**xgb_test, **lgb_test}
best_trees = sorted(all_tree_oof.items(), key=lambda x: mean_absolute_error(actual_W, x[1] * G_train))

print('\nTree models ranked:')
for name, oof in best_trees:
    mae = mean_absolute_error(actual_W, oof * G_train)
    corr = np.corrcoef(oof, ridge_blend_oof)[0, 1]
    print(f'  {name:25s}: CV MAE = {mae:.5f}, corr with Ridge = {corr:.4f}')

# Blend Ridge 6-blend with each tree model at various weights
# Use v9-style integer averaging
print('\n--- Ridge + tree blends ---')
for tree_name, tree_oof in best_trees[:3]:  # top 3 trees only
    tree_mae = mean_absolute_error(actual_W, tree_oof * G_train)

    # Tree test predictions: average the 10 fold predictions (v9 style)
    tree_test_int = np.clip(np.round(
        np.mean(all_tree_test[tree_name], axis=0)
    ).astype(int), 36, 120)

    for tree_weight in [0.10, 0.20, 0.30, 0.40, 0.50]:
        # OOF blend for CV
        blend_oof = (1 - tree_weight) * ridge_blend_oof + tree_weight * tree_oof
        mae_blend = mean_absolute_error(actual_W, blend_oof * G_train)

        # Test blend (v9 style: blend integers, then round)
        blend_test_W = np.clip(np.round(
            (1 - tree_weight) * ridge_blend_W + tree_weight * tree_test_int
        ).astype(int), 36, 120)

        print(f'  Ridge({1-tree_weight:.0%}) + {tree_name}({tree_weight:.0%}): '
              f'CV MAE = {mae_blend:.5f} (Ridge={mae_ridge:.5f}, tree={tree_mae:.5f})')

# ============================================================
# Generate submissions
# ============================================================
print('\n=== Submissions ===')
submissions = {}

# Best standalone tree models
for tree_name in [best_trees[0][0], best_trees[1][0]]:
    tree_test_int = np.clip(np.round(
        np.mean(all_tree_test[tree_name], axis=0)
    ).astype(int), 36, 120)
    submissions[f'v14_{tree_name}'] = tree_test_int

# Best blends: Ridge + best tree at 10%, 20%, 30%
best_tree_name = best_trees[0][0]
best_tree_test_int = np.clip(np.round(
    np.mean(all_tree_test[best_tree_name], axis=0)
).astype(int), 36, 120)

for w in [0.10, 0.20, 0.30]:
    blend_W = np.clip(np.round(
        (1 - w) * ridge_blend_W + w * best_tree_test_int
    ).astype(int), 36, 120)
    submissions[f'v14_ridge{int((1-w)*100)}_tree{int(w*100)}'] = blend_W

# Reference
v9_ref = pd.read_csv('submissions/v9_blend_all6_fixed.csv').sort_values('ID')['W'].values

print(f'\n{"Submission":<35s} {"W mean":>7s} {"diff vs v9":>10s} {"exact":>6s}')
print('-' * 62)
for name, W_arr in submissions.items():
    sub = pd.DataFrame({'ID': test_ids, 'W': W_arr}).sort_values('ID')
    sub.to_csv(f'submissions/{name}.csv', index=False)
    diff = np.abs(sub['W'].values - v9_ref)
    match = (diff == 0).sum()
    print(f'{name:<35s} {W_arr.mean():>7.1f} {diff.mean():>10.3f} {match:>5d}/453')
