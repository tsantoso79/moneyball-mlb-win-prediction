"""
Moneyball v2 — Focus on reducing overfitting gap.
Key changes: simpler ensembles, no weight optimization, try unrounded predictions.
"""
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)

train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")

def engineer_features(df):
    df = df.copy()
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
    df['IP'] = df['IPouts'] / 3
    PA = df['AB'] + df['BB']

    # Batting
    df['BA'] = df['H'] / df['AB']
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = (df['H'] + df['BB']) / PA
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['BB_SO_ratio'] = df['BB'] / (df['SO'] + 1)
    df['XBH'] = df['2B'] + df['3B'] + df['HR']
    df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / PA
    df['BABIP'] = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + 1)
    df['K_pct'] = df['SO'] / PA
    df['BB_pct'] = df['BB'] / PA
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC'] = (df['H'] + df['BB']) * df['TB'] / PA
    df['RC_per_game'] = df['RC'] / df['G']
    df['SecA'] = (df['TB'] - df['H'] + df['BB'] + df['SB']) / df['AB']
    df['power_speed'] = (2 * df['HR'] * df['SB']) / (df['HR'] + df['SB'] + 1)

    # Pitching
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    df['H9'] = 9 * df['HA'] / df['IP']
    df['HR9'] = 9 * df['HRA'] / df['IP']
    df['BB9'] = 9 * df['BBA'] / df['IP']
    df['SO9'] = 9 * df['SOA'] / df['IP']
    df['SO_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
    df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
    df['BABIP_against'] = (df['HA'] - df['HRA']) / (df['IP']*3 - df['SOA'] - df['HRA'] + 1)
    df['pitch_dominance'] = df['SOA'] / (df['HA'] + df['BBA'] + 1)
    df['FIP_ERA_diff'] = df['FIP'] - df['ERA']

    # Run differential (TIER 1)
    df['run_diff'] = df['R'] - df['RA']
    df['R_per_game'] = df['R'] / df['G']
    df['RA_per_game'] = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']

    # Pythagorean wins — 3 variants
    df['pyth_wins'] = df['R']**2 / (df['R']**2 + df['RA']**2) * df['G']
    df['pyth_wins_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83) * df['G']
    rpg = (df['R'] + df['RA']) / df['G']
    pexp = rpg ** 0.287
    df['pyth_wins_pat'] = df['R']**pexp / (df['R']**pexp + df['RA']**pexp) * df['G']

    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))

    # Fielding
    df['DER'] = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))
    df['E_per_game'] = df['E'] / df['G']
    df['DP_per_game'] = df['DP'] / df['G']
    df['UER'] = df['RA'] - df['ER']
    df['UER_pct'] = df['UER'] / (df['RA'] + 1)

    # Era-adjusted
    df['R_vs_mlb'] = df['R_per_game'] - df['mlb_rpg']
    df['RA_vs_mlb'] = df['RA_per_game'] - df['mlb_rpg']
    league_era = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era / (df['FIP'] + 0.001) * 100
    df['R_above_avg'] = (df['R_per_game'] - df['mlb_rpg']) * df['G']
    df['RA_below_avg'] = (df['mlb_rpg'] - df['RA_per_game']) * df['G']
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    # Efficiency
    df['CG_pct'] = df['CG'] / df['G']
    df['SV_pct'] = df['SV'] / df['G']
    df['SHO_pct'] = df['SHO'] / df['G']

    # Interactions
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    # Franchise encoding
    franch_counts = df['franchID'].value_counts().to_dict()
    df['franch_freq'] = df['franchID'].map(franch_counts)

    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

exclude_cols = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
feature_cols = [c for c in train_fe.columns if c not in exclude_cols and train_fe[c].dtype in ['int64', 'float64', 'bool']]
for c in feature_cols:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

train_fe[feature_cols] = train_fe[feature_cols].replace([np.inf, -np.inf], np.nan)
test_fe[feature_cols] = test_fe[feature_cols].replace([np.inf, -np.inf], np.nan)
for c in feature_cols:
    med = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(med)
    test_fe[c] = test_fe[c].fillna(med)

X = train_fe[feature_cols].values
y = train_fe['W'].values
X_test = test_fe[feature_cols].values
test_ids = test_fe['ID'].values

print(f"Features: {len(feature_cols)}")

# Scale for linear models
onehot = [c for c in feature_cols if c.startswith(('era_', 'decade_'))]
other = [c for c in feature_cols if c not in onehot]
oh_idx = [feature_cols.index(c) for c in onehot]
ot_idx = [feature_cols.index(c) for c in other]

scaler = StandardScaler()
X_sc = X.copy(); X_sc[:, ot_idx] = scaler.fit_transform(X[:, ot_idx])
X_test_sc = X_test.copy(); X_test_sc[:, ot_idx] = scaler.transform(X_test[:, ot_idx])

# ============================================================
# OOF predictions with multiple seeds for stability
# ============================================================
def get_oof(model, Xtr, ytr, Xte, n_splits=10, seeds=[42, 123, 456]):
    """Average OOF over multiple random seeds for stability."""
    all_oof = []
    all_test = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(ytr))
        tpreds = np.zeros(len(Xte))
        for tr_idx, val_idx in kf.split(Xtr):
            m = clone(model)
            m.fit(Xtr[tr_idx], ytr[tr_idx])
            oof[val_idx] = m.predict(Xtr[val_idx])
            tpreds += m.predict(Xte) / n_splits
        all_oof.append(oof)
        all_test.append(tpreds)
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    mae = mean_absolute_error(ytr, avg_oof)
    return avg_oof, avg_test, mae

print("\n" + "=" * 60)
print("OOF PREDICTIONS (averaged over 3 seeds)")
print("=" * 60)

# Models
alphas = np.logspace(-3, 3, 100)
models = {
    'Ridge': (RidgeCV(alphas=alphas, cv=5), X_sc, X_test_sc),
    'Lasso': (LassoCV(alphas=alphas, cv=5, max_iter=10000), X_sc, X_test_sc),
    'ElasticNet': (ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, max_iter=10000), X_sc, X_test_sc),
    'XGB': (XGBRegressor(n_estimators=2000, max_depth=4, learning_rate=0.01, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=2.0, min_child_weight=5,
            random_state=42, verbosity=0), X, X_test),
    'LGB': (LGBMRegressor(n_estimators=2000, max_depth=4, learning_rate=0.01, subsample=0.85,
            colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=1.5, min_child_weight=3,
            num_leaves=25, random_state=42, verbose=-1), X, X_test),
    'GBR': (GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42), X, X_test),
}

oof_res = {}
test_res = {}
for name, (model, Xtr, Xte) in models.items():
    oof, tpreds, mae = get_oof(model, Xtr, y, Xte)
    oof_res[name] = oof
    test_res[name] = tpreds
    print(f"  {name:15s} OOF MAE: {mae:.5f}")

# ============================================================
# SIMPLE ENSEMBLES (no weight optimization)
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE STRATEGIES")
print("=" * 60)

combos = {
    'Ridge only':           ['Ridge'],
    'Lasso only':           ['Lasso'],
    'Ridge+Lasso':          ['Ridge', 'Lasso'],
    'Ridge+Lasso+EN':       ['Ridge', 'Lasso', 'ElasticNet'],
    'Ridge+XGB':            ['Ridge', 'XGB'],
    'Ridge+LGB':            ['Ridge', 'LGB'],
    'Ridge+XGB+LGB':        ['Ridge', 'XGB', 'LGB'],
    'Ridge+Lasso+XGB+LGB':  ['Ridge', 'Lasso', 'XGB', 'LGB'],
    'All models':           list(models.keys()),
    'Linear trio':          ['Ridge', 'Lasso', 'ElasticNet'],
}

best_combo_name = None
best_combo_mae = 999
best_combo_preds = None

for combo_name, keys in combos.items():
    avg_oof = np.mean([oof_res[k] for k in keys], axis=0)
    avg_test = np.mean([test_res[k] for k in keys], axis=0)
    mae = mean_absolute_error(y, avg_oof)
    print(f"  {combo_name:30s} MAE: {mae:.5f}")
    if mae < best_combo_mae:
        best_combo_mae = mae
        best_combo_name = combo_name
        best_combo_preds = avg_test

print(f"\n  Best combo: {best_combo_name} (MAE: {best_combo_mae:.5f})")

# ============================================================
# WEIGHTED COMBOS (manual, not optimized)
# ============================================================
print("\n" + "=" * 60)
print("MANUAL WEIGHTED ENSEMBLES")
print("=" * 60)

# Give more weight to Ridge (our best single model) with small contributions from trees
weighted_combos = {
    '70R+15X+15L': (0.70, 'Ridge', 0.15, 'XGB', 0.15, 'LGB'),
    '60R+20X+20L': (0.60, 'Ridge', 0.20, 'XGB', 0.20, 'LGB'),
    '50R+25X+25L': (0.50, 'Ridge', 0.25, 'XGB', 0.25, 'LGB'),
    '80R+10X+10L': (0.80, 'Ridge', 0.10, 'XGB', 0.10, 'LGB'),
    '50R+50La':    (0.50, 'Ridge', 0.50, 'Lasso', 0, None),
    '60R+20La+20X': (0.60, 'Ridge', 0.20, 'Lasso', 0.20, 'XGB'),
}

best_w_name = None
best_w_mae = 999
best_w_preds = None

for wname, wspec in weighted_combos.items():
    w1, n1, w2, n2, w3, n3 = wspec
    oof_blend = w1 * oof_res[n1] + w2 * oof_res[n2]
    test_blend = w1 * test_res[n1] + w2 * test_res[n2]
    if n3 and w3 > 0:
        oof_blend += w3 * oof_res[n3]
        test_blend += w3 * test_res[n3]
    mae = mean_absolute_error(y, oof_blend)
    print(f"  {wname:30s} MAE: {mae:.5f}")
    if mae < best_w_mae:
        best_w_mae = mae
        best_w_name = wname
        best_w_preds = test_blend

print(f"\n  Best weighted: {best_w_name} (MAE: {best_w_mae:.5f})")

# ============================================================
# GENERATE ALL SUBMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSIONS")
print("=" * 60)

def save_submission(preds, filename, desc):
    p = np.round(preds).astype(int)
    p = np.clip(p, 36, 120)
    sub = pd.DataFrame({'ID': test_ids, 'W': p}).sort_values('ID').reset_index(drop=True)
    path = f"{DATA_DIR}/{filename}"
    sub.to_csv(path, index=False)
    mae_info = f"mean={p.mean():.1f}, min={p.min()}, max={p.max()}"
    print(f"  {desc:40s} -> {filename} ({mae_info})")

# Ridge only
save_submission(test_res['Ridge'], 'sub_ridge.csv', 'Ridge only')

# Best simple combo
save_submission(best_combo_preds, 'sub_best_combo.csv', f'Best combo ({best_combo_name})')

# Best weighted
save_submission(best_w_preds, 'sub_best_weighted.csv', f'Best weighted ({best_w_name})')

# Ridge + Lasso average
rl_preds = (test_res['Ridge'] + test_res['Lasso']) / 2
save_submission(rl_preds, 'sub_ridge_lasso.csv', 'Ridge+Lasso avg')

# Linear trio
lt_preds = (test_res['Ridge'] + test_res['Lasso'] + test_res['ElasticNet']) / 3
save_submission(lt_preds, 'sub_linear_trio.csv', 'Linear trio avg')

# All models simple avg
all_preds = np.mean([test_res[k] for k in models], axis=0)
save_submission(all_preds, 'sub_all_avg.csv', 'All models avg')

print("\nDone!")
