"""
Moneyball v3 — Less is more. Fewer features, tighter Ridge.
Key insight: 98 features causes overfitting. Pythagorean wins dominates.
"""
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, make_scorer

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)
train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")

def engineer_features(df):
    df = df.copy()
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
    df['IP'] = df['IPouts'] / 3
    PA = df['AB'] + df['BB']

    # Pythagorean wins (THE key features)
    df['pyth_wins'] = df['R']**2 / (df['R']**2 + df['RA']**2) * df['G']
    df['pyth_wins_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83) * df['G']
    rpg = (df['R'] + df['RA']) / df['G']
    pexp = rpg ** 0.287
    df['pyth_wins_pat'] = df['R']**pexp / (df['R']**pexp + df['RA']**pexp) * df['G']

    # Run-based
    df['run_diff'] = df['R'] - df['RA']
    df['R_per_game'] = df['R'] / df['G']
    df['RA_per_game'] = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))

    # Era-adjusted
    df['R_vs_mlb'] = df['R_per_game'] - df['mlb_rpg']
    df['RA_vs_mlb'] = df['RA_per_game'] - df['mlb_rpg']
    df['total_runs_above_avg'] = (df['R_vs_mlb'] - df['RA_vs_mlb']) * df['G']

    # Key batting
    df['OBP'] = (df['H'] + df['BB']) / PA
    df['SLG'] = df['TB'] / df['AB']
    df['OPS'] = df['OBP'] + df['SLG']
    df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / PA

    # Key pitching
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
    df['ERA_plus'] = (df['mlb_rpg'] * 0.92) / (df['ERA'] + 0.001) * 100

    # Fielding
    df['DER'] = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))

    # Efficiency
    df['SV_pct'] = df['SV'] / df['G']
    df['SHO_pct'] = df['SHO'] / df['G']

    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# ============================================================
# TEST MULTIPLE FEATURE SETS
# ============================================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
alphas = np.logspace(-3, 4, 200)

feature_sets = {
    # Minimal: just Pythagorean
    'pyth_only': ['pyth_wins', 'pyth_wins_183', 'pyth_wins_pat'],

    # Pyth + run context
    'pyth_runs': ['pyth_wins', 'pyth_wins_183', 'pyth_wins_pat',
                  'run_diff', 'run_diff_per_game', 'log_run_ratio',
                  'R_per_game', 'RA_per_game'],

    # Pyth + runs + era
    'pyth_runs_era': ['pyth_wins', 'pyth_wins_183', 'pyth_wins_pat',
                      'run_diff', 'run_diff_per_game', 'log_run_ratio',
                      'R_per_game', 'RA_per_game',
                      'R_vs_mlb', 'RA_vs_mlb', 'total_runs_above_avg', 'mlb_rpg'],

    # Core sabermetrics (compact)
    'core_saber': ['pyth_wins', 'pyth_wins_183', 'pyth_wins_pat',
                   'run_diff', 'run_diff_per_game', 'log_run_ratio',
                   'R_per_game', 'RA_per_game',
                   'R_vs_mlb', 'RA_vs_mlb', 'total_runs_above_avg', 'mlb_rpg',
                   'OPS', 'wOBA', 'WHIP', 'FIP', 'ERA_plus', 'DER',
                   'SV_pct', 'SHO_pct'],

    # Core + raw stats
    'core_plus_raw': ['pyth_wins', 'pyth_wins_183', 'pyth_wins_pat',
                      'run_diff', 'run_diff_per_game', 'log_run_ratio',
                      'R_per_game', 'RA_per_game',
                      'R_vs_mlb', 'RA_vs_mlb', 'total_runs_above_avg', 'mlb_rpg',
                      'OPS', 'wOBA', 'WHIP', 'FIP', 'ERA_plus', 'DER',
                      'SV_pct', 'SHO_pct',
                      'G', 'R', 'RA', 'H', 'HR', 'BB', 'SO', 'SB',
                      'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
                      'E', 'DP', 'FP'],

    # Original default features + sabermetric (what worked for submission_ridge)
    'original_plus_saber': ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                            'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
                            'E', 'DP', 'FP', 'mlb_rpg',
                            'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
                            'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
                            'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
                            'pyth_wins', 'pyth_wins_183', 'pyth_wins_pat',
                            'run_diff', 'run_diff_per_game', 'log_run_ratio',
                            'R_per_game', 'RA_per_game',
                            'R_vs_mlb', 'RA_vs_mlb', 'total_runs_above_avg',
                            'OPS', 'wOBA', 'WHIP', 'FIP', 'ERA_plus', 'DER',
                            'SV_pct', 'SHO_pct'],

    # Use Lasso to find which features matter, then try just those
    'lasso_selected': None,  # will be filled below
}

print("=" * 60)
print("FEATURE SET COMPARISON (Ridge, 10-fold CV)")
print("=" * 60)

best_name = None
best_mae = 999
results = {}

for name, feats in feature_sets.items():
    if feats is None:
        continue

    # Handle booleans
    for c in feats:
        if train_fe[c].dtype == 'bool':
            train_fe[c] = train_fe[c].astype(int)
            test_fe[c] = test_fe[c].astype(int)

    Xtr = train_fe[feats].values.copy()
    Xte = test_fe[feats].values.copy()
    Xtr = np.nan_to_num(Xtr, nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte, nan=0, posinf=0, neginf=0)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    ridge = RidgeCV(alphas=alphas, cv=5)
    scores = cross_val_score(ridge, Xtr_s, train_fe['W'].values, cv=kf, scoring=mae_scorer)
    mae = -scores.mean()
    std = scores.std()
    print(f"  {name:25s} ({len(feats):3d} feats) MAE: {mae:.5f} (+/- {std:.5f})")

    results[name] = {'mae': mae, 'feats': feats, 'Xtr': Xtr_s, 'Xte': Xte_s}
    if mae < best_mae:
        best_mae = mae
        best_name = name

# Now do Lasso feature selection
print("\n  Running Lasso feature selection...")
all_feats = feature_sets['original_plus_saber']
for c in all_feats:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

Xtr_all = train_fe[all_feats].values.copy()
Xtr_all = np.nan_to_num(Xtr_all, nan=0, posinf=0, neginf=0)
sc_all = StandardScaler()
Xtr_all_s = sc_all.fit_transform(Xtr_all)

lasso = LassoCV(alphas=np.logspace(-4, 2, 100), cv=5, max_iter=10000)
lasso.fit(Xtr_all_s, train_fe['W'].values)
selected = [f for f, c in zip(all_feats, lasso.coef_) if abs(c) > 0.01]
print(f"  Lasso selected {len(selected)} features (alpha={lasso.alpha_:.4f})")
print(f"  Selected: {selected}")

if len(selected) >= 3:
    Xtr_sel = train_fe[selected].values.copy()
    Xte_sel = test_fe[selected].values.copy()
    Xtr_sel = np.nan_to_num(Xtr_sel, nan=0, posinf=0, neginf=0)
    Xte_sel = np.nan_to_num(Xte_sel, nan=0, posinf=0, neginf=0)
    sc_sel = StandardScaler()
    Xtr_sel_s = sc_sel.fit_transform(Xtr_sel)
    Xte_sel_s = sc_sel.transform(Xte_sel)

    ridge_sel = RidgeCV(alphas=alphas, cv=5)
    scores = cross_val_score(ridge_sel, Xtr_sel_s, train_fe['W'].values, cv=kf, scoring=mae_scorer)
    mae_sel = -scores.mean()
    print(f"  {'lasso_selected':25s} ({len(selected):3d} feats) MAE: {mae_sel:.5f}")
    results['lasso_selected'] = {'mae': mae_sel, 'feats': selected, 'Xtr': Xtr_sel_s, 'Xte': Xte_sel_s}
    if mae_sel < best_mae:
        best_mae = mae_sel
        best_name = 'lasso_selected'

print(f"\n  BEST: {best_name} (MAE: {best_mae:.5f})")

# ============================================================
# GENERATE SUBMISSIONS FOR TOP FEATURE SETS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSIONS")
print("=" * 60)

y = train_fe['W'].values
test_ids = test_fe['ID'].values

for name, res in sorted(results.items(), key=lambda x: x['mae']):
    ridge = RidgeCV(alphas=alphas, cv=10)
    ridge.fit(res['Xtr'], y)
    preds = np.round(ridge.predict(res['Xte'])).astype(int)
    preds = np.clip(preds, 36, 120)

    fname = f"v3_{name}.csv"
    sub = pd.DataFrame({'ID': test_ids, 'W': preds}).sort_values('ID').reset_index(drop=True)
    sub.to_csv(f"{DATA_DIR}/{fname}", index=False)
    print(f"  {name:25s} MAE: {res['mae']:.5f} -> {fname} (mean={preds.mean():.1f})")

print("\nDone!")
