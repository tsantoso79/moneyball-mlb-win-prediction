"""Generate Ridge-only submission as safe backup."""
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)

train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")

def engineer_features(df):
    df = df.copy()
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
    df['IP'] = df['IPouts'] / 3
    PA = df['AB'] + df['BB']
    df['BA'] = df['H'] / df['AB']
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = (df['H'] + df['BB']) / PA
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['BB_rate'] = df['BB'] / df['AB']
    df['SO_rate'] = df['SO'] / df['AB']
    df['HR_rate'] = df['HR'] / df['AB']
    df['BB_SO_ratio'] = df['BB'] / (df['SO'] + 1)
    df['XBH'] = df['2B'] + df['3B'] + df['HR']
    df['XBH_pct'] = df['XBH'] / (df['H'] + 1)
    df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / PA
    df['BABIP'] = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + 1)
    df['K_pct'] = df['SO'] / PA
    df['BB_pct'] = df['BB'] / PA
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC'] = (df['H'] + df['BB']) * df['TB'] / PA
    df['RC_per_game'] = df['RC'] / df['G']
    df['RC_efficiency'] = df['R'] / (df['RC'] + 1)
    df['SecA'] = (df['TB'] - df['H'] + df['BB'] + df['SB']) / df['AB']
    df['power_speed'] = (2 * df['HR'] * df['SB']) / (df['HR'] + df['SB'] + 1)
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    df['H9'] = 9 * df['HA'] / df['IP']
    df['HR9'] = 9 * df['HRA'] / df['IP']
    df['BB9'] = 9 * df['BBA'] / df['IP']
    df['SO9'] = 9 * df['SOA'] / df['IP']
    df['SO_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)
    df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
    df['BABIP_against'] = (df['HA'] - df['HRA']) / (df['IP']*3 - df['SOA'] - df['HRA'] + 1)
    df['HR_flyball_proxy'] = df['HRA'] / (df['HA'] - df['HRA'] + 1)
    df['pitch_dominance'] = df['SOA'] / (df['HA'] + df['BBA'] + 1)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff'] = df['FIP'] - df['ERA']
    df['run_diff'] = df['R'] - df['RA']
    df['R_per_game'] = df['R'] / df['G']
    df['RA_per_game'] = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['pyth_exp'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['pyth_wins'] = df['pyth_exp'] * df['G']
    df['pyth_exp_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83)
    df['pyth_wins_183'] = df['pyth_exp_183'] * df['G']
    rpg = (df['R'] + df['RA']) / df['G']
    pyth_exp_dyn = rpg ** 0.287
    df['pyth_exp_pat'] = df['R']**pyth_exp_dyn / (df['R']**pyth_exp_dyn + df['RA']**pyth_exp_dyn)
    df['pyth_wins_pat'] = df['pyth_exp_pat'] * df['G']
    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))
    df['run_diff_sq'] = df['run_diff']**2 * np.sign(df['run_diff'])
    df['R_above_RC'] = df['R'] - df['RC']
    df['run_support_era'] = df['R_per_game'] / (df['ERA'] + 0.001)
    df['DER'] = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))
    df['E_per_game'] = df['E'] / df['G']
    df['DP_per_game'] = df['DP'] / df['G']
    df['DP_pct'] = df['DP'] / (df['HA'] + df['BBA'] - df['HRA'] + 1)
    df['UER'] = df['RA'] - df['ER']
    df['UER_per_game'] = df['UER'] / df['G']
    df['UER_pct'] = df['UER'] / (df['RA'] + 1)
    df['R_vs_mlb'] = df['R_per_game'] - df['mlb_rpg']
    df['RA_vs_mlb'] = df['RA_per_game'] - df['mlb_rpg']
    df['R_ratio'] = df['R_per_game'] / (df['mlb_rpg'] + 0.001)
    df['RA_ratio'] = df['RA_per_game'] / (df['mlb_rpg'] + 0.001)
    league_era_approx = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era_approx / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era_approx / (df['FIP'] + 0.001) * 100
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100
    df['R_above_avg'] = (df['R_per_game'] - df['mlb_rpg']) * df['G']
    df['RA_below_avg'] = (df['mlb_rpg'] - df['RA_per_game']) * df['G']
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']
    df['runs_per_hit'] = df['R'] / (df['H'] + 1)
    df['ER_ratio'] = df['ER'] / (df['RA'] + 1)
    df['CG_pct'] = df['CG'] / df['G']
    df['SV_pct'] = df['SV'] / df['G']
    df['SHO_pct'] = df['SHO'] / df['G']
    df['OPS_x_FIP'] = df['OPS'] * (1 / (df['FIP'] + 0.001))
    df['wOBA_x_WHIP'] = df['wOBA'] / (df['WHIP'] + 0.001)
    df['pyth_spread'] = df['pyth_wins_pat'] - df['pyth_wins_183']
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100
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
    median_val = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(median_val)
    test_fe[c] = test_fe[c].fillna(median_val)

X = train_fe[feature_cols].values
y = train_fe['W'].values
X_test = test_fe[feature_cols].values
test_ids = test_fe['ID'].values

onehot_cols = [c for c in feature_cols if c.startswith(('era_', 'decade_'))]
other_cols = [c for c in feature_cols if c not in onehot_cols]
onehot_idx = [feature_cols.index(c) for c in onehot_cols]
other_idx = [feature_cols.index(c) for c in other_cols]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:, other_idx] = scaler.fit_transform(X[:, other_idx])
X_test_scaled = X_test.copy()
X_test_scaled[:, other_idx] = scaler.transform(X_test[:, other_idx])

ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=10)
ridge.fit(X_scaled, y)
print(f"Ridge best alpha: {ridge.alpha_:.4f}")

preds = ridge.predict(X_test_scaled)
preds_rounded = np.round(preds).astype(int)
preds_rounded = np.clip(preds_rounded, 36, 120)

submission = pd.DataFrame({'ID': test_ids, 'W': preds_rounded})
submission = submission.sort_values('ID').reset_index(drop=True)
submission.to_csv(f"{DATA_DIR}/submission_ridge.csv", index=False)
print(f"Ridge submission saved. Predictions: min={preds_rounded.min()}, max={preds_rounded.max()}, mean={preds_rounded.mean():.1f}")
