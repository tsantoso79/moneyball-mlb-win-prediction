"""V4: More features + two-stage Pythagorean residual model."""
import os
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)
train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")

def eng(df):
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

    # Pitching
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

    # Runs
    df['run_diff'] = df['R'] - df['RA']
    df['R_per_game'] = df['R'] / df['G']
    df['RA_per_game'] = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['pyth_wins'] = df['R']**2 / (df['R']**2 + df['RA']**2) * df['G']
    df['pyth_wins_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83) * df['G']
    rpg = (df['R'] + df['RA']) / df['G']
    pexp = rpg ** 0.287
    df['pyth_wins_pat'] = df['R']**pexp / (df['R']**pexp + df['RA']**pexp) * df['G']
    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))
    df['run_diff_sq'] = df['run_diff']**2 * np.sign(df['run_diff'])
    df['R_above_RC'] = df['R'] - df['RC']
    df['run_support_era'] = df['R_per_game'] / (df['ERA'] + 0.001)

    # Fielding
    df['DER'] = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))
    df['E_per_game'] = df['E'] / df['G']
    df['DP_per_game'] = df['DP'] / df['G']
    df['DP_pct'] = df['DP'] / (df['HA'] + df['BBA'] - df['HRA'] + 1)
    df['UER'] = df['RA'] - df['ER']
    df['UER_per_game'] = df['UER'] / df['G']
    df['UER_pct'] = df['UER'] / (df['RA'] + 1)

    # Era-adjusted
    df['R_vs_mlb'] = df['R_per_game'] - df['mlb_rpg']
    df['RA_vs_mlb'] = df['RA_per_game'] - df['mlb_rpg']
    df['R_ratio'] = df['R_per_game'] / (df['mlb_rpg'] + 0.001)
    df['RA_ratio'] = df['RA_per_game'] / (df['mlb_rpg'] + 0.001)
    league_era = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era / (df['FIP'] + 0.001) * 100
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100
    df['R_above_avg'] = (df['R_per_game'] - df['mlb_rpg']) * df['G']
    df['RA_below_avg'] = (df['mlb_rpg'] - df['RA_per_game']) * df['G']
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    # Efficiency
    df['runs_per_hit'] = df['R'] / (df['H'] + 1)
    df['ER_ratio'] = df['ER'] / (df['RA'] + 1)
    df['CG_pct'] = df['CG'] / df['G']
    df['SV_pct'] = df['SV'] / df['G']
    df['SHO_pct'] = df['SHO'] / df['G']

    # Interactions
    df['OPS_x_FIP'] = df['OPS'] * (1 / (df['FIP'] + 0.001))
    df['wOBA_x_WHIP'] = df['wOBA'] / (df['WHIP'] + 0.001)
    df['pyth_spread'] = df['pyth_wins_pat'] - df['pyth_wins_183']
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    # NEW: polynomial/interaction features
    df['pyth_wins_sq'] = df['pyth_wins'] ** 2
    df['run_diff_x_G'] = df['run_diff'] * df['G']
    df['R_x_RA_ratio'] = df['R'] / (df['RA'] + 1)
    df['OPS_sq'] = df['OPS'] ** 2
    df['WHIP_sq'] = df['WHIP'] ** 2
    df['ERA_x_WHIP'] = df['ERA'] * df['WHIP']
    df['pyth_x_ops'] = df['pyth_wins_pat'] * df['OPS']
    df['rdiff_x_era'] = df['run_diff_per_game'] * df['ERA_plus']

    # Franchise
    franch_counts = df['franchID'].value_counts().to_dict()
    df['franch_freq'] = df['franchID'].map(franch_counts)

    return df

train_fe = eng(train)
test_fe = eng(test)

exclude = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
feats = [c for c in train_fe.columns if c not in exclude and train_fe[c].dtype in ['int64', 'float64', 'bool']]
for c in feats:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

X = np.nan_to_num(train_fe[feats].values, nan=0, posinf=0, neginf=0).astype(float)
Xt = np.nan_to_num(test_fe[feats].values, nan=0, posinf=0, neginf=0).astype(float)
y = train_fe['W'].values

sc = StandardScaler()
Xs = sc.fit_transform(X)
Xts = sc.transform(Xt)

print(f"Total features: {len(feats)}")

# Standard Ridge with different alpha ranges
kf = KFold(n_splits=10, shuffle=True, random_state=42)
print("\n=== STANDARD RIDGE (different alpha ranges) ===")
for alpha_max in [3, 4, 5]:
    alphas = np.logspace(-3, alpha_max, 300)
    ridge = RidgeCV(alphas=alphas, cv=10)
    ridge.fit(Xs, y)

    oof = np.zeros(len(y))
    for tr, va in kf.split(Xs):
        r = RidgeCV(alphas=alphas, cv=5)
        r.fit(Xs[tr], y[tr])
        oof[va] = r.predict(Xs[va])
    mae = mean_absolute_error(y, oof)

    preds = np.round(ridge.predict(Xts)).astype(int)
    preds = np.clip(preds, 36, 120)
    fname = f"v4_ridge_{len(feats)}f_a{alpha_max}.csv"
    pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds}).sort_values('ID').reset_index(drop=True).to_csv(f"{DATA_DIR}/{fname}", index=False)
    print(f"  alpha_max=1e{alpha_max}: best_alpha={ridge.alpha_:.2f}, CV MAE={mae:.5f} -> {fname}")

# TWO-STAGE: Pythagorean base + Ridge residual correction
print("\n=== TWO-STAGE: Pythagorean + Ridge residual ===")
pyth_train = train_fe['pyth_wins_pat'].values
pyth_test = test_fe['pyth_wins_pat'].values
residual = y - pyth_train

alphas = np.logspace(-3, 4, 300)
ridge_res = RidgeCV(alphas=alphas, cv=10)
ridge_res.fit(Xs, residual)

oof_res = np.zeros(len(y))
for tr, va in kf.split(Xs):
    r = RidgeCV(alphas=alphas, cv=5)
    r.fit(Xs[tr], residual[tr])
    oof_res[va] = r.predict(Xs[va])
oof_total = pyth_train + oof_res
mae_2s = mean_absolute_error(y, oof_total)

preds_2s = np.round(pyth_test + ridge_res.predict(Xts)).astype(int)
preds_2s = np.clip(preds_2s, 36, 120)
pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds_2s}).sort_values('ID').reset_index(drop=True).to_csv(f"{DATA_DIR}/v4_twostage.csv", index=False)
print(f"  Two-stage CV MAE: {mae_2s:.5f} (alpha={ridge_res.alpha_:.2f}) -> v4_twostage.csv")

print("\nDone!")
