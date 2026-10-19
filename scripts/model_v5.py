"""
V5: Per-game normalize ALL counting stats before feature engineering.
Removes G-dependency confound.
"""
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
    G = df['G']

    # ====== PER-GAME NORMALIZE all counting stats ======
    counting_stats = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                      'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
    for s in counting_stats:
        df[f'{s}_pg'] = df[s] / G

    # Use per-game versions for all derived features
    Rpg = df['R_pg']
    RApg = df['RA_pg']
    Hpg = df['H_pg']
    ABpg = df['AB_pg']
    BBpg = df['BB_pg']
    SOpg = df['SO_pg']
    HRpg = df['HR_pg']
    B2pg = df['2B_pg']
    B3pg = df['3B_pg']
    SBpg = df['SB_pg']
    HApg = df['HA_pg']
    HRApg = df['HRA_pg']
    BBApg = df['BBA_pg']
    SOApg = df['SOA_pg']
    IPpg = df['IPouts_pg'] / 3  # innings per game
    ERpg = df['ER_pg']

    # ====== BATTING (all per-game based) ======
    df['1B_pg'] = Hpg - B2pg - B3pg - HRpg
    df['TB_pg'] = df['1B_pg'] + 2*B2pg + 3*B3pg + 4*HRpg
    PApg = ABpg + BBpg

    df['BA'] = Hpg / (ABpg + 0.001)
    df['SLG'] = df['TB_pg'] / (ABpg + 0.001)
    df['OBP'] = (Hpg + BBpg) / (PApg + 0.001)
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['BB_SO_ratio'] = BBpg / (SOpg + 0.001)
    df['XBH_pg'] = B2pg + B3pg + HRpg
    df['XBH_pct'] = df['XBH_pg'] / (Hpg + 0.001)

    df['wOBA'] = (0.69*BBpg + 0.89*df['1B_pg'] + 1.27*B2pg + 1.62*B3pg + 2.10*HRpg) / (PApg + 0.001)
    df['BABIP'] = (Hpg - HRpg) / (ABpg - SOpg - HRpg + 0.001)
    df['K_pct'] = SOpg / (PApg + 0.001)
    df['BB_pct'] = BBpg / (PApg + 0.001)
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC_pg'] = (Hpg + BBpg) * df['TB_pg'] / (PApg + 0.001)
    df['SecA'] = (df['TB_pg'] - Hpg + BBpg + SBpg) / (ABpg + 0.001)
    df['power_speed'] = (2 * HRpg * SBpg) / (HRpg + SBpg + 0.001)

    # ====== PITCHING (rate stats, already G-independent) ======
    df['WHIP'] = (BBApg + HApg) / (IPpg + 0.001)
    df['H9'] = HApg / (IPpg + 0.001) * 9
    df['HR9'] = HRApg / (IPpg + 0.001) * 9
    df['BB9'] = BBApg / (IPpg + 0.001) * 9
    df['SO9'] = SOApg / (IPpg + 0.001) * 9
    df['SO_BB_ratio'] = SOApg / (BBApg + 0.001)
    df['FIP'] = (13*HRApg + 3*BBApg - 2*SOApg) / (IPpg + 0.001) + 3.10
    df['BABIP_against'] = (HApg - HRApg) / (IPpg*3 - SOApg - HRApg + 0.001)
    df['pitch_dominance'] = SOApg / (HApg + BBApg + 0.001)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff'] = df['FIP'] - df['ERA']

    # ====== RUN DIFFERENTIAL (per-game) ======
    df['run_diff_pg'] = Rpg - RApg

    # Pythagorean (uses totals, which is correct — it predicts win%)
    R = df['R']
    RA = df['RA']
    df['pyth_wins'] = R**2 / (R**2 + RA**2) * G
    df['pyth_wins_183'] = R**1.83 / (R**1.83 + RA**1.83) * G
    rpg_total = (R + RA) / G
    pexp = rpg_total ** 0.287
    df['pyth_wins_pat'] = R**pexp / (R**pexp + RA**pexp) * G

    # Win % versions (G-independent)
    df['pyth_pct'] = R**2 / (R**2 + RA**2)
    df['pyth_pct_183'] = R**1.83 / (R**1.83 + RA**1.83)
    df['pyth_pct_pat'] = R**pexp / (R**pexp + RA**pexp)

    df['log_run_ratio'] = np.log(R / (RA + 1))
    df['run_diff_sq'] = (R - RA)**2 * np.sign(R - RA)
    df['R_above_RC'] = Rpg - df['RC_pg']
    df['run_support_era'] = Rpg / (df['ERA'] + 0.001)

    # ====== FIELDING (per-game) ======
    df['DER'] = 1 - ((HApg - HRApg) / (df['IPouts_pg'] + HApg - HRApg - SOApg + 0.001))
    df['UER_pg'] = RApg - ERpg
    df['UER_pct'] = df['UER_pg'] / (RApg + 0.001)

    # ====== ERA-ADJUSTED ======
    df['R_vs_mlb'] = Rpg - df['mlb_rpg']
    df['RA_vs_mlb'] = RApg - df['mlb_rpg']
    df['R_ratio'] = Rpg / (df['mlb_rpg'] + 0.001)
    df['RA_ratio'] = RApg / (df['mlb_rpg'] + 0.001)
    league_era = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era / (df['FIP'] + 0.001) * 100
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100
    df['R_above_avg'] = df['R_vs_mlb'] * G
    df['RA_below_avg'] = -df['RA_vs_mlb'] * G
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    # ====== EFFICIENCY ======
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    # Keep G, FP, mlb_rpg, ERA as-is (already rates or contextual)
    # Keep era/decade indicators

    # Franchise encoding
    franch_counts = df['franchID'].value_counts().to_dict()
    df['franch_freq'] = df['franchID'].map(franch_counts)

    return df

train_fe = eng(train)
test_fe = eng(test)

# Feature selection: exclude raw counting stats (use _pg versions instead)
raw_counting = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
exclude = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
exclude.update(raw_counting)  # exclude raw totals

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

print(f"Features (per-game normalized, no raw totals): {len(feats)}")

# Ridge CV
alphas = np.logspace(-3, 4, 300)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

ridge = RidgeCV(alphas=alphas, cv=10)
ridge.fit(Xs, y)

oof = np.zeros(len(y))
for tr, va in kf.split(Xs):
    r = RidgeCV(alphas=alphas, cv=5)
    r.fit(Xs[tr], y[tr])
    oof[va] = r.predict(Xs[va])
mae_pg = mean_absolute_error(y, oof)
print(f"Per-game Ridge: alpha={ridge.alpha_:.2f}, CV MAE={mae_pg:.5f}")

preds = np.round(ridge.predict(Xts)).astype(int)
preds = np.clip(preds, 36, 120)
pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds}).sort_values('ID').reset_index(drop=True).to_csv(f"{DATA_DIR}/v5_pergame.csv", index=False)
print(f"Saved v5_pergame.csv (mean={preds.mean():.1f})")

# Also try: per-game + raw totals (let Ridge decide)
feats_both = [c for c in train_fe.columns
              if c not in {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
              and train_fe[c].dtype in ['int64', 'float64', 'bool']]
for c in feats_both:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

X2 = np.nan_to_num(train_fe[feats_both].values, nan=0, posinf=0, neginf=0).astype(float)
Xt2 = np.nan_to_num(test_fe[feats_both].values, nan=0, posinf=0, neginf=0).astype(float)

sc2 = StandardScaler()
X2s = sc2.fit_transform(X2)
Xt2s = sc2.transform(Xt2)

print(f"\nFeatures (per-game + raw totals): {len(feats_both)}")

ridge2 = RidgeCV(alphas=alphas, cv=10)
ridge2.fit(X2s, y)

oof2 = np.zeros(len(y))
for tr, va in kf.split(X2s):
    r = RidgeCV(alphas=alphas, cv=5)
    r.fit(X2s[tr], y[tr])
    oof2[va] = r.predict(X2s[va])
mae_both = mean_absolute_error(y, oof2)
print(f"Both Ridge: alpha={ridge2.alpha_:.2f}, CV MAE={mae_both:.5f}")

preds2 = np.round(ridge2.predict(Xt2s)).astype(int)
preds2 = np.clip(preds2, 36, 120)
pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds2}).sort_values('ID').reset_index(drop=True).to_csv(f"{DATA_DIR}/v5_both.csv", index=False)
print(f"Saved v5_both.csv (mean={preds2.mean():.1f})")

print("\nDone!")
