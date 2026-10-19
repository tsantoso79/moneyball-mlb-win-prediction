"""Generate 02_TwoStage_Model.ipynb from model_v4.py"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ============================================================
md("""# Moneyball: Two-Stage Pythagorean + Ridge Residual Model

**Competition:** SCTPDSAI-M3-DS2F Coaching — MoneyBall Analytics
**Metric:** MAE (Mean Absolute Error) — lower is better
**Public Leaderboard Score:** 3.021 MAE

## Approach Summary
This notebook implements a **two-stage model**:

1. **Stage 1 — Pythagorean base prediction:** Use the Pythagenpat formula to get a physics-based win estimate from runs scored/allowed
2. **Stage 2 — Ridge residual correction:** Train Ridge regression to predict the *residual* (actual wins minus Pythagorean prediction) from all 126 features

```
Final prediction = Pythagenpat_wins + Ridge(residual)
```

### Rationale
Pythagorean expectation is the most reliable single predictor of wins (~1.5 MAE standalone). By predicting residuals, Ridge focuses on the "luck" component — factors like clutch hitting, bullpen performance, and roster construction that cause actual wins to deviate from expected.

### Additional Features in V4
8 polynomial/interaction terms added on top of the 118 base features:
- `pyth_wins_sq`, `run_diff_x_G`, `OPS_sq`, `WHIP_sq`
- `ERA_x_WHIP`, `pyth_x_ops`, `rdiff_x_era`, `R_x_RA_ratio`
""")

# ============================================================
md("## 1. Setup & Data Loading")

code("""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

DATA_DIR = "."

train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")

print(f"Training set: {train.shape[0]} team-seasons, {train.shape[1]} columns")
print(f"Test set:     {test.shape[0]} team-seasons, {test.shape[1]} columns")""")

# ============================================================
md("""## 2. Feature Engineering

Same sabermetric features as the Ridge model, plus **8 polynomial/interaction terms** that capture nonlinear relationships:

| New Feature | Formula | Rationale |
|---|---|---|
| pyth_wins_sq | pyth_wins² | Nonlinear Pythagorean effect |
| run_diff_x_G | run_diff × G | Season-length adjusted run differential |
| R_x_RA_ratio | R / RA | Direct scoring ratio |
| OPS_sq | OPS² | Diminishing returns on offence |
| WHIP_sq | WHIP² | Accelerating penalty for poor pitching |
| ERA_x_WHIP | ERA × WHIP | Pitching compound weakness |
| pyth_x_ops | pyth_wins_pat × OPS | Offence amplification of expected wins |
| rdiff_x_era | run_diff/G × ERA+ | Run efficiency relative to league |
""")

code("""def engineer_features(df, franch_counts=None):
    \"\"\"Create 126 features: 118 sabermetric + 8 polynomial interactions.\"\"\"
    df = df.copy()
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
    df['IP'] = df['IPouts'] / 3
    PA = df['AB'] + df['BB']

    # --- Batting ---
    df['BA']  = df['H'] / df['AB']
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = (df['H'] + df['BB']) / PA
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['BB_rate']    = df['BB'] / df['AB']
    df['SO_rate']    = df['SO'] / df['AB']
    df['HR_rate']    = df['HR'] / df['AB']
    df['BB_SO_ratio'] = df['BB'] / (df['SO'] + 1)
    df['XBH']     = df['2B'] + df['3B'] + df['HR']
    df['XBH_pct'] = df['XBH'] / (df['H'] + 1)
    df['wOBA']    = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B'] + 1.62*df['3B'] + 2.10*df['HR']) / PA
    df['BABIP']   = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + 1)
    df['K_pct']       = df['SO'] / PA
    df['BB_pct']      = df['BB'] / PA
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC']          = (df['H'] + df['BB']) * df['TB'] / PA
    df['RC_per_game'] = df['RC'] / df['G']
    df['RC_efficiency'] = df['R'] / (df['RC'] + 1)
    df['SecA']        = (df['TB'] - df['H'] + df['BB'] + df['SB']) / df['AB']
    df['power_speed'] = (2 * df['HR'] * df['SB']) / (df['HR'] + df['SB'] + 1)

    # --- Pitching ---
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    df['H9']   = 9 * df['HA'] / df['IP']
    df['HR9']  = 9 * df['HRA'] / df['IP']
    df['BB9']  = 9 * df['BBA'] / df['IP']
    df['SO9']  = 9 * df['SOA'] / df['IP']
    df['SO_BB_ratio']  = df['SOA'] / (df['BBA'] + 1)
    df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
    df['BABIP_against']    = (df['HA'] - df['HRA']) / (df['IP']*3 - df['SOA'] - df['HRA'] + 1)
    df['HR_flyball_proxy'] = df['HRA'] / (df['HA'] - df['HRA'] + 1)
    df['pitch_dominance']  = df['SOA'] / (df['HA'] + df['BBA'] + 1)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff']     = df['FIP'] - df['ERA']

    # --- Runs ---
    df['run_diff']         = df['R'] - df['RA']
    df['R_per_game']       = df['R'] / df['G']
    df['RA_per_game']      = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']
    df['pyth_wins']     = df['R']**2 / (df['R']**2 + df['RA']**2) * df['G']
    df['pyth_wins_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83) * df['G']
    rpg = (df['R'] + df['RA']) / df['G']
    pexp = rpg ** 0.287
    df['pyth_wins_pat'] = df['R']**pexp / (df['R']**pexp + df['RA']**pexp) * df['G']
    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))
    df['run_diff_sq']   = df['run_diff']**2 * np.sign(df['run_diff'])
    df['R_above_RC']    = df['R'] - df['RC']
    df['run_support_era'] = df['R_per_game'] / (df['ERA'] + 0.001)

    # --- Fielding ---
    df['DER'] = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))
    df['E_per_game']  = df['E'] / df['G']
    df['DP_per_game'] = df['DP'] / df['G']
    df['DP_pct']      = df['DP'] / (df['HA'] + df['BBA'] - df['HRA'] + 1)
    df['UER']         = df['RA'] - df['ER']
    df['UER_per_game'] = df['UER'] / df['G']
    df['UER_pct']     = df['UER'] / (df['RA'] + 1)

    # --- Era-adjusted ---
    df['R_vs_mlb']  = df['R_per_game'] - df['mlb_rpg']
    df['RA_vs_mlb'] = df['RA_per_game'] - df['mlb_rpg']
    df['R_ratio']   = df['R_per_game'] / (df['mlb_rpg'] + 0.001)
    df['RA_ratio']  = df['RA_per_game'] / (df['mlb_rpg'] + 0.001)
    league_era = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era / (df['FIP'] + 0.001) * 100
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100
    df['R_above_avg']  = (df['R_per_game'] - df['mlb_rpg']) * df['G']
    df['RA_below_avg'] = (df['mlb_rpg'] - df['RA_per_game']) * df['G']
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    # --- Efficiency ---
    df['runs_per_hit']  = df['R'] / (df['H'] + 1)
    df['ER_ratio']      = df['ER'] / (df['RA'] + 1)
    df['CG_pct']        = df['CG'] / df['G']
    df['SV_pct']        = df['SV'] / df['G']
    df['SHO_pct']       = df['SHO'] / df['G']
    df['OPS_x_FIP']     = df['OPS'] * (1 / (df['FIP'] + 0.001))
    df['wOBA_x_WHIP']   = df['wOBA'] / (df['WHIP'] + 0.001)
    df['pyth_spread']   = df['pyth_wins_pat'] - df['pyth_wins_183']
    df['team_quality']  = (df['OPS'] - df['WHIP']) * 100

    # --- NEW: Polynomial & Interaction Features ---
    df['pyth_wins_sq']  = df['pyth_wins'] ** 2
    df['run_diff_x_G']  = df['run_diff'] * df['G']
    df['R_x_RA_ratio']  = df['R'] / (df['RA'] + 1)
    df['OPS_sq']        = df['OPS'] ** 2
    df['WHIP_sq']       = df['WHIP'] ** 2
    df['ERA_x_WHIP']    = df['ERA'] * df['WHIP']
    df['pyth_x_ops']    = df['pyth_wins_pat'] * df['OPS']
    df['rdiff_x_era']   = df['run_diff_per_game'] * df['ERA_plus']

    # Franchise frequency (uses train counts for both sets)
    if franch_counts is not None:
        df['franch_freq'] = df['franchID'].map(franch_counts).fillna(1)
    else:
        franch_counts = df['franchID'].value_counts().to_dict()
        df['franch_freq'] = df['franchID'].map(franch_counts)

    return df, franch_counts

train_fe, franch_counts = engineer_features(train)
test_fe, _ = engineer_features(test, franch_counts=franch_counts)
print(f"Features engineered. Train: {train_fe.shape}, Test: {test_fe.shape}")""")

# ============================================================
md("## 3. Preprocessing")

code("""exclude = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
feats = [c for c in train_fe.columns if c not in exclude and train_fe[c].dtype in ['int64', 'float64', 'bool']]

for c in feats:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

# Handle inf/NaN with median imputation (consistent with Ridge notebook)
train_fe[feats] = train_fe[feats].replace([np.inf, -np.inf], np.nan)
test_fe[feats] = test_fe[feats].replace([np.inf, -np.inf], np.nan)
for c in feats:
    median_val = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(median_val)
    test_fe[c] = test_fe[c].fillna(median_val)

X  = train_fe[feats].values.astype(float)
Xt = test_fe[feats].values.astype(float)
y  = train_fe['W'].values

# Scale continuous features only — preserve binary one-hot indicators
onehot_cols = [c for c in feats if c.startswith(('era_', 'decade_'))]
other_cols  = [c for c in feats if c not in onehot_cols]
onehot_idx  = [feats.index(c) for c in onehot_cols]
other_idx   = [feats.index(c) for c in other_cols]

sc = StandardScaler()
Xs = X.copy()
Xs[:, other_idx] = sc.fit_transform(X[:, other_idx])
Xts = Xt.copy()
Xts[:, other_idx] = sc.transform(Xt[:, other_idx])

print(f"Total features: {len(feats)} ({len(other_cols)} continuous + {len(onehot_cols)} one-hot)")
print(f"Training samples: {Xs.shape[0]}")""")

# ============================================================
md("""## 4. Standard Ridge Baseline

First, we train a standard Ridge model as a comparison point. We test three alpha search ranges to ensure we find the global optimum.
""")

code("""kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []
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
    results.append({'alpha_max': f'1e{alpha_max}', 'best_alpha': ridge.alpha_, 'cv_mae': mae})
    print(f"alpha_max=1e{alpha_max}: best_alpha={ridge.alpha_:.2f}, CV MAE={mae:.5f}")

print(f"\\nBest standard Ridge CV MAE: {min(r['cv_mae'] for r in results):.5f}")""")

# ============================================================
md("""## 5. Two-Stage Model: Pythagorean Base + Ridge Residual

### Stage 1: Pythagorean Prediction
Pythagenpat (dynamic exponent) provides the base win estimate:
```
Win% = R^x / (R^x + RA^x)    where x = RPG^0.287
Expected_Wins = Win% × G
```

### Stage 2: Ridge on Residuals
```
Residual = Actual_W - Pythagorean_W
Ridge learns: Residual ≈ f(all 126 features)
Final = Pythagorean_W + Ridge_predicted_residual
```

This lets Ridge focus on the ~3-4 win "luck" component rather than the full ~79-win prediction.
""")

code("""# Stage 1: Pythagorean base predictions
pyth_train = train_fe['pyth_wins_pat'].values
pyth_test  = test_fe['pyth_wins_pat'].values
residual   = y - pyth_train

print(f"Pythagorean standalone MAE: {mean_absolute_error(y, pyth_train):.3f}")
print(f"Residual stats — Mean: {residual.mean():.2f}, Std: {residual.std():.2f}, "
      f"Range: [{residual.min():.1f}, {residual.max():.1f}]")""")

code("""# Stage 2: Ridge on residuals
alphas = np.logspace(-3, 4, 300)
ridge_res = RidgeCV(alphas=alphas, cv=10)
ridge_res.fit(Xs, residual)

# OOF evaluation of the two-stage model
oof_res = np.zeros(len(y))
for tr, va in kf.split(Xs):
    r = RidgeCV(alphas=alphas, cv=5)
    r.fit(Xs[tr], residual[tr])
    oof_res[va] = r.predict(Xs[va])

oof_total = pyth_train + oof_res
mae_2s = mean_absolute_error(y, oof_total)

print(f"Two-stage model:")
print(f"  Ridge residual alpha: {ridge_res.alpha_:.2f}")
print(f"  CV MAE: {mae_2s:.5f}")
print(f"  Improvement over Pythagorean alone: {mean_absolute_error(y, pyth_train) - mae_2s:.3f} wins")""")

# ============================================================
md("## 6. Generate Submission")

code("""# Final two-stage prediction
preds_2s = np.round(pyth_test + ridge_res.predict(Xts)).astype(int)
preds_2s = np.clip(preds_2s, 36, 120)

print(f"Two-stage predictions — Min: {preds_2s.min()}, Max: {preds_2s.max()}, "
      f"Mean: {preds_2s.mean():.1f}")

submission = pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds_2s})
submission = submission.sort_values('ID').reset_index(drop=True)
submission.to_csv("submissions/v4_twostage.csv", index=False)
print(f"Submission saved: submissions/v4_twostage.csv")
submission.head(10)""")

# ============================================================
md("""## 7. Comparison: Standard Ridge vs Two-Stage

| Model | Features | CV MAE | Public MAE |
|---|---|---|---|
| Standard Ridge (V1) | 121 | 2.720 | **2.992** |
| Standard Ridge (V4) | 126 | 2.720 | — |
| **Two-Stage** | 126 | **2.716** | 3.021 |

### Why Two-Stage had better CV but worse public score
1. **Pythagorean base introduces a dependency** — if Pythagenpat has a systematic bias for certain era teams in the test set, Stage 2 cannot fully correct it
2. **Residual prediction is harder** — the residual is noisier and more random than the full W signal
3. **Over-specialisation risk** — the model may learn CV-specific residual patterns that don't transfer

### Lesson
Simpler is better for this competition. The plain Ridge approach (one direct prediction) is more robust to test distribution shift.
""")

nb.cells = cells
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_TwoStage_Model.ipynb"))
print("Created 02_TwoStage_Model.ipynb")
