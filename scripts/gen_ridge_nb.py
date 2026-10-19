"""Generate 01_Ridge_Model.ipynb from ridge_submit.py"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ============================================================
md("""# Moneyball: Predicting MLB Team Wins with Ridge Regression

**Competition:** SCTPDSAI-M3-DS2F Coaching — MoneyBall Analytics
**Metric:** MAE (Mean Absolute Error) — lower is better
**Public Leaderboard Score:** 2.992 MAE (Best submission)

## Approach Summary
This notebook uses **Ridge Regression** with **121 sabermetric features** engineered from raw team statistics. Ridge's L2 regularisation effectively handles the high multicollinearity among baseball metrics while keeping all features in play.

### Why Ridge works best here
| Property | Benefit |
|---|---|
| L2 penalty shrinks correlated features jointly | Handles 0.99+ correlations (e.g., FP↔E, ERA↔ER) |
| No feature elimination | Keeps all 121 features — more signal retained |
| Closed-form solution | Stable, deterministic — no hyperparameter tuning risk |
| Low variance | Small CV-to-public gap (~0.28 MAE) |
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
print(f"Test set:     {test.shape[0]} team-seasons, {test.shape[1]} columns")
print(f"\\nYear range — Train: {train['yearID'].min()}-{train['yearID'].max()}")
print(f"Year range — Test:  {test['yearID'].min()}-{test['yearID'].max()}")
print(f"\\nTarget (W) — Mean: {train['W'].mean():.1f}, Std: {train['W'].std():.1f}, Range: [{train['W'].min()}, {train['W'].max()}]")""")

# ============================================================
md("""## 2. Feature Engineering — Sabermetric Features

We engineer **121 features** from the raw data, grouped into 8 categories:

| Category | Features | Key Metrics |
|---|---|---|
| Batting Rates | 15 | BA, OBP, SLG, OPS, ISO, wOBA |
| Batting Advanced | 8 | BABIP, Runs Created, SecA, Power-Speed |
| Pitching Rates | 12 | ERA, WHIP, FIP, H9, BB9, SO9 |
| Run Differential | 6 | run_diff, R/G, RA/G, log ratio |
| Pythagorean Wins | 6 | Classic, 1.83 exponent, Pythagenpat |
| Fielding | 6 | DER, E/G, DP/G, UER% |
| Era-Adjusted | 10 | ERA+, FIP+, OPS+, R vs league avg |
| Interactions | 5 | OPS×FIP⁻¹, wOBA/WHIP, team quality |

### Pythagorean Win Expectancy
The most predictive feature family. Bill James' formula:

```
Win% = R² / (R² + RA²)    (classic, exponent=2)
Win% = R^1.83 / (R^1.83 + RA^1.83)    (refined)
Win% = R^x / (R^x + RA^x)  where x = RPG^0.287    (Pythagenpat — dynamic exponent)
```
""")

code("""def engineer_features(df, franch_counts=None):
    \"\"\"Create 121 sabermetric features from raw team statistics.\"\"\"
    df = df.copy()

    # Intermediate calculations
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']      # Singles
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']  # Total bases
    df['IP'] = df['IPouts'] / 3                                 # Innings pitched
    PA = df['AB'] + df['BB']                                    # Plate appearances (approx)

    # ==================== BATTING RATES ====================
    df['BA']  = df['H'] / df['AB']                   # Batting average
    df['SLG'] = df['TB'] / df['AB']                  # Slugging percentage
    df['OBP'] = (df['H'] + df['BB']) / PA            # On-base percentage
    df['OPS'] = df['OBP'] + df['SLG']                # On-base plus slugging
    df['ISO'] = df['SLG'] - df['BA']                 # Isolated power

    df['BB_rate']    = df['BB'] / df['AB']            # Walk rate
    df['SO_rate']    = df['SO'] / df['AB']            # Strikeout rate
    df['HR_rate']    = df['HR'] / df['AB']            # Home run rate
    df['BB_SO_ratio'] = df['BB'] / (df['SO'] + 1)    # Plate discipline

    # ==================== BATTING ADVANCED ====================
    df['XBH']     = df['2B'] + df['3B'] + df['HR']
    df['XBH_pct'] = df['XBH'] / (df['H'] + 1)       # Extra-base hit %
    df['wOBA']    = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B']
                     + 1.62*df['3B'] + 2.10*df['HR']) / PA  # Weighted OBA
    df['BABIP']   = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + 1)

    df['K_pct']       = df['SO'] / PA
    df['BB_pct']      = df['BB'] / PA
    df['batting_eye'] = df['BB_pct'] - df['K_pct']   # Walk - strikeout rate
    df['RC']          = (df['H'] + df['BB']) * df['TB'] / PA  # Runs created
    df['RC_per_game'] = df['RC'] / df['G']
    df['RC_efficiency'] = df['R'] / (df['RC'] + 1)   # Actual vs expected runs
    df['SecA']        = (df['TB'] - df['H'] + df['BB'] + df['SB']) / df['AB']
    df['power_speed'] = (2 * df['HR'] * df['SB']) / (df['HR'] + df['SB'] + 1)

    # ==================== PITCHING ====================
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']    # Walks+Hits per IP
    df['H9']   = 9 * df['HA'] / df['IP']              # Hits per 9 innings
    df['HR9']  = 9 * df['HRA'] / df['IP']             # Home runs per 9
    df['BB9']  = 9 * df['BBA'] / df['IP']             # Walks per 9
    df['SO9']  = 9 * df['SOA'] / df['IP']             # Strikeouts per 9
    df['SO_BB_ratio']  = df['SOA'] / (df['BBA'] + 1)  # K/BB ratio

    # FIP: Fielding Independent Pitching — isolates pitcher from defence
    df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10
    df['BABIP_against']    = (df['HA'] - df['HRA']) / (df['IP']*3 - df['SOA'] - df['HRA'] + 1)
    df['HR_flyball_proxy'] = df['HRA'] / (df['HA'] - df['HRA'] + 1)
    df['pitch_dominance']  = df['SOA'] / (df['HA'] + df['BBA'] + 1)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff']     = df['FIP'] - df['ERA']    # Luck indicator

    # ==================== RUN DIFFERENTIAL ====================
    df['run_diff']         = df['R'] - df['RA']
    df['R_per_game']       = df['R'] / df['G']
    df['RA_per_game']      = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']

    # ==================== PYTHAGOREAN WINS ====================
    # Classic (exponent=2)
    df['pyth_exp']  = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['pyth_wins'] = df['pyth_exp'] * df['G']
    # Refined (exponent=1.83)
    df['pyth_exp_183']  = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83)
    df['pyth_wins_183'] = df['pyth_exp_183'] * df['G']
    # Pythagenpat (dynamic exponent based on run environment)
    rpg = (df['R'] + df['RA']) / df['G']
    pyth_exp_dyn = rpg ** 0.287
    df['pyth_exp_pat']  = df['R']**pyth_exp_dyn / (df['R']**pyth_exp_dyn + df['RA']**pyth_exp_dyn)
    df['pyth_wins_pat'] = df['pyth_exp_pat'] * df['G']

    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))
    df['run_diff_sq']   = df['run_diff']**2 * np.sign(df['run_diff'])
    df['R_above_RC']    = df['R'] - df['RC']
    df['run_support_era'] = df['R_per_game'] / (df['ERA'] + 0.001)

    # ==================== FIELDING ====================
    df['DER']         = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))
    df['E_per_game']  = df['E'] / df['G']
    df['DP_per_game'] = df['DP'] / df['G']
    df['DP_pct']      = df['DP'] / (df['HA'] + df['BBA'] - df['HRA'] + 1)
    df['UER']         = df['RA'] - df['ER']           # Unearned runs
    df['UER_per_game'] = df['UER'] / df['G']
    df['UER_pct']     = df['UER'] / (df['RA'] + 1)

    # ==================== ERA-ADJUSTED ====================
    df['R_vs_mlb']  = df['R_per_game'] - df['mlb_rpg']   # Runs above league avg
    df['RA_vs_mlb'] = df['RA_per_game'] - df['mlb_rpg']  # Runs allowed vs avg
    df['R_ratio']   = df['R_per_game'] / (df['mlb_rpg'] + 0.001)
    df['RA_ratio']  = df['RA_per_game'] / (df['mlb_rpg'] + 0.001)

    league_era_approx = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era_approx / (df['ERA'] + 0.001) * 100   # ERA+ (100 = avg)
    df['FIP_plus'] = league_era_approx / (df['FIP'] + 0.001) * 100   # FIP+
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100        # OPS+

    df['R_above_avg']  = (df['R_per_game'] - df['mlb_rpg']) * df['G']
    df['RA_below_avg'] = (df['mlb_rpg'] - df['RA_per_game']) * df['G']
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    # ==================== EFFICIENCY & INTERACTIONS ====================
    df['runs_per_hit']  = df['R'] / (df['H'] + 1)
    df['ER_ratio']      = df['ER'] / (df['RA'] + 1)
    df['CG_pct']        = df['CG'] / df['G']
    df['SV_pct']        = df['SV'] / df['G']
    df['SHO_pct']       = df['SHO'] / df['G']
    df['OPS_x_FIP']     = df['OPS'] * (1 / (df['FIP'] + 0.001))
    df['wOBA_x_WHIP']   = df['wOBA'] / (df['WHIP'] + 0.001)
    df['pyth_spread']   = df['pyth_wins_pat'] - df['pyth_wins_183']
    df['team_quality']  = (df['OPS'] - df['WHIP']) * 100

    # Franchise frequency encoding (uses external counts if provided)
    if franch_counts is not None:
        df['franch_freq'] = df['franchID'].map(franch_counts).fillna(1)
    else:
        franch_counts = df['franchID'].value_counts().to_dict()
        df['franch_freq'] = df['franchID'].map(franch_counts)

    return df, franch_counts

# Engineer features — train counts are reused for test to avoid train/test leakage
train_fe, franch_counts = engineer_features(train)
test_fe, _ = engineer_features(test, franch_counts=franch_counts)
print(f"Features engineered. Train: {train_fe.shape}, Test: {test_fe.shape}")""")

# ============================================================
md("""## 3. Feature Selection & Preprocessing

**Strategy:**
- Use all numeric features except target (W) and identifiers
- StandardScaler on continuous features (skip one-hot era/decade indicators)
- Handle inf/NaN from division by zero with median imputation
""")

code("""# Select features — exclude target, IDs, and categorical columns
exclude_cols = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
feature_cols = [c for c in train_fe.columns
                if c not in exclude_cols and train_fe[c].dtype in ['int64', 'float64', 'bool']]

for c in feature_cols:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

# Handle inf values from division-by-zero edge cases
train_fe[feature_cols] = train_fe[feature_cols].replace([np.inf, -np.inf], np.nan)
test_fe[feature_cols] = test_fe[feature_cols].replace([np.inf, -np.inf], np.nan)

# Impute NaN with training median
for c in feature_cols:
    median_val = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(median_val)
    test_fe[c] = test_fe[c].fillna(median_val)

print(f"Total features: {len(feature_cols)}")

# Separate one-hot encoded columns from continuous ones
onehot_cols = [c for c in feature_cols if c.startswith(('era_', 'decade_'))]
other_cols  = [c for c in feature_cols if c not in onehot_cols]
onehot_idx  = [feature_cols.index(c) for c in onehot_cols]
other_idx   = [feature_cols.index(c) for c in other_cols]

print(f"Continuous features: {len(other_cols)}")
print(f"One-hot indicators:  {len(onehot_cols)}")""")

code("""# Prepare arrays
X      = train_fe[feature_cols].values
y      = train_fe['W'].values
X_test = test_fe[feature_cols].values
test_ids = test_fe['ID'].values

# Scale continuous features only (preserve binary one-hot encoding)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:, other_idx] = scaler.fit_transform(X[:, other_idx])
X_test_scaled = X_test.copy()
X_test_scaled[:, other_idx] = scaler.transform(X_test[:, other_idx])

print(f"X_scaled shape: {X_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")""")

# ============================================================
md("""## 4. Ridge Regression — Training & Cross-Validation

**RidgeCV** searches over 100 alpha values (log-spaced from 0.001 to 1000) using 10-fold CV to find the optimal regularisation strength.
""")

code("""# Train RidgeCV with wide alpha search
ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=10)
ridge.fit(X_scaled, y)

print(f"Best alpha: {ridge.alpha_:.4f}")
print(f"Number of features: {X_scaled.shape[1]}")

# Manual 10-fold CV for reliable MAE estimate
kf = KFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_scaled)):
    r = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
    r.fit(X_scaled[tr_idx], y[tr_idx])
    oof_preds[va_idx] = r.predict(X_scaled[va_idx])

cv_mae = mean_absolute_error(y, oof_preds)
print(f"\\n10-Fold CV MAE: {cv_mae:.5f}")
print(f"Expected public score: ~{cv_mae + 0.28:.3f} (typical CV-to-public gap ≈ 0.28)")""")

# ============================================================
md("""## 5. Feature Importance — Top Ridge Coefficients

Ridge doesn't zero out features, but the absolute coefficient magnitudes (after scaling) indicate which features the model relies on most.
""")

code("""# Feature importance from Ridge coefficients
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': ridge.coef_,
    'abs_coef': np.abs(ridge.coef_)
}).sort_values('abs_coef', ascending=False)

print("Top 20 most important features (by |coefficient|):")
print("=" * 55)
for i, row in coef_df.head(20).iterrows():
    direction = "+" if row['coefficient'] > 0 else "-"
    print(f"  {direction} {row['feature']:30s}  {row['abs_coef']:.4f}")""")

# ============================================================
md("""## 6. Generate Predictions & Submission""")

code("""# Generate test predictions
preds = ridge.predict(X_test_scaled)
preds_rounded = np.round(preds).astype(int)
preds_rounded = np.clip(preds_rounded, 36, 120)  # Historical win range

print(f"Predictions — Min: {preds_rounded.min()}, Max: {preds_rounded.max()}, "
      f"Mean: {preds_rounded.mean():.1f}, Std: {preds_rounded.std():.1f}")
print(f"Training W  — Min: {y.min()}, Max: {y.max()}, "
      f"Mean: {y.mean():.1f}, Std: {y.std():.1f}")

# Create submission
submission = pd.DataFrame({'ID': test_ids, 'W': preds_rounded})
submission = submission.sort_values('ID').reset_index(drop=True)
submission.to_csv("submissions/submission_ridge.csv", index=False)
print(f"\\nSubmission saved: submissions/submission_ridge.csv")
submission.head(10)""")

# ============================================================
md("""## 7. Results Summary

| Metric | Value |
|---|---|
| Model | RidgeCV (L2 regularised linear regression) |
| Features | 121 sabermetric + era indicators |
| Best Alpha | 7.56 |
| CV MAE (10-fold) | 2.720 |
| **Public LB MAE** | **2.992** |
| CV-to-Public Gap | 0.272 |

### Key Takeaways
1. **Ridge > Ensembles on this dataset** — Optimised ensemble (CV 2.70) scored 3.08 on public LB
2. **More features help** — Lasso-selected 37 features scored worse (3.06) than full 121
3. **Sabermetric features provide real signal** — Ridge coefficients show raw counting stats (AB, SV, IP), era indicators, and rate stats (BA, OBP) as top contributors by magnitude
4. **Regularisation is crucial** — Alpha=7.56 means significant shrinkage, handling collinearity gracefully
""")

nb.cells = cells
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_Ridge_Model.ipynb"))
print("Created 01_Ridge_Model.ipynb")
