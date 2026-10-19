"""Generate 04_PerGame_Normalisation.ipynb from model_v5.py"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "base", "language": "python", "name": "base"}

cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ============================================================
md("""# Moneyball: Per-Game Normalisation — Removing the Games-Played Confound

**Competition:** SCTPDSAI-M3-DS2F Coaching — MoneyBall Analytics
**Metric:** MAE (Mean Absolute Error) — lower is better

## Motivation
Raw counting stats (R, HR, SO, E, etc.) are confounded by **Games Played (G)**, which ranges from 150 to 165 across eras:
- 154-game seasons (pre-1961)
- 162-game seasons (post-1961)
- Strike-shortened seasons (e.g., 1994)

A team with 162 games accumulates ~5% more counting stats than a 154-game team, purely from playing more games. This creates a **spurious signal** that the model may overfit to.

## Approach
1. Divide all 21 counting stats by G to create **per-game rates** before computing any derived features
2. All sabermetric features (OPS, WHIP, FIP, wOBA, etc.) are computed from per-game versions
3. **Two variants tested:**
   - **Variant A (per-game only):** Drop raw totals, use only per-game features (99 features)
   - **Variant B (per-game + raw):** Keep both raw totals and per-game versions (120 features), letting Ridge decide what to weight

### Counting Stats Normalised
```
Batting:   R, AB, H, 2B, 3B, HR, BB, SO, SB
Pitching:  RA, ER, CG, SHO, SV, IPouts, HA, HRA, BBA, SOA
Fielding:  E, DP
```
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
print(f"\\nGames Played (G) — Min: {train['G'].min()}, Max: {train['G'].max()}, "
      f"Mean: {train['G'].mean():.1f}, Std: {train['G'].std():.1f}")
print(f"Teams with G < 154: {(train['G'] < 154).sum()} ({(train['G'] < 154).mean()*100:.1f}%)")""")

# ============================================================
md("""## 2. Feature Engineering — Per-Game Normalisation

### Key Difference from Previous Models
In `01_Ridge_Model.ipynb`, derived features used **raw totals**:
```python
df['BA'] = df['H'] / df['AB']          # Uses raw H and AB
df['RC'] = (df['H'] + df['BB']) * df['TB'] / PA  # Uses raw totals
```

Here, we first normalise to per-game rates, then compute everything from those:
```python
Hpg  = df['H'] / G     # Hits per game
ABpg = df['AB'] / G    # At-bats per game
df['BA'] = Hpg / ABpg  # Same ratio, but computed from per-game values
```

For rate stats like BA, OBP, SLG — the result is mathematically identical. But for **additive** features like Runs Created, the per-game version removes the G-dependency:
```python
# Raw: RC = (H + BB) * TB / PA  → scales with G
# Per-game: RC_pg = (Hpg + BBpg) * TBpg / PApg  → G-independent
```
""")

code("""def engineer_features(df, franch_counts=None):
    \"\"\"Per-game normalised feature engineering.
    All 21 counting stats are divided by G before computing derived features.\"\"\"
    df = df.copy()
    G = df['G']

    # ====== STEP 1: Per-game normalise all counting stats ======
    counting_stats = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                      'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
    for s in counting_stats:
        df[f'{s}_pg'] = df[s] / G

    # Convenient aliases for per-game versions
    Rpg, RApg = df['R_pg'], df['RA_pg']
    Hpg, ABpg, BBpg, SOpg = df['H_pg'], df['AB_pg'], df['BB_pg'], df['SO_pg']
    HRpg, B2pg, B3pg, SBpg = df['HR_pg'], df['2B_pg'], df['3B_pg'], df['SB_pg']
    HApg, HRApg, BBApg, SOApg = df['HA_pg'], df['HRA_pg'], df['BBA_pg'], df['SOA_pg']
    IPpg = df['IPouts_pg'] / 3   # Innings per game
    ERpg = df['ER_pg']

    # ====== STEP 2: Derived features from per-game values ======

    # --- Batting ---
    df['1B_pg'] = Hpg - B2pg - B3pg - HRpg
    df['TB_pg'] = df['1B_pg'] + 2*B2pg + 3*B3pg + 4*HRpg
    PApg = ABpg + BBpg

    df['BA']  = Hpg / (ABpg + 0.001)
    df['SLG'] = df['TB_pg'] / (ABpg + 0.001)
    df['OBP'] = (Hpg + BBpg) / (PApg + 0.001)
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['BB_SO_ratio'] = BBpg / (SOpg + 0.001)
    df['XBH_pg']  = B2pg + B3pg + HRpg
    df['XBH_pct'] = df['XBH_pg'] / (Hpg + 0.001)

    df['wOBA'] = (0.69*BBpg + 0.89*df['1B_pg'] + 1.27*B2pg + 1.62*B3pg + 2.10*HRpg) / (PApg + 0.001)
    df['BABIP'] = (Hpg - HRpg) / (ABpg - SOpg - HRpg + 0.001)
    df['K_pct']  = SOpg / (PApg + 0.001)
    df['BB_pct'] = BBpg / (PApg + 0.001)
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC_pg']  = (Hpg + BBpg) * df['TB_pg'] / (PApg + 0.001)
    df['SecA']   = (df['TB_pg'] - Hpg + BBpg + SBpg) / (ABpg + 0.001)
    df['power_speed'] = (2 * HRpg * SBpg) / (HRpg + SBpg + 0.001)

    # --- Pitching ---
    df['WHIP'] = (BBApg + HApg) / (IPpg + 0.001)
    df['H9']   = HApg / (IPpg + 0.001) * 9
    df['HR9']  = HRApg / (IPpg + 0.001) * 9
    df['BB9']  = BBApg / (IPpg + 0.001) * 9
    df['SO9']  = SOApg / (IPpg + 0.001) * 9
    df['SO_BB_ratio']     = SOApg / (BBApg + 0.001)
    df['FIP']             = (13*HRApg + 3*BBApg - 2*SOApg) / (IPpg + 0.001) + 3.10
    df['BABIP_against']   = (HApg - HRApg) / (IPpg*3 - SOApg - HRApg + 0.001)
    df['pitch_dominance'] = SOApg / (HApg + BBApg + 0.001)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff']    = df['FIP'] - df['ERA']

    # --- Run Differential (per-game) ---
    df['run_diff_pg'] = Rpg - RApg

    # Pythagorean (uses totals — predicts win%, then × G)
    R, RA = df['R'], df['RA']
    df['pyth_wins']     = R**2 / (R**2 + RA**2) * G
    df['pyth_wins_183'] = R**1.83 / (R**1.83 + RA**1.83) * G
    rpg_total = (R + RA) / G
    pexp = rpg_total ** 0.287
    df['pyth_wins_pat'] = R**pexp / (R**pexp + RA**pexp) * G

    # Win % versions (G-independent)
    df['pyth_pct']     = R**2 / (R**2 + RA**2)
    df['pyth_pct_183'] = R**1.83 / (R**1.83 + RA**1.83)
    df['pyth_pct_pat'] = R**pexp / (R**pexp + RA**pexp)

    df['log_run_ratio'] = np.log(R / (RA + 1))
    df['run_diff_sq']   = (R - RA)**2 * np.sign(R - RA)
    df['R_above_RC']    = Rpg - df['RC_pg']
    df['run_support_era'] = Rpg / (df['ERA'] + 0.001)

    # --- Fielding ---
    df['DER'] = 1 - ((HApg - HRApg) / (df['IPouts_pg'] + HApg - HRApg - SOApg + 0.001))
    df['UER_pg']  = RApg - ERpg
    df['UER_pct'] = df['UER_pg'] / (RApg + 0.001)

    # --- Era-Adjusted ---
    df['R_vs_mlb']  = Rpg - df['mlb_rpg']
    df['RA_vs_mlb'] = RApg - df['mlb_rpg']
    df['R_ratio']   = Rpg / (df['mlb_rpg'] + 0.001)
    df['RA_ratio']  = RApg / (df['mlb_rpg'] + 0.001)
    league_era = df['mlb_rpg'] * 0.92
    df['ERA_plus'] = league_era / (df['ERA'] + 0.001) * 100
    df['FIP_plus'] = league_era / (df['FIP'] + 0.001) * 100
    df['OPS_plus'] = df['OPS'] / (df['mlb_rpg'] / 4.5) * 100
    df['R_above_avg']  = df['R_vs_mlb'] * G
    df['RA_below_avg'] = -df['RA_vs_mlb'] * G
    df['total_runs_above_avg'] = df['R_above_avg'] + df['RA_below_avg']

    # --- Efficiency ---
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    # Franchise encoding (use train counts for consistency)
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
md("""## 3. Variant A — Per-Game Features Only (No Raw Totals)

Drop all 21 raw counting stats. The model sees only per-game rates and derived features.

**Rationale:** If per-game rates fully capture the information, raw totals add noise (G-confounded signal). Ridge should perform better with cleaner inputs.
""")

code("""# Exclude raw counting stats + identifiers
raw_counting = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
exclude = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
exclude.update(raw_counting)

feats_pg = [c for c in train_fe.columns if c not in exclude and train_fe[c].dtype in ['int64', 'float64', 'bool']]
for c in feats_pg:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

# Preprocessing: median imputation + selective scaling
train_pg = train_fe[feats_pg].replace([np.inf, -np.inf], np.nan).copy()
test_pg = test_fe[feats_pg].replace([np.inf, -np.inf], np.nan).copy()
for c in feats_pg:
    med = train_pg[c].median()
    train_pg[c] = train_pg[c].fillna(med)
    test_pg[c] = test_pg[c].fillna(med)

X_pg = train_pg.values.astype(float)
Xt_pg = test_pg.values.astype(float)
y = train_fe['W'].values

# Scale continuous only, skip one-hot
onehot = [c for c in feats_pg if c.startswith(('era_', 'decade_'))]
other = [c for c in feats_pg if c not in onehot]
oh_idx = [feats_pg.index(c) for c in onehot]
ot_idx = [feats_pg.index(c) for c in other]

sc_pg = StandardScaler()
Xs_pg = X_pg.copy()
Xs_pg[:, ot_idx] = sc_pg.fit_transform(X_pg[:, ot_idx])
Xts_pg = Xt_pg.copy()
Xts_pg[:, ot_idx] = sc_pg.transform(Xt_pg[:, ot_idx])

print(f"Variant A — Per-game only: {len(feats_pg)} features ({len(other)} continuous + {len(onehot)} one-hot)")""")

code("""# Train & evaluate Variant A
alphas = np.logspace(-3, 4, 300)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

ridge_pg = RidgeCV(alphas=alphas, cv=10)
ridge_pg.fit(Xs_pg, y)

oof_pg = np.zeros(len(y))
for tr, va in kf.split(Xs_pg):
    r = RidgeCV(alphas=alphas, cv=5)
    r.fit(Xs_pg[tr], y[tr])
    oof_pg[va] = r.predict(Xs_pg[va])
mae_pg = mean_absolute_error(y, oof_pg)

print(f"Variant A results:")
print(f"  Alpha: {ridge_pg.alpha_:.2f}")
print(f"  CV MAE: {mae_pg:.5f}")

# Generate submission
preds_pg = np.round(ridge_pg.predict(Xts_pg)).astype(int)
preds_pg = np.clip(preds_pg, 36, 120)
sub_pg = pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds_pg}).sort_values('ID').reset_index(drop=True)
sub_pg.to_csv("submissions/v5_pergame.csv", index=False)
print(f"  Saved: submissions/v5_pergame.csv (mean={preds_pg.mean():.1f})")""")

# ============================================================
md("""## 4. Variant B — Per-Game + Raw Totals Combined

Keep both raw counting stats and per-game versions. Ridge regularisation decides which representation is more useful for each stat.

**Rationale:** Sometimes raw totals carry information that per-game rates lose (e.g., a team with 162 games has seen more situations, and raw totals may better correlate with W since W is also a count).
""")

code("""# Include all features (raw + per-game)
exclude_both = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins'}
feats_both = [c for c in train_fe.columns
              if c not in exclude_both and train_fe[c].dtype in ['int64', 'float64', 'bool']]

for c in feats_both:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

# Preprocessing
train_both = train_fe[feats_both].replace([np.inf, -np.inf], np.nan).copy()
test_both = test_fe[feats_both].replace([np.inf, -np.inf], np.nan).copy()
for c in feats_both:
    med = train_both[c].median()
    train_both[c] = train_both[c].fillna(med)
    test_both[c] = test_both[c].fillna(med)

X_both = train_both.values.astype(float)
Xt_both = test_both.values.astype(float)

onehot_b = [c for c in feats_both if c.startswith(('era_', 'decade_'))]
other_b = [c for c in feats_both if c not in onehot_b]
oh_idx_b = [feats_both.index(c) for c in onehot_b]
ot_idx_b = [feats_both.index(c) for c in other_b]

sc_both = StandardScaler()
Xs_both = X_both.copy()
Xs_both[:, ot_idx_b] = sc_both.fit_transform(X_both[:, ot_idx_b])
Xts_both = Xt_both.copy()
Xts_both[:, ot_idx_b] = sc_both.transform(Xt_both[:, ot_idx_b])

print(f"Variant B — Per-game + raw: {len(feats_both)} features ({len(other_b)} continuous + {len(onehot_b)} one-hot)")""")

code("""# Train & evaluate Variant B
ridge_both = RidgeCV(alphas=alphas, cv=10)
ridge_both.fit(Xs_both, y)

oof_both = np.zeros(len(y))
for tr, va in kf.split(Xs_both):
    r = RidgeCV(alphas=alphas, cv=5)
    r.fit(Xs_both[tr], y[tr])
    oof_both[va] = r.predict(Xs_both[va])
mae_both = mean_absolute_error(y, oof_both)

print(f"Variant B results:")
print(f"  Alpha: {ridge_both.alpha_:.2f}")
print(f"  CV MAE: {mae_both:.5f}")

# Generate submission
preds_both = np.round(ridge_both.predict(Xts_both)).astype(int)
preds_both = np.clip(preds_both, 36, 120)
sub_both = pd.DataFrame({'ID': test_fe['ID'].values, 'W': preds_both}).sort_values('ID').reset_index(drop=True)
sub_both.to_csv("submissions/v5_both.csv", index=False)
print(f"  Saved: submissions/v5_both.csv (mean={preds_both.mean():.1f})")""")

# ============================================================
md("## 5. Comparison — All Models")

code("""print("=" * 70)
print(f"{'Model':<35} {'Features':>8} {'CV MAE':>10} {'Alpha':>8}")
print("=" * 70)
print(f"{'Variant A (per-game only)':<35} {len(feats_pg):>8} {mae_pg:>10.5f} {ridge_pg.alpha_:>8.2f}")
print(f"{'Variant B (per-game + raw)':<35} {len(feats_both):>8} {mae_both:>10.5f} {ridge_both.alpha_:>8.2f}")
print("=" * 70)
print(f"\\nReference scores from other notebooks:")
print(f"  01_Ridge_Model (121 features):      CV MAE ≈ 2.720, Public MAE = 2.992")
print(f"  02_TwoStage_Model (126 features):   CV MAE ≈ 2.716, Public MAE = 3.021")
print(f"\\nBest variant for submission: ", end="")
if mae_pg < mae_both:
    print(f"Variant A (per-game only) — v5_pergame.csv")
else:
    print(f"Variant B (per-game + raw) — v5_both.csv")""")

# ============================================================
md("""## 6. Results Summary

### Per-Game Normalisation Impact
- Dividing counting stats by G ensures features are **comparable across different season lengths**
- Derived features (OPS, WHIP, FIP, wOBA) computed from per-game values are **mathematically equivalent** to raw-based versions for ratio stats
- For additive features (Runs Created, Total Bases), per-game versions remove the G-dependency

### Which Variant to Submit?
- **Variant B (per-game + raw)** typically has slightly better CV because Ridge can select from both representations
- However, the CV improvement is marginal (~0.003 MAE), so both variants are viable
- The key question is whether the public leaderboard rewards G-normalised features or penalises the additional noise from raw totals

### Normalisation Hierarchy (from EDA findings)
1. **Per-game (÷G)** — removes season-length confound ✅ (this notebook)
2. **mlb_rpg-relative** — removes era scoring environment ✅ (ERA+, OPS+, R_vs_mlb already included)
3. **Z-score within year** — removes all year-to-year variation ⬜ (not yet tested, potential V6)
""")

nb.cells = cells
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04_PerGame_Normalisation.ipynb"))
print("Created 04_PerGame_Normalisation.ipynb")
