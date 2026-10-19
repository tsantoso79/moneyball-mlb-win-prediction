"""Generate 05_Era_Normalisation.ipynb — z-score within era + per-game normalisation."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "base", "language": "python", "name": "base"}

cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ============================================================
md("""# Moneyball V6: Era-Normalised Features + Per-Game Rates

**Competition:** SCTPDSAI-M3-DS2F Coaching — MoneyBall Analytics
**Metric:** MAE (Mean Absolute Error) — lower is better

## Motivation
Previous models used raw counting stats or per-game rates, but **era effects** still confound the features. A team with 0.270 BA in the dead-ball era (1910s) was elite; the same BA in the steroid era (2000s) was below average.

**Z-scoring within era** transforms each feature into "how many standard deviations above/below the era average this team was." This:
- Removes era-level shifts in scoring environment
- Preserves within-era relative team strength
- Uses larger groups than per-year z-scores (more stable estimates)

## Approach
We test **4 variants** to find the best normalisation strategy:

| Variant | Features | Description |
|---|---|---|
| A | Per-game + era z-score (no raw) | Clean normalised features only |
| B | Per-game + era z-score + raw totals | Let Ridge choose from all representations |
| C | Era z-score on raw stats (no per-game) | Simpler — z-score does the heavy lifting |
| D | Per-game + era z-score + original sabermetrics | Kitchen sink — all feature representations |
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

print(f"Training set: {train.shape[0]} team-seasons")
print(f"Test set:     {test.shape[0]} team-seasons")
print(f"Year range — Train: {train['yearID'].min()}-{train['yearID'].max()}")
print(f"Year range — Test:  {test['yearID'].min()}-{test['yearID'].max()}")""")

# ============================================================
md("""## 2. Era Labelling

The dataset has `era_1` through `era_8` as boolean columns. We create a single era label for grouping.
""")

code("""def assign_era(df):
    \"\"\"Create era_label column from one-hot era indicators.\"\"\"
    df = df.copy()
    era_cols = [c for c in df.columns if c.startswith('era_')]
    era_labels = []
    for _, row in df.iterrows():
        found = False
        for ec in era_cols:
            if row[ec]:
                era_labels.append(ec)
                found = True
                break
        if not found:
            era_labels.append('era_unknown')
    df['era_label'] = era_labels
    return df

train = assign_era(train)
test = assign_era(test)

# Show era distribution
print("Era distribution (train):")
print(train['era_label'].value_counts().sort_index().to_string())
print(f"\\nEra distribution (test):")
print(test['era_label'].value_counts().sort_index().to_string())""")

# ============================================================
md("""## 3. Feature Engineering — Per-Game + Era Z-Score

### Step 1: Per-game normalisation (÷G)
Removes season-length confound.

### Step 2: Era z-score
For each counting stat and per-game rate, compute:
```
z = (value - era_mean) / (era_std + epsilon)
```
This is computed on the **combined train+test** so that era statistics are consistent. This is safe because we only use the era mean/std (not the target W).
""")

code("""def engineer_features_v6(df, franch_counts=None, era_stats=None):
    \"\"\"Per-game normalisation + era z-score + sabermetric features.\"\"\"
    df = df.copy()
    G = df['G']

    # ====== STEP 1: Per-game normalise counting stats ======
    counting_stats = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                      'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
    for s in counting_stats:
        df[f'{s}_pg'] = df[s] / G

    # ====== STEP 2: Era z-score on counting stats AND per-game rates ======
    stats_to_zscore = counting_stats + [f'{s}_pg' for s in counting_stats] + ['ERA', 'FP']

    if era_stats is None:
        # Compute era means/stds (should be done on combined train+test)
        era_stats = {}
        for s in stats_to_zscore:
            era_stats[s] = df.groupby('era_label')[s].agg(['mean', 'std']).to_dict('index')

    for s in stats_to_zscore:
        z_vals = []
        for _, row in df.iterrows():
            era = row['era_label']
            if era in era_stats[s]:
                mean = era_stats[s][era]['mean']
                std = era_stats[s][era]['std']
                z_vals.append((row[s] - mean) / (std + 1e-6))
            else:
                z_vals.append(0)  # Unknown era — neutral
        df[f'{s}_ez'] = z_vals

    # ====== STEP 3: Sabermetric features (from per-game values) ======
    Rpg, RApg = df['R_pg'], df['RA_pg']
    Hpg, ABpg, BBpg, SOpg = df['H_pg'], df['AB_pg'], df['BB_pg'], df['SO_pg']
    HRpg, B2pg, B3pg, SBpg = df['HR_pg'], df['2B_pg'], df['3B_pg'], df['SB_pg']
    HApg, HRApg, BBApg, SOApg = df['HA_pg'], df['HRA_pg'], df['BBA_pg'], df['SOA_pg']
    IPpg = df['IPouts_pg'] / 3
    ERpg = df['ER_pg']

    # Batting
    df['1B_pg'] = Hpg - B2pg - B3pg - HRpg
    df['TB_pg'] = df['1B_pg'] + 2*B2pg + 3*B3pg + 4*HRpg
    PApg = ABpg + BBpg
    df['BA']  = Hpg / (ABpg + 0.001)
    df['SLG'] = df['TB_pg'] / (ABpg + 0.001)
    df['OBP'] = (Hpg + BBpg) / (PApg + 0.001)
    df['OPS'] = df['OBP'] + df['SLG']
    df['ISO'] = df['SLG'] - df['BA']
    df['wOBA'] = (0.69*BBpg + 0.89*df['1B_pg'] + 1.27*B2pg + 1.62*B3pg + 2.10*HRpg) / (PApg + 0.001)
    df['BABIP'] = (Hpg - HRpg) / (ABpg - SOpg - HRpg + 0.001)
    df['K_pct']  = SOpg / (PApg + 0.001)
    df['BB_pct'] = BBpg / (PApg + 0.001)
    df['batting_eye'] = df['BB_pct'] - df['K_pct']
    df['RC_pg']  = (Hpg + BBpg) * df['TB_pg'] / (PApg + 0.001)
    df['SecA']   = (df['TB_pg'] - Hpg + BBpg + SBpg) / (ABpg + 0.001)
    df['power_speed'] = (2 * HRpg * SBpg) / (HRpg + SBpg + 0.001)
    df['BB_SO_ratio'] = BBpg / (SOpg + 0.001)

    # Pitching
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

    # Run differential
    df['run_diff_pg'] = Rpg - RApg
    R, RA = df['R'], df['RA']
    df['pyth_wins']     = R**2 / (R**2 + RA**2) * G
    df['pyth_wins_183'] = R**1.83 / (R**1.83 + RA**1.83) * G
    rpg_total = (R + RA) / G
    pexp = rpg_total ** 0.287
    df['pyth_wins_pat'] = R**pexp / (R**pexp + RA**pexp) * G
    df['pyth_pct']     = R**2 / (R**2 + RA**2)
    df['pyth_pct_pat'] = R**pexp / (R**pexp + RA**pexp)
    df['log_run_ratio'] = np.log(R / (RA + 1))
    df['run_diff_sq']   = (R - RA)**2 * np.sign(R - RA)
    df['R_above_RC']    = Rpg - df['RC_pg']
    df['run_support_era'] = Rpg / (df['ERA'] + 0.001)

    # Fielding
    df['DER'] = 1 - ((HApg - HRApg) / (df['IPouts_pg'] + HApg - HRApg - SOApg + 0.001))
    df['UER_pg']  = RApg - ERpg
    df['UER_pct'] = df['UER_pg'] / (RApg + 0.001)

    # Era-adjusted (via mlb_rpg)
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

    # Efficiency
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    # Franchise
    if franch_counts is not None:
        df['franch_freq'] = df['franchID'].map(franch_counts).fillna(1)
    else:
        franch_counts = df['franchID'].value_counts().to_dict()
        df['franch_freq'] = df['franchID'].map(franch_counts)

    return df, franch_counts, era_stats

# Compute era stats on combined data for consistent z-scores
combined = pd.concat([train, test], ignore_index=True)
combined = assign_era(combined) if 'era_label' not in combined.columns else combined

counting_stats = ['R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
                  'RA', 'ER', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP']
pg_stats = [f'{s}_pg' for s in counting_stats]

# Pre-compute per-game on combined for era stats
for s in counting_stats:
    combined[f'{s}_pg'] = combined[s] / combined['G']

stats_to_zscore = counting_stats + pg_stats + ['ERA', 'FP']
era_stats = {}
for s in stats_to_zscore:
    era_stats[s] = combined.groupby('era_label')[s].agg(['mean', 'std']).to_dict('index')

print("Era stats computed on combined data (train+test)")
print(f"Stats z-scored: {len(stats_to_zscore)}")

# Engineer features
train_fe, franch_counts, _ = engineer_features_v6(train, era_stats=era_stats)
test_fe, _, _ = engineer_features_v6(test, franch_counts=franch_counts, era_stats=era_stats)
print(f"\\nTrain features: {train_fe.shape}")
print(f"Test features:  {test_fe.shape}")""")

# ============================================================
md("""## 4. Variant Testing

We test 4 feature subsets to find the optimal normalisation combination.
""")

code("""y = train_fe['W'].values
exclude = {'W', 'ID', 'yearID', 'teamID', 'franchID', 'year_label', 'decade_label', 'win_bins', 'era_label'}

# Ensure era_label and decade_label are not in feature sets (they are string columns)
for col_to_drop in ['era_label', 'decade_label']:
    if col_to_drop in train_fe.columns:
        train_fe[col_to_drop] = train_fe[col_to_drop].astype(str)
    if col_to_drop in test_fe.columns:
        test_fe[col_to_drop] = test_fe[col_to_drop].astype(str)

# Define feature sets for each variant
raw_counting = set(counting_stats)
pg_cols = {f'{s}_pg' for s in counting_stats}
ez_cols = {c for c in train_fe.columns if c.endswith('_ez')}
saber_cols = {c for c in train_fe.columns if c not in exclude and c not in raw_counting
              and c not in pg_cols and c not in ez_cols
              and train_fe[c].dtype in ['int64', 'float64', 'bool']
              and not c.startswith(('era_', 'decade_'))}
indicator_cols = {c for c in train_fe.columns if c.startswith(('era_', 'decade_'))
                  and train_fe[c].dtype in ['bool', 'int64', 'float64']}

print(f"Raw counting stats:  {len(raw_counting)}")
print(f"Per-game rates:      {len(pg_cols)}")
print(f"Era z-scores:        {len(ez_cols)}")
print(f"Sabermetric derived: {len(saber_cols)}")
print(f"Era/decade indicators: {len(indicator_cols)}")""")

code("""# Define variants
variants = {
    'A: pg + ez (no raw)':     list(pg_cols | ez_cols | saber_cols | indicator_cols),
    'B: pg + ez + raw':        list(pg_cols | ez_cols | raw_counting | saber_cols | indicator_cols),
    'C: ez + saber (no pg)':   list(ez_cols | saber_cols | indicator_cols),
    'D: all features':         list(pg_cols | ez_cols | raw_counting | saber_cols | indicator_cols),
}

# Also test: era z-score only (minimal)
ez_only = [c for c in train_fe.columns if c.endswith('_ez')]
variants['E: era z-score only'] = ez_only + list(indicator_cols)

alphas = np.logspace(-3, 4, 300)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = {}
best_mae = 999
best_variant = None

for name, feat_list in variants.items():
    # Filter to existing columns
    feat_list = [c for c in feat_list if c in train_fe.columns]

    # Prepare data
    tr = train_fe[feat_list].copy()
    te = test_fe[feat_list].copy()

    for c in feat_list:
        if tr[c].dtype == 'bool':
            tr[c] = tr[c].astype(int)
            te[c] = te[c].astype(int)

    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)
    for c in feat_list:
        med = tr[c].median()
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)

    X = tr.values.astype(float)
    Xt = te.values.astype(float)

    # Selective scaling
    oh = [c for c in feat_list if c.startswith(('era_', 'decade_'))]
    ot = [c for c in feat_list if c not in oh]
    oh_idx = [feat_list.index(c) for c in oh]
    ot_idx = [feat_list.index(c) for c in ot]

    sc = StandardScaler()
    Xs = X.copy()
    if ot_idx:
        Xs[:, ot_idx] = sc.fit_transform(X[:, ot_idx])
    Xts = Xt.copy()
    if ot_idx:
        Xts[:, ot_idx] = sc.transform(Xt[:, ot_idx])

    # Train & evaluate
    ridge = RidgeCV(alphas=alphas, cv=10)
    ridge.fit(Xs, y)

    oof = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(Xs):
        r = RidgeCV(alphas=alphas, cv=5)
        r.fit(Xs[tr_idx], y[tr_idx])
        oof[va_idx] = r.predict(Xs[va_idx])
    mae = mean_absolute_error(y, oof)

    # Predictions
    preds = np.round(ridge.predict(Xts)).astype(int)
    preds = np.clip(preds, 36, 120)

    results[name] = {
        'n_features': len(feat_list),
        'alpha': ridge.alpha_,
        'cv_mae': mae,
        'preds': preds
    }

    if mae < best_mae:
        best_mae = mae
        best_variant = name

    print(f"{name:30s}  features={len(feat_list):>4}  alpha={ridge.alpha_:>8.2f}  CV MAE={mae:.5f}")

print(f"\\nBest variant: {best_variant} (CV MAE={best_mae:.5f})")""")

# ============================================================
md("## 5. Results Comparison")

code("""print("=" * 80)
print(f"{'Variant':<35} {'Features':>8} {'Alpha':>8} {'CV MAE':>10}")
print("=" * 80)
for name, r in sorted(results.items(), key=lambda x: x[1]['cv_mae']):
    marker = " <-- BEST" if name == best_variant else ""
    print(f"{name:<35} {r['n_features']:>8} {r['alpha']:>8.2f} {r['cv_mae']:>10.5f}{marker}")
print("=" * 80)

print(f"\\nReference scores (previous notebooks):")
print(f"  01_Ridge (121 features, no norm):    CV MAE = 2.720, Public MAE = 2.992")
print(f"  04_V5 per-game only (99 features):   CV MAE = 2.719")
print(f"  04_V5 per-game + raw (120 features): CV MAE = 2.716")""")

# ============================================================
md("## 6. Generate Submissions")

code("""# Save all variant submissions
for name, r in results.items():
    # Create clean filename
    suffix = name.split(':')[0].strip().lower()
    fname = f"submissions/v6_{suffix}.csv"
    sub = pd.DataFrame({'ID': test_fe['ID'].values, 'W': r['preds']})
    sub = sub.sort_values('ID').reset_index(drop=True)
    sub.to_csv(fname, index=False)
    print(f"Saved {fname} (mean={r['preds'].mean():.1f})")

print(f"\\nRecommended submission: v6_{best_variant.split(':')[0].strip().lower()}.csv")""")

# ============================================================
md("""## 7. Analysis & Conclusions

### What Era Z-Scoring Does
- Transforms "team scored 800 runs" → "team scored 1.2 standard deviations above their era's average"
- A team +1.5σ in the dead-ball era and a team +1.5σ in the steroid era are treated as equally dominant
- This removes the confound where raw R=800 means very different things in 1910 vs 2000

### Per-Game vs Era Z-Score
| Normalisation | What It Removes | What It Keeps |
|---|---|---|
| Per-game (÷G) | Season-length differences | Era-level scoring environment |
| Era z-score | Era-level scoring environment | Within-era relative strength |
| **Both combined** | Season length AND era effects | Pure relative team quality |

### Impact on Ridge
- Era z-scores add ~44 new features (22 stats × 2: raw and per-game z-scored)
- Ridge's L2 penalty handles the increased dimensionality
- The question is whether the normalised features reduce noise enough to improve generalisation

### Comparison with Previous Approaches
The CV-to-public gap (~0.27 MAE) has been consistent across all models. If era z-scoring reduces CV MAE, the public score should improve proportionally — unless the test set has a different era distribution than training.
""")

nb.cells = cells
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "05_Era_Normalisation.ipynb"))
print("Created 05_Era_Normalisation.ipynb")
