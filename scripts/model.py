"""
Moneyball Kaggle Competition — Full Pipeline
Predict W (wins) | Metric: MAE | Target: beat 2.959
Built with sabermetric domain knowledge.
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.base import clone
from scipy.optimize import minimize

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. FEATURE ENGINEERING (Sabermetric-driven)
# ============================================================
def engineer_features(df):
    """Create comprehensive baseball sabermetric features."""
    df = df.copy()

    # --- Intermediate calculations ---
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']
    df['IP'] = df['IPouts'] / 3
    PA = df['AB'] + df['BB']  # plate appearances (approx, no HBP/SF)

    # ==================== BATTING ====================
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

    # wOBA (weighted On-Base Average — best single batting metric)
    df['wOBA'] = (0.69*df['BB'] + 0.89*df['1B'] + 1.27*df['2B']
                  + 1.62*df['3B'] + 2.10*df['HR']) / PA

    # BABIP (luck indicator)
    df['BABIP'] = (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + 1)

    # True K% and BB% (using PA denominator)
    df['K_pct'] = df['SO'] / PA
    df['BB_pct'] = df['BB'] / PA
    df['batting_eye'] = df['BB_pct'] - df['K_pct']

    # Runs Created (Bill James)
    df['RC'] = (df['H'] + df['BB']) * df['TB'] / PA
    df['RC_per_game'] = df['RC'] / df['G']
    df['RC_efficiency'] = df['R'] / (df['RC'] + 1)

    # Secondary Average (non-BA offensive production)
    df['SecA'] = (df['TB'] - df['H'] + df['BB'] + df['SB']) / df['AB']

    # Power-Speed Number (Bill James)
    df['power_speed'] = (2 * df['HR'] * df['SB']) / (df['HR'] + df['SB'] + 1)

    # ==================== PITCHING ====================
    df['WHIP'] = (df['BBA'] + df['HA']) / df['IP']
    df['H9'] = 9 * df['HA'] / df['IP']
    df['HR9'] = 9 * df['HRA'] / df['IP']
    df['BB9'] = 9 * df['BBA'] / df['IP']
    df['SO9'] = 9 * df['SOA'] / df['IP']
    df['SO_BB_ratio'] = df['SOA'] / (df['BBA'] + 1)

    # FIP (Fielding Independent Pitching — best advanced pitching metric)
    df['FIP'] = (13*df['HRA'] + 3*df['BBA'] - 2*df['SOA']) / df['IP'] + 3.10

    # BABIP against (pitching luck)
    df['BABIP_against'] = (df['HA'] - df['HRA']) / (df['IP']*3 - df['SOA'] - df['HRA'] + 1)

    # HR/flyball proxy
    df['HR_flyball_proxy'] = df['HRA'] / (df['HA'] - df['HRA'] + 1)

    # Pitching dominance & resilience
    df['pitch_dominance'] = df['SOA'] / (df['HA'] + df['BBA'] + 1)
    df['pitch_resilience'] = df['SO9'] / (df['WHIP'] + 0.001)
    df['FIP_ERA_diff'] = df['FIP'] - df['ERA']

    # ==================== RUN DIFFERENTIAL (TIER 1) ====================
    df['run_diff'] = df['R'] - df['RA']
    df['R_per_game'] = df['R'] / df['G']
    df['RA_per_game'] = df['RA'] / df['G']
    df['run_diff_per_game'] = df['run_diff'] / df['G']

    # Pythagorean Win Expectation — 3 variants
    # Original (exponent = 2)
    df['pyth_exp'] = df['R']**2 / (df['R']**2 + df['RA']**2)
    df['pyth_wins'] = df['pyth_exp'] * df['G']

    # Bill James revised (exponent = 1.83)
    df['pyth_exp_183'] = df['R']**1.83 / (df['R']**1.83 + df['RA']**1.83)
    df['pyth_wins_183'] = df['pyth_exp_183'] * df['G']

    # Pythagenpat (Baseball Prospectus — dynamic exponent, best variant)
    rpg = (df['R'] + df['RA']) / df['G']
    pyth_exp_dyn = rpg ** 0.287
    df['pyth_exp_pat'] = df['R']**pyth_exp_dyn / (df['R']**pyth_exp_dyn + df['RA']**pyth_exp_dyn)
    df['pyth_wins_pat'] = df['pyth_exp_pat'] * df['G']

    # Additional run-based
    df['log_run_ratio'] = np.log(df['R'] / (df['RA'] + 1))
    df['run_diff_sq'] = df['run_diff']**2 * np.sign(df['run_diff'])
    df['R_above_RC'] = df['R'] - df['RC']
    df['run_support_era'] = df['R_per_game'] / (df['ERA'] + 0.001)

    # ==================== FIELDING ====================
    df['DER'] = 1 - ((df['HA'] - df['HRA']) / (df['IPouts'] + df['HA'] - df['HRA'] - df['SOA']))
    df['E_per_game'] = df['E'] / df['G']
    df['DP_per_game'] = df['DP'] / df['G']
    df['DP_pct'] = df['DP'] / (df['HA'] + df['BBA'] - df['HRA'] + 1)
    df['UER'] = df['RA'] - df['ER']
    df['UER_per_game'] = df['UER'] / df['G']
    df['UER_pct'] = df['UER'] / (df['RA'] + 1)

    # ==================== ERA-ADJUSTED ====================
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

    # ==================== EFFICIENCY ====================
    df['runs_per_hit'] = df['R'] / (df['H'] + 1)
    df['ER_ratio'] = df['ER'] / (df['RA'] + 1)
    df['CG_pct'] = df['CG'] / df['G']
    df['SV_pct'] = df['SV'] / df['G']
    df['SHO_pct'] = df['SHO'] / df['G']

    # ==================== INTERACTIONS ====================
    df['OPS_x_FIP'] = df['OPS'] * (1 / (df['FIP'] + 0.001))
    df['wOBA_x_WHIP'] = df['wOBA'] / (df['WHIP'] + 0.001)
    df['pyth_spread'] = df['pyth_wins_pat'] - df['pyth_wins_183']
    df['team_quality'] = (df['OPS'] - df['WHIP']) * 100

    # ==================== FRANCHISE ENCODING ====================
    franch_counts = df['franchID'].value_counts().to_dict()
    df['franch_freq'] = df['franchID'].map(franch_counts)

    return df


train_fe = engineer_features(train)
test_fe = engineer_features(test)

# ============================================================
# 3. FEATURE SELECTION
# ============================================================
exclude_cols = {'W', 'ID', 'yearID', 'teamID', 'franchID',
                'year_label', 'decade_label', 'win_bins'}

feature_cols = [c for c in train_fe.columns
                if c not in exclude_cols
                and (train_fe[c].dtype in ['int64', 'float64', 'bool'])]

for c in feature_cols:
    if train_fe[c].dtype == 'bool':
        train_fe[c] = train_fe[c].astype(int)
        test_fe[c] = test_fe[c].astype(int)

# Handle inf/nan
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

print(f"Features: {len(feature_cols)}")
print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

# ============================================================
# 4. CROSS-VALIDATION SETUP
# ============================================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

def evaluate_model(model, name, X_data, y_data):
    scores = cross_val_score(model, X_data, y_data, cv=kf, scoring=mae_scorer)
    mae = -scores.mean()
    std = scores.std()
    print(f"  {name:35s} MAE: {mae:.5f} (+/- {std:.5f})")
    return mae

# ============================================================
# 5. SCALE FEATURES (for linear models)
# ============================================================
onehot_cols = [c for c in feature_cols if c.startswith(('era_', 'decade_'))]
other_cols = [c for c in feature_cols if c not in onehot_cols]
onehot_idx = [feature_cols.index(c) for c in onehot_cols]
other_idx = [feature_cols.index(c) for c in other_cols]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:, other_idx] = scaler.fit_transform(X[:, other_idx])
X_test_scaled = X_test.copy()
X_test_scaled[:, other_idx] = scaler.transform(X_test[:, other_idx])

# ============================================================
# 6. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON (10-fold CV)")
print("=" * 60)

alphas = np.logspace(-3, 3, 100)

evaluate_model(RidgeCV(alphas=alphas, cv=5), "RidgeCV", X_scaled, y)
evaluate_model(LassoCV(alphas=alphas, cv=5, max_iter=10000), "LassoCV", X_scaled, y)
evaluate_model(RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5,
               random_state=42, n_jobs=-1), "Random Forest", X, y)
evaluate_model(GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
               subsample=0.8, min_samples_leaf=5, random_state=42), "Gradient Boosting", X, y)
evaluate_model(XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
               min_child_weight=5, random_state=42, verbosity=0), "XGBoost (baseline)", X, y)
evaluate_model(LGBMRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
               min_child_weight=5, random_state=42, verbose=-1), "LightGBM (baseline)", X, y)

# ============================================================
# 7. HYPERPARAMETER SEARCH
# ============================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER SEARCH — XGBoost")
print("=" * 60)

xgb_configs = [
    {'n_estimators': 1500, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 3},
    {'n_estimators': 2000, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.85,
     'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5, 'min_child_weight': 5},
    {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.7,
     'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 7},
    {'n_estimators': 1500, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.75,
     'colsample_bytree': 0.65, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 4},
    {'n_estimators': 2000, 'max_depth': 4, 'learning_rate': 0.01, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 2.0, 'min_child_weight': 5},
    {'n_estimators': 3000, 'max_depth': 3, 'learning_rate': 0.01, 'subsample': 0.85,
     'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 3.0, 'min_child_weight': 3},
]

best_xgb_mae = 999
best_xgb_cfg = None
for i, cfg in enumerate(xgb_configs):
    m = XGBRegressor(**cfg, random_state=42, verbosity=0)
    mae = evaluate_model(m, f"XGB config {i+1}", X, y)
    if mae < best_xgb_mae:
        best_xgb_mae = mae
        best_xgb_cfg = cfg.copy()

print(f"\n  Best XGB CV MAE: {best_xgb_mae:.5f}")
print(f"  Best config: {best_xgb_cfg}")

print("\n" + "=" * 60)
print("HYPERPARAMETER SEARCH — LightGBM")
print("=" * 60)

lgb_configs = [
    {'n_estimators': 1500, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 3,
     'num_leaves': 31},
    {'n_estimators': 2000, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.85,
     'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5, 'min_child_weight': 5,
     'num_leaves': 15},
    {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.7,
     'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 7,
     'num_leaves': 50},
    {'n_estimators': 1500, 'max_depth': -1, 'learning_rate': 0.03, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 5,
     'num_leaves': 20},
    {'n_estimators': 2000, 'max_depth': 4, 'learning_rate': 0.01, 'subsample': 0.85,
     'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5, 'min_child_weight': 3,
     'num_leaves': 25},
    {'n_estimators': 3000, 'max_depth': 3, 'learning_rate': 0.01, 'subsample': 0.8,
     'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 3.0, 'min_child_weight': 5,
     'num_leaves': 15},
]

best_lgb_mae = 999
best_lgb_cfg = None
for i, cfg in enumerate(lgb_configs):
    m = LGBMRegressor(**cfg, random_state=42, verbose=-1)
    mae = evaluate_model(m, f"LGB config {i+1}", X, y)
    if mae < best_lgb_mae:
        best_lgb_mae = mae
        best_lgb_cfg = cfg.copy()

print(f"\n  Best LGB CV MAE: {best_lgb_mae:.5f}")
print(f"  Best config: {best_lgb_cfg}")

# ============================================================
# 8. ENSEMBLE via Out-of-Fold Predictions
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE (OOF Blending)")
print("=" * 60)

def get_oof_predictions(model, X_data, y_data, X_test_data, n_splits=10):
    kf_oof = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_data))
    test_preds = np.zeros(len(X_test_data))

    for fold, (train_idx, val_idx) in enumerate(kf_oof.split(X_data)):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]
        m = clone(model)
        m.fit(X_tr, y_tr)
        oof_preds[val_idx] = m.predict(X_val)
        test_preds += m.predict(X_test_data) / n_splits

    oof_mae = mean_absolute_error(y_data, oof_preds)
    return oof_preds, test_preds, oof_mae

# Build ensemble candidates
best_xgb = XGBRegressor(**best_xgb_cfg, random_state=42, verbosity=0)
best_lgb = LGBMRegressor(**best_lgb_cfg, random_state=42, verbose=-1)
ridge = RidgeCV(alphas=alphas, cv=5)
gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, min_samples_leaf=5, random_state=42)
rf = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)

# Additional diverse XGB/LGB with different seeds for variance reduction
xgb_s2 = XGBRegressor(**best_xgb_cfg, random_state=123, verbosity=0)
lgb_s2 = LGBMRegressor(**best_lgb_cfg, random_state=123, verbose=-1)

ensemble_candidates = {
    'Ridge':    (ridge, X_scaled, X_test_scaled),
    'XGB':      (best_xgb, X, X_test),
    'LGB':      (best_lgb, X, X_test),
    'GBR':      (gbr, X, X_test),
    'RF':       (rf, X, X_test),
    'XGB_s2':   (xgb_s2, X, X_test),
    'LGB_s2':   (lgb_s2, X, X_test),
}

oof_results = {}
test_results = {}

for name, (model, X_data, X_test_data) in ensemble_candidates.items():
    oof_preds, test_preds, oof_mae = get_oof_predictions(model, X_data, y, X_test_data)
    oof_results[name] = oof_preds
    test_results[name] = test_preds
    print(f"  {name:20s} OOF MAE: {oof_mae:.5f}")

# ============================================================
# 9. OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================
print("\n" + "=" * 60)
print("WEIGHT OPTIMIZATION")
print("=" * 60)

oof_matrix = np.column_stack([oof_results[k] for k in ensemble_candidates])
test_matrix = np.column_stack([test_results[k] for k in ensemble_candidates])

def ensemble_mae(weights):
    w = np.array(weights)
    w = w / w.sum()
    preds = oof_matrix @ w
    return mean_absolute_error(y, preds)

n_models = len(ensemble_candidates)
initial_weights = [1.0 / n_models] * n_models
bounds = [(0, 1)] * n_models

# Try multiple optimization runs
best_opt_mae = 999
best_opt_weights = None

for seed in [42, 123, 456, 789, 0]:
    np.random.seed(seed)
    init_w = np.random.dirichlet(np.ones(n_models))
    result = minimize(ensemble_mae, init_w, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-10})
    if result.fun < best_opt_mae:
        best_opt_mae = result.fun
        best_opt_weights = result.x

# Also try from equal weights
result_eq = minimize(ensemble_mae, initial_weights, method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-10})
if result_eq.fun < best_opt_mae:
    best_opt_mae = result_eq.fun
    best_opt_weights = result_eq.x

opt_weights = np.array(best_opt_weights)
opt_weights = np.maximum(opt_weights, 0)  # no negative weights
opt_weights = opt_weights / opt_weights.sum()

print(f"\nOptimized ensemble weights:")
for name, w in zip(ensemble_candidates.keys(), opt_weights):
    print(f"  {name:20s}: {w:.4f}")
print(f"Ensemble OOF MAE: {best_opt_mae:.5f}")

# Compare with simple averages
print("\nSimple average comparisons:")
for combo_name, combo_keys in [
    ("All models", list(ensemble_candidates.keys())),
    ("XGB+LGB", ['XGB', 'LGB']),
    ("XGB+LGB+GBR", ['XGB', 'LGB', 'GBR']),
    ("XGB+LGB+XGB_s2+LGB_s2", ['XGB', 'LGB', 'XGB_s2', 'LGB_s2']),
    ("XGB+LGB+Ridge", ['XGB', 'LGB', 'Ridge']),
]:
    avg_oof = np.mean([oof_results[k] for k in combo_keys], axis=0)
    avg_mae = mean_absolute_error(y, avg_oof)
    print(f"  {combo_name:35s} MAE: {avg_mae:.5f}")

# ============================================================
# 10. GENERATE SUBMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSIONS")
print("=" * 60)

# Primary: optimized weights ensemble
final_preds = test_matrix @ opt_weights
final_preds_rounded = np.round(final_preds).astype(int)
final_preds_rounded = np.clip(final_preds_rounded, 36, 120)

submission = pd.DataFrame({'ID': test_ids, 'W': final_preds_rounded})
submission = submission.sort_values('ID').reset_index(drop=True)
submission.to_csv(f"{DATA_DIR}/submission_ensemble_opt.csv", index=False)
print(f"Optimized ensemble -> submission_ensemble_opt.csv")

# Also save best single model (XGB)
best_xgb_full = XGBRegressor(**best_xgb_cfg, random_state=42, verbosity=0)
best_xgb_full.fit(X, y)
xgb_preds = np.round(best_xgb_full.predict(X_test)).astype(int)
xgb_preds = np.clip(xgb_preds, 36, 120)
submission_xgb = pd.DataFrame({'ID': test_ids, 'W': xgb_preds})
submission_xgb = submission_xgb.sort_values('ID').reset_index(drop=True)
submission_xgb.to_csv(f"{DATA_DIR}/submission_xgb.csv", index=False)
print(f"Best XGB single model -> submission_xgb.csv")

# Also save simple XGB+LGB average
avg_xgb_lgb = (test_results['XGB'] + test_results['LGB'] +
               test_results['XGB_s2'] + test_results['LGB_s2']) / 4
avg_preds = np.round(avg_xgb_lgb).astype(int)
avg_preds = np.clip(avg_preds, 36, 120)
submission_avg = pd.DataFrame({'ID': test_ids, 'W': avg_preds})
submission_avg = submission_avg.sort_values('ID').reset_index(drop=True)
submission_avg.to_csv(f"{DATA_DIR}/submission_avg_xgb_lgb.csv", index=False)
print(f"XGB+LGB average -> submission_avg_xgb_lgb.csv")

# Print feature importance from best XGB
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_xgb_full.feature_importances_
}).sort_values('Importance', ascending=False)
print(f"\nTop 20 features (XGB importance):")
print(importance.head(20).to_string(index=False))

print(f"\nPrediction stats:")
print(f"  Ensemble: min={final_preds_rounded.min()}, max={final_preds_rounded.max()}, "
      f"mean={final_preds_rounded.mean():.1f}")
print(f"  XGB:      min={xgb_preds.min()}, max={xgb_preds.max()}, mean={xgb_preds.mean():.1f}")

print("\nDone! Files ready for Kaggle submission.")
