"""
EDA & Data Profiling for Moneyball Kaggle Competition
Goal: Predict W (wins) — evaluated by MAE
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ is one level down)

# Load all files
train = pd.read_csv(f"{DATA_DIR}/data_year_team_franchise.csv")
test = pd.read_csv(f"{DATA_DIR}/predict_year_team_franchise.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("=" * 60)
print("SHAPES")
print("=" * 60)
print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Sample submission: {sample_sub.shape}")

print("\n" + "=" * 60)
print("TRAIN COLUMNS")
print("=" * 60)
print(train.columns.tolist())

print("\n" + "=" * 60)
print("TEST COLUMNS")
print("=" * 60)
print(test.columns.tolist())

# Columns in train but not test
train_only = set(train.columns) - set(test.columns)
test_only = set(test.columns) - set(train.columns)
print(f"\nTrain-only columns: {train_only}")
print(f"Test-only columns:  {test_only}")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(train.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES (TRAIN)")
print("=" * 60)
missing = train.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

print("\n" + "=" * 60)
print("MISSING VALUES (TEST)")
print("=" * 60)
missing_test = test.isnull().sum()
print(missing_test[missing_test > 0] if missing_test.sum() > 0 else "No missing values")

print("\n" + "=" * 60)
print("TARGET VARIABLE (W) STATS")
print("=" * 60)
print(train['W'].describe())
print(f"\nSkewness: {train['W'].skew():.4f}")
print(f"Kurtosis: {train['W'].kurtosis():.4f}")

print("\n" + "=" * 60)
print("NUMERIC FEATURES SUMMARY")
print("=" * 60)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
print(train[numeric_cols].describe().T.to_string())

print("\n" + "=" * 60)
print("CATEGORICAL/OBJECT COLUMNS")
print("=" * 60)
obj_cols = train.select_dtypes(include=['object', 'bool']).columns.tolist()
for col in obj_cols:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts().head(10))

print("\n" + "=" * 60)
print("CORRELATIONS WITH W (top 20)")
print("=" * 60)
corr_with_w = train[numeric_cols].corr()['W'].drop('W').abs().sort_values(ascending=False)
print(corr_with_w.head(20).to_string())

print("\n" + "=" * 60)
print("YEAR RANGE")
print("=" * 60)
print(f"Train: {train['yearID'].min()} - {train['yearID'].max()}")
print(f"Test:  {test['yearID'].min()} - {test['yearID'].max()}")

print("\n" + "=" * 60)
print("TEAM/FRANCHISE COUNTS")
print("=" * 60)
print(f"Train teams: {train['teamID'].nunique()}, franchises: {train['franchID'].nunique()}")
print(f"Test teams:  {test['teamID'].nunique()}, franchises: {test['franchID'].nunique()}")

# Check if test teams/franchises are all seen in train
test_teams_unseen = set(test['teamID']) - set(train['teamID'])
test_franch_unseen = set(test['franchID']) - set(train['franchID'])
print(f"\nUnseen test teams: {test_teams_unseen if test_teams_unseen else 'None'}")
print(f"Unseen test franchises: {test_franch_unseen if test_franch_unseen else 'None'}")

print("\n" + "=" * 60)
print("GAMES PLAYED (G) DISTRIBUTION")
print("=" * 60)
print(f"Train G: min={train['G'].min()}, max={train['G'].max()}, median={train['G'].median()}")
print(f"Test G:  min={test['G'].min()}, max={test['G'].max()}, median={test['G'].median()}")

print("\n" + "=" * 60)
print("FEATURE CORRELATIONS (pairwise > 0.95)")
print("=" * 60)
# Check for highly correlated feature pairs
feature_cols = [c for c in numeric_cols if c not in ['W', 'ID', 'yearID']]
corr_matrix = train[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col, row, upper.loc[row, col])
             for col in upper.columns for row in upper.index
             if upper.loc[row, col] > 0.95]
high_corr.sort(key=lambda x: x[2], reverse=True)
for c1, c2, corr in high_corr:
    print(f"  {c1} <-> {c2}: {corr:.4f}")

print("\n" + "=" * 60)
print("SAMPLE SUBMISSION CHECK")
print("=" * 60)
print(sample_sub.head())
print(f"\nTest IDs range: {test['ID'].min()} - {test['ID'].max()}")
print(f"Sample sub IDs range: {sample_sub['ID'].min()} - {sample_sub['ID'].max()}")
print(f"IDs match: {set(test['ID']) == set(sample_sub['ID'])}")
