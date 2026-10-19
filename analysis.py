import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data_year_team_franchise.csv')
era_1 = df[df['era_1'] == True].copy()
other_eras = df[df['era_1'] == False].copy()

print('DEAD-BALL ERA ANALYSIS')
print('='*80)

era_1['pyth_pct'] = (era_1['R']**2) / (era_1['R']**2 + era_1['RA']**2)
era_1['W_per_G'] = era_1['W'] / era_1['G']
era_1['run_diff_pg'] = (era_1['R'] - era_1['RA']) / era_1['G']
era_1['R_per_G'] = era_1['R'] / era_1['G']
era_1['RA_per_G'] = era_1['RA'] / era_1['G']

other_eras['W_per_G'] = other_eras['W'] / other_eras['G']
other_eras['R_per_G'] = other_eras['R'] / other_eras['G']
other_eras['RA_per_G'] = other_eras['RA'] / other_eras['G']

print('Q1: Pythagorean Win Pct vs W/G')
clean1 = era_1[['pyth_pct', 'W_per_G']].dropna()
print('Correlation:', clean1['pyth_pct'].corr(clean1['W_per_G']))
X = clean1['pyth_pct'].values
y = clean1['W_per_G'].values
z1 = np.polyfit(X, y, 1)
z2 = np.polyfit(X, y, 2)
r2_lin = 1 - np.sum((y-np.polyval(z1, X))**2) / np.sum((y-y.mean())**2)
r2_quad = 1 - np.sum((y-np.polyval(z2, X))**2) / np.sum((y-y.mean())**2)
print('Linear R2:', r2_lin, '| Quad R2:', r2_quad, '| Delta:', r2_quad-r2_lin)
print('Result: LINEAR' if abs(r2_quad-r2_lin) < 0.02 else 'Result: CURVED')

print()
print('Q2: Run Differential per Game vs W/G')
clean2 = era_1[['run_diff_pg', 'W_per_G']].dropna()
print('Correlation:', clean2['run_diff_pg'].corr(clean2['W_per_G']))
X = clean2['run_diff_pg'].values
y = clean2['W_per_G'].values
z1 = np.polyfit(X, y, 1)
z2 = np.polyfit(X, y, 2)
r2_lin = 1 - np.sum((y-np.polyval(z1, X))**2) / np.sum((y-y.mean())**2)
r2_quad = 1 - np.sum((y-np.polyval(z2, X))**2) / np.sum((y-y.mean())**2)
print('Linear R2:', r2_lin, '| Quad R2:', r2_quad, '| Delta:', r2_quad-r2_lin)

print()
print('Q3: Nonlinear Features')
for feat in ['R_per_G', 'RA_per_G', 'ERA']:
    clean = era_1[[feat, 'W_per_G']].dropna()
    X = clean[feat].values
    y = clean['W_per_G'].values
    z1 = np.polyfit(X, y, 1)
    z2 = np.polyfit(X, y, 2)
    r2_lin = 1 - np.sum((y-np.polyval(z1, X))**2) / np.sum((y-y.mean())**2)
    r2_quad = 1 - np.sum((y-np.polyval(z2, X))**2) / np.sum((y-y.mean())**2)
    print(f'{feat:10} dR2: {r2_quad-r2_lin:.4f}')

print()
print('Q4: W/G Distribution')
print('era_1 - Mean:', era_1['W_per_G'].mean(), 'Std:', era_1['W_per_G'].std())
print('other - Mean:', other_eras['W_per_G'].mean(), 'Std:', other_eras['W_per_G'].std())
cv1 = era_1['W_per_G'].std() / era_1['W_per_G'].mean()
cv2 = other_eras['W_per_G'].std() / other_eras['W_per_G'].mean()
print('CV - era_1:', cv1, '| CV - other:', cv2)
print('Result: MORE spread' if cv1 > cv2 else 'Result: LESS spread')

print()
print('Q5: Key Stats')
era_2 = df[df['era_1'] == False].copy()
era_2['R_per_G'] = era_2['R'] / era_2['G']
era_2['RA_per_G'] = era_2['RA'] / era_2['G']
r_pct = (era_1['R_per_G'].mean() - era_2['R_per_G'].mean()) / era_2['R_per_G'].mean() * 100
ra_pct = (era_1['RA_per_G'].mean() - era_2['RA_per_G'].mean()) / era_2['RA_per_G'].mean() * 100
era_pct = (era_1['ERA'].mean() - era_2['ERA'].mean()) / era_2['ERA'].mean() * 100
print(f'R/G: {r_pct:+.1f}% | RA/G: {ra_pct:+.1f}% | ERA: {era_pct:+.1f}%')

print()
print('Q6: Outliers')
print('Top W/G:')
for idx, row in era_1.nlargest(3, 'W_per_G').iterrows():
    print(f'  {int(row["yearID"])} {row["teamID"]} | W/G: {row["W_per_G"]:.4f}')
print('Bottom W/G:')
for idx, row in era_1.nsmallest(3, 'W_per_G').iterrows():
    print(f'  {int(row["yearID"])} {row["teamID"]} | W/G: {row["W_per_G"]:.4f}')

