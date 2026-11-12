import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------- 0. Load ----------
FILE = 'sales_data_sample.csv'
df = pd.read_csv(FILE, encoding='latin1')
print("Initial shape:", df.shape)

quant_cols = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'SALES']
cat_cols   = ['PRODUCTLINE', 'DEALSIZE', 'COUNTRY', 'STATUS']

# ---------- 1. Structural audit and cleaning ----------
print(df[quant_cols + cat_cols].dtypes)
print(df[quant_cols + cat_cols].isnull().sum())
d = df[quant_cols + cat_cols].dropna().reset_index(drop=True)
print("Clean shape:", d.shape)
print("corr(SALES, QTY*PRICE):", d['SALES'].corr(d['QUANTITYORDERED'] * d['PRICEEACH']))

# ---------- 2. Univariate audit ----------
desc = d[quant_cols].agg(['count', 'mean', 'std', 'median', 'min', 'max']).T
desc['skewness'] = d[quant_cols].skew()
print(desc.round(3))

for c in cat_cols:
    f = d[c].value_counts(normalize=True) * 100
    print(f"\n{c} ({d[c].nunique()} modalities)")
    print(f.round(2).to_string())
    rare = f[f < 2].index.tolist()
    if rare:
        print("  Rare modalities (<2%):", rare)

d[quant_cols].hist(bins=30, figsize=(10, 6))
plt.tight_layout(); plt.show()

# ---------- 3. Bivariate analysis ----------
corr = d[quant_cols].corr()
print(corr.round(3))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(4), quant_cols, rotation=45); plt.yticks(range(4), quant_cols)
plt.colorbar(); plt.title("Pearson correlation"); plt.show()

ct = pd.crosstab(d['PRODUCTLINE'], d['DEALSIZE'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(ct)
print(f"Chi2={chi2:.2f}, dof={dof}, p={p:.3e}, Cramer's V={cramers_v:.3f}")

# ---------- 4. Multivariate outliers (Mahalanobis) ----------
X3 = d[['QUANTITYORDERED', 'PRICEEACH', 'MSRP']]
diff = X3 - X3.mean()
inv_cov = np.linalg.inv(np.cov(X3.values.T))
d2 = np.einsum('ij,jk,ik->i', diff.values, inv_cov, diff.values)
thr = stats.chi2.ppf(0.999, df=3)
print("Outliers (Mahalanobis, p<0.001):", (d2 > thr).sum(), "of", len(d))
# To drop them: d = d[d2 <= thr].reset_index(drop=True)

# ---------- 5. OLS: raw model ----------
y = d['SALES']
X = sm.add_constant(d[['QUANTITYORDERED', 'PRICEEACH', 'MSRP']])
ols = sm.OLS(y, X).fit()
print(ols.summary())

vif_df = pd.DataFrame({'Variable': X.columns,
                       'VIF': [vif(X.values, i) for i in range(X.shape[1])]})
print(vif_df.to_string(index=False))

jb = stats.jarque_bera(ols.resid)
bp = het_breuschpagan(ols.resid, X)
print(f"Jarque-Bera: stat={jb[0]:.1f}, p={jb[1]:.3g}")
print(f"Breusch-Pagan: LM p-value={bp[1]:.3g}")
print(f"Durbin-Watson: {durbin_watson(ols.resid):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(ols.fittedvalues, ols.resid, alpha=0.3, s=10)
ax[0].axhline(0, color='k'); ax[0].set_title("Residuals vs fitted")
sm.qqplot(ols.resid, line='s', ax=ax[1]); ax[1].set_title("QQ plot")
plt.tight_layout(); plt.show()

# ---------- 6. OLS: robust (HC3) and log-log ----------
ols_rob = sm.OLS(y, X).fit(cov_type='HC3')
print(ols_rob.summary().tables[1])

Xl = sm.add_constant(np.log(d[['QUANTITYORDERED', 'PRICEEACH', 'MSRP']]))
ols_log = sm.OLS(np.log(y), Xl).fit(cov_type='HC3')
print(ols_log.summary())

# ---------- 7. PCA on the 3 explanatory variables ----------
exp_cols = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP']
Z3 = StandardScaler().fit_transform(d[exp_cols])
pca3 = PCA().fit(Z3)

eig = pca3.explained_variance_
ratio = pca3.explained_variance_ratio_ * 100
print(pd.DataFrame({'Axis': [f'PC{i+1}' for i in range(3)],
                    'Eigenvalue': eig.round(4),
                    'Inertia %': ratio.round(2),
                    'Cumulative %': ratio.cumsum().round(2),
                    'Kaiser (>1)': eig > 1}).to_string(index=False))

plt.bar(range(1, 4), eig); plt.axhline(1, color='r', ls='--')
plt.xlabel("Axis"); plt.ylabel("Eigenvalue"); plt.title("Scree plot"); plt.show()

loadings3 = pd.DataFrame(pca3.components_.T, index=exp_cols,
                         columns=[f'PC{i+1}' for i in range(3)])
print(loadings3.round(3))

# ---------- 8. PCR ----------
K = int((eig > 1).sum())
print("Components retained:", K)
scores = PCA(n_components=K).fit_transform(Z3)
Xp = sm.add_constant(pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(K)]))
pcr = sm.OLS(y.values, Xp).fit()
print(pcr.summary())
print(pd.DataFrame({'Variable': Xp.columns,
                    'VIF': [vif(Xp.values, i) for i in range(Xp.shape[1])]}).to_string(index=False))

# ---------- 9. PCA on 4 variables: circle, individuals, biplot ----------
Z4 = StandardScaler().fit_transform(d[quant_cols])
pca4 = PCA(n_components=2)
coords = pca4.fit_transform(Z4)
load4 = pca4.components_.T * np.sqrt(pca4.explained_variance_)
v1, v2 = pca4.explained_variance_ratio_ * 100
print(f"PC1 {v1:.2f}% + PC2 {v2:.2f}% = {v1+v2:.2f}%")
print(pd.DataFrame(load4, index=quant_cols, columns=['PC1', 'PC2']).round(3))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
a1.add_patch(plt.Circle((0, 0), 1, color='gray', fill=False, ls='--'))
for i, v in enumerate(quant_cols):
    a1.arrow(0, 0, load4[i, 0], load4[i, 1], head_width=0.03, fc='red', ec='red')
    a1.text(load4[i, 0] * 1.1, load4[i, 1] * 1.1, v, color='darkred', fontsize=9)
a1.axhline(0, color='grey'); a1.axvline(0, color='grey')
a1.set_xlim(-1.3, 1.3); a1.set_ylim(-1.3, 1.3); a1.set_aspect('equal')
a1.set_title("Correlation circle")
a1.set_xlabel(f"Axis 1 ({v1:.2f}%)"); a1.set_ylabel(f"Axis 2 ({v2:.2f}%)")

a2.scatter(coords[:, 0], coords[:, 1], alpha=0.3, s=10, color='royalblue')
a2.axhline(0, color='grey'); a2.axvline(0, color='grey')
a2.set_title("Individuals (transactions)")
a2.set_xlabel(f"Axis 1 ({v1:.2f}%)"); a2.set_ylabel(f"Axis 2 ({v2:.2f}%)")
plt.tight_layout(); plt.show()

sf = 3.0
plt.figure(figsize=(9, 9))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.25, s=12, color='royalblue')
for i, v in enumerate(quant_cols):
    plt.arrow(0, 0, load4[i, 0] * sf, load4[i, 1] * sf, head_width=0.08, fc='red', ec='red')
    plt.text(load4[i, 0] * sf * 1.1, load4[i, 1] * sf * 1.1, v, color='darkred', weight='bold')
plt.axhline(0, color='grey', ls='--'); plt.axvline(0, color='grey', ls='--')
plt.xlabel(f"Axis 1 ({v1:.2f}%)"); plt.ylabel(f"Axis 2 ({v2:.2f}%)")
plt.title("PCA biplot (PC1 vs PC2)"); plt.show()

# ---------- 10. Model comparison ----------
comp = pd.DataFrame({
    'Model': ['OLS raw', 'PCR', 'OLS log-log'],
    'R2': [ols.rsquared, pcr.rsquared, ols_log.rsquared],
    'AIC': [ols.aic, pcr.aic, np.nan],
    'Max VIF': [vif_df['VIF'][1:].max(), 1.0, np.nan]
})
print(comp.round(3).to_string(index=False))
