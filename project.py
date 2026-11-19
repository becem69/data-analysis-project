#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sales Data Analysis - OLS Regression, PCA, MCA and SVM
======================================================

Single self-contained script. Run it with:

    python3 project.py | tee output.txt

Every figure is displayed with plt.show() and saved to the images/ folder.
The console output is also written to output.txt automatically.

Sections
--------
 1  Structural audit and cleaning
 2  Univariate audit
 3  Bivariate analysis (Pearson + chi-square)
 4  Multivariate outliers (Mahalanobis)
 5  OLS: raw model and diagnostics
 6  Robust (HC3) and log-log OLS
 7  PCA on the three explanatory variables
 8  Principal component regression (PCR)
 9  PCA on four variables: circle, individuals, biplot
10  Model comparison
11  Multiple Correspondence Analysis (MCA / ACM) + hierarchical clustering
12  Support Vector Machine (SVM) classification + ROC curve
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, f1_score)
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated",
                        category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")
os.makedirs(IMG_DIR, exist_ok=True)


class Tee:
    """Print to the console and to output.txt at the same time."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.console = sys.stdout

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


sys.stdout = Tee(os.path.join(HERE, "output.txt"))


def show(filename):
    """Save the current figure to images/ and display it."""
    plt.savefig(os.path.join(IMG_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()


def section(number, title):
    print("\n" + "=" * 78)
    print(f"SECTION {number} - {title}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 0. Load
# --------------------------------------------------------------------------- #
FILE = "sales_data_sample.csv"
path = FILE if os.path.exists(FILE) else os.path.join(HERE, FILE)
df = pd.read_csv(path, encoding="latin1")
print("Initial shape:", df.shape)

quant_cols = ["QUANTITYORDERED", "PRICEEACH", "MSRP", "SALES"]
cat_cols = ["PRODUCTLINE", "DEALSIZE", "COUNTRY", "STATUS"]

# --------------------------------------------------------------------------- #
# 1. Structural audit and cleaning
# --------------------------------------------------------------------------- #
section(1, "STRUCTURAL AUDIT AND CLEANING")
print(df[quant_cols + cat_cols].dtypes.to_string())
print("\nMissing values per selected column:")
print(df[quant_cols + cat_cols].isnull().sum().to_string())
d = df[quant_cols + cat_cols].dropna().reset_index(drop=True)
print("Clean shape:", d.shape)
print("corr(SALES, QUANTITYORDERED x PRICEEACH):",
      round(d["SALES"].corr(d["QUANTITYORDERED"] * d["PRICEEACH"]), 4))
print("Note: SALES is largely, but not exactly, quantity x unit price.")

# --------------------------------------------------------------------------- #
# 2. Univariate audit
# --------------------------------------------------------------------------- #
section(2, "UNIVARIATE AUDIT")
desc = d[quant_cols].agg(["count", "mean", "std", "median", "min", "max"]).T
desc["skewness"] = d[quant_cols].skew()
print(desc.round(3).to_string())

for c in cat_cols:
    f = d[c].value_counts(normalize=True) * 100
    print(f"\n{c} ({d[c].nunique()} modalities)")
    print(f.round(2).to_string())
    rare = f[f < 2].index.tolist()
    if rare:
        print("  Rare modalities (< 2%):", rare)

d[quant_cols].hist(bins=30, figsize=(10, 6))
plt.suptitle("Distributions of quantitative variables")
plt.tight_layout()
show("fig_01_histograms.png")

# --------------------------------------------------------------------------- #
# 3. Bivariate analysis
# --------------------------------------------------------------------------- #
section(3, "BIVARIATE ANALYSIS")
corr = d[quant_cols].corr()
print(corr.round(3).to_string())

plt.figure(figsize=(7, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(4), quant_cols, rotation=45)
plt.yticks(range(4), quant_cols)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar()
plt.title("Pearson correlation matrix")
plt.tight_layout()
show("fig_02_correlation_heatmap.png")

ct = pd.crosstab(d["PRODUCTLINE"], d["DEALSIZE"])
chi2, pval, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print("\nContingency table PRODUCTLINE x DEALSIZE:")
print(ct.to_string())
print(f"Chi2={chi2:.2f}, dof={dof}, p={pval:.3e}, Cramer's V={cramers_v:.3f}")

# --------------------------------------------------------------------------- #
# 4. Multivariate outliers (Mahalanobis)
# --------------------------------------------------------------------------- #
section(4, "MULTIVARIATE OUTLIERS (MAHALANOBIS)")
X3 = d[["QUANTITYORDERED", "PRICEEACH", "MSRP"]]
diff = X3 - X3.mean()
inv_cov = np.linalg.inv(np.cov(X3.values.T))
d2 = np.einsum("ij,jk,ik->i", diff.values, inv_cov, diff.values)
thr = stats.chi2.ppf(0.999, df=3)
print("Outliers (Mahalanobis, p < 0.001):", (d2 > thr).sum(), "of", len(d))
print("They are kept so that all 2,823 observations remain in the analysis.")
print("To remove them: d = d[d2 <= thr].reset_index(drop=True)")

# --------------------------------------------------------------------------- #
# 5. OLS: raw model
# --------------------------------------------------------------------------- #
section(5, "OLS: RAW MODEL")
y = d["SALES"]
X = sm.add_constant(d[["QUANTITYORDERED", "PRICEEACH", "MSRP"]])
ols = sm.OLS(y, X).fit()
print(ols.summary())

vif_df = pd.DataFrame({"Variable": X.columns,
                       "VIF": [vif(X.values, i) for i in range(X.shape[1])]})
print("\nVariance inflation factors:")
print(vif_df.to_string(index=False))

jb = stats.jarque_bera(ols.resid)
bp = het_breuschpagan(ols.resid, X)
print(f"Jarque-Bera: stat={jb[0]:.1f}, p={jb[1]:.3g}")
print(f"Breusch-Pagan: LM p-value={bp[1]:.3g}")
print(f"Durbin-Watson: {durbin_watson(ols.resid):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(ols.fittedvalues, ols.resid, alpha=0.3, s=10)
ax[0].axhline(0, color="k")
ax[0].set_title("Residuals vs fitted")
sm.qqplot(ols.resid, line="s", ax=ax[1])
ax[1].set_title("QQ plot")
plt.tight_layout()
show("fig_03_ols_residual_diagnostics.png")

# --------------------------------------------------------------------------- #
# 6. Robust (HC3) and log-log OLS
# --------------------------------------------------------------------------- #
section(6, "ROBUST (HC3) AND LOG-LOG OLS")
ols_rob = sm.OLS(y, X).fit(cov_type="HC3")
print("OLS with HC3 standard errors (coefficient table):")
print(ols_rob.summary().tables[1])

Xl = sm.add_constant(np.log(d[["QUANTITYORDERED", "PRICEEACH", "MSRP"]]))
ols_log = sm.OLS(np.log(y), Xl).fit(cov_type="HC3")
print("\nLog-log model (elasticities, HC3 errors):")
print(ols_log.summary())

# --------------------------------------------------------------------------- #
# 7. PCA on the three explanatory variables
# --------------------------------------------------------------------------- #
section(7, "PCA ON THE THREE EXPLANATORY VARIABLES")
exp_cols = ["QUANTITYORDERED", "PRICEEACH", "MSRP"]
Z3 = StandardScaler().fit_transform(d[exp_cols])
pca3 = PCA().fit(Z3)

eig = pca3.explained_variance_
ratio = pca3.explained_variance_ratio_ * 100
print(pd.DataFrame({"Axis": [f"PC{i+1}" for i in range(3)],
                    "Eigenvalue": eig.round(4),
                    "Inertia %": ratio.round(2),
                    "Cumulative %": ratio.cumsum().round(2),
                    "Kaiser (>1)": eig > 1}).to_string(index=False))

plt.figure(figsize=(7, 4))
plt.bar(range(1, 4), eig, color="steelblue")
plt.axhline(1, color="r", ls="--", label="Kaiser threshold = 1")
plt.xticks(range(1, 4), ["PC1", "PC2", "PC3"])
plt.xlabel("Axis")
plt.ylabel("Eigenvalue")
plt.title("Scree plot (3 explanatory variables)")
plt.legend()
plt.tight_layout()
show("fig_04_scree_plot.png")

loadings3 = pd.DataFrame(pca3.components_.T, index=exp_cols,
                         columns=[f"PC{i+1}" for i in range(3)])
print("\nLoadings:")
print(loadings3.round(3).to_string())
print("Interpretation: PC1 = unit value, PC2 = physical volume, "
      "PC3 = price vs MSRP (discount depth).")

# --------------------------------------------------------------------------- #
# 8. Principal component regression
# --------------------------------------------------------------------------- #
section(8, "PRINCIPAL COMPONENT REGRESSION (PCR)")
K = int((eig > 1).sum())
print("Components retained (Kaiser rule):", K)
scores = PCA(n_components=K).fit_transform(Z3)
Xp = sm.add_constant(pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(K)]))
pcr = sm.OLS(y.values, Xp).fit()
print(pcr.summary())
print(pd.DataFrame({"Variable": Xp.columns,
                    "VIF": [vif(Xp.values, i) for i in range(Xp.shape[1])]}).to_string(index=False))
print("The PCA used here excludes SALES: the dependent variable never enters the regressors.")

# --------------------------------------------------------------------------- #
# 9. PCA on four variables: circle, individuals, biplot
# --------------------------------------------------------------------------- #
section(9, "PCA ON FOUR VARIABLES: CIRCLE, INDIVIDUALS, BIPLOT")
Z4 = StandardScaler().fit_transform(d[quant_cols])
pca4 = PCA(n_components=2)
coords = pca4.fit_transform(Z4)
load4 = pca4.components_.T * np.sqrt(pca4.explained_variance_)
v1, v2 = pca4.explained_variance_ratio_ * 100
print(f"PC1 {v1:.2f}% + PC2 {v2:.2f}% = {v1 + v2:.2f}% of inertia")
print(pd.DataFrame(load4, index=quant_cols, columns=["PC1", "PC2"]).round(3).to_string())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
a1.add_patch(plt.Circle((0, 0), 1, color="gray", fill=False, ls="--"))
for i, v in enumerate(quant_cols):
    a1.arrow(0, 0, load4[i, 0], load4[i, 1], head_width=0.03, fc="red", ec="red")
    a1.text(load4[i, 0] * 1.1, load4[i, 1] * 1.1, v, color="darkred", fontsize=9)
a1.axhline(0, color="grey")
a1.axvline(0, color="grey")
a1.set_xlim(-1.3, 1.3)
a1.set_ylim(-1.3, 1.3)
a1.set_aspect("equal")
a1.set_title("Correlation circle")
a1.set_xlabel(f"Axis 1 ({v1:.2f}%)")
a1.set_ylabel(f"Axis 2 ({v2:.2f}%)")

a2.scatter(coords[:, 0], coords[:, 1], alpha=0.3, s=10, color="royalblue")
a2.axhline(0, color="grey")
a2.axvline(0, color="grey")
a2.set_title("Individuals (2,823 transactions)")
a2.set_xlabel(f"Axis 1 ({v1:.2f}%)")
a2.set_ylabel(f"Axis 2 ({v2:.2f}%)")
plt.tight_layout()
show("fig_05_correlation_circle_individuals.png")

sf = 3.0
plt.figure(figsize=(9, 9))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.25, s=12, color="royalblue")
for i, v in enumerate(quant_cols):
    plt.arrow(0, 0, load4[i, 0] * sf, load4[i, 1] * sf,
              head_width=0.08, fc="red", ec="red")
    plt.text(load4[i, 0] * sf * 1.1, load4[i, 1] * sf * 1.1, v,
             color="darkred", weight="bold")
plt.axhline(0, color="grey", ls="--")
plt.axvline(0, color="grey", ls="--")
plt.xlabel(f"Axis 1 ({v1:.2f}%)")
plt.ylabel(f"Axis 2 ({v2:.2f}%)")
plt.title("PCA biplot (PC1 vs PC2)")
plt.tight_layout()
show("fig_06_biplot.png")

# --------------------------------------------------------------------------- #
# 10. Model comparison
# --------------------------------------------------------------------------- #
section(10, "MODEL COMPARISON")
comp = pd.DataFrame({
    "Model": ["OLS raw", "PCR", "OLS log-log"],
    "R2": [ols.rsquared, pcr.rsquared, ols_log.rsquared],
    "AIC": [ols.aic, pcr.aic, np.nan],
    "Max VIF": [vif_df["VIF"][1:].max(), 1.0, np.nan]
})
print(comp.round(3).to_string(index=False))
print("Note: the log-log AIC is not comparable with the others (different response).")

# --------------------------------------------------------------------------- #
# 11. Multiple Correspondence Analysis + CAH
# --------------------------------------------------------------------------- #
section(11, "MULTIPLE CORRESPONDENCE ANALYSIS (MCA / ACM) + CAH")
d["TERRITORY"] = df["TERRITORY"].fillna("NA_Americas").values
d["TRIMESTRE"] = "T" + df["QTR_ID"].astype(str).values
acm_cols = ["PRODUCTLINE", "DEALSIZE", "TERRITORY", "TRIMESTRE"]
Z = pd.get_dummies(d[acm_cols], dtype=float)
n_acm, Q = Z.shape
p = len(acm_cols)
Ftab = Z.values / (n_acm * p)
r = Ftab.sum(axis=1)
c = Ftab.sum(axis=0)
S = (Ftab - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, s, Vt = np.linalg.svd(S, full_matrices=False)
rank = Q - p
eig_acm = s[:rank] ** 2
inertie_brute = 100 * eig_acm / eig_acm.sum()
eig_benz = np.where(eig_acm > 1 / p, ((p / (p - 1)) * (eig_acm - 1 / p)) ** 2, 0)
inertie_benz = 100 * eig_benz / eig_benz.sum()
print(f"Disjunctive table: {n_acm} x {Q}, rank = {rank}, Benzecri threshold = 1/p = {1/p:.2f}")
print(pd.DataFrame({"Axis": [f"Dim{i+1}" for i in range(rank)],
                    "Eigenvalue": eig_acm.round(4),
                    "Raw inertia %": inertie_brute.round(2),
                    "Benzecri %": inertie_benz.round(2)}).head(6).to_string(index=False))
print(f"First plane after correction: {inertie_benz[0]:.2f}% + "
      f"{inertie_benz[1]:.2f}% = {inertie_benz[:2].sum():.2f}%")

G = Vt.T[:, :rank] * s[:rank] / np.sqrt(c)[:, None]
contrib = 100 * c * G[:, 0] ** 2 / eig_acm[0]
mod_table = pd.DataFrame({"Coord1": G[:, 0], "Coord2": G[:, 1],
                          "Contrib1_%": contrib}, index=Z.columns)
print("\nModalities with the largest contribution to axis 1:")
print(mod_table.sort_values("Contrib1_%", ascending=False).head(8).round(3).to_string())

fig, (b1, b2) = plt.subplots(1, 2, figsize=(15, 6))
x = np.arange(1, 9)
w = 0.38
b1.bar(x - w / 2, inertie_brute[:8], w, label="Raw inertia", color="steelblue")
b1.bar(x + w / 2, inertie_benz[:8], w, label="Benzecri corrected", color="darkorange")
b1.set_title("MCA scree plot: raw vs corrected inertia")
b1.set_xlabel("Dimension")
b1.set_ylabel("% inertia")
b1.legend()
for mod, row in mod_table.iterrows():
    color = {"PRODUCTLINE": "green", "DEALSIZE": "orange",
             "TERRITORY": "purple", "TRIMESTRE": "crimson"}[mod.split("_")[0]]
    b2.scatter(row.Coord1, row.Coord2, color=color, s=60, edgecolors="k")
    b2.annotate(mod.split("_", 1)[1], (row.Coord1, row.Coord2), fontsize=8)
b2.axhline(0, color="grey", ls="--")
b2.axvline(0, color="grey", ls="--")
b2.set_title("Modalities plane (axes 1-2)")
plt.tight_layout()
show("fig_07_mca_benzecri.png")

Zc = linkage(G[:, :5], method="ward")
plt.figure(figsize=(11, 6))
dendrogram(Zc, labels=[m.split("_", 1)[1] for m in Z.columns],
           leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical clustering (Ward) of MCA modalities")
plt.tight_layout()
show("fig_08_cah_dendrogram.png")

# --------------------------------------------------------------------------- #
# 12. Support Vector Machine
# --------------------------------------------------------------------------- #
section(12, "SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION")
features = ["QUANTITYORDERED", "PRICEEACH", "MSRP", "SALES"]
y_s = (d["DEALSIZE"] == "Large").astype(int)
print("Target IS_LARGE:", int(y_s.sum()), "Large /", int((1 - y_s).sum()), "other")
print("Warning: DEALSIZE is a binning of SALES (<3000, 3000-7000, >7000); "
      "this classification partly relearns a business rule.")

Xtr, Xte, ytr, yte = train_test_split(d[features], y_s, test_size=0.25,
                                      random_state=42, stratify=y_s)
scaler = StandardScaler().fit(Xtr)  # fitted on training data only
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)
svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
svm.fit(Xtr_s, ytr)
pred = svm.predict(Xte_s)
prob = svm.predict_proba(Xte_s)[:, 1]
print(f"Accuracy = {accuracy_score(yte, pred):.4f}")
print(f"AUC-ROC  = {roc_auc_score(yte, prob):.4f}")
print(f"F1 (Large) = {f1_score(yte, pred):.4f}")
print(pd.DataFrame(confusion_matrix(yte, pred),
                   index=["Actual Non-Large", "Actual Large"],
                   columns=["Pred Non-Large", "Pred Large"]).to_string())
print(classification_report(yte, pred, target_names=["Non-Large", "Large"], digits=4))

pca_vis = PCA(n_components=2, random_state=42)
Xtr_pca = pca_vis.fit_transform(Xtr_s)
svm2d = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42).fit(Xtr_pca, ytr)
gx, gy = np.meshgrid(np.linspace(Xtr_pca[:, 0].min() - 1, Xtr_pca[:, 0].max() + 1, 200),
                     np.linspace(Xtr_pca[:, 1].min() - 1, Xtr_pca[:, 1].max() + 1, 200))
gz = svm2d.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
fpr, tpr, _ = roc_curve(yte, prob)
auc = roc_auc_score(yte, prob)

fig, (c1, c2) = plt.subplots(1, 2, figsize=(15, 6))
c1.contourf(gx, gy, gz, alpha=0.15, cmap=plt.cm.coolwarm)
c1.contour(gx, gy, gz, colors="k", linestyles="--", linewidths=1)
for val, color, label in [(0, "steelblue", "Non-Large"), (1, "darkorange", "Large")]:
    m = ytr.values == val
    c1.scatter(Xtr_pca[m, 0], Xtr_pca[m, 1], s=20, alpha=0.6, color=color, label=label)
c1.set_title("SVM RBF decision boundary (PCA 2D projection)")
c1.set_xlabel("PC1")
c1.set_ylabel("PC2")
c1.legend()
c2.plot(fpr, tpr, color="darkorange", lw=2, label=f"SVM (AUC = {auc:.4f})")
c2.plot([0, 1], [0, 1], "k--", label="Chance")
c2.set_title("ROC curve (test set)")
c2.set_xlabel("False positive rate")
c2.set_ylabel("True positive rate")
c2.legend(loc="lower right")
plt.tight_layout()
show("fig_09_svm_decision_boundary_roc.png")

print("\nAnalysis complete. Figures saved to images/, console output saved to output.txt.")
sys.stdout.console.write("Analysis complete. See output.txt and images/.\n")
