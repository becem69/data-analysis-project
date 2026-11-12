# Sales Data Analysis: OLS Regression and Principal Component Analysis

**Repository:** [github.com/becem69/data-analysis-project](https://github.com/becem69/data-analysis-project)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-OLS%20%2B%20PCA%20done%2C%20MCA%20%2B%20SVM%20in%20progress-orange)

A statistical study of transaction-level sales revenue using **Ordinary Least Squares (OLS / MCO)** and **Principal Component Analysis (PCA / ACP)**, with a principal component regression (PCR) and a log-log robust model as comparisons. The work follows a case-study format: an executive asks *why* sales move and *how* to steer them, and the analyst team must answer with statistical evidence.

This repository covers the first two methods. Multiple Correspondence Analysis (MCA / ACM) and Support Vector Machines (SVM) are planned (see [Roadmap](#roadmap)).

## Table of contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Methodology](#methodology)
- [Key results](#key-results)
- [Figures](#figures)
- [Getting started](#getting-started)
- [Reading the output](#reading-the-output)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Sources](#sources)
- [Author](#author)

## Overview

The analysis answers three questions:

1. **What is the effect of price and volume on revenue?** Estimated with OLS on `QUANTITYORDERED`, `PRICEEACH` and `MSRP`.
2. **Are the estimates reliable?** Tested with Fisher and Student tests, VIF, Jarque-Bera, Breusch-Pagan and Durbin-Watson, and re-estimated with HC3 robust errors and a log-log specification.
3. **What is the structure of the quantitative space?** Explored with PCA (eigenvalues, Kaiser rule, loadings, correlation circle, individuals plane, biplot) and used in a PCR.

## Dataset

- **Name:** Sample Sales Data (Kaggle, author `kyanyoga`)
- **Link:** https://www.kaggle.com/datasets/kyanyoga/sample-sales-data
- **File:** `sales_data_sample.csv`, 2,823 rows and 25 columns, read with `encoding='latin1'`

Variables used:

| Role | Variables |
|---|---|
| Quantitative (OLS, PCA) | `QUANTITYORDERED`, `PRICEEACH`, `MSRP`, `SALES` |
| Qualitative (MCA, SVM, later stages) | `PRODUCTLINE`, `DEALSIZE`, `COUNTRY`, `STATUS` |

No missing values were found in these eight columns, so all 2,823 observations are kept.

> The dataset is not committed to this repository. Download it from Kaggle (see [Getting started](#getting-started)).

## Repository structure

```
.
├── project.py                  # Full analysis pipeline (run top to bottom)
├── Sales_OLS_PCA_Report.pdf    # Short scientific report (3 pages)
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
├── images/                     # Figures saved by project.py (see Figures)
│   ├── fig_01_histograms.png
│   ├── fig_02_correlation_heatmap.png
│   ├── fig_03_ols_residual_diagnostics.png
│   ├── fig_04_scree_plot.png
│   ├── fig_05_correlation_circle_individuals.png
│   └── fig_06_biplot.png
└── sales_data_sample.csv       # Not tracked, download from Kaggle
```

## Methodology

`project.py` runs ten sections in order:

| # | Step | What it does |
|---|---|---|
| 1 | Structural audit | Types, missing values, clean subset, check of `SALES` against `QUANTITYORDERED x PRICEEACH` |
| 2 | Univariate audit | Mean, std, median, min, max, skewness, category frequencies, rare modalities (< 2%) |
| 3 | Bivariate analysis | Pearson matrix and heatmap, chi-square test and Cramér's V (`PRODUCTLINE` vs `DEALSIZE`) |
| 4 | Outliers | Mahalanobis distance on the explanatory variables (chi-square threshold, p < 0.001) |
| 5 | OLS | Fit, VIF, Jarque-Bera, Breusch-Pagan, Durbin-Watson, residual and QQ plots |
| 6 | Robust and log-log OLS | HC3 standard errors and elasticities from `log(SALES)` on logged regressors |
| 7 | PCA (3 explanatory variables) | Eigenvalues, inertia, Kaiser rule, scree plot, loadings |
| 8 | PCR | OLS of `SALES` on the retained components, VIF check |
| 9 | PCA (4 variables) | Correlation circle, individuals plane, biplot |
| 10 | Model comparison | R², AIC and maximum VIF for OLS, PCR and log-log |

Standardisation (zero mean, unit variance) is applied before every PCA. The PCA used for PCR is fitted on the three explanatory variables only, so the dependent variable never enters the regressors.

## Key results

**Exploratory findings**

- `QUANTITYORDERED` is uncorrelated with price (r = 0.006) and with `MSRP` (r = 0.018).
- `PRICEEACH` and `MSRP` are moderately correlated (r = 0.671).
- `SALES` is strongly but not exactly `QUANTITYORDERED x PRICEEACH` (correlation 0.864 in this file).
- 13 multivariate outliers (0.46%) were detected and kept.
- `PRODUCTLINE` and `DEALSIZE` are significantly associated (chi-square = 135.36, 12 d.f.), but the association is weak (Cramér's V = 0.155).

**Model comparison**

| Model | R² | AIC | Max VIF | Remark |
|---|---|---|---|---|
| OLS (raw) | 0.796 | 45,981.5 | 1.82 | Non-normal, heteroscedastic residuals |
| PCR (2 components) | 0.794 | 46,001.6 | 1.00 | No gain, discards discount information |
| Log-log (HC3) | 0.906 | -2,323* | 1.82 | Best fit, interpretable elasticities |

\* Not comparable with the other AICs because the dependent variable is `log(SALES)`.

**OLS coefficients (raw model)**

| Term | Coefficient | Std. error | t | VIF |
|---|---|---|---|---|
| Intercept | -4878.88 | 87.54 | -55.74 | |
| `QUANTITYORDERED` | 102.67 | 1.61 | 63.81 | 1.00 |
| `PRICEEACH` | 39.04 | 1.05 | 37.28 | 1.82 |
| `MSRP` | 15.52 | 0.53 | 29.53 | 1.82 |

**PCA**

- Three explanatory variables: eigenvalues 1.6716, 1.0001 and 0.3294, which retain 89.02% of the inertia with two axes.
- PC1 is a *unit value* axis (`PRICEEACH` and `MSRP`), PC2 is a *volume* axis (`QUANTITYORDERED`), and PC3 contrasts price with `MSRP`, which can be read as discount depth.
- Four variables, first plane: 59.82% + 28.84% = 88.66% of the inertia.

**Main conclusions**

1. Price and volume are independent and both drive revenue strongly, so they can be managed as separate levers.
2. Multicollinearity is moderate (maximum VIF 1.82, below the usual thresholds of 5 or 10), so PCR is not needed and slightly degrades the model. A VIF of 1.0 in a PCR is guaranteed by orthogonality and does not prove a better model.
3. The preferred specification is the log-log model with HC3 errors. Elasticities are about 1.00 for quantity, 1.06 for price and 0.28 for `MSRP`.
4. Because price and quantity do not co-move, these elasticities reflect how revenue is composed. They are **not** a demand elasticity.
5. Residual non-normality persists in every model, so prediction intervals should be treated with caution.

The full discussion is in [`Sales_OLS_PCA_Report.pdf`](Sales_OLS_PCA_Report.pdf).

## Figures

`project.py` saves every plot to the `images/` folder as it displays it. The figures below are listed in the order the script shows them.

### 1. Distributions of the quantitative variables (section 2)

![Histograms of QUANTITYORDERED, PRICEEACH, MSRP and SALES](images/fig_01_histograms.png)

Shows the right skew of `SALES` and the left skew of `PRICEEACH` seen in the univariate audit.

### 2. Pearson correlation heatmap (section 3)

![Pearson correlation heatmap](images/fig_02_correlation_heatmap.png)

`QUANTITYORDERED` is independent of price, while `PRICEEACH` and `MSRP` are moderately correlated (r = 0.671).

### 3. OLS residual diagnostics (section 5)

![Residuals vs fitted values and QQ plot](images/fig_03_ols_residual_diagnostics.png)

Residuals vs fitted values reveal heteroscedasticity, and the QQ plot shows heavy tails (kurtosis 11.48).

### 4. Scree plot (section 7)

![Scree plot of the PCA eigenvalues](images/fig_04_scree_plot.png)

Eigenvalues of the three explanatory variables with the Kaiser threshold at 1.

### 5. Correlation circle and individuals plane (section 9)

![Correlation circle and plane of individuals](images/fig_05_correlation_circle_individuals.png)

Left: variables on the first factorial plane (value axis and volume axis). Right: the 2,823 transactions as points.

### 6. PCA biplot (section 9)

![PCA biplot, PC1 vs PC2](images/fig_06_biplot.png)

Variables and individuals superimposed on the plane that explains 88.66% of the inertia.

> The images are generated when you run the script. Run it once and commit the `images/` folder so the figures appear on GitHub.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/becem69/data-analysis-project.git
cd data-analysis-project
```

### 2. Create an environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
matplotlib
scipy
statsmodels
scikit-learn
kagglehub
```

### 3. Download the dataset

Option A, with `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("kyanyoga/sample-sales-data")
print(path)
```

Then copy `sales_data_sample.csv` from the printed folder into the project root.

Option B, manually: download it from https://www.kaggle.com/datasets/kyanyoga/sample-sales-data, unzip it, and place `sales_data_sample.csv` next to `project.py`.

Check that it loads correctly:

```bash
python3 -c "import pandas as pd; print(pd.read_csv('sales_data_sample.csv', encoding='latin1').shape)"
# (2823, 25)
```

### 4. Run the analysis

```bash
python project.py | tee output.txt
```

Each plot window pauses the script until it is closed, and each figure is saved to `images/`. The console output is saved in `output.txt`. On a headless machine, run `MPLBACKEND=Agg python project.py` to save the figures without opening windows.

## Reading the output

| Output | What to check |
|---|---|
| OLS summary | R², F-statistic, signs and sizes of coefficients, t-tests |
| VIF table | Values above 5 or 10 signal harmful multicollinearity |
| Jarque-Bera | A small p-value rejects normal residuals |
| Breusch-Pagan | A small p-value rejects constant error variance, so use HC3 errors |
| Durbin-Watson | Meaningful only if rows are ordered in time (not the case here) |
| Eigenvalue table | Keep axes with eigenvalue above 1 and enough cumulative inertia (about 70 to 80%) |
| Loadings | Name each axis from the variables that load heavily on it |
| Model comparison | Compare R² and AIC for models with the same dependent variable |

## Limitations

- The data are cross-sectional and the rows have no usable time order, so autocorrelation tests have no clear meaning.
- `SALES` is largely a function of the regressors, which inflates R² and makes the elasticities partly arithmetic.
- Results describe association, not causation.
- Chi-square tests on `PRODUCTLINE` vs `DEALSIZE` involve empty or very small cells (for example, no Large deals for Ships).
- Six countries and four order statuses fall below 2% frequency and should be grouped before MCA.

## Roadmap

- [x] Data preparation and exploratory analysis
- [x] OLS with diagnostics, robust errors and log-log model
- [x] PCA, PCR and graphical representations
- [x] Short scientific report
- [ ] Multiple Correspondence Analysis (MCA / ACM) on `PRODUCTLINE`, `DEALSIZE`, `COUNTRY`, `STATUS`, with rare modalities grouped
- [ ] Support Vector Machine (SVM) classification with standardised and one-hot encoded features

## Sources

- **Dataset:** Kyanyoga, *Sample Sales Data*, Kaggle. https://www.kaggle.com/datasets/kyanyoga/sample-sales-data
- **Library:** scikit-learn developers, *scikit-learn: Machine Learning in Python*. https://github.com/scikit-learn/scikit-learn

## Author

**Bacem El Manai** ([@becem69](https://github.com/becem69))
ISET'COM, Management of Information and Communication Technologies
