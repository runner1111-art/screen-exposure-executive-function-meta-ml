# Screen exposure and executive function: supplementary code

This repository contains the analysis code and numerical outputs for the manuscript **"Effects of Different Types of Screen Exposure on the Development of Executive Function in Children and Adolescents: A Machine Learning-Enhanced Meta-Analysis."**

## Repository structure

```
.
├── machine_learning/        # Python pipeline for the machine-learning analyses
│   ├── 01_run_machine_learning.py
│   ├── 02_render_existing_results.py
│   ├── 03_additional_machine_learning_figures.py
│   ├── requirements.txt
│   ├── Figure_ML*.png / .svg   # publication-ready figures
│   └── ML_*.csv                # numerical outputs
└── traditional_meta/        # R code for the three-level meta-analysis
    ├── three_level_meta_analysis.R
    ├── 01–14_*.svg          # publication-ready figures
    └── *_results.csv        # numerical outputs
```

## Reproducing the traditional meta-analysis

- Software: R 4.6.1, `metafor` 5.0.1
- Script: `traditional_meta/three_level_meta_analysis.R`
- The script fits three-level random-effects models (report, cohort/sample cluster and effect levels) with cluster-robust inference, runs subgroup and meta-regression analyses, and renders the figures.
- Key outputs: `main_results.csv`, `subgroup_results.csv`, `meta_regression_results.csv`, `publication_bias_results.csv`, `sensitivity_results.csv`.

## Reproducing the machine-learning analysis

- Software: Python 3.12, `scikit-learn` (see `requirements.txt`)
- Entry point: `machine_learning/01_run_machine_learning.py`
- The pipeline compares ridge, elastic-net, random-forest and histogram-gradient-boosting regressors against a training-fold mean baseline, using nested cohort-grouped cross-validation (five outer folds repeated five times, four inner folds for hyperparameter tuning). All effect sizes from the same cohort/sample cluster are held out together to prevent leakage.
- Figures are rendered by `02_render_existing_results.py` and `03_additional_machine_learning_figures.py`.

## Data availability

All effect-level inputs and analysis outputs are provided as CSV files. The primary effect-size data used in the manuscript are available from the corresponding author upon request, or as specified in the manuscript's Data Availability statement.

## Note

The `ML_*.csv` and `*_results.csv` files are machine-readable summaries of the results reported in the manuscript. The figures are provided in both SVG (editable) and PNG (raster) formats.
