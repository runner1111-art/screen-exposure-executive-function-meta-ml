# Screen exposure–executive function machine-learning analysis

This directory contains the reproducible code, analysis datasets, numerical outputs and publication-ready figures for the machine-learning component of the meta-analysis.

## Primary estimand

The outcome is the harmonized effect size on the Fisher-z scale. Negative values indicate poorer executive function with greater screen exposure. The primary ML dataset is restricted to effect sizes that are valid for the conventional meta-analysis. The larger proxy-augmented dataset is analysed only as a sensitivity analysis because regression coefficients and other proxy statistics are not necessarily commensurate with correlations.

## Leakage control

All effects from the same cohort/sample cluster are assigned to the same fold. Hyperparameter tuning is performed inside the outer training fold. Missing-value imputation, scaling and one-hot encoding are fitted inside each training split. Training and performance metrics are group-balanced, so each independent cluster contributes equal total weight.

## Models and outputs

Ridge, elastic-net, random-forest and histogram-gradient-boosting regressors are compared with a training-fold mean comparator. Performance is summarized by RMSE, MAE, predictive Q², correlation R², calibration intercept/slope and mean bias. Group-block permutation importance and partial dependence are exploratory descriptions of predictive dependence, not causal effects.

The final predictor set excludes `Quality_Class`. The source field was an effect-eligibility label rather than a validated report-level risk-of-bias assessment, so retaining it would have introduced a non-comparable methodological label into prediction.

## Run

The committed prepared primary and proxy-sensitivity CSV files are used by default, so the workflow is portable and does not depend on the original local directory structure.

```powershell
python -m pip install -r requirements.txt
python 01_run_machine_learning.py
```

After a completed model run, figures and the narrative report can be regenerated without refitting:

```powershell
python 02_render_existing_results.py
```

The five extended diagnostic figures can then be generated from the locked
primary data, predictions and tuning audit:

```powershell
python 03_additional_machine_learning_figures.py
```

Input and output paths can be overridden with `SCREEN_EF_PRIMARY_PREPARED_CSV`, `SCREEN_EF_PROXY_PREPARED_CSV`, `SCREEN_EF_META_XLSX`, `SCREEN_EF_PROXY_XLSX`, `SCREEN_EF_AGE_XLSX` and `SCREEN_EF_ML_OUTPUT`. The random seed and resampling settings are recorded in `ML_run_config_and_session.json`.

## Figure set

- `Figure_ML1_model_performance`: repeated nested grouped-validation performance, out-of-fold predictions and residual calibration.
- `Figure_ML2_predictive_drivers`: held-out group-block permutation importance, partial dependence and domain-specific error.
- `Figure_ML3_sensitivity_and_error`: proxy-augmented sensitivity analysis, data composition and screen-category error.
- `Figure_ML4_learning_curve`: cross-cohort learning curves as the number of independent training clusters increases.
- `Figure_ML5_cohort_error_audit`: cohort/sample-level out-of-fold error and directional prediction bias.
- `Figure_ML6_error_support_heatmap`: prediction error and evidence support jointly classified by executive-function domain and screen-exposure category.
- `Figure_ML7_model_selection_stability`: inner-validation model wins, outer-fold improvement over the comparator and selected ridge penalties.
- `Figure_ML8_linear_contribution_stability`: magnitude and cluster-level direction of contributions in the full-data ridge refit.

For a four-figure main-text presentation, the recommended set is ML1 (overall
performance), ML2 (predictive drivers), ML4 (data sufficiency) and ML6
(clinically interpretable error/evidence matrix). ML3, ML5, ML7 and ML8 are
recommended for Supplementary Information. This division retains the complete
analysis while avoiding redundant main-text panels.

ROC curves and PCA cluster plots were not added. The outcome is continuous
Fisher's z, and dichotomising it solely to produce an ROC curve would discard
information and create an unregistered classification target. PCA separation
would likewise describe the predictor matrix rather than validate prediction of
the meta-analytic outcome.

The SVG files are the editable vector originals; the PNG copies are rendered at 300 dpi.

## Methodological references used in the implementation

- ViewWay, *Clinical decision support systems*: grouped/temporal splitting, pipeline-contained preprocessing, calibration, subgroup evaluation and non-causal interpretation of SHAP-type explanations. https://github.com/ViewWay/academic-research-skill/blob/main/references/tools/clinical-decision-support.md
- Aperivue, *medsci-skills*: medical prediction-model reporting, calibration and validation checks, and compact journal-ready multi-panel figure conventions. https://github.com/Aperivue/medsci-skills

These repositories informed the validation and reporting safeguards; no study result was imported from them.
