# Screen exposure and executive function: meta-analysis and machine learning

This repository contains the final, publication-oriented supplementary analysis package for the systematic review of screen exposure and executive function in children and adolescents. It combines the locked input datasets, reproducible analysis code, numerical result tables and publication-ready figures for the three-level meta-analysis and the machine-learning analyses.

## Repository structure

- `data/` — locked analysis workbook and the final verified study-characteristics/quality-assessment table.
- `traditional_meta/` — R code, three-level meta-analysis results, sensitivity and multiplicity checks, and final figures.
- `machine_learning/` — Python code, prepared analysis datasets, nested grouped-validation outputs, diagnostics and final figures.

Only final deliverables are retained. Revision folders, caches, compiled files, serialized workspaces and superseded scripts are intentionally excluded.

## Reproduce the analyses

Run the traditional meta-analysis from the repository root:

```powershell
Rscript traditional_meta/01_three_level_meta_analysis.R
Rscript traditional_meta/02_sensitivity_and_multiplicity.R
```

Run the machine-learning analysis:

```powershell
python -m pip install -r machine_learning/requirements.txt
python machine_learning/01_run_machine_learning.py
python machine_learning/03_additional_machine_learning_figures.py
```

To redraw the three primary machine-learning figures and report from the committed numerical outputs without refitting the models:

```powershell
python machine_learning/02_render_existing_results.py
```

All scripts resolve inputs relative to the repository. Environment variables documented in the scripts can override the default paths and resampling settings.

## Statistical notes

The traditional synthesis models Fisher-transformed correlations with random effects at the report, cohort/sample-cluster and effect-size levels. Machine-learning validation is grouped by independent cohort/sample cluster; preprocessing and model selection occur within the resampling procedure. The proxy-augmented machine-learning dataset is retained as a separately labelled sensitivity analysis and is not treated as interchangeable with the meta-grade primary estimand.

The committed figures and numerical outputs correspond to the final analysis version dated 26 September 2026.
