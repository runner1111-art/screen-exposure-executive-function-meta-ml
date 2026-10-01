# Screen exposure and executive function: meta-analysis and machine learning

This repository contains the final, publication-oriented supplementary analysis package for the systematic review of screen exposure and executive function in children and adolescents. It combines the locked input datasets, reproducible analysis code and publication-ready figures for the three-level meta-analysis and the machine-learning analyses.

## Repository structure

- `data/` — locked analysis workbook, the final verified study-characteristics/quality-assessment table and the frozen proxy-sensitivity input.
- `traditional_meta/` — final R code and publication-ready figures.
- `machine_learning/` — final Python code and publication-ready figures.

Only final deliverables are retained. Revision folders, caches, compiled files, serialized workspaces, run logs, intermediate CSV outputs and superseded scripts are intentionally excluded. The analysis scripts regenerate numerical outputs locally when run.

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

After completing the full machine-learning run, redraw the three primary figures without refitting the models:

```powershell
python machine_learning/02_render_existing_results.py
```

All scripts resolve inputs relative to the repository. Environment variables documented in the scripts can override the default paths and resampling settings.

## Statistical notes

The traditional synthesis models Fisher-transformed correlations with random effects at the report, cohort/sample-cluster and effect-size levels. The expanded primary set admits direct correlations and effect measures that can be converted to a signed correlation with a documented formula and valid sampling variance. The original direct/validated set is retained as a stricter sensitivity analysis. Machine-learning validation is grouped by independent cohort/sample cluster; preprocessing and model selection occur within the resampling procedure.

The committed figures and numerical outputs correspond to the expanded convertible-effect analysis updated 1 October 2026. The effect-level workbook contains 376 effects from 69 quantitatively represented reports and 69 independent cohort/sample clusters. Altun (2022) and Sinvani (2023) were excluded at full-text eligibility because no retrievable full text or verifiable extractable data were available; their ten secondary-source rows were removed before analysis. The three core-domain models contain 261 effects from 43 reports and 45 cohort/sample clusters: 90 inhibitory-control, 120 working-memory and 51 cognitive-flexibility effects. The original strict set contains 153 effects overall and 104 core-domain effects from 19 reports and is retained as a labelled sensitivity analysis.

The systematic review contains 103 included reports. Both unavailable-full-text exclusions are documented in the workbook Fulltext_Status sheet and in the PRISMA accounting; neither contributes to Table 1, risk-of-bias summaries, meta-analysis or machine learning.
