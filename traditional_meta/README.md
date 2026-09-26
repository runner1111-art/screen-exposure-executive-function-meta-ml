# Three-level meta-analysis

`01_three_level_meta_analysis.R` fits the domain-specific three-level random-effects models, subgroup models, univariable meta-regressions, leave-one-report-out analyses and publication-bias diagnostics. It also creates the final forest, moderator, regression, sensitivity, funnel and geographic-distribution figures.

`02_sensitivity_and_multiplicity.R` performs the report-level trim-and-fill sensitivity analysis, applies Benjamini–Hochberg correction to the prespecified families of moderator tests and derives variance shares and intraclass correlations from the three-level models.

Both scripts use `../data/Meta-analysis_data_full_age_updated.xlsx`; the main script additionally uses `../data/Table_1_final_verified.xlsx`. Set `SCREEN_EF_META_OUTPUT` to write regenerated outputs to another directory. Input paths may be overridden with `SCREEN_EF_META_XLSX` and `SCREEN_EF_INCLUDED_REPORTS_XLSX`.

The committed CSV files are the exact numerical outputs supporting the figures. Fisher's z is used for model fitting; pooled estimates are back-transformed to correlation coefficients for presentation.
