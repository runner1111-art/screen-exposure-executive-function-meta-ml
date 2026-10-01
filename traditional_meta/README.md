# Three-level meta-analysis

`01_three_level_meta_analysis.R` fits the domain-specific three-level random-effects models, subgroup models, univariable meta-regressions, leave-one-report-out analyses and publication-bias diagnostics. Main models are restricted to `Expanded_Primary_Model = 1`; the original `Primary_Model = 1` direct/validated subset is retained in the explicit analysis-set sensitivity comparison. The script also creates the final forest, moderator, regression, sensitivity, funnel and geographic-distribution figures.

`02_sensitivity_and_multiplicity.R` performs the report-level trim-and-fill sensitivity analysis, applies Benjamini–Hochberg correction to the prespecified families of moderator tests and derives variance shares and intraclass correlations from the three-level models.

Both scripts use `../data/Meta-analysis_data_full_age_updated.xlsx`; the main script additionally uses `../data/Table_1_final_verified.xlsx`. Set `SCREEN_EF_META_OUTPUT` to write regenerated outputs to another directory. Input paths may be overridden with `SCREEN_EF_META_XLSX` and `SCREEN_EF_INCLUDED_REPORTS_XLSX`.

The scripts regenerate the numerical CSV outputs supporting the figures; those run-specific files are not committed. Fisher's z is used for model fitting; pooled estimates are back-transformed to correlation coefficients for presentation. The expanded core-domain analysis comprises 261 effects from 43 reports and 45 cohort/sample clusters: 90 inhibitory-control, 120 working-memory and 51 cognitive-flexibility effects. The strict sensitivity set comprises 104 core-domain effects from 19 reports. The geographic figure counts all 103 included reports in Table 1 and therefore should not be interpreted as the quantitative-model denominator.
