suppressPackageStartupMessages({
  library(dplyr)
  library(metafor)
  library(readr)
  library(readxl)
})

script_file <- sub(
  "^--file=", "",
  grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
)
source_dir <- if (length(script_file) > 0L) {
  dirname(normalizePath(script_file[[1]], winslash = "/", mustWork = TRUE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
repo_root <- normalizePath(file.path(source_dir, ".."), winslash = "/", mustWork = TRUE)
input_xlsx <- Sys.getenv(
  "SCREEN_EF_META_XLSX",
  file.path(repo_root, "data", "Meta-analysis_data_full_age_updated.xlsx")
)
output_dir <- Sys.getenv("SCREEN_EF_META_OUTPUT", source_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

bt_r <- function(z) tanh(z)

meta <- read_excel(input_xlsx, sheet = "Meta_Data") %>%
  mutate(
    EF_dimension = case_when(
      grepl("^inhibitory", EF_Domain, ignore.case = TRUE) ~ "Inhibitory control",
      grepl("^working memory", EF_Domain, ignore.case = TRUE) ~ "Working memory",
      grepl("^cognitive flexibility", EF_Domain, ignore.case = TRUE) ~ "Cognitive flexibility",
      TRUE ~ NA_character_
    ),
    Fisher_z = as.numeric(Fisher_z),
    Sampling_Variance_z = as.numeric(Sampling_Variance_z),
    Expanded_Primary_Model = as.numeric(Expanded_Primary_Model)
  ) %>%
  filter(
    Expanded_Primary_Model == 1,
    !is.na(EF_dimension), is.finite(Fisher_z),
    is.finite(Sampling_Variance_z), Sampling_Variance_z > 0
  )

# Trim-and-fill is not defined for multilevel dependent estimates. For this
# sensitivity analysis only, collapse each report/domain to one conservative
# estimate. The point estimate uses inverse-sampling-variance weights; its
# variance is the larger of the conventional fixed-effect variance and the
# mean component variance, avoiding artificial precision from multiple highly
# correlated outcomes within a report.
report_level <- meta %>%
  group_by(EF_dimension, Report_ID) %>%
  summarise(
    yi = weighted.mean(Fisher_z, w = 1 / Sampling_Variance_z),
    vi_fe = 1 / sum(1 / Sampling_Variance_z),
    vi_typical = mean(Sampling_Variance_z),
    vi = pmax(vi_fe, vi_typical),
    effects_collapsed = n(),
    .groups = "drop"
  )

trim_rows <- lapply(unique(report_level$EF_dimension), function(domain) {
  dat <- filter(report_level, EF_dimension == domain)
  fit <- rma.uni(yi = yi, vi = vi, data = dat, method = "REML")
  tf <- trimfill(fit, estimator = "L0")
  tibble(
    EF_dimension = domain,
    reports = nrow(dat),
    original_z = as.numeric(fit$b[1]),
    original_r = bt_r(as.numeric(fit$b[1])),
    original_ci_lb_r = bt_r(as.numeric(fit$ci.lb)),
    original_ci_ub_r = bt_r(as.numeric(fit$ci.ub)),
    imputed_reports = as.integer(tf$k0),
    adjusted_z = as.numeric(tf$b[1]),
    adjusted_r = bt_r(as.numeric(tf$b[1])),
    adjusted_ci_lb_r = bt_r(as.numeric(tf$ci.lb)),
    adjusted_ci_ub_r = bt_r(as.numeric(tf$ci.ub)),
    estimator = "L0",
    note = "Exploratory trim-and-fill after conservative report-level aggregation; not a replacement for the multilevel primary model."
  )
})
trim_results <- bind_rows(trim_rows)
write_csv(report_level, file.path(output_dir, "report_level_inputs_for_trimfill.csv"))
write_csv(trim_results, file.path(output_dir, "trimfill_report_level_sensitivity.csv"))

subgroup <- read_csv(file.path(source_dir, "subgroup_results.csv"), show_col_types = FALSE) %>%
  distinct(EF_dimension, moderator, moderator_label, interaction_p) %>%
  mutate(
    p_fdr_bh_21_subgroup_tests = p.adjust(interaction_p, method = "BH"),
    significant_fdr_0_05 = p_fdr_bh_21_subgroup_tests < 0.05
  )
meta_reg_file <- if (file.exists(file.path(source_dir, "meta_regression_results.csv"))) {
  file.path(source_dir, "meta_regression_results.csv")
} else {
  file.path(source_dir, "meta_regression_coefficients.csv")
}
meta_reg <- read_csv(meta_reg_file, show_col_types = FALSE) %>%
  mutate(
    p_fdr_bh_9_meta_regressions = p.adjust(p, method = "BH"),
    significant_fdr_0_05 = p_fdr_bh_9_meta_regressions < 0.05
  )
write_csv(subgroup, file.path(output_dir, "subgroup_omnibus_with_fdr.csv"))
write_csv(meta_reg, file.path(output_dir, "meta_regression_with_fdr.csv"))

variance <- read_csv(file.path(source_dir, "main_results.csv"), show_col_types = FALSE) %>%
  mutate(
    random_variance_total = tau2_report + tau2_cohort + tau2_effect,
    report_variance_share = tau2_report / random_variance_total,
    cohort_variance_share = tau2_cohort / random_variance_total,
    effect_variance_share = tau2_effect / random_variance_total,
    same_report_different_cohort_icc = report_variance_share,
    same_cohort_icc = (tau2_report + tau2_cohort) / random_variance_total
  )
write_csv(variance, file.path(output_dir, "variance_components_and_icc.csv"))

writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_session_info_sensitivity.txt"))
