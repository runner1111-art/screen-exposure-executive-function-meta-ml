options(stringsAsFactors = FALSE, scipen = 999, encoding = "UTF-8")

# Three-level meta-analysis of screen exposure and executive function
# R version used for the verified run: 4.5.1
# Effect sizes are modelled on Fisher's z scale and displayed as correlations.

required_packages <- c(
  "readxl", "metafor", "dplyr", "tidyr", "ggplot2", "patchwork",
  "svglite", "purrr", "stringr", "forcats", "MASS", "maps"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Missing required R packages: ", paste(missing_packages, collapse = ", "),
    ". Install them before running this script."
  )
}

suppressPackageStartupMessages({
  library(readxl)
  library(metafor)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(svglite)
  library(purrr)
  library(stringr)
  library(forcats)
})

script_file <- sub(
  "^--file=", "",
  grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
)
script_dir <- if (length(script_file) > 0L) {
  dirname(normalizePath(script_file[[1]], winslash = "/", mustWork = TRUE))
} else {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
repo_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = TRUE)

input_file <- Sys.getenv(
  "SCREEN_EF_META_XLSX",
  file.path(repo_root, "data", "Meta-analysis_data_full_age_updated.xlsx")
)
included_reports_file <- Sys.getenv(
  "SCREEN_EF_INCLUDED_REPORTS_XLSX",
  file.path(repo_root, "data", "Table_1_final_verified.xlsx")
)
output_dir <- Sys.getenv("SCREEN_EF_META_OUTPUT", script_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input_file)) stop("Input workbook not found: ", input_file)
if (!file.exists(included_reports_file)) {
  stop("Final included-report workbook not found: ", included_reports_file)
}

theme_nhb <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 3, hjust = 0),
      plot.subtitle = element_text(size = base_size, colour = "#4D4D4D", hjust = 0),
      plot.caption = element_text(size = base_size - 1, colour = "#4D4D4D", hjust = 0),
      axis.title = element_text(size = base_size + 1),
      axis.text = element_text(size = base_size),
      strip.text = element_text(face = "bold", size = base_size + 1),
      legend.position = "none",
      plot.margin = margin(3, 3, 3, 3)
    )
}

clean_text <- function(x) {
  x <- as.character(x)
  x <- str_replace_all(x, "\u2013|\u2014", "-")
  x <- str_replace_all(x, "\u2192", " to ")
  x <- str_replace_all(x, "\u2018|\u2019", "'")
  x <- str_replace_all(x, "\u00A0", " ")
  str_squish(x)
}

fmt_p <- function(p) {
  ifelse(
    is.na(p), "NA",
    ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
  )
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
}

fmt_n <- function(x) {
  ifelse(is.na(x), "NR", format(round(x), big.mark = ",", scientific = FALSE))
}

bt_r <- function(z) tanh(z)

safe_file <- function(x) {
  x <- str_replace_all(x, "[^A-Za-z0-9]+", "_")
  str_replace_all(x, "^_|_$", "")
}

meta_raw <- read_excel(input_file, sheet = "Meta_Data")

required_columns <- c(
  "Effect_ID", "Study_Group_ID", "Report_ID", "First_Author", "Year", "Country",
  "Study_design", "Longitudinal", "N", "Age_Group", "Screen_Category",
  "Screen_Nature", "Detailed_EF_Domain", "EF_Domain", "EF_Instrument",
  "Adjusted", "Harmonized_r", "Fisher_z", "Sampling_Variance_z",
  "Analysis_Tier", "Primary_Model", "Age_Mean_years", "Age_SD_years"
)
missing_columns <- setdiff(required_columns, names(meta_raw))
if (length(missing_columns) > 0L) {
  stop("Required columns missing from Meta_Data: ", paste(missing_columns, collapse = ", "))
}

if (anyDuplicated(meta_raw$Effect_ID)) stop("Effect_ID is not unique.")
if (any(!is.finite(meta_raw$Fisher_z))) stop("Non-finite Fisher_z values found.")
if (any(!is.finite(meta_raw$Sampling_Variance_z) | meta_raw$Sampling_Variance_z <= 0)) {
  stop("Sampling_Variance_z must be finite and greater than zero.")
}

region_from_country <- function(country) {
  case_when(
    country %in% c("United States", "Canada") ~ "North America",
    country %in% c("China", "Turkey") ~ "Asia",
    country %in% c("Australia") ~ "Oceania",
    country %in% c("Netherlands", "Romania", "Russia", "Spain", "United Kingdom") ~ "Europe",
    TRUE ~ "Other / not reported"
  )
}

measurement_mode <- function(instrument) {
  case_when(
    str_detect(str_to_lower(instrument), "brief|self-report|questionnaire|rating") ~
      "Rating scale / report",
    TRUE ~ "Performance task"
  )
}

age_stage <- function(age_group) {
  case_when(
    str_detect(str_to_lower(age_group), "early childhood") ~ "Early childhood",
    str_detect(str_to_lower(age_group), "middle childhood") ~ "Middle childhood",
    str_detect(str_to_lower(age_group), "adolescence") ~ "Adolescence",
    TRUE ~ "Mixed / not reported"
  )
}

screen_type <- function(category) {
  case_when(
    str_detect(str_to_lower(category), "passive|background") ~ "Passive / background viewing",
    str_detect(str_to_lower(category), "social|multitask") ~ "Social media / multitasking",
    str_detect(str_to_lower(category), "interactive|gaming|touchscreen") ~ "Interactive / gaming / apps",
    TRUE ~ "Total / mixed exposure"
  )
}

screen_feature <- function(nature) {
  case_when(
    str_detect(str_to_lower(nature), "passive") ~ "Passive",
    str_detect(str_to_lower(nature), "interactive|entertainment|social|multitask") ~ "Interactive / entertainment",
    TRUE ~ "Mixed / unspecified"
  )
}

meta_data <- meta_raw %>%
  mutate(
    across(where(is.character), clean_text),
    Year = suppressWarnings(as.numeric(Year)),
    Longitudinal = suppressWarnings(as.numeric(Longitudinal)),
    Adjusted = suppressWarnings(as.numeric(Adjusted)),
    N = suppressWarnings(as.numeric(N)),
    Harmonized_r = suppressWarnings(as.numeric(Harmonized_r)),
    Fisher_z = suppressWarnings(as.numeric(Fisher_z)),
    Sampling_Variance_z = suppressWarnings(as.numeric(Sampling_Variance_z)),
    Primary_Model = suppressWarnings(as.numeric(Primary_Model)),
    Age_Mean_years = suppressWarnings(as.numeric(Age_Mean_years)),
    Age_SD_years = suppressWarnings(as.numeric(Age_SD_years)),
    EF_dimension = case_when(
      str_detect(str_to_lower(EF_Domain), "^inhibitory") ~ "Inhibitory control",
      str_detect(str_to_lower(EF_Domain), "^working memory") ~ "Working memory",
      str_detect(str_to_lower(EF_Domain), "^cognitive flexibility") ~ "Cognitive flexibility",
      TRUE ~ NA_character_
    ),
    Region = region_from_country(Country),
    Age_stage = age_stage(Age_Group),
    Screen_type = screen_type(Screen_Category),
    Screen_feature = screen_feature(Screen_Nature),
    Measurement_mode = measurement_mode(EF_Instrument),
    Design_group = if_else(Longitudinal == 1, "Longitudinal", "Cross-sectional"),
    Adjustment_group = if_else(Adjusted == 1, "Adjusted", "Unadjusted"),
    Publication_period = case_when(
      Year <= 2019 ~ "2010-2019",
      Year <= 2022 ~ "2020-2022",
      TRUE ~ "2023-2026"
    ),
    Author_year = paste0(First_Author, ", ", Year),
    Effect_label = paste0(Author_year, " (", Effect_ID, ")"),
    sei = sqrt(Sampling_Variance_z),
    r_ci_lb = bt_r(Fisher_z - qnorm(0.975) * sei),
    r_ci_ub = bt_r(Fisher_z + qnorm(0.975) * sei),
    log_N = log(N),
    Age_Mean_centered = Age_Mean_years - mean(Age_Mean_years, na.rm = TRUE),
    Year_centered = Year - mean(Year, na.rm = TRUE),
    log_N_centered = log_N - mean(log_N, na.rm = TRUE)
  ) %>%
  filter(!is.na(EF_dimension))

domain_order <- c("Inhibitory control", "Working memory", "Cognitive flexibility")
meta_data$EF_dimension <- factor(meta_data$EF_dimension, levels = domain_order)

make_random_terms <- function(dat) {
  out <- list()
  if (n_distinct(dat$Report_ID) > 1L) out <- c(out, list(~ 1 | Report_ID))
  if (n_distinct(dat$Study_Group_ID) > 1L) out <- c(out, list(~ 1 | Study_Group_ID))
  if (n_distinct(dat$Effect_ID) > 1L) out <- c(out, list(~ 1 | Effect_ID))
  out
}

fit_meta <- function(dat, mods = NULL) {
  dat <- dat %>%
    filter(
      is.finite(Fisher_z), is.finite(Sampling_Variance_z),
      Sampling_Variance_z > 0
    )
  if (nrow(dat) < 2L) return(list(model = NULL, robust = NULL, data = dat))

  random_terms <- make_random_terms(dat)
  fit <- tryCatch(
    {
      if (length(random_terms) == 0L) {
        if (is.null(mods)) {
          rma.uni(
            yi = Fisher_z, vi = Sampling_Variance_z,
            data = dat, method = "REML"
          )
        } else {
          rma.uni(
            yi = Fisher_z, vi = Sampling_Variance_z, mods = mods,
            data = dat, method = "REML"
          )
        }
      } else {
        if (is.null(mods)) {
          rma.mv(
            yi = Fisher_z, V = Sampling_Variance_z,
            random = random_terms, data = dat, method = "REML",
            control = list(optimizer = "optim", optmethod = "BFGS")
          )
        } else {
          rma.mv(
            yi = Fisher_z, V = Sampling_Variance_z, mods = mods,
            random = random_terms, data = dat, method = "REML",
            control = list(optimizer = "optim", optmethod = "BFGS")
          )
        }
      }
    },
    error = function(e) {
      message("Primary optimizer failed; retrying with nlminb: ", conditionMessage(e))
      if (length(random_terms) == 0L) {
        if (is.null(mods)) {
          rma.uni(
            yi = Fisher_z, vi = Sampling_Variance_z,
            data = dat, method = "REML"
          )
        } else {
          rma.uni(
            yi = Fisher_z, vi = Sampling_Variance_z, mods = mods,
            data = dat, method = "REML"
          )
        }
      } else {
        if (is.null(mods)) {
          rma.mv(
            yi = Fisher_z, V = Sampling_Variance_z,
            random = random_terms, data = dat, method = "REML",
            control = list(optimizer = "nlminb")
          )
        } else {
          rma.mv(
            yi = Fisher_z, V = Sampling_Variance_z, mods = mods,
            random = random_terms, data = dat, method = "REML",
            control = list(optimizer = "nlminb")
          )
        }
      }
    }
  )

  robust_fit <- NULL
  if (n_distinct(dat$Report_ID) >= 3L) {
    robust_fit <- tryCatch(
      suppressWarnings(robust(fit, cluster = dat$Report_ID, adjust = TRUE)),
      error = function(e) NULL
    )
  }
  list(model = fit, robust = robust_fit, data = dat)
}

extract_intercept <- function(fit_object) {
  fit <- fit_object$model
  rob <- fit_object$robust
  if (is.null(fit)) {
    dat <- fit_object$data
    if (nrow(dat) == 1L) {
      est <- dat$Fisher_z[1]
      se <- sqrt(dat$Sampling_Variance_z[1])
      return(tibble(
        estimate_z = est, se = se, ci_lb_z = est - 1.96 * se,
        ci_ub_z = est + 1.96 * se, p = 2 * pnorm(-abs(est / se)),
        inference = "Single effect"
      ))
    }
    return(tibble(
      estimate_z = NA_real_, se = NA_real_, ci_lb_z = NA_real_,
      ci_ub_z = NA_real_, p = NA_real_, inference = "Not estimable"
    ))
  }
  source <- if (!is.null(rob)) rob else fit
  tibble(
    estimate_z = as.numeric(source$beta[1]),
    se = as.numeric(source$se[1]),
    ci_lb_z = as.numeric(source$ci.lb[1]),
    ci_ub_z = as.numeric(source$ci.ub[1]),
    p = as.numeric(source$pval[1]),
    inference = if (!is.null(rob)) "Cluster-robust CR1" else "Model-based"
  )
}

variance_summary <- function(fit_object) {
  fit <- fit_object$model
  dat <- fit_object$data
  sigma2 <- if (!is.null(fit$sigma2)) as.numeric(fit$sigma2) else numeric(0)
  sigma2 <- c(sigma2, rep(NA_real_, max(0, 3L - length(sigma2))))[1:3]
  total_tau2 <- sum(sigma2, na.rm = TRUE)
  mean_vi <- mean(dat$Sampling_Variance_z, na.rm = TRUE)
  i2 <- if ((total_tau2 + mean_vi) > 0) 100 * total_tau2 / (total_tau2 + mean_vi) else 0
  tibble(
    tau2_report = sigma2[1], tau2_cohort = sigma2[2], tau2_effect = sigma2[3],
    total_I2_percent = i2,
    Q = if (!is.null(fit$QE)) as.numeric(fit$QE) else NA_real_,
    Q_df = if (!is.null(fit$k)) as.numeric(fit$k - fit$p) else NA_real_,
    Q_p = if (!is.null(fit$QEp)) as.numeric(fit$QEp) else NA_real_
  )
}

make_diamonds <- function(rows) {
  pools <- rows %>% filter(row_type %in% c("Pooled", "Overall"), is.finite(estimate))
  if (nrow(pools) == 0L) return(tibble())
  map_dfr(seq_len(nrow(pools)), function(i) {
    z <- pools[i, ]
    tibble(
      x = c(z$ci_lb, z$estimate, z$ci_ub, z$estimate),
      y = c(z$y, z$y + 0.30, z$y, z$y - 0.30),
      diamond_id = i
    )
  })
}

forest_table_plot <- function(rows, figure_title, figure_subtitle, figure_caption,
                              output_file = NULL, height = NULL) {
  rows <- rows %>% mutate(y = rev(seq_len(n())))
  diamonds <- make_diamonds(rows)

  estimated_rows <- rows %>% filter(is.finite(estimate), is.finite(ci_lb), is.finite(ci_ub))
  x_min <- max(-0.99, min(c(estimated_rows$ci_lb, -0.05), na.rm = TRUE))
  x_max <- min(0.99, max(c(estimated_rows$ci_ub, 0.05), na.rm = TRUE))
  x_pad <- max(0.03, 0.06 * (x_max - x_min))
  x_limits <- c(max(-1, x_min - x_pad), min(1, x_max + x_pad))

  header_rows <- rows %>% filter(row_type == "ColumnHeader")
  label_rows <- rows %>% filter(row_type != "ColumnHeader")

  author_col <- ggplot(rows, aes(y = y)) +
    geom_text(
      data = label_rows,
      aes(x = 0, label = label),
      hjust = 0,
      family = "Arial",
      size = 3.20,
      fontface = ifelse(label_rows$row_type %in% c("Header", "Pooled", "Overall"), "bold", "plain")
    ) +
    geom_text(
      data = header_rows,
      aes(x = 0, label = label), hjust = 0, family = "Arial", size = 3.40, fontface = "bold"
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, max(rows$y) + 0.5), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(2, 0, 2, 2))

  n_col <- ggplot(rows, aes(y = y)) +
    geom_text(
      data = label_rows,
      aes(x = 1, label = N_label),
      hjust = 1,
      family = "Arial",
      size = 3.20,
      fontface = ifelse(label_rows$row_type %in% c("Pooled", "Overall"), "bold", "plain")
    ) +
    geom_text(
      data = header_rows,
      aes(x = 1, label = N_label), hjust = 1, family = "Arial", size = 3.40, fontface = "bold"
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, max(rows$y) + 0.5), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(2, 2, 2, 0))

  forest <- ggplot(rows, aes(y = y)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, colour = "#7F7F7F") +
    geom_segment(
      data = estimated_rows %>% filter(row_type == "Effect"),
      aes(x = ci_lb, xend = ci_ub, yend = y),
      linewidth = 0.38, colour = "#777777"
    ) +
    geom_point(
      data = estimated_rows %>% filter(row_type == "Effect"),
      aes(x = estimate, size = point_size),
      shape = 15, colour = "#19558C"
    ) +
    geom_polygon(
      data = diamonds,
      aes(x = x, y = y, group = diamond_id),
      inherit.aes = FALSE, fill = "#C7362F", colour = "#C7362F"
    ) +
    scale_size_continuous(range = c(1.1, 2.7), guide = "none") +
    scale_x_continuous(limits = x_limits, breaks = pretty(x_limits, n = 5)) +
    scale_y_continuous(limits = c(0.5, max(rows$y) + 0.5), expand = c(0, 0)) +
    labs(x = "Correlation coefficient (r)", y = NULL) +
    theme_nhb(9.5) +
    theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank(),
      panel.grid.major.y = element_line(colour = "#EEEEEE", linewidth = 0.25),
      plot.margin = margin(2, 4, 2, 4)
    )

  right <- ggplot(rows, aes(y = y)) +
    geom_text(
      data = label_rows,
      aes(x = 0, label = result_label),
      hjust = 0, family = "Arial", size = 3.20,
      fontface = ifelse(label_rows$row_type %in% c("Pooled", "Overall"), "bold", "plain")
    ) +
    geom_text(
      data = header_rows,
      aes(x = 0, label = result_label),
      hjust = 0, family = "Arial", size = 3.40, fontface = "bold"
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, max(rows$y) + 0.5), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(2, 2, 2, 4))

  assembled <- (author_col | n_col | forest | right) +
    plot_layout(widths = c(0.15, 0.055, 0.565, 0.23)) +
    plot_annotation(
      title = figure_title,
      subtitle = figure_subtitle,
      caption = str_wrap(figure_caption, width = 175),
      theme = theme(
        text = element_text(family = "Arial", colour = "#111111"),
        plot.title = element_text(face = "bold", size = 14, hjust = 0),
        plot.subtitle = element_text(size = 10, colour = "#555555", hjust = 0),
        plot.caption = element_text(size = 8.5, colour = "#555555", hjust = 0),
        plot.margin = margin(4, 4, 4, 4)
      )
    )

  if (is.null(height)) height <- max(5.0, 1.9 + 0.215 * nrow(rows))
  if (!is.null(output_file)) {
    ggsave(
      filename = output_file, plot = assembled, device = svglite,
      width = 12, height = height, units = "in", bg = "white"
    )
  }
  invisible(list(plot = assembled, height = height))
}

main_result_rows <- list()
main_models <- list()
main_plot_objects <- list()
main_panel_letters <- setNames(c("A", "B", "C"), domain_order)

for (domain in domain_order) {
  dat <- meta_data %>% filter(EF_dimension == domain) %>% arrange(Year, First_Author, Effect_ID)
  fit <- fit_meta(dat)
  main_models[[domain]] <- fit
  pooled <- extract_intercept(fit)
  variance <- variance_summary(fit)

  main_result_rows[[domain]] <- bind_cols(
    tibble(
      EF_dimension = domain,
      effects = nrow(dat), reports = n_distinct(dat$Report_ID),
      cohorts = n_distinct(dat$Study_Group_ID)
    ),
    pooled %>% mutate(
      pooled_r = bt_r(estimate_z), ci_lb_r = bt_r(ci_lb_z), ci_ub_r = bt_r(ci_ub_z)
    ),
    variance
  )

  effect_rows <- dat %>%
    transmute(
      row_type = "Effect", label = Author_year, N_label = fmt_n(N),
      estimate = Harmonized_r, ci_lb = r_ci_lb, ci_ub = r_ci_ub,
      result_label = sprintf("%.3f [%.3f, %.3f]", Harmonized_r, r_ci_lb, r_ci_ub),
      point_size = pmin(1 / sqrt(Sampling_Variance_z), quantile(1 / sqrt(Sampling_Variance_z), 0.90))
    )

  overall_row <- tibble(
    row_type = "Overall", label = "Overall", N_label = "",
    estimate = bt_r(pooled$estimate_z), ci_lb = bt_r(pooled$ci_lb_z), ci_ub = bt_r(pooled$ci_ub_z),
    result_label = sprintf(
      "%.3f [%.3f, %.3f]", bt_r(pooled$estimate_z), bt_r(pooled$ci_lb_z), bt_r(pooled$ci_ub_z)
    ),
    point_size = NA_real_
  )

  rows <- bind_rows(
    tibble(
      row_type = "ColumnHeader", label = "Author, year", N_label = "N",
      estimate = NA_real_, ci_lb = NA_real_, ci_ub = NA_real_,
      result_label = "r [95% CI]", point_size = NA_real_
    ),
    effect_rows,
    overall_row
  )

  heterogeneity <- sprintf(
    paste0(
      "Heterogeneity: Q(%d) = %.2f, p %s; tau2 report = %.4f, ",
      "tau2 cohort = %.4f, tau2 effect = %.4f; variance-based total I2 = %.1f%%."
    ),
    variance$Q_df, variance$Q,
    ifelse(variance$Q_p < 0.001, "< 0.001", paste0("= ", fmt_p(variance$Q_p))),
    variance$tau2_report, variance$tau2_cohort, variance$tau2_effect,
    variance$total_I2_percent
  )

  main_plot_objects[[domain]] <- forest_table_plot(
    rows = rows,
    figure_title = paste0(
      main_panel_letters[[domain]], "  Screen exposure and ", str_to_lower(domain)
    ),
    figure_subtitle = sprintf(
      "Three-level random-effects model: %d effects, %d reports, %d cohort/sample clusters",
      nrow(dat), n_distinct(dat$Report_ID), n_distinct(dat$Study_Group_ID)
    ),
    figure_caption = paste0(
      heterogeneity,
      " Pooled confidence intervals use report-clustered CR1 inference; individual confidence intervals are sampling intervals."
    ),
    output_file = NULL
  )
}

main_results <- bind_rows(main_result_rows)

main_file_names <- c(
  "Inhibitory control" = "01A_Main_inhibitory_control.svg",
  "Working memory" = "01B_Main_working_memory.svg",
  "Cognitive flexibility" = "01C_Main_cognitive_flexibility.svg"
)
for (domain in domain_order) {
  ggsave(
    file.path(output_dir, main_file_names[[domain]]),
    main_plot_objects[[domain]]$plot,
    device = svglite, width = 10.8,
    height = main_plot_objects[[domain]]$height,
    units = "in", bg = "white", limitsize = FALSE
  )
}
unlink(file.path(output_dir, "01_Main_analysis_three_domains.svg"))

collapse_sparse_levels <- function(dat, variable, minimum_reports = 3L) {
  level_counts <- dat %>%
    mutate(.level = if_else(is.na(.data[[variable]]) | .data[[variable]] == "", "Not reported", as.character(.data[[variable]]))) %>%
    group_by(.level) %>%
    summarise(reports = n_distinct(Report_ID), .groups = "drop")
  sparse <- level_counts %>% filter(reports < minimum_reports) %>% pull(.level)
  dat %>%
    mutate(
      Subgroup_level = if_else(
        is.na(.data[[variable]]) | .data[[variable]] == "", "Not reported", as.character(.data[[variable]])
      ),
      Subgroup_level = if_else(Subgroup_level %in% sparse, "Other / sparse", Subgroup_level)
    )
}

robust_moderator_test <- function(fit_object) {
  fit <- fit_object$model
  rob <- fit_object$robust
  if (is.null(fit) || is.null(rob) || nrow(rob$beta) <= 1L) {
    return(tibble(F = NA_real_, df1 = NA_real_, df2 = NA_real_, p = NA_real_))
  }
  idx <- seq.int(2L, nrow(rob$beta))
  b <- as.numeric(rob$beta[idx, , drop = FALSE])
  V <- rob$vb[idx, idx, drop = FALSE]
  q <- length(b)
  W <- as.numeric(t(b) %*% MASS::ginv(V) %*% b)
  df2 <- max(1, n_distinct(fit_object$data$Report_ID) - nrow(rob$beta))
  F_stat <- W / q
  tibble(F = F_stat, df1 = q, df2 = df2, p = pf(F_stat, q, df2, lower.tail = FALSE))
}

subgroup_variables <- c(
  Age_stage = "Developmental stage",
  Region = "Geographic region",
  Screen_type = "Screen exposure type",
  Screen_feature = "Screen exposure feature",
  Design_group = "Study design",
  Adjustment_group = "Adjustment status",
  Measurement_mode = "Executive-function measurement"
)

subgroup_result_rows <- list()
subgroup_index <- 0L

for (domain in domain_order) {
  domain_data <- meta_data %>% filter(EF_dimension == domain)
  for (variable in names(subgroup_variables)) {
    subgroup_index <- subgroup_index + 1L
    display_name <- subgroup_variables[[variable]]
    dat <- collapse_sparse_levels(domain_data, variable, minimum_reports = 1L)
    dat$Subgroup_level <- factor(dat$Subgroup_level, levels = unique(dat$Subgroup_level))

    moderator_fit <- NULL
    moderator_test <- tibble(F = NA_real_, df1 = NA_real_, df2 = NA_real_, p = NA_real_)
    if (n_distinct(dat$Subgroup_level) >= 2L) {
      moderator_fit <- fit_meta(dat, mods = ~ Subgroup_level)
      moderator_test <- robust_moderator_test(moderator_fit)
    }

    level_results <- list()
    for (level in levels(dat$Subgroup_level)) {
      level_data <- dat %>% filter(Subgroup_level == level) %>% arrange(Year, First_Author, Effect_ID)
      if (nrow(level_data) == 0L) next
      level_fit <- fit_meta(level_data)
      pooled <- extract_intercept(level_fit)

      level_results[[level]] <- tibble(
        EF_dimension = domain, moderator = variable, moderator_label = display_name,
        level = as.character(level), effects = nrow(level_data),
        reports = n_distinct(level_data$Report_ID), cohorts = n_distinct(level_data$Study_Group_ID),
        estimate_z = pooled$estimate_z, ci_lb_z = pooled$ci_lb_z, ci_ub_z = pooled$ci_ub_z,
        pooled_r = bt_r(pooled$estimate_z), ci_lb_r = bt_r(pooled$ci_lb_z),
        ci_ub_r = bt_r(pooled$ci_ub_z), p = pooled$p, inference = pooled$inference,
        interaction_F = moderator_test$F, interaction_df1 = moderator_test$df1,
        interaction_df2 = moderator_test$df2, interaction_p = moderator_test$p
      )
    }

    subgroup_result_rows[[subgroup_index]] <- bind_rows(level_results)
  }
}

subgroup_results <- bind_rows(subgroup_result_rows)

compact_subgroup_panel <- function(result_data, overall_data, panel_letter, domain) {
  panel_rows <- result_data %>%
    arrange(level) %>%
    transmute(
      level, effects, reports,
      estimate = pooled_r, ci_lb = ci_lb_r, ci_ub = ci_ub_r, p,
      row_type = "Subgroup"
    ) %>%
    bind_rows(
      overall_data %>%
        transmute(
          level = "Overall", effects, reports,
          estimate = pooled_r, ci_lb = ci_lb_r, ci_ub = ci_ub_r, p,
          row_type = "Overall"
        )
    ) %>%
    mutate(
      y = rev(seq_len(n())),
      estimate_label = sprintf("%.3f", estimate),
      ci_label = sprintf("(%.3f, %.3f)", ci_lb, ci_ub),
      p_label = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
    )

  moderator_row <- result_data %>% slice(1)
  moderator_text <- if (nrow(moderator_row) == 1L && is.finite(moderator_row$interaction_p)) {
    sprintf(
      "Omnibus moderator test: F(%d, %.0f) = %.2f, p %s",
      moderator_row$interaction_df1, moderator_row$interaction_df2,
      moderator_row$interaction_F,
      ifelse(
        moderator_row$interaction_p < 0.001, "< 0.001",
        paste0("= ", fmt_p(moderator_row$interaction_p))
      )
    )
  } else {
    "Omnibus moderator test not estimable"
  }

  header_y <- max(panel_rows$y) + 1.0
  subtitle_y <- header_y + 0.48
  title_y <- header_y + 0.94
  y_upper <- header_y + 1.16
  table_plot <- ggplot(panel_rows, aes(y = y)) +
    geom_segment(
      aes(x = 0, xend = 1, y = header_y - 0.42, yend = header_y - 0.42),
      inherit.aes = FALSE, colour = "#8A8A8A", linewidth = 0.35
    ) +
    geom_text(
      aes(x = 0.00, label = level), hjust = 0, family = "Arial", size = 3.45,
      fontface = ifelse(panel_rows$row_type == "Overall", "bold", "plain")
    ) +
    geom_text(
      aes(x = 0.37, label = effects), hjust = 1, family = "Arial", size = 3.45,
      fontface = ifelse(panel_rows$row_type == "Overall", "bold", "plain")
    ) +
    geom_text(
      aes(x = 0.50, label = reports), hjust = 1, family = "Arial", size = 3.45,
      fontface = ifelse(panel_rows$row_type == "Overall", "bold", "plain")
    ) +
    geom_text(
      aes(x = 0.64, label = estimate_label), hjust = 1, family = "Arial", size = 3.45,
      fontface = ifelse(panel_rows$row_type == "Overall", "bold", "plain")
    ) +
    geom_text(
      aes(x = 0.91, label = ci_label), hjust = 1, family = "Arial", size = 3.45,
      fontface = ifelse(panel_rows$row_type == "Overall", "bold", "plain")
    ) +
    geom_text(
      aes(x = 1.00, label = p_label), hjust = 1, family = "Arial", size = 3.45,
      fontface = ifelse(panel_rows$row_type == "Overall", "bold", "plain")
    ) +
    annotate("text", x = 0.00, y = title_y, label = paste0(panel_letter, "  ", domain),
             hjust = 0, family = "Arial", fontface = "bold", size = 4.10) +
    annotate("text", x = 0.00, y = subtitle_y, label = moderator_text,
             hjust = 0, family = "Arial", colour = "#555555", size = 3.00) +
    annotate("text", x = 0.00, y = header_y, label = "Category", hjust = 0,
             family = "Arial", fontface = "bold", size = 3.45) +
    annotate("text", x = 0.37, y = header_y, label = "k", hjust = 1,
             family = "Arial", fontface = "bold", size = 3.45) +
    annotate("text", x = 0.50, y = header_y, label = "Reports", hjust = 1,
             family = "Arial", fontface = "bold", size = 3.45) +
    annotate("text", x = 0.64, y = header_y, label = "Estimate", hjust = 1,
             family = "Arial", fontface = "bold", size = 3.45) +
    annotate("text", x = 0.91, y = header_y, label = "95% CI", hjust = 1,
             family = "Arial", fontface = "bold", size = 3.45) +
    annotate("text", x = 1.00, y = header_y, label = "P", hjust = 1,
             family = "Arial", fontface = "bold", size = 3.45) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.45, y_upper), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(2, 7, 2, 2))

  overall_diamond <- panel_rows %>%
    filter(row_type == "Overall") %>%
    transmute(
      x = list(c(ci_lb, estimate, ci_ub, estimate)),
      y_poly = list(c(y, y + 0.21, y, y - 0.21))
    ) %>%
    unnest(c(x, y_poly))

  forest_plot <- ggplot(panel_rows, aes(y = y)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "#8C8C8C", linewidth = 0.45) +
    geom_vline(
      xintercept = c(-0.6, -0.4, -0.2, 0.2), linetype = "dotted",
      colour = "#E4E4E4", linewidth = 0.35
    ) +
    geom_segment(
      data = panel_rows %>% filter(row_type == "Subgroup"),
      aes(x = ci_lb, xend = ci_ub, yend = y), colour = "#9A9A9A", linewidth = 0.55
    ) +
    geom_point(
      data = panel_rows %>% filter(row_type == "Subgroup"),
      aes(x = estimate), shape = 15, size = 2.8, colour = "#19558C"
    ) +
    geom_polygon(
      data = overall_diamond, aes(x = x, y = y_poly), inherit.aes = FALSE,
      fill = "#C7362F", colour = "#C7362F"
    ) +
    scale_x_continuous(
      limits = c(-0.65, 0.30), breaks = c(-0.6, -0.4, -0.2, 0, 0.2),
      expand = c(0, 0)
    ) +
    scale_y_continuous(limits = c(0.45, y_upper), expand = c(0, 0)) +
    labs(x = "Correlation coefficient (r)", y = NULL) +
    theme_nhb(10) +
    theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank(),
      axis.line.y = element_blank(), plot.margin = margin(2, 2, 2, 7)
    )

  assembled <- (table_plot | forest_plot) +
    plot_layout(widths = c(0.66, 0.34))

  list(plot = assembled, height = max(2.55, 1.55 + 0.34 * nrow(panel_rows)))
}

subgroup_plot_store <- setNames(vector("list", length(subgroup_variables)), names(subgroup_variables))
panel_letters <- c("A", "B", "C")
for (variable in names(subgroup_variables)) {
  for (i in seq_along(domain_order)) {
    domain <- domain_order[[i]]
    panel_results <- subgroup_results %>%
      filter(moderator == variable, EF_dimension == domain)
    overall_results <- main_results %>% filter(EF_dimension == domain)
    subgroup_plot_store[[variable]][[domain]] <- compact_subgroup_panel(
      result_data = panel_results,
      overall_data = overall_results,
      panel_letter = panel_letters[[i]],
      domain = domain
    )
  }
}

subgroup_file_names <- c(
  Age_stage = "02_Subgroup_age_stage.svg",
  Region = "03_Subgroup_region.svg",
  Screen_type = "04_Subgroup_screen_type.svg",
  Screen_feature = "05_Subgroup_screen_feature.svg",
  Design_group = "06_Subgroup_study_design.svg",
  Adjustment_group = "07_Subgroup_adjustment_status.svg",
  Measurement_mode = "08_Subgroup_measurement_mode.svg"
)

for (variable in names(subgroup_plot_store)) {
  objects <- subgroup_plot_store[[variable]][domain_order]
  figure <- wrap_plots(lapply(objects, `[[`, "plot"), ncol = 1) +
    plot_annotation(
      title = paste("Subgroup analysis by", str_to_lower(subgroup_variables[[variable]])),
      subtitle = "Three-level pooled subgroup estimates; individual study estimates are omitted for readability",
      caption = str_wrap(paste0(
        "k denotes the number of effect sizes. Reports denotes unique reports. Estimates are pooled correlations with ",
        "report-clustered robust 95% confidence intervals when at least three independent reports were available. ",
        "For smaller categories, the displayed confidence interval is model-based or the sampling interval for a single effect."
      ), width = 145),
      theme = theme(
        text = element_text(family = "Arial", colour = "#111111"),
        plot.title = element_text(face = "bold", size = 15, hjust = 0),
        plot.subtitle = element_text(size = 10.5, colour = "#555555", hjust = 0),
        plot.caption = element_text(size = 8.5, colour = "#555555", hjust = 0),
        plot.margin = margin(5, 5, 5, 5)
      )
    )
  ggsave(
    file.path(output_dir, subgroup_file_names[[variable]]),
    figure, device = svglite, width = 12,
    height = sum(vapply(objects, `[[`, numeric(1), "height")) + 1.05,
    units = "in", bg = "white", limitsize = FALSE
  )
}

fit_regression <- function(dat, xvar, x_label, domain) {
  reg_data <- dat %>% filter(is.finite(.data[[xvar]]))
  if (n_distinct(reg_data[[xvar]]) < 3L || n_distinct(reg_data$Report_ID) < 5L) {
    return(list(plot = NULL, result = NULL, fit = NULL))
  }
  formula_mod <- as.formula(paste("~", xvar))
  fit <- fit_meta(reg_data, mods = formula_mod)
  source <- if (!is.null(fit$robust)) fit$robust else fit$model
  beta <- as.numeric(source$beta)
  Vbeta <- source$vb
  df <- if (!is.null(fit$robust)) max(1, n_distinct(reg_data$Report_ID) - length(beta)) else Inf
  crit <- if (is.finite(df)) qt(0.975, df) else qnorm(0.975)

  x_seq <- seq(min(reg_data[[xvar]]), max(reg_data[[xvar]]), length.out = 120)
  X <- cbind(1, x_seq)
  pred <- as.numeric(X %*% beta)
  pred_se <- sqrt(pmax(0, diag(X %*% Vbeta %*% t(X))))
  pred_data <- tibble(
    x = x_seq, pred = pred, ci_lb = pred - crit * pred_se, ci_ub = pred + crit * pred_se
  )

  slope <- beta[2]
  slope_se <- as.numeric(source$se[2])
  slope_p <- as.numeric(source$pval[2])
  slope_ci_lb <- as.numeric(source$ci.lb[2])
  slope_ci_ub <- as.numeric(source$ci.ub[2])

  plot <- ggplot(reg_data, aes(x = .data[[xvar]], y = Fisher_z)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "#777777", linewidth = 0.45) +
    geom_ribbon(
      data = pred_data, aes(x = x, ymin = ci_lb, ymax = ci_ub),
      inherit.aes = FALSE, fill = "#C8D7E6", alpha = 0.65
    ) +
    geom_line(
      data = pred_data, aes(x = x, y = pred), inherit.aes = FALSE,
      colour = "#1F5A87", linewidth = 1.05
    ) +
    geom_point(
      aes(size = pmin(1 / sqrt(Sampling_Variance_z), quantile(1 / sqrt(Sampling_Variance_z), 0.90))),
      colour = "#6D6D6D", alpha = 0.78
    ) +
    annotate(
      "text", x = -Inf, y = Inf,
      label = sprintf("b = %.3f [%.3f, %.3f], p %s", slope, slope_ci_lb, slope_ci_ub,
                      ifelse(slope_p < 0.001, "< 0.001", paste0("= ", fmt_p(slope_p)))),
      hjust = -0.05, vjust = 1.35, family = "Arial", fontface = "italic", size = 3.2
    ) +
    scale_size_continuous(range = c(1.4, 4.0), guide = "none") +
    labs(x = x_label, y = "Fisher's z", title = x_label) +
    theme_nhb(9.5)

  result <- tibble(
    EF_dimension = domain, moderator = xvar, moderator_label = x_label,
    effects = nrow(reg_data), reports = n_distinct(reg_data$Report_ID),
    beta = slope, se = slope_se, ci_lb = slope_ci_lb, ci_ub = slope_ci_ub,
    p = slope_p,
    inference = if (!is.null(fit$robust)) "Cluster-robust CR1" else "Model-based"
  )
  list(plot = plot, result = result, fit = fit)
}

regression_results_list <- list()
regression_models <- list()
regression_index <- 0L
regression_plot_store <- list()

for (i in seq_along(domain_order)) {
  domain <- domain_order[[i]]
  dat <- meta_data %>% filter(EF_dimension == domain)
  age_fit <- fit_regression(dat, "Age_Mean_years", "Baseline mean age (years)", domain)
  year_fit <- fit_regression(dat, "Year", "Publication year", domain)
  n_fit <- fit_regression(dat, "log_N", "Log sample size", domain)

  if (!is.null(age_fit$plot)) {
    age_fit$plot <- age_fit$plot +
      labs(title = paste0(main_panel_letters[[domain]], "  ", domain, "\nBaseline mean age (years)"))
  }

  regression_plot_store[[domain]] <- list(age_fit$plot, year_fit$plot, n_fit$plot)

  for (obj in list(age_fit, year_fit, n_fit)) {
    if (!is.null(obj$result)) {
      regression_index <- regression_index + 1L
      regression_results_list[[regression_index]] <- obj$result
    }
  }
  regression_models[[domain]] <- list(
    baseline_mean_age = age_fit$fit,
    publication_year = year_fit$fit,
    log_sample_size = n_fit$fit
  )
}

regression_results <- bind_rows(regression_results_list)

regression_plots <- unlist(regression_plot_store[domain_order], recursive = FALSE)
regression_plots <- regression_plots[!vapply(regression_plots, is.null, logical(1))]
regression_figure <- wrap_plots(regression_plots, ncol = 3) +
  plot_annotation(
    title = "Meta-regression across the three core executive-function domains",
    subtitle = paste0(
      "A, inhibitory control; B, working memory; C, cognitive flexibility. Columns show baseline mean age, ",
      "publication year and log sample size. Longitudinal cohorts use baseline/exposure-wave age."
    ),
    theme = theme(
      text = element_text(family = "Arial"),
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, colour = "#555555")
    )
  )
ggsave(
  file.path(output_dir, "09_Meta_regression_age_year_sample_size.svg"),
  regression_figure, device = svglite, width = 15, height = 12.2,
  units = "in", bg = "white"
)

leave_one_cluster <- function(dat, cluster_var, domain, full_fit) {
  clusters <- unique(dat[[cluster_var]])
  full_pool <- extract_intercept(full_fit)
  out <- map_dfr(clusters, function(cluster_value) {
    reduced <- dat[dat[[cluster_var]] != cluster_value, , drop = FALSE]
    fit <- fit_meta(reduced)
    pooled <- extract_intercept(fit)
    excluded_rows <- dat[dat[[cluster_var]] == cluster_value, , drop = FALSE]
    excluded_label <- if (cluster_var == "Report_ID") {
      first(excluded_rows$Author_year)
    } else {
      as.character(cluster_value)
    }
    tibble(
      EF_dimension = domain, exclusion_level = cluster_var,
      excluded_id = as.character(cluster_value), excluded_label = excluded_label,
      excluded_effects = nrow(excluded_rows),
      estimate_z = pooled$estimate_z, ci_lb_z = pooled$ci_lb_z, ci_ub_z = pooled$ci_ub_z,
      pooled_r = bt_r(pooled$estimate_z), ci_lb_r = bt_r(pooled$ci_lb_z), ci_ub_r = bt_r(pooled$ci_ub_z),
      full_r = bt_r(full_pool$estimate_z), full_ci_lb_r = bt_r(full_pool$ci_lb_z),
      full_ci_ub_r = bt_r(full_pool$ci_ub_z)
    )
  })
  out
}

plot_leave_one_out <- function(results, title, subtitle, output_file = NULL) {
  dat <- results %>% arrange(pooled_r) %>% mutate(y = seq_len(n()))
  full_r <- first(dat$full_r)
  full_lb <- first(dat$full_ci_lb_r)
  full_ub <- first(dat$full_ci_ub_r)
  header_y <- nrow(dat) + 0.60
  subtitle_y <- nrow(dat) + 1.18
  title_y <- nrow(dat) + 1.72
  y_upper <- nrow(dat) + 2.02

  x_min <- max(-1, min(dat$ci_lb_r, full_lb, na.rm = TRUE) - 0.03)
  x_max <- min(1, max(dat$ci_ub_r, full_ub, na.rm = TRUE) + 0.03)

  left <- ggplot(dat, aes(y = y)) +
    geom_text(aes(x = 0, label = excluded_label), hjust = 0, family = "Arial", size = 4.00) +
    annotate(
      "text", x = 0, y = title_y, label = title,
      hjust = 0, family = "Arial", fontface = "bold", size = 4.60
    ) +
    annotate(
      "text", x = 0, y = subtitle_y, label = subtitle,
      hjust = 0, family = "Arial", colour = "#555555", size = 3.40
    ) +
    annotate(
      "text", x = 0, y = header_y, label = "Excluded author, year",
      hjust = 0, family = "Arial", fontface = "bold", size = 4.00
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, y_upper), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(2, 1, 2, 2))

  center <- ggplot(dat, aes(y = y)) +
    geom_vline(xintercept = full_r, colour = "#C7362F", linewidth = 0.7) +
    geom_vline(xintercept = c(full_lb, full_ub), colour = "#C7362F", linetype = "dotted", linewidth = 0.45) +
    geom_segment(aes(x = ci_lb_r, xend = ci_ub_r, yend = y), colour = "#555555", linewidth = 0.45) +
    geom_point(aes(x = pooled_r), shape = 15, colour = "#222222", size = 2.35) +
    scale_x_continuous(limits = c(x_min, x_max), breaks = pretty(c(x_min, x_max), n = 5)) +
    scale_y_continuous(limits = c(0.5, y_upper), expand = c(0, 0)) +
    labs(x = "Pooled correlation after exclusion (r)", y = NULL) +
    theme_nhb(11.5) +
    theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank(),
      panel.grid.major.y = element_line(colour = "#EEEEEE", linewidth = 0.25),
      plot.margin = margin(2, 2, 2, 1)
    )

  right <- ggplot(dat, aes(y = y)) +
    geom_text(
      aes(x = 0, label = sprintf("%.3f [%.3f, %.3f]", pooled_r, ci_lb_r, ci_ub_r)),
      hjust = 0, family = "Arial", size = 4.00
    ) +
    annotate(
      "text", x = 0, y = header_y, label = "r [95% CI]",
      hjust = 0, family = "Arial", fontface = "bold", size = 4.00
    ) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.5, y_upper), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(2, 2, 2, 3))

  assembled <- (left | center | right) +
    plot_layout(widths = c(0.21, 0.54, 0.25))

  plot_height <- max(5.2, 2.55 + 0.25 * nrow(dat))
  if (!is.null(output_file)) {
    ggsave(
      output_file, assembled, device = svglite, width = 12,
      height = plot_height, units = "in", bg = "white"
    )
  }
  invisible(list(plot = assembled, height = plot_height))
}

sensitivity_results_list <- list()
sensitivity_index <- 0L
report_loo_plots <- list()

for (i in seq_along(domain_order)) {
  domain <- domain_order[[i]]
  dat <- meta_data %>% filter(EF_dimension == domain)
  full_fit <- main_models[[domain]]

  report_loo <- leave_one_cluster(dat, "Report_ID", domain, full_fit)
  sensitivity_index <- sensitivity_index + 1L
  sensitivity_results_list[[sensitivity_index]] <- report_loo
  report_loo_plots[[domain]] <- plot_leave_one_out(
    report_loo,
    title = paste0(panel_letters[[i]], "  ", domain),
    subtitle = sprintf(
      "Full model: r = %.3f [%.3f, %.3f]",
      first(report_loo$full_r), first(report_loo$full_ci_lb_r), first(report_loo$full_ci_ub_r)
    ),
    output_file = NULL
  )
}

sensitivity_results <- bind_rows(sensitivity_results_list)

report_loo_figure <- wrap_plots(lapply(report_loo_plots[domain_order], `[[`, "plot"), ncol = 1) +
  plot_annotation(
    title = "Leave-one-report-out sensitivity analysis",
    subtitle = "Each refit removes all effect sizes contributed by one report",
    caption = str_wrap(paste0(
      "Solid red lines show the corresponding full-model estimate; dotted red lines show its 95% CI. ",
      "Report-level omission is retained as the primary leave-one-out analysis because it maps directly to individual publications."
    ), width = 145),
    theme = theme(
      text = element_text(family = "Arial"),
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 11.5, colour = "#555555"),
      plot.caption = element_text(size = 9.5, colour = "#555555", hjust = 0)
    )
  )
ggsave(
  file.path(output_dir, "10_Sensitivity_leave_one_report.svg"),
  report_loo_figure, device = svglite, width = 12,
  height = sum(vapply(report_loo_plots[domain_order], `[[`, numeric(1), "height")) + 1.0,
  units = "in", bg = "white", limitsize = FALSE
)
unlink(file.path(output_dir, "11_Sensitivity_leave_one_cohort.svg"))

model_set_results <- map_dfr(domain_order, function(domain) {
  all_dat <- meta_data %>% filter(EF_dimension == domain)
  primary_dat <- all_dat %>% filter(Primary_Model == 1)
  fits <- list(
    "All eligible effects" = fit_meta(all_dat),
    "Prespecified primary subset" = fit_meta(primary_dat)
  )
  imap_dfr(fits, function(fit, set_name) {
    pooled <- extract_intercept(fit)
    tibble(
      EF_dimension = domain, analysis_set = set_name,
      effects = nrow(fit$data), reports = n_distinct(fit$data$Report_ID),
      pooled_r = bt_r(pooled$estimate_z), ci_lb_r = bt_r(pooled$ci_lb_z), ci_ub_r = bt_r(pooled$ci_ub_z)
    )
  })
})

model_set_plot <- model_set_results %>%
  mutate(
    analysis_set = factor(analysis_set, levels = c("All eligible effects", "Prespecified primary subset")),
    EF_dimension = factor(EF_dimension, levels = rev(domain_order))
  ) %>%
  ggplot(aes(x = pooled_r, y = EF_dimension, colour = analysis_set, shape = analysis_set)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "#777777", linewidth = 0.45) +
  geom_errorbar(
    aes(xmin = ci_lb_r, xmax = ci_ub_r), orientation = "y",
    position = position_dodge(width = 0.45), width = 0.12, linewidth = 0.55
  ) +
  geom_point(position = position_dodge(width = 0.45), size = 2.8) +
  scale_colour_manual(values = c("#19558C", "#C7362F")) +
  labs(
    x = "Pooled correlation coefficient (r)", y = NULL,
    title = "Sensitivity to the prespecified analysis set",
    subtitle = "All eligible effects are compared with the prespecified primary subset",
    colour = NULL, shape = NULL
  ) +
  theme_nhb(10) +
  theme(legend.position = "top", legend.justification = "left")

ggsave(
  file.path(output_dir, "12_Sensitivity_analysis_set_comparison.svg"),
  model_set_plot, device = svglite, width = 8.5, height = 4.6, units = "in", bg = "white"
)

publication_bias_rows <- list()
publication_bias_index <- 0L
publication_bias_plots <- list()

funnel_triangle <- function(center, zcrit, max_se) {
  tibble(x = c(center, center - zcrit * max_se, center + zcrit * max_se), y = c(0, max_se, max_se))
}

for (i in seq_along(domain_order)) {
  domain <- domain_order[[i]]
  dat <- meta_data %>% filter(EF_dimension == domain)
  full_fit <- main_models[[domain]]
  pooled <- extract_intercept(full_fit)
  center <- pooled$estimate_z
  max_se <- max(dat$sei) * 1.04
  x_data_min <- min(dat$Fisher_z, center - 2.8 * max_se)
  x_data_max <- max(dat$Fisher_z, center + 2.8 * max_se)

  egger_fit <- fit_meta(dat, mods = ~ sei)
  egger_source <- if (!is.null(egger_fit$robust)) egger_fit$robust else egger_fit$model
  egger_beta <- as.numeric(egger_source$beta[2])
  egger_se <- as.numeric(egger_source$se[2])
  egger_p <- as.numeric(egger_source$pval[2])
  egger_lb <- as.numeric(egger_source$ci.lb[2])
  egger_ub <- as.numeric(egger_source$ci.ub[2])

  publication_bias_index <- publication_bias_index + 1L
  publication_bias_rows[[publication_bias_index]] <- tibble(
    EF_dimension = domain, effects = nrow(dat), reports = n_distinct(dat$Report_ID),
    egger_SE_slope = egger_beta, se = egger_se, ci_lb = egger_lb, ci_ub = egger_ub,
    p = egger_p,
    inference = if (!is.null(egger_fit$robust)) "Cluster-robust CR1" else "Model-based"
  )

  tri95 <- funnel_triangle(center, qnorm(0.975), max_se)
  standard <- ggplot(dat, aes(x = Fisher_z, y = sei)) +
    geom_polygon(data = tri95, aes(x = x, y = y), inherit.aes = FALSE, fill = "#EFEFEF", colour = NA) +
    geom_line(data = tri95[c(1, 2), ], aes(x = x, y = y), inherit.aes = FALSE, linetype = "dashed", colour = "#777777") +
    geom_line(data = tri95[c(1, 3), ], aes(x = x, y = y), inherit.aes = FALSE, linetype = "dashed", colour = "#777777") +
    geom_vline(xintercept = center, colour = "#C7362F", linewidth = 0.75) +
    geom_point(colour = "#666666", alpha = 0.85, size = 2.1) +
    scale_y_reverse(limits = c(max_se, 0)) +
    coord_cartesian(xlim = c(x_data_min, x_data_max)) +
    labs(
      title = paste0(panel_letters[[i]], "  ", domain, "\na  Funnel plot"),
      x = "Fisher's z", y = "Standard error"
    ) +
    theme_nhb(10)

  tri01 <- funnel_triangle(center, qnorm(0.995), max_se)
  tri05 <- funnel_triangle(center, qnorm(0.975), max_se)
  tri10 <- funnel_triangle(center, qnorm(0.95), max_se)
  contour <- ggplot(dat, aes(x = Fisher_z, y = sei)) +
    theme_nhb(10) +
    theme(panel.background = element_rect(fill = "#BDBDBD", colour = NA)) +
    geom_polygon(data = tri01, aes(x = x, y = y), inherit.aes = FALSE, fill = "#D7D7D7", colour = NA) +
    geom_polygon(data = tri05, aes(x = x, y = y), inherit.aes = FALSE, fill = "#ECECEC", colour = NA) +
    geom_polygon(data = tri10, aes(x = x, y = y), inherit.aes = FALSE, fill = "#FFFFFF", colour = NA) +
    geom_vline(xintercept = center, linetype = "dashed", colour = "#777777", linewidth = 0.55) +
    geom_point(colour = "#555555", alpha = 0.88, size = 2.1) +
    scale_y_reverse(limits = c(max_se, 0)) +
    coord_cartesian(xlim = c(x_data_min, x_data_max)) +
    labs(title = "b  Contour-enhanced funnel", x = "Fisher's z", y = "Standard error")

  funnel_figure <- (standard | contour) +
    plot_annotation(
      title = paste("Publication-bias diagnostics:", domain),
      subtitle = sprintf(
        "Multilevel Egger SE slope = %.3f [%.3f, %.3f], p %s",
        egger_beta, egger_lb, egger_ub,
        ifelse(egger_p < 0.001, "< 0.001", paste0("= ", fmt_p(egger_p)))
      ),
      caption = str_wrap(paste0(
        "All eligible effects are shown. The Egger test retains the three-level structure and uses report-clustered inference. ",
        "Funnel plots are exploratory because effect sizes within reports are dependent and residual heterogeneity is present."
      ), width = 185),
      theme = theme(
        text = element_text(family = "Arial"),
        plot.title = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(size = 9.5, colour = "#555555"),
        plot.caption = element_text(size = 8, colour = "#555555", hjust = 0)
      )
    )

  publication_bias_plots[[domain]] <- funnel_figure
}

publication_bias_results <- bind_rows(publication_bias_rows)

publication_bias_figure <- wrap_plots(publication_bias_plots[domain_order], ncol = 1)
ggsave(
  file.path(output_dir, "13_Publication_bias_three_domains.svg"),
  publication_bias_figure, device = svglite, width = 12, height = 15.8,
  units = "in", bg = "white"
)

# World map: all reports retained after final full-text adjudication, including
# reports without an extractable effect for the quantitative synthesis.
included_sheets <- excel_sheets(included_reports_file)
if ("Table 1" %in% included_sheets) {
  included_reports <- read_excel(included_reports_file, sheet = "Table 1", skip = 2)
} else if ("Table1" %in% included_sheets) {
  included_reports <- read_excel(included_reports_file, sheet = "Table1", skip = 4)
} else {
  stop("No Table 1/Table1 sheet found in included-report workbook.")
}
country_candidates <- intersect(c("Country", "Country / cohort"), names(included_reports))
if (length(country_candidates) == 0L) {
  stop("No Country or Country / cohort column found in included-report workbook.")
}
country_col <- country_candidates[[1]]
included_reports <- included_reports %>%
  filter(!is.na(.data[[country_col]]), str_squish(.data[[country_col]]) != "") %>%
  mutate(Included_Report_ID = row_number())

country_long <- included_reports %>%
  transmute(Included_Report_ID, Country_raw = .data[[country_col]]) %>%
  filter(!is.na(Country_raw), str_squish(Country_raw) != "", Country_raw != "NR") %>%
  separate_rows(Country_raw, sep = ";") %>%
  mutate(
    Country = str_trim(str_remove(Country_raw, "\\s*/.*$")),
    map_region = case_when(
      Country == "United States" ~ "USA",
      Country == "United Kingdom" ~ "UK",
      TRUE ~ Country
    )
  ) %>%
  distinct(Included_Report_ID, Country, map_region)

country_counts <- country_long %>%
  count(Country, map_region, name = "reports", sort = TRUE)

n_included_reports <- n_distinct(included_reports$Included_Report_ID)
n_geocoded_reports <- n_distinct(country_long$Included_Report_ID)
n_country_attributions <- sum(country_counts$reports)
n_countries <- n_distinct(country_counts$Country)

world_data <- map_data("world") %>%
  left_join(country_counts, by = c("region" = "map_region")) %>%
  mutate(reports = replace_na(reports, 0L))

map_panel <- ggplot(world_data, aes(long, lat, group = group)) +
  geom_polygon(aes(fill = reports), colour = "white", linewidth = 0.16) +
  scale_fill_gradientn(
    colours = c("#F1F3F5", "#C7D9E8", "#6F9FC2", "#19558C", "#0D3558"),
    values = scales::rescale(c(0, 1, 3, 8, max(country_counts$reports))),
    breaks = c(0, 1, 3, 5, 10, 15, 20),
    name = "Reports"
  ) +
  coord_quickmap(xlim = c(-170, 360), ylim = c(-58, 84), expand = FALSE) +
  theme_void(base_family = "Arial") +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    plot.margin = margin(4, 2, 4, 4)
  )

country_rank <- country_counts %>%
  arrange(reports, Country) %>%
  mutate(Country = factor(Country, levels = Country))

country_panel <- ggplot(country_rank, aes(x = reports, y = Country)) +
  geom_col(width = 0.66, fill = "#19558C") +
  geom_text(
    aes(label = reports), hjust = -0.35, family = "Arial", size = 2.8,
    colour = "#222222"
  ) +
  scale_x_continuous(
    limits = c(0, max(country_rank$reports) + 3),
    breaks = seq(0, max(country_rank$reports), by = 5), expand = c(0, 0)
  ) +
  labs(title = "Included reports by country", x = "Number of reports", y = NULL) +
  theme_nhb(8.5) +
  theme(
    plot.title = element_text(face = "bold", size = 10.5),
    axis.text.y = element_text(size = 7.6),
    panel.grid.major.x = element_line(colour = "#E6E6E6", linewidth = 0.3),
    axis.line.y = element_blank(), axis.ticks.y = element_blank(),
    panel.background = element_rect(fill = alpha("white", 0.94), colour = NA),
    plot.background = element_rect(fill = alpha("white", 0.94), colour = "#B8B8B8", linewidth = 0.35),
    plot.margin = margin(5, 5, 5, 5)
  )

world_map_figure <- map_panel +
  inset_element(
    country_panel,
    left = 0.685, bottom = 0.075, right = 0.995, top = 0.93,
    align_to = "full"
  ) +
  plot_annotation(
    title = "Geographic distribution of all included reports",
    subtitle = sprintf(
      "%d included reports; %d geocoded across %d countries (%d country attributions; one multi-country report)",
      n_included_reports, n_geocoded_reports, n_countries, n_country_attributions
    ),
    caption = paste0(
      "Counts are based on the final adjudicated Table 1, including reports without an extractable core-effect estimate. ",
      "They therefore exceed the 24 reports contributing to at least one of the three quantitative core-domain models."
    ),
    theme = theme(
      text = element_text(family = "Arial", colour = "#111111"),
      plot.title = element_text(face = "bold", size = 14, hjust = 0),
      plot.subtitle = element_text(size = 9.5, colour = "#555555", hjust = 0),
      plot.caption = element_text(size = 8, colour = "#555555", hjust = 0),
      plot.margin = margin(5, 5, 5, 5)
    )
  )

ggsave(
  file.path(output_dir, "14_World_map_study_distribution.svg"),
  world_map_figure, device = svglite, width = 14, height = 8.2,
  units = "in", bg = "white"
)

data_coverage <- meta_data %>%
  group_by(EF_dimension) %>%
  summarise(
    effects = n(), reports = n_distinct(Report_ID), cohorts = n_distinct(Study_Group_ID),
    countries = n_distinct(Country), year_min = min(Year), year_max = max(Year),
    .groups = "drop"
  )

moderator_availability <- tibble(
  requested_variable = c(
    "Mean age", "Female percentage", "Study quality score", "Publication year",
    "Geographic region", "Screen exposure type", "Screen exposure feature",
    "Screen-time dose", "Follow-up duration", "EF measurement mode"
  ),
  status = c(
    "Verified from full texts; longitudinal cohorts use baseline age", "Not present", "Not present", "Available",
    "Derived from Country", "Available and harmonized", "Available and harmonized",
    "Not present in a harmonized numeric unit", "Not present as a numeric variable",
    "Derived from EF_Instrument"
  ),
  analysis = c(
    "Meta-regression", "Not run", "Not run", "Subgroup and meta-regression",
    "Subgroup", "Subgroup", "Subgroup", "Not run", "Not run", "Subgroup"
  )
)

results_workbook <- list(
  Main_results = main_results,
  Subgroup_results = subgroup_results,
  Meta_regression = regression_results,
  Sensitivity = sensitivity_results,
  Analysis_set_comparison = model_set_results,
  Publication_bias = publication_bias_results,
  Data_coverage = data_coverage,
  Moderator_availability = moderator_availability
)

write.csv(main_results, file.path(output_dir, "main_results.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(subgroup_results, file.path(output_dir, "subgroup_results.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(regression_results, file.path(output_dir, "meta_regression_results.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(sensitivity_results, file.path(output_dir, "sensitivity_results.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(publication_bias_results, file.path(output_dir, "publication_bias_results.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(
  country_counts %>% select(Country, reports),
  file.path(output_dir, "included_report_country_counts.csv"),
  row.names = FALSE, fileEncoding = "UTF-8"
)

svg_files <- list.files(output_dir, pattern = "\\.svg$", full.names = FALSE)
run_manifest <- tibble(
  file = svg_files,
  category = case_when(
    str_detect(file, "^01[A-C]_") ~ "Main analysis",
    str_detect(file, "^0[2-8]_Subgroup") ~ "Subgroup analysis",
    str_starts(file, "09_") ~ "Meta-regression",
    str_detect(file, "^1[0-2]_Sensitivity") ~ "Sensitivity analysis",
    str_starts(file, "13_") ~ "Publication-bias analysis",
    str_starts(file, "14_") ~ "Geographic distribution",
    TRUE ~ "Other"
  )
)
write.csv(run_manifest, file.path(output_dir, "figure_manifest.csv"), row.names = FALSE, fileEncoding = "UTF-8")

session_lines <- capture.output(sessionInfo())
writeLines(session_lines, file.path(output_dir, "R_session_info.txt"), useBytes = TRUE)

cat("\nAnalysis complete.\n")
cat("Input:", input_file, "\n")
cat("Output directory:", output_dir, "\n")
cat("SVG files created:", length(svg_files), "\n\n")
print(main_results %>% select(EF_dimension, effects, reports, cohorts, pooled_r, ci_lb_r, ci_ub_r, p))
