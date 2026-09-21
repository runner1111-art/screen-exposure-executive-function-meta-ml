#!/usr/bin/env python3
"""Regenerate figures and the narrative report from completed ML CSV outputs."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "screen_ef_ml", SCRIPT_DIR / "01_run_machine_learning.py"
)
ml = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ml
spec.loader.exec_module(ml)


def main() -> None:
    out = ml.OUTPUT_DIR
    meta, proxy_raw, age = ml.read_inputs()
    primary = ml.prepare_primary(meta, proxy_raw, age)
    proxy = ml.prepare_proxy(proxy_raw)

    primary_metrics = ml.pd.read_csv(out / "ML_primary_performance_by_repeat.csv")
    proxy_metrics = ml.pd.read_csv(out / "ML_proxy_sensitivity_performance_by_repeat.csv")
    combined_metrics = ml.pd.concat([primary_metrics, proxy_metrics], ignore_index=True)
    summary = ml.pd.read_csv(out / "ML_model_performance_summary.csv")
    predictions = ml.pd.read_csv(
        out / "ML_primary_nested_OOF_predictions_all_repeats.csv"
    )
    importance = ml.pd.read_csv(out / "ML_group_block_permutation_importance.csv")
    pdp = ml.pd.read_csv(out / "ML_partial_dependence.csv")
    errors = ml.pd.read_csv(out / "ML_subgroup_prediction_error.csv")
    tuning = ml.pd.read_csv(out / "ML_final_model_tuning_audit.csv")

    best_model = ml.select_final_model(summary, "Meta-grade primary")
    chosen = tuning.loc[tuning["Inner_fold"] == 0].sort_values("RMSE").iloc[0]
    best_params = json.loads(chosen["Parameters"])
    pdp_age = pdp.loc[pdp["Feature"] == "Age_Mean_years"].copy()
    pdp_n = pdp.loc[pdp["Feature"] == "Log_N"].copy()
    domain_error = errors.loc[errors["Category_variable"] == "EF_Domain"].copy()
    screen_error = errors.loc[
        errors["Category_variable"] == "Screen_Category"
    ].copy()

    ml.plot_model_performance(primary_metrics, predictions, primary, best_model)
    ml.plot_interpretation(importance, pdp_age, pdp_n, domain_error, primary)
    ml.plot_sensitivity(combined_metrics, best_model, proxy, screen_error)
    ml.write_report(primary, proxy, summary, best_model, best_params, importance)

    config = {
        "seed": ml.SEED,
        "outer_folds": ml.OUTER_FOLDS,
        "outer_repeats": ml.OUTER_REPEATS,
        "inner_folds": ml.INNER_FOLDS,
        "permutations_per_outer_fold": ml.PERMUTATIONS,
        "primary_input": str(ml.META_XLSX),
        "proxy_input": str(ml.ML_XLSX),
        "age_input": str(ml.AGE_XLSX),
        "output_dir": str(out),
        "best_model": best_model,
        "best_parameters": best_params,
        "python": ml.sys.version,
        "platform": platform.platform(),
        "pandas": ml.pd.__version__,
        "numpy": ml.np.__version__,
        "scikit_learn": ml.sklearn.__version__,
        "matplotlib": ml.mpl.__version__,
        "rendered_from_cached_results": True,
    }
    (out / "ML_run_config_and_session.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
