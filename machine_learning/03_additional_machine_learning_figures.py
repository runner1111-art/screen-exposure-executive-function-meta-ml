#!/usr/bin/env python3
"""Generate five additional, conclusion-oriented ML figures.

This script reuses the locked primary dataset and repeated nested cohort-grouped
predictions produced by 01_run_machine_learning.py. It does not create an arbitrary
classification endpoint or reuse the target as a predictor.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "screen_ef_ml", SCRIPT_DIR / "01_run_machine_learning.py"
)
ml = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ml
spec.loader.exec_module(ml)

np = ml.np
pd = ml.pd
plt = ml.plt
mpl = ml.mpl

OUTPUT_DIR = ml.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def q025(values: Any) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    return float(np.quantile(values, 0.025)) if len(values) else np.nan


def q975(values: Any) -> float:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    return float(np.quantile(values, 0.975)) if len(values) else np.nan


def display_feature(name: str) -> str:
    mapping = {
        "Age_Mean_years": "Mean age",
        "Log_N": "Log sample size",
        "EF_Measure_Type": "EF measure type",
        "Study_design": "Study design",
        "Quality_Class": "Quality class",
        "Screen_Nature": "Screen nature",
        "EF_Domain": "EF domain",
        "Screen_Category": "Screen category",
    }
    return mapping.get(name, name.replace("_", " "))


def load_locked_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    primary = pd.read_csv(OUTPUT_DIR / "ML_primary_analysis_dataset.csv")
    predictions = pd.read_csv(
        OUTPUT_DIR / "ML_primary_nested_OOF_predictions_all_repeats.csv"
    )
    tuning = pd.read_csv(OUTPUT_DIR / "ML_primary_nested_tuning_audit.csv")
    config = json.loads(
        (OUTPUT_DIR / "ML_run_config_and_session.json").read_text(encoding="utf-8")
    )
    return primary, predictions, tuning, config


def run_learning_curve(
    primary: pd.DataFrame,
    best_model: str,
    best_params: dict[str, Any],
    repeats: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.array(pd.unique(primary["Group"]), dtype=object)
    n_test_groups = 5
    max_train = len(groups) - n_test_groups
    requested = [8, 12, 16, 20, 24, max_train]
    train_sizes = sorted({n for n in requested if 4 <= n <= max_train})
    x = primary[ml.PRIMARY_SPEC.all].copy()
    y = primary["Target"].to_numpy(float)
    group_values = primary["Group"].to_numpy(object)
    records: list[dict[str, Any]] = []

    for repeat in range(1, repeats + 1):
        rng = np.random.default_rng(ml.SEED + 700000 + repeat)
        shuffled = rng.permutation(groups)
        test_groups = shuffled[:n_test_groups]
        training_pool = shuffled[n_test_groups:]
        test_idx = np.flatnonzero(np.isin(group_values, test_groups))
        for n_groups in train_sizes:
            train_groups = training_pool[:n_groups]
            train_idx = np.flatnonzero(np.isin(group_values, train_groups))
            fit = ml.fit_pipeline(
                best_model,
                ml.PRIMARY_SPEC,
                best_params,
                x.iloc[train_idx],
                y[train_idx],
                group_values[train_idx],
            )
            predicted = fit.predict(x.iloc[test_idx])
            null_mean = ml.weighted_mean(
                y[train_idx], ml.group_balanced_weights(group_values[train_idx])
            )
            null_predicted = np.repeat(null_mean, len(test_idx))
            for model, pred in (
                (best_model, predicted),
                ("Null (training mean)", null_predicted),
            ):
                metrics = ml.regression_metrics(
                    y[test_idx], pred, group_values[test_idx]
                )
                records.append(
                    {
                        "Repeat": repeat,
                        "Training_groups": n_groups,
                        "Test_groups": n_test_groups,
                        "Model": model,
                        **metrics,
                    }
                )
    raw = pd.DataFrame(records)
    summary = (
        raw.groupby(["Training_groups", "Model"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_low=("RMSE", q025),
            RMSE_high=("RMSE", q975),
            Q2_mean=("Predictive_Q2", "mean"),
            Q2_low=("Predictive_Q2", q025),
            Q2_high=("Predictive_Q2", q975),
            MAE_mean=("MAE", "mean"),
        )
        .sort_values(["Model", "Training_groups"])
    )
    paired = raw.pivot_table(
        index=["Repeat", "Training_groups"], columns="Model", values="RMSE"
    ).reset_index()
    paired["Delta_RMSE_model_minus_null"] = (
        paired[best_model] - paired["Null (training mean)"]
    )
    delta = (
        paired.groupby("Training_groups", as_index=False)["Delta_RMSE_model_minus_null"]
        .agg(Delta_mean="mean", Delta_low=q025, Delta_high=q975)
    )
    summary = summary.merge(delta, on="Training_groups", how="left")
    return raw, summary


def plot_learning_curve(
    raw: pd.DataFrame, summary: pd.DataFrame, best_model: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    colors = {best_model: ml.BLUE, "Null (training mean)": ml.GREY}
    labels = {best_model: best_model, "Null (training mean)": "Training-mean comparator"}

    ax = axes[0]
    for model in ("Null (training mean)", best_model):
        part = summary.loc[summary["Model"] == model].sort_values("Training_groups")
        ax.plot(
            part["Training_groups"],
            part["RMSE_mean"],
            marker="o",
            linewidth=1.5,
            color=colors[model],
            label=labels[model],
        )
        ax.fill_between(
            part["Training_groups"].to_numpy(float),
            part["RMSE_low"].to_numpy(float),
            part["RMSE_high"].to_numpy(float),
            color=colors[model],
            alpha=0.14,
            linewidth=0,
        )
    ax.set_xlabel("Independent training clusters")
    ax.set_ylabel("Held-out RMSE (Fisher's z)")
    ax.legend(frameon=False, fontsize=6.8)
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "A")

    ax = axes[1]
    delta = summary.drop_duplicates("Training_groups").sort_values("Training_groups")
    ax.errorbar(
        delta["Training_groups"],
        delta["Delta_mean"],
        yerr=np.vstack(
            [
                delta["Delta_mean"] - delta["Delta_low"],
                delta["Delta_high"] - delta["Delta_mean"],
            ]
        ),
        fmt="s-",
        color=ml.ORANGE,
        ecolor=ml.GREY,
        linewidth=1.2,
        markersize=4.5,
        capsize=2,
    )
    ax.axhline(0, color=ml.GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Independent training clusters")
    ax.set_ylabel("ΔRMSE: model minus comparator")
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "B")

    ax = axes[2]
    for model in ("Null (training mean)", best_model):
        part = summary.loc[summary["Model"] == model].sort_values("Training_groups")
        ax.plot(
            part["Training_groups"],
            part["Q2_mean"],
            marker="o",
            linewidth=1.5,
            color=colors[model],
        )
        ax.fill_between(
            part["Training_groups"].to_numpy(float),
            part["Q2_low"].to_numpy(float),
            part["Q2_high"].to_numpy(float),
            color=colors[model],
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0, color=ml.GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Independent training clusters")
    ax.set_ylabel(r"Held-out predictive $Q^2$")
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "C")
    fig.subplots_adjust(wspace=0.42, left=0.08, right=0.99, bottom=0.20, top=0.96)
    ml.save_figure(fig, "Figure_ML4_learning_curve")


def cohort_labels(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in primary.groupby("Group", sort=False):
        authors = list(pd.unique(part["First_Author"].dropna().astype(str)))
        years = pd.to_numeric(part["Year"], errors="coerce").dropna()
        reports = part["Report_ID"].nunique()
        if reports == 1 and authors:
            year = str(int(years.iloc[0])) if len(years) else "NR"
            label = f"{authors[0]}, {year}"
        elif reports > 1:
            label = f"{group} ({reports} reports)"
        else:
            year = str(int(years.iloc[0])) if len(years) else "NR"
            label = f"{group}, {year}"
        rows.append({"Group": group, "Display_label": label})
    return pd.DataFrame(rows)


def cohort_error_audit(
    primary: pd.DataFrame, predictions: pd.DataFrame, best_model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = predictions.loc[predictions["Model"] == best_model].copy()
    part["Absolute_error"] = np.abs(part["Predicted"] - part["Observed"])
    part["Squared_error"] = (part["Predicted"] - part["Observed"]) ** 2
    part["Signed_error"] = part["Predicted"] - part["Observed"]
    repeat_group = (
        part.groupby(["Repeat", "Group"], as_index=False)
        .agg(
            MAE=("Absolute_error", "mean"),
            RMSE=("Squared_error", lambda x: math.sqrt(float(np.mean(x)))),
            Mean_bias=("Signed_error", "mean"),
            Effects=("Effect_ID", "nunique"),
        )
    )
    summary = (
        repeat_group.groupby("Group", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            MAE_low=("MAE", q025),
            MAE_high=("MAE", q975),
            RMSE=("RMSE", "mean"),
            Bias=("Mean_bias", "mean"),
            Bias_low=("Mean_bias", q025),
            Bias_high=("Mean_bias", q975),
            Effects=("Effects", "max"),
        )
        .merge(cohort_labels(primary), on="Group", how="left")
        .sort_values("MAE", ascending=False)
    )
    return repeat_group, summary


def plot_cohort_error(summary: pd.DataFrame) -> None:
    show = summary.sort_values("MAE", ascending=True).reset_index(drop=True)
    y = np.arange(len(show))
    fig, axes = plt.subplots(
        1, 2, figsize=(7.8, 8.8), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    ax = axes[0]
    ax.errorbar(
        show["MAE"],
        y,
        xerr=np.vstack(
            [show["MAE"] - show["MAE_low"], show["MAE_high"] - show["MAE"]]
        ),
        fmt="s",
        color=ml.BLUE,
        ecolor=ml.GREY,
        markersize=4.5,
        linewidth=0.9,
        capsize=2,
    )
    ax.set_yticks(y, show["Display_label"])
    ax.set_xlabel("Out-of-fold MAE (Fisher's z)")
    ax.set_ylabel("Cohort or independent sample cluster")
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "A")

    ax = axes[1]
    ax.errorbar(
        show["Bias"],
        y,
        xerr=np.vstack(
            [show["Bias"] - show["Bias_low"], show["Bias_high"] - show["Bias"]]
        ),
        fmt="s",
        color=ml.RED,
        ecolor=ml.GREY,
        markersize=4.5,
        linewidth=0.9,
        capsize=2,
    )
    ax.axvline(0, color=ml.GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean prediction bias\n(predicted minus observed z)")
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "B")
    fig.subplots_adjust(wspace=0.12, left=0.31, right=0.99, bottom=0.07, top=0.98)
    ml.save_figure(fig, "Figure_ML5_cohort_error_audit")


def error_matrix(
    primary: pd.DataFrame, predictions: pd.DataFrame, best_model: str
) -> pd.DataFrame:
    oof = ml.aggregate_oof(predictions, best_model)
    merged = oof.merge(
        primary[["Effect_ID", "EF_Domain", "Screen_Category"]],
        on="Effect_ID",
        how="left",
    )
    merged["Absolute_error"] = np.abs(merged["Predicted"] - merged["Observed"])
    group_cell = (
        merged.groupby(["EF_Domain", "Screen_Category", "Group"], as_index=False)
        .agg(Group_MAE=("Absolute_error", "mean"), Effects=("Effect_ID", "nunique"))
    )
    return (
        group_cell.groupby(["EF_Domain", "Screen_Category"], as_index=False)
        .agg(
            MAE=("Group_MAE", "mean"),
            Independent_groups=("Group", "nunique"),
            Effects=("Effects", "sum"),
        )
    )


def plot_error_matrix(matrix: pd.DataFrame) -> None:
    domain_order = [
        "Inhibitory control",
        "Working memory or updating",
        "Cognitive flexibility or shifting",
        "Global executive function",
        "Hot EF or self-regulation",
    ]
    screen_order = [
        "Passive or background viewing",
        "Interactive or gaming",
        "Social media or multitasking",
        "Total or mixed screen exposure",
    ]
    domain_order = [x for x in domain_order if x in set(matrix["EF_Domain"])]
    screen_order = [x for x in screen_order if x in set(matrix["Screen_Category"])]
    mae = matrix.pivot(index="EF_Domain", columns="Screen_Category", values="MAE").reindex(
        index=domain_order, columns=screen_order
    )
    groups = matrix.pivot(
        index="EF_Domain", columns="Screen_Category", values="Independent_groups"
    ).reindex(index=domain_order, columns=screen_order)
    effects = matrix.pivot(
        index="EF_Domain", columns="Screen_Category", values="Effects"
    ).reindex(index=domain_order, columns=screen_order)

    display_domains = {
        "Working memory or updating": "Working memory",
        "Cognitive flexibility or shifting": "Cognitive flexibility",
        "Global executive function": "Global EF",
        "Hot EF or self-regulation": "Hot EF/self-regulation",
    }
    display_screens = {
        "Passive or background viewing": "Passive/background",
        "Interactive or gaming": "Interactive/gaming",
        "Social media or multitasking": "Social/multitasking",
        "Total or mixed screen exposure": "Total/mixed",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.5))
    ax = axes[0]
    values = mae.to_numpy(float)
    support = groups.to_numpy(float)
    displayed = values.copy()
    displayed[(support < 3) | ~np.isfinite(support)] = np.nan
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad("#E6E6E6")
    im = ax.imshow(displayed, aspect="auto", cmap=cmap, vmin=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            g = support[i, j]
            if not np.isfinite(g):
                text = "No data"
            elif g < 3:
                text = f"Sparse\n(g={int(g)})"
            else:
                text = f"{values[i, j]:.3f}\n(g={int(g)})"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)
    ax.set_xticks(
        np.arange(len(screen_order)),
        [display_screens.get(x, x) for x in screen_order],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(domain_order)),
        [display_domains.get(x, x) for x in domain_order],
    )
    ax.set_xlabel("Screen-exposure category")
    ax.set_ylabel("Executive-function domain")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Group-balanced MAE")
    ml.add_panel_label(ax, "A")

    ax = axes[1]
    support_values = groups.fillna(0).to_numpy(float)
    im2 = ax.imshow(support_values, aspect="auto", cmap="Blues", vmin=0)
    effect_values = effects.fillna(0).to_numpy(float)
    for i in range(support_values.shape[0]):
        for j in range(support_values.shape[1]):
            ax.text(
                j,
                i,
                f"g={int(support_values[i, j])}\nk={int(effect_values[i, j])}",
                ha="center",
                va="center",
                fontsize=7,
            )
    ax.set_xticks(
        np.arange(len(screen_order)),
        [display_screens.get(x, x) for x in screen_order],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(domain_order)),
        ["" for _ in domain_order],
    )
    ax.set_xlabel("Screen-exposure category")
    ax.set_ylabel("")
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
    cbar2.set_label("Independent clusters")
    ml.add_panel_label(ax, "B")
    fig.subplots_adjust(wspace=0.34, left=0.15, right=0.98, bottom=0.28, top=0.96)
    ml.save_figure(fig, "Figure_ML6_error_support_heatmap")


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (repeat, fold, model), part in predictions.groupby(
        ["Repeat", "Fold", "Model"], sort=False
    ):
        metrics = ml.regression_metrics(
            part["Observed"].to_numpy(),
            part["Predicted"].to_numpy(),
            part["Group"],
        )
        rows.append({"Repeat": repeat, "Fold": fold, "Model": model, **metrics})
    out = pd.DataFrame(rows)
    null = out.loc[
        out["Model"] == "Null (training mean)", ["Repeat", "Fold", "RMSE"]
    ].rename(columns={"RMSE": "Null_RMSE"})
    out = out.merge(null, on=["Repeat", "Fold"], how="left")
    out["Delta_RMSE_vs_null"] = out["RMSE"] - out["Null_RMSE"]
    return out


def selection_stability(tuning: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    winners = (
        tuning.sort_values("Inner_RMSE")
        .groupby(["Repeat", "Fold"], as_index=False)
        .first()[["Repeat", "Fold", "Model", "Inner_RMSE"]]
    )
    winners_summary = (
        winners.groupby("Model", as_index=False)
        .agg(Outer_folds_won=("Fold", "count"))
        .sort_values("Outer_folds_won", ascending=False)
    )
    ridge = tuning.loc[tuning["Model"] == "Ridge"].copy()
    ridge["Alpha"] = ridge["Parameters"].map(
        lambda x: float(json.loads(x)["model__alpha"])
    )
    ridge_summary = (
        ridge.groupby("Alpha", as_index=False)
        .size()
        .rename(columns={"size": "Outer_folds_selected"})
        .sort_values("Alpha")
    )
    return winners_summary, ridge_summary


def plot_selection_stability(
    fold_data: pd.DataFrame,
    winners: pd.DataFrame,
    ridge_alpha: pd.DataFrame,
) -> None:
    model_order = ["Ridge", "Elastic net", "Random forest", "Gradient boosting"]
    model_order = [m for m in model_order if m in set(fold_data["Model"])]
    colors = [ml.LIGHT_BLUE, "#5A91BF", ml.ORANGE, ml.RED]
    fig, axes = plt.subplots(1, 3, figsize=(7.9, 2.9))

    ax = axes[0]
    win_map = dict(zip(winners["Model"], winners["Outer_folds_won"]))
    counts = [win_map.get(m, 0) for m in model_order]
    ax.barh(np.arange(len(model_order)), counts, color=colors[: len(model_order)])
    ax.set_yticks(np.arange(len(model_order)), model_order)
    ax.invert_yaxis()
    ax.set_xlabel("Inner-validation wins\n(out of 25 outer folds)")
    for i, value in enumerate(counts):
        ax.text(value + 0.2, i, str(value), va="center", fontsize=8)
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "A")

    ax = axes[1]
    data = [
        fold_data.loc[fold_data["Model"] == m, "Delta_RMSE_vs_null"].to_numpy()
        for m in model_order
    ]
    bp = ax.boxplot(
        data,
        tick_labels=model_order,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    rng = np.random.default_rng(ml.SEED + 810000)
    for i, values in enumerate(data, 1):
        ax.scatter(
            i + rng.normal(0, 0.035, len(values)),
            values,
            s=10,
            color=ml.BLACK,
            alpha=0.45,
            zorder=3,
        )
    ax.axhline(0, color=ml.GREY, linestyle="--", linewidth=0.8)
    ax.set_ylabel("Outer-fold ΔRMSE versus comparator")
    ax.tick_params(axis="x", rotation=38)
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "B")

    ax = axes[2]
    x = np.arange(len(ridge_alpha))
    ax.bar(x, ridge_alpha["Outer_folds_selected"], color=ml.BLUE, width=0.65)
    ax.set_xticks(x, [f"{v:g}" for v in ridge_alpha["Alpha"]])
    ax.set_xlabel("Ridge penalty α")
    ax.set_ylabel("Outer folds selected")
    for i, value in enumerate(ridge_alpha["Outer_folds_selected"]):
        ax.text(i, value + 0.25, str(value), ha="center", fontsize=8)
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "C")
    fig.subplots_adjust(wspace=0.52, left=0.12, right=0.99, bottom=0.27, top=0.96)
    ml.save_figure(fig, "Figure_ML7_model_selection_stability")


def map_transformed_features(preprocessor: Any) -> list[str]:
    names = list(preprocessor.get_feature_names_out())
    raw_features: list[str] = []
    numeric = list(ml.PRIMARY_SPEC.numeric + ml.PRIMARY_SPEC.binary)
    categorical = sorted(ml.PRIMARY_SPEC.categorical, key=len, reverse=True)
    for name in names:
        section, encoded = name.split("__", 1)
        if section == "num":
            encoded = encoded.replace("missingindicator_", "")
            matches = [raw for raw in numeric if raw in encoded]
            raw_features.append(max(matches, key=len) if matches else encoded)
        else:
            matches = [raw for raw in categorical if encoded.startswith(raw + "_")]
            raw_features.append(matches[0] if matches else encoded)
    return raw_features


def linear_contributions(
    primary: pd.DataFrame, best_model: str, best_params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = ml.fit_pipeline(
        best_model,
        ml.PRIMARY_SPEC,
        best_params,
        primary[ml.PRIMARY_SPEC.all],
        primary["Target"].to_numpy(float),
        primary["Group"],
    )
    transformed = fit.named_steps["preprocess"].transform(
        primary[ml.PRIMARY_SPEC.all]
    )
    coefficients = np.asarray(fit.named_steps["model"].coef_, float).ravel()
    weights = ml.group_balanced_weights(primary["Group"])
    center = np.average(transformed, axis=0, weights=weights)
    encoded_contributions = (transformed - center) * coefficients
    mapping = map_transformed_features(fit.named_steps["preprocess"])
    raw = pd.DataFrame(index=primary.index)
    for feature in ml.PRIMARY_SPEC.all:
        columns = [i for i, mapped in enumerate(mapping) if mapped == feature]
        raw[feature] = (
            encoded_contributions[:, columns].sum(axis=1) if columns else 0.0
        )
    raw["Effect_ID"] = primary["Effect_ID"].to_numpy()
    raw["Group"] = primary["Group"].to_numpy()

    rows = []
    for feature in ml.PRIMARY_SPEC.all:
        values = raw[feature].to_numpy(float)
        rows.append(
            {
                "Feature": feature,
                "Mean_absolute_contribution": ml.weighted_mean(np.abs(values), weights),
                "Mean_signed_contribution": ml.weighted_mean(values, weights),
                "Positive_fraction": ml.weighted_mean((values > 0).astype(float), weights),
                "Q025": float(np.quantile(values, 0.025)),
                "Q975": float(np.quantile(values, 0.975)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        "Mean_absolute_contribution", ascending=False
    )
    group_values = raw.groupby("Group", as_index=False)[ml.PRIMARY_SPEC.all].mean()
    return summary, group_values


def plot_linear_contributions(
    summary: pd.DataFrame, group_values: pd.DataFrame
) -> None:
    top = summary.head(10).sort_values("Mean_absolute_contribution", ascending=True)
    features = list(top["Feature"])
    y = np.arange(len(features))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.5))

    ax = axes[0]
    ax.barh(y, top["Mean_absolute_contribution"], color=ml.BLUE, alpha=0.88)
    ax.set_yticks(y, [display_feature(x) for x in features])
    ax.set_xlabel("Mean absolute linear contribution\nto predicted Fisher's z")
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "A")

    ax = axes[1]
    values = [group_values[feature].to_numpy(float) for feature in features]
    bp = ax.boxplot(
        values,
        vert=False,
        tick_labels=[display_feature(x) for x in features],
        patch_artist=True,
        widths=0.58,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(ml.LIGHT_BLUE)
        patch.set_alpha(0.78)
    rng = np.random.default_rng(ml.SEED + 820000)
    for i, vals in enumerate(values, 1):
        ax.scatter(
            vals,
            i + rng.normal(0, 0.055, len(vals)),
            s=9,
            color=ml.BLACK,
            alpha=0.38,
            zorder=3,
        )
    ax.axvline(0, color=ml.GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cluster-mean signed linear contribution")
    ml.clean_axis(ax)
    ml.add_panel_label(ax, "B")
    fig.subplots_adjust(wspace=0.64, left=0.19, right=0.99, bottom=0.15, top=0.97)
    ml.save_figure(fig, "Figure_ML8_linear_contribution_stability")


def main() -> None:
    primary, predictions, tuning, config = load_locked_outputs()
    best_model = str(config["best_model"])
    best_params = dict(config["best_parameters"])

    learning_raw, learning_summary = run_learning_curve(
        primary, best_model, best_params
    )
    learning_raw.to_csv(
        OUTPUT_DIR / "ML_learning_curve_resamples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    learning_summary.to_csv(
        OUTPUT_DIR / "ML_learning_curve_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_learning_curve(learning_raw, learning_summary, best_model)

    cohort_raw, cohort_summary = cohort_error_audit(
        primary, predictions, best_model
    )
    cohort_raw.to_csv(
        OUTPUT_DIR / "ML_cohort_error_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cohort_summary.to_csv(
        OUTPUT_DIR / "ML_cohort_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_cohort_error(cohort_summary)

    matrix = error_matrix(primary, predictions, best_model)
    matrix.to_csv(
        OUTPUT_DIR / "ML_EF_by_screen_error_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_error_matrix(matrix)

    outer_fold_metrics = fold_metrics(predictions)
    winners, ridge_alpha = selection_stability(tuning)
    outer_fold_metrics.to_csv(
        OUTPUT_DIR / "ML_outer_fold_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    winners.to_csv(
        OUTPUT_DIR / "ML_inner_model_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ridge_alpha.to_csv(
        OUTPUT_DIR / "ML_ridge_penalty_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_selection_stability(outer_fold_metrics, winners, ridge_alpha)

    # Figure ML8 is deliberately a transparent linear audit even when a
    # nonlinear algorithm wins the predictive comparison. Use the ridge
    # penalty selected most often across inner resamples rather than trying to
    # read nonexistent coefficients from a random-forest or boosting model.
    ridge_alpha_choice = float(
        ridge_alpha.sort_values(
            ["Outer_folds_selected", "Alpha"], ascending=[False, True]
        ).iloc[0]["Alpha"]
    )
    contribution_summary, group_contributions = linear_contributions(
        primary, "Ridge", {"model__alpha": ridge_alpha_choice}
    )
    contribution_summary.to_csv(
        OUTPUT_DIR / "ML_linear_contribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_contributions.to_csv(
        OUTPUT_DIR / "ML_linear_contributions_by_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_linear_contributions(contribution_summary, group_contributions)

    manifest = pd.DataFrame(
        [
            {
                "Figure": "Figure_ML4_learning_curve",
                "Purpose": "Data sufficiency and generalization as independent training clusters increase",
                "Recommended_location": "Main text",
            },
            {
                "Figure": "Figure_ML5_cohort_error_audit",
                "Purpose": "Identify cohorts or samples with concentrated prediction error and bias",
                "Recommended_location": "Supplementary information",
            },
            {
                "Figure": "Figure_ML6_error_support_heatmap",
                "Purpose": "Joint EF-domain by screen-category error with evidence support",
                "Recommended_location": "Main text",
            },
            {
                "Figure": "Figure_ML7_model_selection_stability",
                "Purpose": "Show algorithm and hyperparameter selection instability across grouped folds",
                "Recommended_location": "Supplementary information",
            },
            {
                "Figure": "Figure_ML8_linear_contribution_stability",
                "Purpose": "Describe magnitude and direction of contributions in the final ridge refit",
                "Recommended_location": "Supplementary information",
            },
        ]
    )
    manifest.to_csv(
        OUTPUT_DIR / "ML_additional_figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
