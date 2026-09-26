#!/usr/bin/env python3
"""Leakage-resistant machine-learning meta-analysis for screen exposure and EF.

The unit of validation is the independent cohort/sample cluster, not the effect-size
row. The primary target is a comparable Fisher-z effect size. Non-meta-analytic
proxy statistics are evaluated only in a separately labelled sensitivity analysis.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

LOCAL_LIB = os.environ.get("SCREEN_EF_ML_LIB", "")
if LOCAL_LIB and Path(LOCAL_LIB).exists():
    sys.path.insert(0, LOCAL_LIB)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = int(os.environ.get("SCREEN_EF_ML_SEED", "20260917"))
OUTER_FOLDS = int(os.environ.get("SCREEN_EF_ML_OUTER_FOLDS", "5"))
OUTER_REPEATS = int(os.environ.get("SCREEN_EF_ML_OUTER_REPEATS", "5"))
INNER_FOLDS = int(os.environ.get("SCREEN_EF_ML_INNER_FOLDS", "4"))
PERMUTATIONS = int(os.environ.get("SCREEN_EF_ML_PERMUTATIONS", "5"))

META_XLSX = Path(
    os.environ.get(
        "SCREEN_EF_META_XLSX",
        str(DATA_DIR / "Meta-analysis_data_full_age_updated.xlsx"),
    )
)
ML_XLSX = Path(
    os.environ.get(
        "SCREEN_EF_PROXY_XLSX",
        str(DATA_DIR / "Machine-learning_data.xlsx"),
    )
)
AGE_XLSX = Path(
    os.environ.get(
        "SCREEN_EF_AGE_XLSX",
        str(DATA_DIR / "Meta-analysis_data_full_age_updated.xlsx"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "SCREEN_EF_ML_OUTPUT",
        str(SCRIPT_DIR),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRIMARY_PREPARED_CSV = os.environ.get(
    "SCREEN_EF_PRIMARY_PREPARED_CSV",
    str(SCRIPT_DIR / "ML_primary_analysis_dataset.csv"),
)
PROXY_PREPARED_CSV = os.environ.get(
    "SCREEN_EF_PROXY_PREPARED_CSV",
    str(SCRIPT_DIR / "ML_proxy_sensitivity_dataset.csv"),
)


BLUE = "#1F5A89"
LIGHT_BLUE = "#8FB9D4"
ORANGE = "#D77A28"
RED = "#B23A2B"
GREY = "#6F6F6F"
LIGHT_GREY = "#D9D9D9"
BLACK = "#222222"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    binary: tuple[str, ...]

    @property
    def all(self) -> list[str]:
        return list(self.numeric + self.categorical + self.binary)


PRIMARY_SPEC = FeatureSpec(
    numeric=("Year", "Log_N", "Age_Mean_years"),
    categorical=(
        "Region",
        "Study_design",
        "Screen_Category",
        "Screen_Nature",
        "EF_Domain",
        "EF_Measure_Type",
    ),
    binary=("Adjusted",),
)

PROXY_SPEC = FeatureSpec(
    numeric=("Year", "Log_N"),
    categorical=(
        "Region",
        "Design_Group",
        "Age_Group",
        "Screen_Category",
        "Screen_Nature",
        "EF_Domain",
        "Outcome_Tier",
        "Metric_Class",
    ),
    binary=("Longitudinal", "Experimental", "Adjusted", "Proxy_Flag"),
)


def clean_text(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    replacements = {
        "�C": "–",
        "â€“": "–",
        "â€”": "—",
        "Â": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text if text else np.nan


def derive_region(country: Any) -> str:
    value = clean_text(country)
    if not isinstance(value, str):
        return "Not reported"
    countries = {part.strip().lower() for part in value.replace("/", ";").split(";")}
    maps = {
        "North America": {"united states", "canada"},
        "Europe": {
            "spain",
            "romania",
            "united kingdom",
            "netherlands",
            "russia",
            "portugal",
            "croatia",
            "germany",
            "italy",
        },
        "East Asia": {"china", "south korea", "japan", "hong kong", "taiwan"},
        "Middle East": {"israel", "iran", "saudi arabia", "turkey"},
        "South Asia": {"india", "pakistan", "bangladesh"},
        "Oceania": {"australia", "new zealand"},
        "Latin America": {"chile", "brazil", "mexico", "argentina"},
    }
    matched = [region for region, members in maps.items() if countries & members]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return "Multinational"
    return "Other / not reported"


def derive_measure_type(instrument: Any) -> str:
    value = "" if pd.isna(instrument) else str(instrument).lower()
    rating_tokens = (
        "brief",
        "questionnaire",
        "rating",
        "inventory",
        "scale",
        "chexi",
        "cbcl",
        "sdq",
        "parent",
        "teacher",
        "self-report",
        "self report",
    )
    if any(token in value for token in rating_tokens):
        return "Rating / questionnaire"
    if value.strip() in ("", "nr", "not reported"):
        return "Not reported"
    return "Performance task"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (META_XLSX, ML_XLSX, AGE_XLSX):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    meta = pd.read_excel(AGE_XLSX, sheet_name="Meta_Data")
    proxy = pd.read_excel(ML_XLSX, sheet_name="ML_Data", skiprows=3)
    age = pd.read_excel(AGE_XLSX, sheet_name="Meta_Data")
    return meta, proxy, age


def prepare_primary(
    meta: pd.DataFrame, proxy: pd.DataFrame, age: pd.DataFrame
) -> pd.DataFrame:
    age_cols = ["Effect_ID", "Age_Mean_years", "Age_SD_years", "Age_Verification_Status"]
    quality_cols = ["Effect_ID", "Quality_Class", "Metric_Class", "Proxy_Flag"]
    if set(age_cols).issubset(meta.columns):
        out = meta.copy()
    else:
        out = meta.merge(age[age_cols].drop_duplicates("Effect_ID"), on="Effect_ID", how="left")
    out = out.merge(
        proxy[quality_cols].drop_duplicates("Effect_ID"), on="Effect_ID", how="left"
    )
    out["Target"] = pd.to_numeric(out["Fisher_z"], errors="coerce")
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["N"] = pd.to_numeric(out["N"], errors="coerce")
    out["Log_N"] = np.log(out["N"].where(out["N"] > 0))
    out["Age_Mean_years"] = pd.to_numeric(out["Age_Mean_years"], errors="coerce")
    out["Age_SD_years"] = pd.to_numeric(out["Age_SD_years"], errors="coerce")
    out["Region"] = out["Country"].map(derive_region)
    out["EF_Measure_Type"] = out["EF_Instrument"].map(derive_measure_type)
    out["Group"] = out["Study_Group_ID"].astype(str)
    for col in PRIMARY_SPEC.categorical:
        out[col] = out[col].map(clean_text)
    for col in PRIMARY_SPEC.binary:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[np.isfinite(out["Target"]) & out["Group"].notna()].copy()
    if out["Effect_ID"].duplicated().any():
        raise ValueError("Primary input contains duplicated Effect_ID values.")
    return out.reset_index(drop=True)


def prepare_proxy(proxy: pd.DataFrame) -> pd.DataFrame:
    out = proxy.copy()
    out["Target"] = pd.to_numeric(out["Target_Fisher_z_or_proxy"], errors="coerce")
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["N"] = pd.to_numeric(out["N"], errors="coerce")
    out["Log_N"] = pd.to_numeric(out["Log_N"], errors="coerce")
    out.loc[~np.isfinite(out["Log_N"]), "Log_N"] = np.log(
        out.loc[~np.isfinite(out["Log_N"]), "N"].where(lambda x: x > 0)
    )
    out["Region"] = out["Country"].map(derive_region)
    out["Group"] = out["Grouped_CV_Unit"].fillna(out["Study_Group_ID"]).astype(str)
    for col in PROXY_SPEC.categorical:
        out[col] = out[col].map(clean_text)
    for col in PROXY_SPEC.binary:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[np.isfinite(out["Target"]) & out["Group"].notna()].copy()
    if out["Effect_ID"].duplicated().any():
        raise ValueError("Proxy-augmented input contains duplicated Effect_ID values.")
    return out.reset_index(drop=True)


def group_balanced_weights(groups: Iterable[Any]) -> np.ndarray:
    s = pd.Series(list(groups), dtype="object")
    counts = s.value_counts(dropna=False)
    w = s.map(lambda g: 1.0 / counts.loc[g]).to_numpy(float)
    return w * len(w) / w.sum()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx = weighted_mean(x, w)
    my = weighted_mean(y, w)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / math.sqrt(vx * vy))


def calibration(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(pred)), pred])
    sw = np.sqrt(w)
    try:
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
        return float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def regression_metrics(y: np.ndarray, pred: np.ndarray, groups: Iterable[Any]) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    w = group_balanced_weights(groups)
    resid = pred - y
    mse = weighted_mean(resid**2, w)
    mae = weighted_mean(np.abs(resid), w)
    ybar = weighted_mean(y, w)
    sst = np.sum(w * (y - ybar) ** 2)
    sse = np.sum(w * resid**2)
    q2 = 1.0 - sse / sst if sst > 0 else np.nan
    corr = weighted_corr(y, pred, w)
    intercept, slope = calibration(y, pred, w)
    return {
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(mae),
        "Predictive_Q2": float(q2),
        "Correlation_R2": float(corr**2) if np.isfinite(corr) else np.nan,
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
        "Mean_bias": weighted_mean(resid, w),
    }


def balanced_group_folds(
    groups: Iterable[Any], n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(list(groups), dtype=object)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < n_splits:
        raise ValueError(f"Only {len(unique)} groups for {n_splits} folds.")
    rng = np.random.default_rng(seed)
    tie = rng.random(len(unique))
    order = np.lexsort((tie, -counts))
    fold_groups: list[list[Any]] = [[] for _ in range(n_splits)]
    fold_loads = np.zeros(n_splits, dtype=float)
    for idx in order:
        candidates = np.flatnonzero(fold_loads == fold_loads.min())
        chosen = int(rng.choice(candidates))
        fold_groups[chosen].append(unique[idx])
        fold_loads[chosen] += counts[idx]
    splits = []
    for held_out in fold_groups:
        test = np.flatnonzero(np.isin(groups, held_out))
        train = np.flatnonzero(~np.isin(groups, held_out))
        splits.append((train, test))
    return splits


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_cols = list(spec.numeric + spec.binary)
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, list(spec.categorical))],
        remainder="drop",
        sparse_threshold=0,
    )


def make_pipeline(model_name: str, spec: FeatureSpec) -> Pipeline:
    if model_name == "Ridge":
        model = Ridge()
    elif model_name == "Elastic net":
        model = ElasticNet(max_iter=20000, selection="cyclic", random_state=SEED)
    elif model_name == "Random forest":
        model = RandomForestRegressor(
            n_estimators=250,
            random_state=SEED,
            n_jobs=-1,
            bootstrap=True,
        )
    elif model_name == "Gradient boosting":
        model = HistGradientBoostingRegressor(
            max_iter=180,
            early_stopping=False,
            random_state=SEED,
        )
    else:
        raise KeyError(model_name)
    return Pipeline([("preprocess", build_preprocessor(spec)), ("model", model)])


MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "Ridge": [{"model__alpha": x} for x in (0.03, 0.1, 1.0, 10.0, 100.0)],
    "Elastic net": [
        {"model__alpha": a, "model__l1_ratio": l1}
        for a, l1 in product((0.01, 0.1, 0.5), (0.1, 0.5))
    ],
    "Random forest": [
        {"model__max_features": mf, "model__min_samples_leaf": leaf}
        for mf, leaf in product((0.5, 1.0), (2, 5))
    ],
    "Gradient boosting": [
        {
            "model__learning_rate": lr,
            "model__max_leaf_nodes": leaves,
            "model__l2_regularization": 1.0,
            "model__min_samples_leaf": 5,
        }
        for lr, leaves in product((0.03, 0.08), (7, 15))
    ],
}


def fit_pipeline(
    model_name: str,
    spec: FeatureSpec,
    params: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: Iterable[Any],
) -> Pipeline:
    pipe = make_pipeline(model_name, spec).set_params(**params)
    weights = group_balanced_weights(groups)
    pipe.fit(x, y, model__sample_weight=weights)
    return pipe


def select_hyperparameters(
    model_name: str,
    spec: FeatureSpec,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    inner = balanced_group_folds(groups, INNER_FOLDS, seed)
    records = []
    for parameter_index, params in enumerate(MODEL_GRIDS[model_name]):
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(inner, 1):
            try:
                fit = fit_pipeline(
                    model_name,
                    spec,
                    params,
                    x.iloc[train_idx],
                    y[train_idx],
                    groups[train_idx],
                )
                pred = fit.predict(x.iloc[test_idx])
                score = regression_metrics(y[test_idx], pred, groups[test_idx])["RMSE"]
            except Exception:
                score = np.inf
            fold_scores.append(score)
            records.append(
                {
                    "Parameter_index": parameter_index,
                    "Inner_fold": fold,
                    "RMSE": score,
                    "Parameters": json.dumps(params, sort_keys=True),
                }
            )
        mean_score = float(np.mean(fold_scores))
        records.append(
            {
                "Parameter_index": parameter_index,
                "Inner_fold": 0,
                "RMSE": mean_score,
                "Parameters": json.dumps(params, sort_keys=True),
            }
        )
    summary = pd.DataFrame(records)
    means = summary.loc[summary["Inner_fold"] == 0].copy()
    chosen = means.sort_values(["RMSE", "Parameter_index"]).iloc[0]
    params = MODEL_GRIDS[model_name][int(chosen["Parameter_index"])]
    return params, float(chosen["RMSE"]), summary


def block_permute_feature(
    x: pd.DataFrame, groups: np.ndarray, feature: str, rng: np.random.Generator
) -> pd.DataFrame:
    out = x.copy()
    unique = np.array(pd.unique(groups), dtype=object)
    donors = rng.permutation(unique)
    for recipient, donor in zip(unique, donors):
        recipient_rows = np.flatnonzero(groups == recipient)
        donor_values = x.loc[groups == donor, feature].to_numpy()
        if len(donor_values) == 0:
            continue
        out.loc[recipient_rows, feature] = rng.choice(
            donor_values, size=len(recipient_rows), replace=True
        )
    return out


def run_nested_cv(
    data: pd.DataFrame,
    spec: FeatureSpec,
    dataset_name: str,
    model_names: list[str],
    outer_repeats: int,
    compute_permutation: bool,
) -> dict[str, pd.DataFrame]:
    x = data[spec.all].copy()
    y = data["Target"].to_numpy(float)
    groups = data["Group"].to_numpy(object)
    effect_ids = data["Effect_ID"].astype(str).to_numpy()

    prediction_records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []
    permutation_records: list[dict[str, Any]] = []

    for repeat in range(1, outer_repeats + 1):
        splits = balanced_group_folds(groups, OUTER_FOLDS, SEED + repeat * 101)
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            train_mean = weighted_mean(y[train_idx], group_balanced_weights(groups[train_idx]))
            null_pred = np.repeat(train_mean, len(test_idx))
            for row, pred in zip(test_idx, null_pred):
                prediction_records.append(
                    {
                        "Dataset": dataset_name,
                        "Repeat": repeat,
                        "Fold": fold,
                        "Effect_ID": effect_ids[row],
                        "Group": groups[row],
                        "Observed": y[row],
                        "Predicted": pred,
                        "Model": "Null (training mean)",
                    }
                )

            for model_index, model_name in enumerate(model_names):
                params, inner_rmse, _ = select_hyperparameters(
                    model_name,
                    spec,
                    x.iloc[train_idx].reset_index(drop=True),
                    y[train_idx],
                    groups[train_idx],
                    SEED + repeat * 1000 + fold * 100 + model_index,
                )
                fit = fit_pipeline(
                    model_name,
                    spec,
                    params,
                    x.iloc[train_idx],
                    y[train_idx],
                    groups[train_idx],
                )
                pred = fit.predict(x.iloc[test_idx])
                parameter_records.append(
                    {
                        "Dataset": dataset_name,
                        "Repeat": repeat,
                        "Fold": fold,
                        "Model": model_name,
                        "Inner_RMSE": inner_rmse,
                        "Parameters": json.dumps(params, sort_keys=True),
                    }
                )
                for row, predicted in zip(test_idx, pred):
                    prediction_records.append(
                        {
                            "Dataset": dataset_name,
                            "Repeat": repeat,
                            "Fold": fold,
                            "Effect_ID": effect_ids[row],
                            "Group": groups[row],
                            "Observed": y[row],
                            "Predicted": predicted,
                            "Model": model_name,
                        }
                    )

                if compute_permutation:
                    base_rmse = regression_metrics(
                        y[test_idx], pred, groups[test_idx]
                    )["RMSE"]
                    for feature_index, feature in enumerate(spec.all):
                        for permutation in range(1, PERMUTATIONS + 1):
                            rng = np.random.default_rng(
                                SEED
                                + repeat * 100000
                                + fold * 10000
                                + model_index * 1000
                                + feature_index * 50
                                + permutation
                            )
                            permuted = block_permute_feature(
                                x.iloc[test_idx].reset_index(drop=True),
                                groups[test_idx],
                                feature,
                                rng,
                            )
                            perm_pred = fit.predict(permuted)
                            perm_rmse = regression_metrics(
                                y[test_idx], perm_pred, groups[test_idx]
                            )["RMSE"]
                            permutation_records.append(
                                {
                                    "Dataset": dataset_name,
                                    "Repeat": repeat,
                                    "Fold": fold,
                                    "Model": model_name,
                                    "Feature": feature,
                                    "Permutation": permutation,
                                    "Base_RMSE": base_rmse,
                                    "Permuted_RMSE": perm_rmse,
                                    "Delta_RMSE": perm_rmse - base_rmse,
                                }
                            )

    predictions = pd.DataFrame(prediction_records)
    parameters = pd.DataFrame(parameter_records)
    permutations = pd.DataFrame(permutation_records)
    repeat_metrics = []
    for (dataset, repeat, model), part in predictions.groupby(
        ["Dataset", "Repeat", "Model"], sort=False
    ):
        metrics = regression_metrics(
            part["Observed"].to_numpy(), part["Predicted"].to_numpy(), part["Group"]
        )
        repeat_metrics.append(
            {"Dataset": dataset, "Repeat": repeat, "Model": model, **metrics}
        )
    metrics = pd.DataFrame(repeat_metrics)
    return {
        "predictions": predictions,
        "parameters": parameters,
        "permutations": permutations,
        "metrics": metrics,
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "RMSE",
        "MAE",
        "Predictive_Q2",
        "Correlation_R2",
        "Calibration_intercept",
        "Calibration_slope",
        "Mean_bias",
    ]
    rows = []
    for (dataset, model), part in metrics.groupby(["Dataset", "Model"], sort=False):
        row: dict[str, Any] = {
            "Dataset": dataset,
            "Model": model,
            "Repeats": part["Repeat"].nunique(),
        }
        for col in numeric:
            vals = pd.to_numeric(part[col], errors="coerce").dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            row[f"{col}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{col}_q025"] = float(np.quantile(vals, 0.025)) if len(vals) else np.nan
            row[f"{col}_q975"] = float(np.quantile(vals, 0.975)) if len(vals) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    null = summary.loc[
        summary["Model"] == "Null (training mean)", ["Dataset", "RMSE_mean"]
    ].rename(columns={"RMSE_mean": "Null_RMSE_mean"})
    summary = summary.merge(null, on="Dataset", how="left")
    summary["RMSE_improvement_vs_null"] = (
        summary["Null_RMSE_mean"] - summary["RMSE_mean"]
    )
    return summary


def aggregate_oof(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    part = predictions.loc[predictions["Model"] == model].copy()
    return (
        part.groupby(["Effect_ID", "Group"], as_index=False)
        .agg(
            Observed=("Observed", "first"),
            Predicted=("Predicted", "mean"),
            Prediction_SD=("Predicted", "std"),
            Repeats=("Repeat", "nunique"),
        )
        .reset_index(drop=True)
    )


def summarize_permutation(permutations: pd.DataFrame, model: str) -> pd.DataFrame:
    part = permutations.loc[permutations["Model"] == model].copy()
    if part.empty:
        return pd.DataFrame()
    return (
        part.groupby("Feature", as_index=False)["Delta_RMSE"]
        .agg(
            Mean_Delta_RMSE="mean",
            Median_Delta_RMSE="median",
            SD_Delta_RMSE="std",
            Q025_Delta_RMSE=lambda x: np.quantile(x, 0.025),
            Q975_Delta_RMSE=lambda x: np.quantile(x, 0.975),
            Positive_fraction=lambda x: np.mean(np.asarray(x) > 0),
            Evaluations="count",
        )
        .sort_values("Mean_Delta_RMSE", ascending=False)
    )


def select_final_model(summary: pd.DataFrame, dataset: str) -> str:
    candidates = summary.loc[
        (summary["Dataset"] == dataset) & (summary["Model"] != "Null (training mean)")
    ].copy()
    if candidates.empty:
        raise ValueError("No non-null candidate model was evaluated.")
    return str(
        candidates.sort_values(
            ["RMSE_mean", "Predictive_Q2_mean"], ascending=[True, False]
        ).iloc[0]["Model"]
    )


def fit_final_model(
    data: pd.DataFrame, spec: FeatureSpec, model_name: str
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    x = data[spec.all].copy()
    y = data["Target"].to_numpy(float)
    groups = data["Group"].to_numpy(object)
    params, score, audit = select_hyperparameters(
        model_name, spec, x, y, groups, SEED + 999999
    )
    fit = fit_pipeline(model_name, spec, params, x, y, groups)
    audit["Full_data_inner_RMSE"] = score
    return fit, params, audit


def partial_dependence_data(
    model: Pipeline,
    data: pd.DataFrame,
    spec: FeatureSpec,
    feature: str,
    points: int = 50,
) -> pd.DataFrame:
    observed = pd.to_numeric(data[feature], errors="coerce").dropna()
    if observed.nunique() < 3:
        return pd.DataFrame()
    lower, upper = np.quantile(observed, [0.05, 0.95])
    grid = np.linspace(lower, upper, points)
    weights = group_balanced_weights(data["Group"])
    rows = []
    x = data[spec.all].copy()
    for value in grid:
        modified = x.copy()
        modified[feature] = value
        pred = model.predict(modified)
        rows.append(
            {
                "Feature": feature,
                "Grid_value": value,
                "Mean_prediction": weighted_mean(pred, weights),
            }
        )
    return pd.DataFrame(rows)


def group_category_error(
    oof: pd.DataFrame,
    data: pd.DataFrame,
    category: str,
    min_groups: int = 3,
    bootstraps: int = 1000,
) -> pd.DataFrame:
    merged = oof.merge(data[["Effect_ID", category]], on="Effect_ID", how="left")
    merged["Absolute_error"] = np.abs(merged["Predicted"] - merged["Observed"])
    group_error = (
        merged.groupby([category, "Group"], dropna=False, as_index=False)["Absolute_error"]
        .mean()
        .rename(columns={"Absolute_error": "Group_MAE"})
    )
    rng = np.random.default_rng(SEED + sum(map(ord, category)))
    rows = []
    for label, part in group_error.groupby(category, dropna=False):
        values = part["Group_MAE"].to_numpy(float)
        if len(values) < min_groups:
            continue
        boot = np.array(
            [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(bootstraps)]
        )
        rows.append(
            {
                "Category_variable": category,
                "Category": str(label),
                "Independent_groups": len(values),
                "MAE": float(np.mean(values)),
                "MAE_CI_low": float(np.quantile(boot, 0.025)),
                "MAE_CI_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE") if rows else pd.DataFrame()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.svg")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_model_performance(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    primary: pd.DataFrame,
    best_model: str,
) -> None:
    model_order = [
        "Null (training mean)",
        "Ridge",
        "Elastic net",
        "Random forest",
        "Gradient boosting",
    ]
    available = [m for m in model_order if m in set(metrics["Model"])]
    palette = {
        "Null (training mean)": GREY,
        "Ridge": LIGHT_BLUE,
        "Elastic net": "#5A91BF",
        "Random forest": ORANGE,
        "Gradient boosting": RED,
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4))
    ax = axes[0, 0]
    data = [metrics.loc[metrics["Model"] == m, "RMSE"].to_numpy() for m in available]
    bp = ax.boxplot(data, tick_labels=available, patch_artist=True, widths=0.58, showfliers=False)
    for patch, model in zip(bp["boxes"], available):
        patch.set_facecolor(palette[model])
        patch.set_alpha(0.85)
    for i, model in enumerate(available, 1):
        vals = metrics.loc[metrics["Model"] == model, "RMSE"].to_numpy()
        ax.scatter(np.full(len(vals), i), vals, s=14, color=BLACK, alpha=0.6, zorder=3)
    ax.set_ylabel("Group-balanced RMSE (Fisher's z)")
    ax.tick_params(axis="x", rotation=35)
    clean_axis(ax)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    data = [metrics.loc[metrics["Model"] == m, "Predictive_Q2"].to_numpy() for m in available]
    bp = ax.boxplot(data, tick_labels=available, patch_artist=True, widths=0.58, showfliers=False)
    for patch, model in zip(bp["boxes"], available):
        patch.set_facecolor(palette[model])
        patch.set_alpha(0.85)
    ax.axhline(0, color=GREY, linewidth=0.9, linestyle="--")
    for i, model in enumerate(available, 1):
        vals = metrics.loc[metrics["Model"] == model, "Predictive_Q2"].to_numpy()
        ax.scatter(np.full(len(vals), i), vals, s=14, color=BLACK, alpha=0.6, zorder=3)
    ax.set_ylabel(r"Predictive $Q^2$")
    ax.tick_params(axis="x", rotation=35)
    clean_axis(ax)
    add_panel_label(ax, "B")

    oof = aggregate_oof(predictions, best_model).merge(
        primary[["Effect_ID", "EF_Domain"]], on="Effect_ID", how="left"
    )
    domain_levels = list(pd.unique(oof["EF_Domain"]))
    domain_colors = [BLUE, ORANGE, "#5F8D4E", "#8E6C9E", GREY]
    color_map = {d: domain_colors[i % len(domain_colors)] for i, d in enumerate(domain_levels)}
    ax = axes[1, 0]
    for domain, part in oof.groupby("EF_Domain", dropna=False):
        ax.scatter(
            part["Observed"],
            part["Predicted"],
            s=20,
            alpha=0.72,
            color=color_map[domain],
            label=str(domain),
            edgecolor="white",
            linewidth=0.25,
        )
    lo = min(oof["Observed"].min(), oof["Predicted"].min())
    hi = max(oof["Observed"].max(), oof["Predicted"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=GREY, linewidth=0.9)
    metrics_oof = regression_metrics(oof["Observed"], oof["Predicted"], oof["Group"])
    repeated = metrics.loc[metrics["Model"] == best_model]
    repeated_rmse = repeated["RMSE"].mean()
    repeated_q2 = repeated["Predictive_Q2"].mean()
    repeated_slope = repeated["Calibration_slope"].mean()
    repeated_bias = repeated["Mean_bias"].mean()
    ax.text(
        0.03,
        0.97,
        f"Mean repeated-CV RMSE = {repeated_rmse:.3f}\nMean repeated-CV Q² = {repeated_q2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    ax.set_xlabel("Observed Fisher's z")
    ax.set_ylabel("Out-of-fold predicted Fisher's z")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5, handletextpad=0.3)
    clean_axis(ax)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    oof["Residual"] = oof["Predicted"] - oof["Observed"]
    ax.scatter(oof["Predicted"], oof["Residual"], color=BLUE, alpha=0.68, s=20)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.9)
    coef = np.polyfit(oof["Predicted"], oof["Residual"], deg=1)
    grid = np.linspace(oof["Predicted"].min(), oof["Predicted"].max(), 100)
    ax.plot(grid, coef[0] * grid + coef[1], color=RED, linewidth=1.2)
    ax.text(
        0.03,
        0.97,
        f"Mean CV calibration slope = {repeated_slope:.2f}\nMean CV bias = {repeated_bias:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    ax.set_xlabel("Out-of-fold predicted Fisher's z")
    ax.set_ylabel("Prediction residual")
    clean_axis(ax)
    add_panel_label(ax, "D")
    fig.subplots_adjust(wspace=0.32, hspace=0.42, left=0.10, right=0.98, bottom=0.09, top=0.98)
    save_figure(fig, "Figure_ML1_model_performance")


def plot_interpretation(
    importance: pd.DataFrame,
    pdp_age: pd.DataFrame,
    pdp_n: pd.DataFrame,
    domain_error: pd.DataFrame,
    primary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4))
    ax = axes[0, 0]
    show = importance.head(10).sort_values("Mean_Delta_RMSE")
    y = np.arange(len(show))
    xerr = np.vstack(
        [
            show["Mean_Delta_RMSE"] - show["Q025_Delta_RMSE"],
            show["Q975_Delta_RMSE"] - show["Mean_Delta_RMSE"],
        ]
    )
    ax.errorbar(
        show["Mean_Delta_RMSE"],
        y,
        xerr=xerr,
        fmt="s",
        color=BLUE,
        ecolor=GREY,
        markersize=4.5,
        linewidth=0.9,
        capsize=2,
    )
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, show["Feature"])
    ax.set_xlabel("Increase in validation RMSE after\ngroup-block permutation")
    clean_axis(ax)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    if not pdp_age.empty:
        ax.plot(pdp_age["Grid_value"], pdp_age["Mean_prediction"], color=BLUE, linewidth=1.6)
        observed = pd.to_numeric(primary["Age_Mean_years"], errors="coerce").dropna()
        ymin = pdp_age["Mean_prediction"].min()
        ax.plot(observed, np.repeat(ymin, len(observed)), "|", color=GREY, alpha=0.35)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean age at baseline (years)")
    ax.set_ylabel("Predicted Fisher's z")
    clean_axis(ax)
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    if not pdp_n.empty:
        ax.plot(pdp_n["Grid_value"], pdp_n["Mean_prediction"], color=ORANGE, linewidth=1.6)
        observed = pd.to_numeric(primary["Log_N"], errors="coerce").dropna()
        ymin = pdp_n["Mean_prediction"].min()
        ax.plot(observed, np.repeat(ymin, len(observed)), "|", color=GREY, alpha=0.35)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Log sample size")
    ax.set_ylabel("Predicted Fisher's z")
    clean_axis(ax)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    if not domain_error.empty:
        show = domain_error.sort_values("MAE")
        y = np.arange(len(show))
        xerr = np.vstack(
            [show["MAE"] - show["MAE_CI_low"], show["MAE_CI_high"] - show["MAE"]]
        )
        ax.errorbar(
            show["MAE"],
            y,
            xerr=xerr,
            fmt="s",
            color=RED,
            ecolor=GREY,
            markersize=4.5,
            linewidth=0.9,
            capsize=2,
        )
        display = {
            "Global executive function": "Global EF",
            "Cognitive flexibility or shifting": "Cognitive flexibility",
            "Working memory or updating": "Working memory",
            "Hot EF or self-regulation": "Hot EF/self-regulation",
        }
        labels = [
            f"{display.get(c, c)} (g={g})"
            for c, g in zip(show["Category"], show["Independent_groups"])
        ]
        ax.set_yticks(y, labels)
    ax.set_xlabel("Group-balanced MAE (Fisher's z)")
    clean_axis(ax)
    add_panel_label(ax, "D")
    fig.subplots_adjust(wspace=0.64, hspace=0.40, left=0.16, right=0.99, bottom=0.09, top=0.98)
    save_figure(fig, "Figure_ML2_predictive_drivers")


def plot_sensitivity(
    combined_metrics: pd.DataFrame,
    best_model: str,
    proxy: pd.DataFrame,
    screen_error: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.2))
    show_models = ["Null (training mean)", best_model]
    datasets = ["Meta-grade primary", "Proxy-augmented sensitivity"]
    colors = {"Meta-grade primary": BLUE, "Proxy-augmented sensitivity": ORANGE}

    ax = axes[0, 0]
    positions = np.arange(len(show_models))
    width = 0.32
    for j, dataset in enumerate(datasets):
        means, lows, highs = [], [], []
        for model in show_models:
            vals = combined_metrics.loc[
                (combined_metrics["Dataset"] == dataset) & (combined_metrics["Model"] == model),
                "RMSE",
            ].to_numpy()
            means.append(np.mean(vals))
            lows.append(np.mean(vals) - np.quantile(vals, 0.025))
            highs.append(np.quantile(vals, 0.975) - np.mean(vals))
        ax.errorbar(
            positions + (j - 0.5) * width,
            means,
            yerr=np.vstack([lows, highs]),
            fmt="s",
            color=colors[dataset],
            markersize=5,
            capsize=2,
            linewidth=1,
            label=dataset,
        )
    ax.set_xticks(positions, show_models, rotation=20)
    ax.set_ylabel("Group-balanced RMSE (Fisher's z)")
    ax.legend(frameon=False, fontsize=7)
    clean_axis(ax)
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    for j, dataset in enumerate(datasets):
        means, lows, highs = [], [], []
        for model in show_models:
            vals = combined_metrics.loc[
                (combined_metrics["Dataset"] == dataset) & (combined_metrics["Model"] == model),
                "Predictive_Q2",
            ].to_numpy()
            means.append(np.mean(vals))
            lows.append(np.mean(vals) - np.quantile(vals, 0.025))
            highs.append(np.quantile(vals, 0.975) - np.mean(vals))
        ax.errorbar(
            positions + (j - 0.5) * width,
            means,
            yerr=np.vstack([lows, highs]),
            fmt="s",
            color=colors[dataset],
            markersize=5,
            capsize=2,
            linewidth=1,
        )
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xticks(positions, show_models, rotation=20)
    ax.set_ylabel(r"Predictive $Q^2$")
    clean_axis(ax)
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    composition = (
        proxy.groupby(["EF_Domain", "Proxy_Flag"]).size().unstack(fill_value=0).sort_values(0)
    )
    exact = composition.get(0, pd.Series(0, index=composition.index))
    prox = composition.get(1, pd.Series(0, index=composition.index))
    y = np.arange(len(composition))
    ax.barh(y, exact, color=BLUE, label="Meta-grade effect")
    ax.barh(y, prox, left=exact, color=ORANGE, label="Proxy statistic")
    ax.set_yticks(y, composition.index)
    ax.set_xlabel("Number of effect-size rows")
    ax.legend(frameon=False, fontsize=7)
    clean_axis(ax)
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    if not screen_error.empty:
        show = screen_error.sort_values("MAE")
        y = np.arange(len(show))
        xerr = np.vstack(
            [show["MAE"] - show["MAE_CI_low"], show["MAE_CI_high"] - show["MAE"]]
        )
        ax.errorbar(
            show["MAE"],
            y,
            xerr=xerr,
            fmt="s",
            color=BLUE,
            ecolor=GREY,
            markersize=4.5,
            capsize=2,
        )
        display = {
            "Interactive or gaming": "Interactive",
            "Social media or multitasking": "Social/multitasking",
            "Passive or background viewing": "Passive/background",
            "Total or mixed screen exposure": "Total/mixed",
        }
        labels = [
            f"{display.get(c, c)} (g={g})"
            for c, g in zip(show["Category"], show["Independent_groups"])
        ]
        ax.set_yticks(y, labels)
    ax.set_xlabel("Group-balanced MAE (Fisher's z)")
    clean_axis(ax)
    add_panel_label(ax, "D")
    fig.subplots_adjust(wspace=0.78, hspace=0.40, left=0.18, right=0.99, bottom=0.10, top=0.98)
    save_figure(fig, "Figure_ML3_sensitivity_and_error")


def write_data_dictionary() -> None:
    rows = [
        ("Target", "Outcome", "Fisher's z of the harmonized correlation; negative values indicate poorer EF with greater screen exposure."),
        ("Group", "Validation cluster", "Independent cohort/sample cluster kept intact in every resampling split."),
        ("Year", "Predictor", "Publication year; median-imputed within each training split."),
        ("Log_N", "Predictor", "Natural logarithm of effect-level sample size."),
        ("Age_Mean_years", "Predictor", "Mean baseline/exposure-wave age in years; training-fold median imputation plus missingness indicator."),
        ("Region", "Predictor", "Broad geographic region derived from the reported country."),
        ("Study_design", "Predictor", "Cross-sectional, longitudinal observational or experimental design."),
        ("Screen_Category", "Predictor", "Screen exposure category."),
        ("Screen_Nature", "Predictor", "Passive, interactive/entertainment, educational or mixed exposure."),
        ("EF_Domain", "Predictor", "Executive-function domain."),
        ("EF_Measure_Type", "Predictor", "Performance task versus rating/questionnaire."),
        ("Quality_Class", "Excluded predictor", "Excluded in the revision because the prior field was an effect-eligibility label, not a validated report-level risk-of-bias assessment."),
        ("Adjusted", "Predictor", "Whether the effect estimate was adjusted."),
        ("Sampling_Variance_z", "Excluded", "Excluded to avoid outcome-derived leakage; sample size is represented by Log_N."),
        ("Harmonized_r / Fisher_z", "Excluded predictor", "Outcome-defining quantities; never supplied to the model as predictors."),
        ("First_Author / Report_ID / Effect_ID", "Excluded predictor", "Identifiers retained for traceability but not used for prediction."),
    ]
    pd.DataFrame(rows, columns=["Variable", "Role", "Definition_or_reason"]).to_csv(
        OUTPUT_DIR / "ML_data_dictionary.csv", index=False, encoding="utf-8-sig"
    )


def write_report(
    primary: pd.DataFrame,
    proxy: pd.DataFrame,
    summary: pd.DataFrame,
    best_model: str,
    best_params: dict[str, Any],
    importance: pd.DataFrame,
) -> None:
    primary_row = summary.loc[
        (summary["Dataset"] == "Meta-grade primary") & (summary["Model"] == best_model)
    ].iloc[0]
    null_row = summary.loc[
        (summary["Dataset"] == "Meta-grade primary")
        & (summary["Model"] == "Null (training mean)")
    ].iloc[0]
    improved = primary_row["RMSE_mean"] < null_row["RMSE_mean"]
    top = importance.head(5)
    top_text = ", ".join(
        f"{row.Feature} (ΔRMSE={row.Mean_Delta_RMSE:.3f})" for row in top.itertuples()
    )
    conclusion = (
        "The selected model improved on the training-mean comparator under cohort-grouped validation."
        if improved
        else "No candidate model improved on the training-mean comparator under cohort-grouped validation; the ML findings should therefore be treated as exploratory heterogeneity mapping rather than a validated prediction tool."
    )
    text = f"""# Machine-learning meta-analysis: locked analysis report

## Analysis population

The primary analysis used {len(primary)} comparable effect sizes from {primary['Report_ID'].nunique()} reports and {primary['Group'].nunique()} independent cohort/sample clusters. The proxy-augmented sensitivity analysis used {len(proxy)} rows from {proxy['Report_ID'].nunique()} reports and {proxy['Group'].nunique()} clusters; {int((proxy['Proxy_Flag'] == 1).sum())} rows were explicitly flagged as proxy statistics and were not mixed into the primary evidence set.

## Validation and model selection

All rows from the same cohort/sample cluster were kept in the same fold. Four regression algorithms (ridge, elastic net, random forest and histogram gradient boosting) were compared with a training-fold mean comparator using {OUTER_FOLDS}-fold outer validation repeated {OUTER_REPEATS} times. Hyperparameters were selected exclusively in {INNER_FOLDS}-fold grouped inner validation. Training and evaluation were group-balanced so that each independent cluster contributed equal total weight. No effect size, confidence interval, standard error, sampling variance or record identifier was supplied as a predictor.

## Primary result

The lowest-RMSE non-null algorithm was {best_model}. Its mean group-balanced RMSE was {primary_row['RMSE_mean']:.3f} (repeat range approximated by the 2.5th–97.5th percentiles, {primary_row['RMSE_q025']:.3f} to {primary_row['RMSE_q975']:.3f}), compared with {null_row['RMSE_mean']:.3f} for the null comparator. Mean MAE was {primary_row['MAE_mean']:.3f}, mean predictive Q² was {primary_row['Predictive_Q2_mean']:.3f}, and the mean calibration slope was {primary_row['Calibration_slope_mean']:.2f}. {conclusion}

The final full-data refit used: `{json.dumps(best_params, sort_keys=True)}`. This refit is provided for interpretation and future external validation; its apparent fit is not used as evidence of performance.

## Exploratory predictor dependence

Cross-validated group-block permutation ranked the leading predictors as {top_text}. These values quantify loss of held-out predictive accuracy after disrupting a predictor while preserving clustered validation. They are not causal effects and should not be interpreted as intervention targets. Partial-dependence plots are likewise descriptive full-data refits.

## Recommended manuscript interpretation

The ML analysis estimates whether study, exposure and outcome characteristics generalize to entirely unseen cohorts. It does not classify individual children and does not establish causal moderators. Proxy-statistic results are sensitivity evidence only because regression coefficients and other non-correlation estimates are not guaranteed to share a common scale with Fisher-transformed correlations.
"""
    (OUTPUT_DIR / "Machine_learning_analysis_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    if PRIMARY_PREPARED_CSV and PROXY_PREPARED_CSV:
        primary = pd.read_csv(PRIMARY_PREPARED_CSV)
        proxy = pd.read_csv(PROXY_PREPARED_CSV)
    else:
        meta, proxy_raw, age = read_inputs()
        primary = prepare_primary(meta, proxy_raw, age)
        proxy = prepare_proxy(proxy_raw)

    manifest = pd.DataFrame(
        [
            {
                "Dataset": "Meta-grade primary",
                "Rows": len(primary),
                "Reports": primary["Report_ID"].nunique(),
                "Independent_groups": primary["Group"].nunique(),
                "Proxy_rows": 0,
                "Target": "Fisher_z",
            },
            {
                "Dataset": "Proxy-augmented sensitivity",
                "Rows": len(proxy),
                "Reports": proxy["Report_ID"].nunique(),
                "Independent_groups": proxy["Group"].nunique(),
                "Proxy_rows": int((proxy["Proxy_Flag"] == 1).sum()),
                "Target": "Fisher_z_or_explicit_proxy",
            },
        ]
    )
    manifest.to_csv(OUTPUT_DIR / "ML_analysis_manifest.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(OUTPUT_DIR / "ML_primary_analysis_dataset.csv", index=False, encoding="utf-8-sig")
    proxy.to_csv(OUTPUT_DIR / "ML_proxy_sensitivity_dataset.csv", index=False, encoding="utf-8-sig")
    write_data_dictionary()

    model_names = ["Ridge", "Elastic net", "Random forest", "Gradient boosting"]
    primary_results = run_nested_cv(
        primary,
        PRIMARY_SPEC,
        "Meta-grade primary",
        model_names,
        OUTER_REPEATS,
        compute_permutation=True,
    )
    primary_summary = summarize_metrics(primary_results["metrics"])
    best_model = select_final_model(primary_summary, "Meta-grade primary")

    proxy_results = run_nested_cv(
        proxy,
        PROXY_SPEC,
        "Proxy-augmented sensitivity",
        [best_model],
        OUTER_REPEATS,
        compute_permutation=False,
    )
    combined_metrics = pd.concat(
        [primary_results["metrics"], proxy_results["metrics"]], ignore_index=True
    )
    combined_summary = summarize_metrics(combined_metrics)

    final_model, best_params, final_tuning = fit_final_model(
        primary, PRIMARY_SPEC, best_model
    )
    importance = summarize_permutation(primary_results["permutations"], best_model)
    pdp_age = partial_dependence_data(
        final_model, primary, PRIMARY_SPEC, "Age_Mean_years"
    )
    pdp_n = partial_dependence_data(final_model, primary, PRIMARY_SPEC, "Log_N")
    oof = aggregate_oof(primary_results["predictions"], best_model)
    domain_error = group_category_error(oof, primary, "EF_Domain")
    screen_error = group_category_error(oof, primary, "Screen_Category")

    primary_results["predictions"].to_csv(
        OUTPUT_DIR / "ML_primary_nested_OOF_predictions_all_repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof.to_csv(
        OUTPUT_DIR / "ML_primary_OOF_predictions_averaged.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_results["metrics"].to_csv(
        OUTPUT_DIR / "ML_primary_performance_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined_summary.to_csv(
        OUTPUT_DIR / "ML_model_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_results["parameters"].to_csv(
        OUTPUT_DIR / "ML_primary_nested_tuning_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    proxy_results["predictions"].to_csv(
        OUTPUT_DIR / "ML_proxy_sensitivity_OOF_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    proxy_results["metrics"].to_csv(
        OUTPUT_DIR / "ML_proxy_sensitivity_performance_by_repeat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance.to_csv(
        OUTPUT_DIR / "ML_group_block_permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat([pdp_age, pdp_n], ignore_index=True).to_csv(
        OUTPUT_DIR / "ML_partial_dependence.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([domain_error, screen_error], ignore_index=True).to_csv(
        OUTPUT_DIR / "ML_subgroup_prediction_error.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_tuning.to_csv(
        OUTPUT_DIR / "ML_final_model_tuning_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_model_performance(
        primary_results["metrics"],
        primary_results["predictions"],
        primary,
        best_model,
    )
    plot_interpretation(importance, pdp_age, pdp_n, domain_error, primary)
    plot_sensitivity(combined_metrics, best_model, proxy, screen_error)
    write_report(primary, proxy, combined_summary, best_model, best_params, importance)

    config = {
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": OUTER_REPEATS,
        "inner_folds": INNER_FOLDS,
        "permutations_per_outer_fold": PERMUTATIONS,
        "primary_input": str(META_XLSX),
        "proxy_input": str(ML_XLSX),
        "age_input": str(AGE_XLSX),
        "output_dir": str(OUTPUT_DIR),
        "best_model": best_model,
        "best_parameters": best_params,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": mpl.__version__,
    }
    (OUTPUT_DIR / "ML_run_config_and_session.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
