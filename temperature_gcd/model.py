from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .io import GCDTemperatureRecord


MAX_CURVE_FIT_EVALUATIONS = 25_000


def charge_equation(t, a, b, c):
    t = np.asarray(t, dtype=float)
    a = max(float(a), 1e-12)
    return b * (1.0 - np.exp(-t / a)) + c * t


def discharge_equation(t, a, b, c):
    t = np.asarray(t, dtype=float)
    a = max(float(a), 1e-12)
    return b * np.exp(-t / a) + c * t


@dataclass
class BranchFit:
    a: float
    b: float
    c: float
    rmse_V: float
    success: bool
    message: str


@dataclass
class TimePeriodModelBundle:
    comparison: pd.DataFrame
    tuned_tree: DecisionTreeRegressor
    best_params: dict[str, Any]
    cv_mse: float
    cv_r2: float
    cv_splits: int


@dataclass
class ParameterForestBundle:
    models: dict[str, RandomForestRegressor]
    training_table: pd.DataFrame


@dataclass
class PaperPrediction:
    target_temperature_C: float
    target_current_mA: float
    predicted_total_period_s: float
    predicted_parameters: dict[str, float]
    predicted_curve: pd.DataFrame
    charge_duration_s: float
    discharge_duration_s: float
    potential_min_V: float
    potential_max_V: float


def _normalize_branch_time(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df["time_s"].to_numpy(dtype=float)
    v = df["potential_V"].to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    t = t[finite]
    v = v[finite]

    if len(t) < 5:
        raise ValueError("用于拟合的充电/放电数据点不足。")

    t = t - t[0]

    if float(t[-1]) <= 0:
        raise ValueError("充电/放电段时间长度必须大于 0。")

    return t, v


def _fit_branch(
    df: pd.DataFrame,
    branch: str,
    max_nfev: int = MAX_CURVE_FIT_EVALUATIONS,
) -> BranchFit:
    t, v = _normalize_branch_time(df)
    duration = max(float(t[-1]), 1e-6)

    if branch == "charge":
        func = charge_equation
        b0 = max(float(v[-1]), float(np.max(v)), 1e-6)
        c0 = (float(v[-1]) - float(v[0])) / duration * 0.1
    elif branch == "discharge":
        func = discharge_equation
        b0 = max(float(v[0]), float(np.max(v)), 1e-6)
        c0 = (float(v[-1]) - float(v[0])) / duration * 0.1
    else:
        raise ValueError("branch 必须是 'charge' 或 'discharge'。")

    a0 = max(duration * 0.3, 1e-6)
    x0 = np.array([a0, b0, c0], dtype=float)

    def residual(x):
        a, b, c = x
        return func(t, a, b, c) - v

    result = least_squares(
        residual,
        x0=x0,
        bounds=([1e-12, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
        max_nfev=max_nfev,
        method="trf",
        x_scale="jac",
    )

    a, b, c = [float(x) for x in result.x]
    pred = func(t, a, b, c)
    rmse = float(np.sqrt(np.mean((pred - v) ** 2)))

    return BranchFit(
        a=a,
        b=b,
        c=c,
        rmse_V=rmse,
        success=bool(result.success),
        message=str(result.message),
    )


def build_feature_table(
    records: list[GCDTemperatureRecord],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in records:
        if record.temperature_C is None:
            raise ValueError(f"{record.filename} 缺少温度。")
        if record.current_mA is None:
            raise ValueError(f"{record.filename} 缺少电流。")

        charge_fit = _fit_branch(record.charge_df, "charge")
        discharge_fit = _fit_branch(record.discharge_df, "discharge")
        total = float(record.total_period_s)

        if total <= 0:
            raise ValueError(f"{record.filename} 的总周期时间无效。")

        rows.append(
            {
                "record_id": record.record_id,
                "filename": record.filename,
                "temperature_C": float(record.temperature_C),
                "current_mA": float(record.current_mA),
                "time_period_s": total,
                "charge_duration_s": float(record.charge_duration_s),
                "discharge_duration_s": float(record.discharge_duration_s),
                "charge_fraction": float(record.charge_duration_s / total),
                "discharge_fraction": float(record.discharge_duration_s / total),
                "potential_min_V": float(record.potential_min_V),
                "potential_max_V": float(record.potential_max_V),
                "charge_a": charge_fit.a,
                "charge_b": charge_fit.b,
                "charge_c": charge_fit.c,
                "charge_fit_rmse_V": charge_fit.rmse_V,
                "charge_fit_success": charge_fit.success,
                "discharge_a": discharge_fit.a,
                "discharge_b": discharge_fit.b,
                "discharge_c": discharge_fit.c,
                "discharge_fit_rmse_V": discharge_fit.rmse_V,
                "discharge_fit_success": discharge_fit.success,
            }
        )

    table = pd.DataFrame(rows)
    if len(table) == 0:
        raise ValueError("没有可用于建模的 GCD 数据。")

    return table.sort_values(
        ["temperature_C", "current_mA", "filename"],
        kind="stable",
    ).reset_index(drop=True)


def _cv_splitter(n_samples: int) -> KFold:
    if n_samples < 4:
        raise ValueError("至少需要 4 条温度-电流 GCD 数据才能进行模型比较。")

    return KFold(
        n_splits=min(5, n_samples),
        shuffle=True,
        random_state=42,
    )


def compare_time_period_models(feature_table: pd.DataFrame) -> pd.DataFrame:
    data = feature_table.dropna(
        subset=["temperature_C", "current_mA", "time_period_s"]
    ).copy()

    if (data["current_mA"] <= 0).any():
        raise ValueError("电流必须大于 0 mA。")

    y = data["time_period_s"].to_numpy(dtype=float)
    X_tree = data[["temperature_C", "current_mA"]].to_numpy(dtype=float)

    X_ratio = (
        data["temperature_C"].to_numpy(dtype=float)
        / data["current_mA"].to_numpy(dtype=float)
    ).reshape(-1, 1)

    models = [
        ("Linear Regression", LinearRegression(), X_ratio),
        ("Support Vector Regression", SVR(kernel="linear"), X_ratio),
        ("Decision Tree", DecisionTreeRegressor(random_state=42), X_tree),
        ("Random Forest", RandomForestRegressor(random_state=42), X_tree),
    ]

    cv = _cv_splitter(len(data))
    rows: list[dict[str, Any]] = []

    for name, model, X in models:
        pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=None)
        rows.append(
            {
                "model": name,
                "mse_s2": float(mean_squared_error(y, pred)),
                "r2": float(r2_score(y, pred)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "mse_s2",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)


def tune_decision_tree(feature_table: pd.DataFrame) -> TimePeriodModelBundle:
    data = feature_table.dropna(
        subset=["temperature_C", "current_mA", "time_period_s"]
    ).copy()

    X = data[["temperature_C", "current_mA"]].to_numpy(dtype=float)
    y = data["time_period_s"].to_numpy(dtype=float)
    cv = _cv_splitter(len(data))
    comparison = compare_time_period_models(data)

    param_grid = {
        "max_depth": [None, 2, 3, 4, 5, 6, 8, 10],
        "min_samples_split": [2, 3, 4, 5, 6],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    }

    search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=None,
    )
    search.fit(X, y)

    best = search.best_estimator_
    cv_pred = cross_val_predict(clone(best), X, y, cv=cv, n_jobs=None)

    cv_mse = float(mean_squared_error(y, cv_pred))
    cv_r2 = float(r2_score(y, cv_pred))

    best.fit(X, y)

    return TimePeriodModelBundle(
        comparison=comparison,
        tuned_tree=best,
        best_params=dict(search.best_params_),
        cv_mse=cv_mse,
        cv_r2=cv_r2,
        cv_splits=cv.n_splits,
    )


def train_parameter_random_forests(
    feature_table: pd.DataFrame,
) -> ParameterForestBundle:
    columns = [
        "charge_a",
        "charge_b",
        "charge_c",
        "discharge_a",
        "discharge_b",
        "discharge_c",
    ]

    clean = feature_table.dropna(
        subset=["time_period_s", *columns]
    ).copy()

    if len(clean) < 3:
        raise ValueError("至少需要 3 条成功拟合的 GCD 数据来训练 a、b、c Random Forest。")

    X = clean[["time_period_s"]].to_numpy(dtype=float)
    models: dict[str, RandomForestRegressor] = {}

    for column in columns:
        model = RandomForestRegressor(random_state=42)
        model.fit(X, clean[column].to_numpy(dtype=float))
        models[column] = model

    return ParameterForestBundle(
        models=models,
        training_table=clean,
    )


def _predict_parameters(
    bundle: ParameterForestBundle,
    predicted_time_period_s: float,
) -> dict[str, float]:
    X = np.array([[float(predicted_time_period_s)]], dtype=float)
    return {
        name: float(model.predict(X)[0])
        for name, model in bundle.models.items()
    }


def _median_duration_fraction(
    feature_table: pd.DataFrame,
    target_current_mA: float,
) -> float:
    mask = np.isclose(
        feature_table["current_mA"].to_numpy(dtype=float),
        float(target_current_mA),
        rtol=1e-6,
        atol=1e-9,
    )
    same_current = feature_table[mask]
    source = same_current if len(same_current) >= 2 else feature_table
    fraction = float(np.median(source["charge_fraction"].to_numpy(dtype=float)))
    return float(np.clip(fraction, 0.05, 0.95))


def _target_voltage_window(
    feature_table: pd.DataFrame,
    target_current_mA: float,
) -> tuple[float, float]:
    mask = np.isclose(
        feature_table["current_mA"].to_numpy(dtype=float),
        float(target_current_mA),
        rtol=1e-6,
        atol=1e-9,
    )
    same_current = feature_table[mask]
    source = same_current if len(same_current) >= 1 else feature_table

    vmin = float(np.median(source["potential_min_V"].to_numpy(dtype=float)))
    vmax = float(np.median(source["potential_max_V"].to_numpy(dtype=float)))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("无法确定预测曲线的电位窗口。")

    return vmin, vmax


def _affine_endpoint_scale(
    values: np.ndarray,
    target_start: float,
    target_end: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    y0 = float(values[0])
    y1 = float(values[-1])

    if not np.isfinite(y0) or not np.isfinite(y1):
        raise ValueError("预测方程产生了非有限电压。")

    if abs(y1 - y0) < 1e-12:
        return np.linspace(target_start, target_end, len(values))

    scale = (target_end - target_start) / (y1 - y0)
    return target_start + (values - y0) * scale


def reconstruct_predicted_curve(
    predicted_time_period_s: float,
    parameters: dict[str, float],
    feature_table: pd.DataFrame,
    target_current_mA: float,
    points_per_branch: int = 300,
) -> tuple[pd.DataFrame, float, float, float, float]:
    if predicted_time_period_s <= 0:
        raise ValueError("预测得到的 time period 必须大于 0。")

    charge_fraction = _median_duration_fraction(
        feature_table,
        target_current_mA,
    )

    charge_duration = predicted_time_period_s * charge_fraction
    discharge_duration = predicted_time_period_s - charge_duration

    vmin, vmax = _target_voltage_window(
        feature_table,
        target_current_mA,
    )

    t_charge_local = np.linspace(0.0, charge_duration, int(points_per_branch))
    t_discharge_local = np.linspace(0.0, discharge_duration, int(points_per_branch))

    charge_raw = charge_equation(
        t_charge_local,
        parameters["charge_a"],
        parameters["charge_b"],
        parameters["charge_c"],
    )

    discharge_raw = discharge_equation(
        t_discharge_local,
        parameters["discharge_a"],
        parameters["discharge_b"],
        parameters["discharge_c"],
    )

    charge_v = _affine_endpoint_scale(
        charge_raw,
        target_start=vmin,
        target_end=vmax,
    )

    discharge_v = _affine_endpoint_scale(
        discharge_raw,
        target_start=vmax,
        target_end=vmin,
    )

    charge = pd.DataFrame(
        {
            "time_s": t_charge_local,
            "potential_V": charge_v,
            "branch": "Charging",
        }
    )

    discharge = pd.DataFrame(
        {
            "time_s": charge_duration + t_discharge_local,
            "potential_V": discharge_v,
            "branch": "Discharging",
        }
    )

    curve = pd.concat(
        [charge, discharge.iloc[1:]],
        ignore_index=True,
    )

    return curve, charge_duration, discharge_duration, vmin, vmax


def predict_paper_gcd(
    feature_table: pd.DataFrame,
    time_model: TimePeriodModelBundle,
    parameter_bundle: ParameterForestBundle,
    target_temperature_C: float,
    target_current_mA: float,
) -> PaperPrediction:
    if target_current_mA <= 0:
        raise ValueError("目标电流必须大于 0 mA。")

    X = np.array(
        [[float(target_temperature_C), float(target_current_mA)]],
        dtype=float,
    )

    predicted_period = float(time_model.tuned_tree.predict(X)[0])

    if predicted_period <= 0:
        raise ValueError(
            "Decision Tree 得到了非正的 time period，当前数据不适合该目标条件。"
        )

    parameters = _predict_parameters(
        parameter_bundle,
        predicted_period,
    )

    curve, charge_duration, discharge_duration, vmin, vmax = (
        reconstruct_predicted_curve(
            predicted_time_period_s=predicted_period,
            parameters=parameters,
            feature_table=feature_table,
            target_current_mA=target_current_mA,
        )
    )

    return PaperPrediction(
        target_temperature_C=float(target_temperature_C),
        target_current_mA=float(target_current_mA),
        predicted_total_period_s=predicted_period,
        predicted_parameters=parameters,
        predicted_curve=curve,
        charge_duration_s=charge_duration,
        discharge_duration_s=discharge_duration,
        potential_min_V=vmin,
        potential_max_V=vmax,
    )
