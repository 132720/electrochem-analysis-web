"""Paper stage 3.3 PCA-GPR only: model directly from uploaded research source.

Original: paper-stage3.3/统一论文结果/_stage1_recomputed/
_platform_source/electrochem_app/core/gcd_pca_gpr_final.py
Contains exactly one prediction model; no PCA-Ridge or Piecewise-Ridge.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import warnings
import io
import zipfile
import re
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from core.gcd_temperature_record_parser import GCDTemperatureRecord, parse_temperature_gcd_upload

NORMALIZED_POINTS=401
DEFAULT_VARIANCE_TARGET=0.995
DEFAULT_MAX_COMPONENTS=10

@dataclass
class PCAGPRFullCurveModel:
    grid: np.ndarray
    pca: PCA
    x_scaler: StandardScaler
    score_gps: list[GaussianProcessRegressor]
    time_gp: GaussianProcessRegressor
    training_temperatures_C: np.ndarray
    training_currents_A_g: np.ndarray
    training_curve_min_V: float
    training_curve_max_V: float
    variance_target: float
    explained_variance_ratio_sum: float

@dataclass
class PCAGPRPrediction:
    target_temperature_C: float
    target_current_A_g: float
    predicted_total_period_s: float
    total_period_ci95_low_s: float
    total_period_ci95_high_s: float
    predicted_charge_duration_s: float
    predicted_discharge_duration_s: float
    predicted_charge_specific_charge_C_g: float
    predicted_discharge_specific_charge_C_g: float
    predicted_peak_potential_V: float
    predicted_peak_tau: float
    normalized_curve: pd.DataFrame
    physical_curve: pd.DataFrame
    diagnostics: dict[str, Any]

def _record_feature(record: GCDTemperatureRecord) -> np.ndarray:
    if record.current_value is None:
        raise ValueError(
            f"{record.filename} 无法识别电流密度，不能训练 PCA-GPR。"
        )

    current = float(record.current_value)

    if current <= 0:
        raise ValueError("电流密度必须大于 0。")

    return np.asarray(
        [
            float(record.temperature_C),
            float(np.log(current)),
        ],
        dtype=float,
    )

def _features(records: list[GCDTemperatureRecord]) -> np.ndarray:
    return np.vstack(
        [
            _record_feature(record)
            for record in records
        ]
    )

def _cycle_grid(
    record: GCDTemperatureRecord,
    grid: np.ndarray,
) -> np.ndarray:
    df = record.cycle_df

    t = df["time_s"].to_numpy(dtype=float)
    v = df["potential_V"].to_numpy(dtype=float)

    finite = np.isfinite(t) & np.isfinite(v)
    t = t[finite]
    v = v[finite]

    if len(t) < 10:
        raise ValueError(
            f"{record.filename} 有效 GCD 点过少。"
        )

    t = t - t[0]
    duration = float(t[-1])

    if duration <= 0:
        raise ValueError(
            f"{record.filename} 周期时间无效。"
        )

    tau = t / duration

    keep = np.r_[
        True,
        np.diff(tau) > 1e-12,
    ]

    return np.interp(
        grid,
        tau[keep],
        v[keep],
    )

def _curve_matrix(
    records: list[GCDTemperatureRecord],
    grid: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            _cycle_grid(record, grid)
            for record in records
        ]
    )

def _select_pca(
    matrix: np.ndarray,
    variance_target: float,
    max_components: int,
) -> PCA:
    if matrix.shape[0] < 3:
        raise ValueError(
            "PCA-GPR 至少需要 3 条 GCD 曲线。"
        )

    max_n = min(
        int(max_components),
        int(matrix.shape[0] - 1),
        int(matrix.shape[1]),
    )

    if max_n < 1:
        raise ValueError("PCA 可用主成分数量不足。")

    probe = PCA(
        n_components=max_n,
        svd_solver="full",
    ).fit(matrix)

    cumulative = np.cumsum(
        probe.explained_variance_ratio_
    )

    n_components = int(
        np.searchsorted(
            cumulative,
            float(variance_target),
        )
        + 1
    )

    n_components = max(
        1,
        min(
            n_components,
            max_n,
        ),
    )

    return PCA(
        n_components=n_components,
        svd_solver="full",
    ).fit(matrix)

def _gpr_kernel() -> Any:
    # ARD Matérn 3/2 is deliberately used as a conservative small-sample
    # kernel. The two length scales correspond to temperature and log(current).
    return (
        ConstantKernel(
            1.0,
            constant_value_bounds=(1e-3, 1e3),
        )
        * Matern(
            length_scale=np.asarray(
                [1.0, 1.0],
                dtype=float,
            ),
            length_scale_bounds=(0.2, 20.0),
            nu=1.5,
        )
        + WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-8, 2e-1),
        )
    )

def _fit_gp(
    x_scaled: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=_gpr_kernel(),
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=int(random_state),
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )

        gp.fit(
            x_scaled,
            np.asarray(
                y,
                dtype=float,
            ),
        )

    return gp

def train_pca_gpr_full_curve(
    records: list[GCDTemperatureRecord],
    variance_target: float = DEFAULT_VARIANCE_TARGET,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    random_state: int = 42,
) -> PCAGPRFullCurveModel:
    if len(records) < 8:
        raise ValueError(
            "PCA-GPR 建议至少使用 8 条曲线；当前数据不足。"
        )

    temperatures = sorted(
        {
            float(record.temperature_C)
            for record in records
        }
    )

    if len(temperatures) < 2:
        raise ValueError(
            "PCA-GPR 至少需要两个温度条件。"
        )

    grid = np.linspace(
        0.0,
        1.0,
        NORMALIZED_POINTS,
    )

    curves = _curve_matrix(
        records,
        grid,
    )

    pca = _select_pca(
        curves,
        variance_target=float(
            variance_target
        ),
        max_components=int(
            max_components
        ),
    )

    scores = pca.transform(
        curves
    )

    x = _features(
        records
    )

    x_scaler = StandardScaler().fit(
        x
    )

    x_scaled = x_scaler.transform(
        x
    )

    score_gps = []

    for component in range(
        scores.shape[1]
    ):
        score_gps.append(
            _fit_gp(
                x_scaled,
                scores[:, component],
                random_state=
                    int(random_state)
                    + component,
            )
        )

    periods = np.asarray(
        [
            float(record.total_period_s)
            for record in records
        ],
        dtype=float,
    )

    if np.any(
        periods <= 0
    ):
        raise ValueError(
            "训练数据存在非正周期时间。"
        )

    time_gp = _fit_gp(
        x_scaled,
        np.log(periods),
        random_state=
            int(random_state)
            + 100,
    )

    return PCAGPRFullCurveModel(
        grid=grid,
        pca=pca,
        x_scaler=x_scaler,
        score_gps=score_gps,
        time_gp=time_gp,
        training_temperatures_C=np.asarray(
            temperatures,
            dtype=float,
        ),
        training_currents_A_g=np.asarray(
            sorted(
                {
                    float(record.current_value)
                    for record in records
                    if record.current_value is not None
                }
            ),
            dtype=float,
        ),
        training_curve_min_V=float(
            np.min(curves)
        ),
        training_curve_max_V=float(
            np.max(curves)
        ),
        variance_target=float(
            variance_target
        ),
        explained_variance_ratio_sum=float(
            np.sum(
                pca.explained_variance_ratio_
            )
        ),
    )

def _predict_latent(
    model: PCAGPRFullCurveModel,
    temperature_C: float,
    current_A_g: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    current = float(
        current_A_g
    )

    if current <= 0:
        raise ValueError(
            "目标电流密度必须大于 0。"
        )

    x = np.asarray(
        [
            [
                float(
                    temperature_C
                ),
                float(
                    np.log(
                        current
                    )
                ),
            ]
        ],
        dtype=float,
    )

    xs = model.x_scaler.transform(
        x
    )

    means = []
    stds = []

    for gp in model.score_gps:
        mean, std = gp.predict(
            xs,
            return_std=True,
        )

        means.append(
            float(
                mean[0]
            )
        )

        stds.append(
            float(
                std[0]
            )
        )

    log_time_mean, log_time_std = (
        model.time_gp.predict(
            xs,
            return_std=True,
        )
    )

    return (
        np.asarray(
            means,
            dtype=float,
        ),
        np.asarray(
            stds,
            dtype=float,
        ),
        float(
            log_time_mean[0]
        ),
        float(
            log_time_std[0]
        ),
    )

def predict_pca_gpr_full_curve(
    model: PCAGPRFullCurveModel,
    temperature_C: float,
    current_A_g: float,
    uncertainty_samples: int = 400,
    random_state: int = 12345,
) -> PCAGPRPrediction:
    (
        score_mean,
        score_std,
        log_time_mean,
        log_time_std,
    ) = _predict_latent(
        model,
        temperature_C=
            float(
                temperature_C
            ),
        current_A_g=
            float(
                current_A_g
            ),
    )

    mean_curve = (
        model.pca.inverse_transform(
            score_mean.reshape(
                1,
                -1,
            )
        )[0]
    )

    # Independent latent-GP approximation; PCA is a linear decoder.
    latent_variance = (
        (
            score_std ** 2
        )
        @ (
            model.pca.components_ ** 2
        )
    )

    residual_variance = float(
        getattr(
            model.pca,
            "noise_variance_",
            0.0,
        )
        or 0.0
    )

    shape_std = np.sqrt(
        np.maximum(
            latent_variance
            + residual_variance,
            0.0,
        )
    )

    total_period = float(
        np.exp(
            log_time_mean
        )
    )

    time_ci_low = float(
        np.exp(
            log_time_mean
            - 1.96 * log_time_std
        )
    )

    time_ci_high = float(
        np.exp(
            log_time_mean
            + 1.96 * log_time_std
        )
    )

    normalized = pd.DataFrame(
        {
            "tau":
                model.grid,

            "predicted_potential_V":
                mean_curve,

            "lower95_shape_V":
                mean_curve
                - 1.96
                * shape_std,

            "upper95_shape_V":
                mean_curve
                + 1.96
                * shape_std,
        }
    )

    # Full V(t) interval by posterior sampling of both latent scores and
    # log(total period). Score GPs are treated as conditionally independent.
    samples = max(
        0,
        int(
            uncertainty_samples
        ),
    )

    physical_time = (
        model.grid
        * total_period
    )

    if samples >= 20:
        rng = np.random.default_rng(
            int(
                random_state
            )
        )

        score_draws = rng.normal(
            loc=
                score_mean[
                    None,
                    :
                ],
            scale=
                np.maximum(
                    score_std[
                        None,
                        :
                    ],
                    1e-12,
                ),
            size=(
                samples,
                len(
                    score_mean
                ),
            ),
        )

        curve_draws = (
            model.pca.inverse_transform(
                score_draws
            )
        )

        log_time_draws = rng.normal(
            loc=
                log_time_mean,
            scale=
                max(
                    log_time_std,
                    1e-12,
                ),
            size=
                samples,
        )

        time_draws = np.exp(
            log_time_draws
        )

        physical_draws = np.empty(
            (
                samples,
                len(
                    model.grid
                ),
            ),
            dtype=float,
        )

        for i in range(
            samples
        ):
            sample_time = (
                model.grid
                * float(
                    time_draws[i]
                )
            )

            physical_draws[i] = np.interp(
                physical_time,
                sample_time,
                curve_draws[i],
            )

        physical_low = np.percentile(
            physical_draws,
            2.5,
            axis=0,
        )

        physical_high = np.percentile(
            physical_draws,
            97.5,
            axis=0,
        )

    else:
        physical_low = (
            mean_curve
            - 1.96
            * shape_std
        )

        physical_high = (
            mean_curve
            + 1.96
            * shape_std
        )

    physical = pd.DataFrame(
        {
            "time_s":
                physical_time,

            "predicted_potential_V":
                mean_curve,

            "lower95_V":
                physical_low,

            "upper95_V":
                physical_high,
        }
    )

    peak_index = int(
        np.argmax(
            mean_curve
        )
    )

    peak_tau = float(
        model.grid[
            peak_index
        ]
    )

    charge_duration = float(
        peak_tau
        * total_period
    )

    discharge_duration = float(
        max(
            total_period
            - charge_duration,
            0.0,
        )
    )

    current = float(
        current_A_g
    )

    outside_temperature = bool(
        float(
            temperature_C
        )
        <
        float(
            np.min(
                model.training_temperatures_C
            )
        )
        or
        float(
            temperature_C
        )
        >
        float(
            np.max(
                model.training_temperatures_C
            )
        )
    )

    diagnostics = {
        "n_pca_components":
            int(
                model.pca.n_components_
            ),

        "pca_explained_variance":
            float(
                model.explained_variance_ratio_sum
            ),

        "mean_shape_ci95_width_mV":
            float(
                np.mean(
                    normalized[
                        "upper95_shape_V"
                    ]
                    -
                    normalized[
                        "lower95_shape_V"
                    ]
                )
                * 1000.0
            ),

        "mean_physical_ci95_width_mV":
            float(
                np.mean(
                    physical[
                        "upper95_V"
                    ]
                    -
                    physical[
                        "lower95_V"
                    ]
                )
                * 1000.0
            ),

        "relative_total_period_ci95_width_pct":
            float(
                (
                    time_ci_high
                    - time_ci_low
                )
                /
                max(
                    total_period,
                    1e-12,
                )
                * 100.0
            ),

        "outside_training_temperature":
            outside_temperature,

        "training_temperature_min_C":
            float(
                np.min(
                    model.training_temperatures_C
                )
            ),

        "training_temperature_max_C":
            float(
                np.max(
                    model.training_temperatures_C
                )
            ),

        "training_current_min_A_g":
            float(
                np.min(
                    model.training_currents_A_g
                )
            ),

        "training_current_max_A_g":
            float(
                np.max(
                    model.training_currents_A_g
                )
            ),
    }

    return PCAGPRPrediction(
        target_temperature_C=float(
            temperature_C
        ),
        target_current_A_g=current,
        predicted_total_period_s=
            total_period,
        total_period_ci95_low_s=
            time_ci_low,
        total_period_ci95_high_s=
            time_ci_high,
        predicted_charge_duration_s=
            charge_duration,
        predicted_discharge_duration_s=
            discharge_duration,
        predicted_charge_specific_charge_C_g=
            float(
                current
                * charge_duration
            ),
        predicted_discharge_specific_charge_C_g=
            float(
                current
                * discharge_duration
            ),
        predicted_peak_potential_V=
            float(
                mean_curve[
                    peak_index
                ]
            ),
        predicted_peak_tau=
            peak_tau,
        normalized_curve=
            normalized,
        physical_curve=
            physical,
        diagnostics=
            diagnostics,
    )


# Trusted GCD input parsing and strict held-out evaluation.
class _Upload:
    def __init__(self,name,raw):
        self.name=name
        self._raw=raw
    def getvalue(self):
        return self._raw

def training_manifest(source_zip:bytes):
    """List concentration / temperature / current without reading experiment data."""
    output=[]
    with zipfile.ZipFile(io.BytesIO(source_zip)) as z:
        infos=z.infolist()
        if len(infos)>1500 or sum(i.file_size for i in infos)>160_000_000:
            raise ValueError("训练数据ZIP过大或条目过多")
        for item in infos:
            if item.is_dir() or item.file_size>3_000_000:continue
            path=item.filename.replace("\\\\","/")
            if ".." in PurePosixPath(path).parts or path.startswith("/"):
                raise ValueError("ZIP中存在不安全路径")
            m=re.search(r"(?:^|/)([124])\s*mol/([0-9]+(?:\.[0-9]+)?)/[^/]+/gcd/([0-9]+)[-_]([0-9]+)\.txt$",path,re.I)
            if not m:continue
            concentration=int(m.group(1))
            temperature=float(m.group(2))
            density=float(m.group(3)+"."+m.group(4))
            output.append({"filename":path,
                           "concentration_mol_L":concentration,
                           "temperature_C":temperature,
                           "current_A_g":density})
    if not output:raise ValueError(
        "没有识别到训练文件；应使用testwendu3/[1|2|4]mol/温度/质量/gcd/1-0.txt等结构")
    frame=pd.DataFrame(output)
    if frame.duplicated(["concentration_mol_L","temperature_C","current_A_g"]).any():
        raise ValueError("相同浓度/温度/电流密度存在重复文件；请先核对数据")
    return frame.sort_values(["concentration_mol_L","temperature_C","current_A_g"]).reset_index(drop=True)

def load_training_records(source_zip, manifest, concentration, target_temperature):
    """No test-temperature data can enter training; fail on any malformed training record."""
    subset=manifest[(manifest.concentration_mol_L==int(concentration)) &
                    (manifest.temperature_C<float(target_temperature))]
    if subset.empty:
        raise ValueError("没有低于目标温度的训练记录")
    if subset.temperature_C.nunique()<2 or len(subset)<3:
        raise ValueError("至少需要两个不同的较低训练温度及三条有效GCD")
    output=[]
    with zipfile.ZipFile(io.BytesIO(source_zip)) as z:
        for row in subset.itertuples(index=False):
            filename=str(row.filename)
            raw=z.read(filename)
            rec=parse_temperature_gcd_upload(
                _Upload(PurePosixPath(filename).name,raw),float(row.temperature_C))
            rec.current_value=float(row.current_A_g)
            rec.metadata["concentration_mol_L"]=int(concentration)
            output.append(rec)
    return output

def find_heldout_record(source_zip,manifest,concentration,temperature,current):
    """Optional independent measured comparison, never used in training."""
    matches=manifest[
        (manifest.concentration_mol_L==int(concentration))&
        np.isclose(manifest.temperature_C,float(temperature))&
        np.isclose(manifest.current_A_g,float(current))
    ]
    if len(matches)!=1:return None
    row=matches.iloc[0]
    with zipfile.ZipFile(io.BytesIO(source_zip)) as z:
        raw=z.read(str(row.filename))
    record=parse_temperature_gcd_upload(
        _Upload(PurePosixPath(str(row.filename)).name,raw),float(temperature))
    record.current_value=float(current)
    return record

def compare_heldout(predicted,record):
    """Score on measured time samples; beyond predicted endpoint hold end potential."""
    t=record.cycle_df.time_s.to_numpy(dtype=float)
    t=t-t[0]
    observed=record.cycle_df.potential_V.to_numpy(dtype=float)
    predicted_v=np.interp(t,predicted.time_s.to_numpy(float),
                          predicted.predicted_potential_V.to_numpy(float))
    errors=predicted_v-observed
    df=pd.DataFrame({
        "measured_time_s":t,"measured_potential_V":observed,
        "predicted_potential_V":predicted_v,
        "residual_V":errors,
    })
    true_period=float(record.total_period_s)
    pred_period=float(predicted.time_s.iloc[-1])
    summary={
        "Measured total period (s)":true_period,
        "Predicted total period (s)":pred_period,
        "Period absolute error (s)":abs(pred_period-true_period),
        "Period relative error (%)":100*(pred_period/true_period-1),
        "Curve RMSE (mV)":float(np.sqrt(np.mean(errors**2))*1000),
        "Curve MAE (mV)":float(np.mean(np.abs(errors))*1000),
        "Test condition withheld from training":True,
    }
    return df,summary

def fit_and_predict_paper33(records,temperature_C,current_A_g,
                            variance_target=DEFAULT_VARIANCE_TARGET,
                            max_components=DEFAULT_MAX_COMPONENTS,
                            uncertainty_samples=120):
    """Fit paper's original PCA-GPR on STRICTLY LOWER temperatures."""
    target=float(temperature_C)
    current=float(current_A_g)
    if not records or target<=max(float(r.temperature_C) for r in records):
        raise ValueError("目标温度必须高于全部训练温度，防止目标条件泄漏")
    if not np.isfinite(current) or current<=0:
        raise ValueError("电流密度必须大于0")
    model=train_pca_gpr_full_curve(
        records,variance_target=variance_target,
        max_components=max_components,random_state=42)
    predicted=predict_pca_gpr_full_curve(
        model,target,current,uncertainty_samples=uncertainty_samples,
        random_state=12345)
    return model,predicted


def physical_prediction_as_frame(prediction):
    df=prediction.physical_curve.copy()
    df["normalized_time"]=prediction.normalized_curve["tau"].to_numpy(float)
    return df


def paper33_heldout_comparison(prediction,record):
    """Score GPR prediction on measured time samples of withheld condition."""
    frame=physical_prediction_as_frame(prediction)
    curve=frame.rename(columns={"time_s":"time_s"})
    return compare_heldout(curve,record)
