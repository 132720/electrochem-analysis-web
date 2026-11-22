from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any, Iterable
import math
import re
import uuid

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress


R_GAS_J_MOL_K = 8.31446261815324

CIRCUIT_ONE_ARC = "Rs + (R1 || CPE1)"
CIRCUIT_TWO_ARC = "Rs + (R1 || CPE1) + (R2 || CPE2)"
CIRCUIT_ONE_ARC_W = "Rs + (R1 || CPE1) + W"

SUPPORTED_TEMPERATURE_EIS_CIRCUITS = [
    CIRCUIT_TWO_ARC,
    CIRCUIT_ONE_ARC,
    CIRCUIT_ONE_ARC_W,
]


@dataclass
class TemperatureEISRecord:
    record_id: str
    filename: str
    temperature_C: float
    replicate_index: int
    metadata: dict[str, Any]
    data: pd.DataFrame


@dataclass
class KKCompatibilityResult:
    record_id: str
    filename: str
    temperature_C: float
    replicate_index: int
    n_terms: int
    relative_rmse: float
    rmse_ohm: float
    max_normalized_error: float
    screening_threshold: float
    screening_pass: bool
    message: str
    fitted_data: pd.DataFrame


@dataclass
class EISTemperatureFitResult:
    record_id: str
    filename: str
    temperature_C: float
    replicate_index: int
    circuit: str
    success: bool
    message: str
    parameters: dict[str, float]
    initial_parameters: dict[str, float]
    rmse_ohm: float
    relative_rmse: float
    n_points_used: int
    weighting: str
    excluded_inductive_points: bool
    fitted_data: pd.DataFrame


@dataclass
class ArrheniusResult:
    parameter_name: str
    process_label: str
    n_temperatures: int
    slope: float
    intercept: float
    r2: float
    activation_energy_kJ_mol: float
    pre_exponential_inverse_resistance: float
    table: pd.DataFrame


# ---------------------------------------------------------------------------
# Temperature grid
# ---------------------------------------------------------------------------

def build_temperature_grid(
    start_C: float,
    end_C: float,
    step_C: float,
) -> list[float]:
    start_C = float(start_C)
    end_C = float(end_C)
    step_C = float(step_C)

    if not np.isfinite([start_C, end_C, step_C]).all():
        raise ValueError("温度参数必须是有限数值。")

    if step_C <= 0:
        raise ValueError("温度步长必须大于 0。")

    if end_C < start_C:
        raise ValueError("终止温度不能低于起始温度。")

    values = []
    value = start_C
    guard = 0

    while value <= end_C + step_C * 1e-9:
        values.append(float(round(value, 10)))
        value += step_C
        guard += 1

        if guard > 1000:
            raise ValueError("温度序列过长，请检查起始温度、终止温度和步长。")

    if not np.isclose(values[-1], end_C, rtol=1e-9, atol=1e-9):
        values.append(float(end_C))

    return values


# ---------------------------------------------------------------------------
# CHI / text parser
# ---------------------------------------------------------------------------

def _decode_bytes(raw: bytes) -> str:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"无法解码文本文件：{last_error}")


def _parse_numeric_text(value: str) -> Any:
    text = str(value).strip()

    if not text:
        return ""

    normalized = text.replace("−", "-")

    try:
        if re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?",
            normalized,
        ):
            return float(normalized)
    except Exception:
        pass

    return text


def _parse_metadata(lines: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if "=" in stripped:
            key, value = stripped.split("=", 1)
            metadata[key.strip()] = _parse_numeric_text(value)

    return metadata


def _find_eis_header(lines: list[str]) -> int:
    candidates = [
        ("Freq/Hz", "Z'/ohm", 'Z"/ohm'),
        ("Frequency/Hz", "Zreal/ohm", "Zimag/ohm"),
    ]

    for index, line in enumerate(lines):
        compact = line.replace(" ", "")

        if (
            "Freq/Hz" in compact
            and "Z'/ohm" in compact
            and (
                'Z"/ohm' in compact
                or "Z''/ohm" in compact
            )
        ):
            return index

        if (
            "Frequency" in compact
            and (
                "Zreal" in compact
                or "Z'" in compact
            )
        ):
            return index

    raise ValueError(
        "未找到 EIS 数据表头。CHI 文件应包含类似 "
        "Freq/Hz, Z'/ohm, Z\"/ohm 的表头。"
    )


def _split_row(line: str) -> list[str]:
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = re.split(r"\s+", line.strip())

    return [part.strip() for part in parts if part.strip() != ""]


def _read_eis_table(lines: list[str], header_index: int) -> pd.DataFrame:
    header = _split_row(lines[header_index])

    rows: list[list[float]] = []

    for line in lines[header_index + 1:]:
        stripped = line.strip()

        if not stripped:
            continue

        parts = _split_row(stripped)

        if len(parts) < 3:
            continue

        try:
            numeric = [
                float(part.replace("−", "-"))
                for part in parts[: max(5, len(header))]
            ]
        except Exception:
            continue

        rows.append(numeric)

    if len(rows) < 6:
        raise ValueError("EIS 数据点过少，至少需要 6 个有效频率点。")

    max_columns = max(len(row) for row in rows)
    array = np.full((len(rows), max_columns), np.nan, dtype=float)

    for i, row in enumerate(rows):
        array[i, : len(row)] = row

    # CHI660E standard export:
    # Freq/Hz, Z'/ohm, Z"/ohm, Z/ohm, Phase/deg
    data = pd.DataFrame(
        {
            "frequency_Hz": array[:, 0],
            "z_real_ohm": array[:, 1],
            "z_imag_ohm": array[:, 2],
        }
    )

    if max_columns >= 4:
        data["z_modulus_ohm"] = array[:, 3]
    else:
        data["z_modulus_ohm"] = np.sqrt(
            data["z_real_ohm"] ** 2
            + data["z_imag_ohm"] ** 2
        )

    if max_columns >= 5:
        data["phase_deg"] = array[:, 4]
    else:
        data["phase_deg"] = np.degrees(
            np.arctan2(
                data["z_imag_ohm"],
                data["z_real_ohm"],
            )
        )

    finite = (
        np.isfinite(data["frequency_Hz"])
        & np.isfinite(data["z_real_ohm"])
        & np.isfinite(data["z_imag_ohm"])
        & (data["frequency_Hz"] > 0)
    )

    data = data.loc[finite].copy()

    if len(data) < 6:
        raise ValueError("清理无效值后 EIS 数据点不足 6 个。")

    # Preserve the raw Z'' sign. Only sort frequency for numerical analysis.
    data = (
        data.sort_values(
            "frequency_Hz",
            ascending=False,
            kind="stable",
        )
        .drop_duplicates(
            subset=["frequency_Hz"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return data


def parse_temperature_eis_upload(
    uploaded_file,
    temperature_C: float,
    replicate_index: int = 1,
) -> TemperatureEISRecord:
    if hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
        filename = getattr(uploaded_file, "name", "uploaded.txt")
    elif isinstance(uploaded_file, (bytes, bytearray)):
        raw = bytes(uploaded_file)
        filename = "uploaded.txt"
    else:
        raise TypeError("uploaded_file 必须是 Streamlit UploadedFile 或 bytes。")

    text = _decode_bytes(raw)
    lines = text.splitlines()

    header_index = _find_eis_header(lines)
    metadata = _parse_metadata(lines[:header_index])
    data = _read_eis_table(lines, header_index)

    return TemperatureEISRecord(
        record_id=str(uuid.uuid4()),
        filename=str(filename),
        temperature_C=float(temperature_C),
        replicate_index=int(replicate_index),
        metadata=metadata,
        data=data,
    )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _metadata_lookup(
    metadata: dict[str, Any],
    possible_keys: Iterable[str],
) -> float | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(key).lower()): value
        for key, value in metadata.items()
    }

    for key in possible_keys:
        norm_key = re.sub(r"[^a-z0-9]", "", key.lower())

        if norm_key in normalized:
            try:
                value = float(normalized[norm_key])

                if np.isfinite(value):
                    return value
            except Exception:
                continue

    return None


def extract_ac_amplitude_V(record: TemperatureEISRecord) -> float | None:
    return _metadata_lookup(
        record.metadata,
        [
            "Amplitude/V",
            "Amplitude",
            "AC Amplitude/V",
            "A.C. Amplitude/V",
        ],
    )


def extract_high_frequency_Hz(record: TemperatureEISRecord) -> float:
    value = _metadata_lookup(
        record.metadata,
        [
            "High Freq/Hz",
            "High Frequency/Hz",
            "High Freq",
            "High Frequency",
        ],
    )

    if value is not None:
        return float(value)

    return float(record.data["frequency_Hz"].max())


def extract_low_frequency_Hz(record: TemperatureEISRecord) -> float:
    value = _metadata_lookup(
        record.metadata,
        [
            "Low Freq/Hz",
            "Low Frequency/Hz",
            "Low Freq",
            "Low Frequency",
        ],
    )

    if value is not None:
        return float(value)

    return float(record.data["frequency_Hz"].min())


# ---------------------------------------------------------------------------
# Dataset diagnostics
# ---------------------------------------------------------------------------

def summarize_temperature_eis_records(
    records: list[TemperatureEISRecord],
) -> pd.DataFrame:
    rows = []

    for record in records:
        rows.append(
            {
                "temperature_C": record.temperature_C,
                "replicate": record.replicate_index,
                "filename": record.filename,
                "n_points": len(record.data),
                "high_frequency_Hz": extract_high_frequency_Hz(record),
                "low_frequency_Hz": extract_low_frequency_Hz(record),
                "ac_amplitude_V": extract_ac_amplitude_V(record),
                "z_real_min_ohm": float(record.data["z_real_ohm"].min()),
                "z_real_max_ohm": float(record.data["z_real_ohm"].max()),
            }
        )

    return pd.DataFrame(rows)


def temperature_replicate_counts(
    records: list[TemperatureEISRecord],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "temperature_C",
                "replicate_count",
            ]
        )

    table = summarize_temperature_eis_records(records)

    return (
        table.groupby("temperature_C", as_index=False)
        .agg(replicate_count=("record_id" if "record_id" in table.columns else "filename", "count"))
        if False
        else table.groupby("temperature_C", as_index=False)
        .agg(replicate_count=("filename", "count"))
    )


def check_ac_amplitude_consistency(
    records: list[TemperatureEISRecord],
    relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    values = [
        extract_ac_amplitude_V(record)
        for record in records
    ]

    available = [
        value
        for value in values
        if value is not None
        and np.isfinite(value)
    ]

    if len(available) < 2:
        return {
            "available": len(available),
            "consistent": None,
            "min_V": np.nan,
            "max_V": np.nan,
            "relative_span": np.nan,
            "message": "文件元数据中可识别的 AC amplitude 不足，无法自动比较。",
        }

    min_value = float(min(available))
    max_value = float(max(available))
    mean_value = float(np.mean(available))

    if abs(mean_value) < 1e-15:
        relative_span = np.inf
    else:
        relative_span = abs(max_value - min_value) / abs(mean_value)

    consistent = bool(relative_span <= float(relative_tolerance))

    return {
        "available": len(available),
        "consistent": consistent,
        "min_V": min_value,
        "max_V": max_value,
        "relative_span": relative_span,
        "message": (
            "AC amplitude 基本一致。"
            if consistent
            else "不同谱的 AC amplitude 存在明显差异，请确认实验设计。"
        ),
    }


def common_frequency_overlap(
    records: list[TemperatureEISRecord],
) -> tuple[float, float] | None:
    if not records:
        return None

    low = max(
        float(record.data["frequency_Hz"].min())
        for record in records
    )

    high = min(
        float(record.data["frequency_Hz"].max())
        for record in records
    )

    if low >= high:
        return None

    return float(low), float(high)


# ---------------------------------------------------------------------------
# Linear KK compatibility screening
# ---------------------------------------------------------------------------

def _kk_design_matrix(
    frequency_Hz: np.ndarray,
    tau: np.ndarray,
    include_inductance: bool,
) -> np.ndarray:
    omega = 2.0 * np.pi * frequency_Hz

    columns_real = [
        np.ones_like(omega),
    ]

    columns_imag = [
        np.zeros_like(omega),
    ]

    for tau_k in tau:
        wt = omega * tau_k
        denominator = 1.0 + wt * wt

        columns_real.append(
            1.0 / denominator
        )

        columns_imag.append(
            -wt / denominator
        )

    # Optional series capacitance-like term:
    # coefficient p_C = 1/C, Z_C = -j p_C / omega.
    columns_real.append(
        np.zeros_like(omega)
    )
    columns_imag.append(
        -1.0 / omega
    )

    if include_inductance:
        # coefficient is L in H: Z_L = j omega L
        columns_real.append(
            np.zeros_like(omega)
        )
        columns_imag.append(
            omega
        )

    real = np.column_stack(columns_real)
    imag = np.column_stack(columns_imag)

    return np.vstack([real, imag])


def _fit_linear_kk_for_terms(
    frequency_Hz: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
    n_terms: int,
    include_inductance: bool,
    ridge_strength: float,
) -> dict[str, Any]:
    f_max = float(np.max(frequency_Hz))
    f_min = float(np.min(frequency_Hz))

    tau_min = 1.0 / (2.0 * np.pi * f_max) / 3.0
    tau_max = 1.0 / (2.0 * np.pi * f_min) * 3.0

    tau = np.logspace(
        np.log10(tau_min),
        np.log10(tau_max),
        int(n_terms),
    )

    design = _kk_design_matrix(
        frequency_Hz,
        tau,
        include_inductance=include_inductance,
    )

    target = np.concatenate(
        [
            z_real,
            z_imag,
        ]
    )

    modulus = np.sqrt(z_real ** 2 + z_imag ** 2)
    floor = max(
        float(np.nanmedian(modulus)) * 1e-6,
        1e-12,
    )
    weights = 1.0 / np.maximum(modulus, floor)
    stacked_weights = np.concatenate([weights, weights])

    weighted_design = design * stacked_weights[:, None]
    weighted_target = target * stacked_weights

    # Ridge regularization is applied only to distributed RC coefficients.
    n_coefficients = design.shape[1]
    penalty = np.zeros((n_coefficients, n_coefficients), dtype=float)

    # index 0 = Rinf; middle terms = RC coefficients; last 1 or 2 are C/L.
    rc_end = 1 + int(n_terms)
    for index in range(1, rc_end):
        penalty[index, index] = float(ridge_strength)

    augmented_design = np.vstack(
        [
            weighted_design,
            penalty,
        ]
    )

    augmented_target = np.concatenate(
        [
            weighted_target,
            np.zeros(n_coefficients),
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        augmented_design,
        augmented_target,
        rcond=None,
    )

    predicted = design @ coefficients
    n = len(frequency_Hz)

    z_real_fit = predicted[:n]
    z_imag_fit = predicted[n:]

    residual_complex = (
        (z_real_fit - z_real)
        + 1j * (z_imag_fit - z_imag)
    )

    rmse_ohm = float(
        np.sqrt(
            np.mean(
                np.abs(residual_complex) ** 2
            )
        )
    )

    normalized_error = (
        np.abs(residual_complex)
        / np.maximum(modulus, floor)
    )

    relative_rmse = float(
        np.sqrt(
            np.mean(
                normalized_error ** 2
            )
        )
    )

    max_normalized_error = float(
        np.max(normalized_error)
    )

    weighted_residual = (
        np.concatenate(
            [
                z_real_fit - z_real,
                z_imag_fit - z_imag,
            ]
        )
        * stacked_weights
    )

    rss = float(
        np.sum(weighted_residual ** 2)
    )

    n_observations = 2 * n
    k = n_coefficients

    bic = (
        n_observations
        * np.log(
            max(
                rss / n_observations,
                1e-30,
            )
        )
        + k * np.log(n_observations)
    )

    return {
        "tau": tau,
        "coefficients": coefficients,
        "z_real_fit": z_real_fit,
        "z_imag_fit": z_imag_fit,
        "rmse_ohm": rmse_ohm,
        "relative_rmse": relative_rmse,
        "max_normalized_error": max_normalized_error,
        "bic": float(bic),
    }


def linear_kk_compatibility_check(
    record: TemperatureEISRecord,
    screening_threshold: float = 0.05,
    min_terms: int = 3,
    max_terms: int = 14,
    include_inductance: bool = True,
    ridge_strength: float = 1e-5,
) -> KKCompatibilityResult:
    """
    Practical linear Kramers-Kronig compatibility screening.

    Civan et al. (2026) recommend KK analysis before equivalent-circuit fitting,
    but the review does not prescribe one specific KK implementation. This
    software uses a linear measurement-model approximation consisting of a
    distribution of Debye RC time constants, plus optional series C/L terms.

    The screening threshold is a software QA threshold, not a universal
    electrochemical standard.
    """
    df = record.data

    frequency = df["frequency_Hz"].to_numpy(dtype=float)
    z_real = df["z_real_ohm"].to_numpy(dtype=float)
    z_imag = df["z_imag_ohm"].to_numpy(dtype=float)

    if len(df) < 8:
        raise ValueError("KK 兼容性检查至少需要 8 个有效频率点。")

    min_terms = max(2, int(min_terms))
    max_terms = max(min_terms, int(max_terms))
    max_terms = min(max_terms, max(3, len(df) // 2))

    candidates = []

    for n_terms in range(min_terms, max_terms + 1):
        try:
            result = _fit_linear_kk_for_terms(
                frequency_Hz=frequency,
                z_real=z_real,
                z_imag=z_imag,
                n_terms=n_terms,
                include_inductance=include_inductance,
                ridge_strength=ridge_strength,
            )
            result["n_terms"] = n_terms
            candidates.append(result)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("线性 KK measurement-model 拟合失败。")

    best = min(
        candidates,
        key=lambda item: item["bic"],
    )

    fitted_data = pd.DataFrame(
        {
            "frequency_Hz": frequency,
            "z_real_ohm": z_real,
            "z_imag_ohm": z_imag,
            "z_real_kk_ohm": best["z_real_fit"],
            "z_imag_kk_ohm": best["z_imag_fit"],
        }
    )

    screening_pass = bool(
        best["relative_rmse"]
        <= float(screening_threshold)
    )

    message = (
        "相对偏差低于当前软件筛查阈值。"
        if screening_pass
        else (
            "相对偏差高于当前软件筛查阈值；建议检查 AC amplitude、"
            "OCP/热平衡、非稳态行为、频率范围和异常点。"
        )
    )

    return KKCompatibilityResult(
        record_id=record.record_id,
        filename=record.filename,
        temperature_C=record.temperature_C,
        replicate_index=record.replicate_index,
        n_terms=int(best["n_terms"]),
        relative_rmse=float(best["relative_rmse"]),
        rmse_ohm=float(best["rmse_ohm"]),
        max_normalized_error=float(best["max_normalized_error"]),
        screening_threshold=float(screening_threshold),
        screening_pass=screening_pass,
        message=message,
        fitted_data=fitted_data,
    )


def run_kk_checks(
    records: list[TemperatureEISRecord],
    screening_threshold: float = 0.05,
) -> list[KKCompatibilityResult]:
    return [
        linear_kk_compatibility_check(
            record,
            screening_threshold=screening_threshold,
        )
        for record in records
    ]


def kk_results_table(
    results: list[KKCompatibilityResult],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "temperature_C": result.temperature_C,
                "replicate": result.replicate_index,
                "filename": result.filename,
                "n_terms": result.n_terms,
                "relative_rmse": result.relative_rmse,
                "rmse_ohm": result.rmse_ohm,
                "max_normalized_error": result.max_normalized_error,
                "screening_threshold": result.screening_threshold,
                "screening_pass": result.screening_pass,
                "message": result.message,
            }
            for result in results
        ]
    )


# ---------------------------------------------------------------------------
# Replicate consistency
# ---------------------------------------------------------------------------

def _interpolate_complex_log_frequency(
    df: pd.DataFrame,
    log_frequency_grid: np.ndarray,
) -> np.ndarray:
    frequency = df["frequency_Hz"].to_numpy(dtype=float)
    z_real = df["z_real_ohm"].to_numpy(dtype=float)
    z_imag = df["z_imag_ohm"].to_numpy(dtype=float)

    order = np.argsort(frequency)
    frequency = frequency[order]
    z_real = z_real[order]
    z_imag = z_imag[order]

    log_frequency = np.log10(frequency)

    real_interp = np.interp(
        log_frequency_grid,
        log_frequency,
        z_real,
    )

    imag_interp = np.interp(
        log_frequency_grid,
        log_frequency,
        z_imag,
    )

    return real_interp + 1j * imag_interp


def replicate_consistency_table(
    records: list[TemperatureEISRecord],
    grid_points: int = 120,
) -> pd.DataFrame:
    rows = []

    temperatures = sorted(
        {
            float(record.temperature_C)
            for record in records
        }
    )

    for temperature in temperatures:
        group = [
            record
            for record in records
            if np.isclose(
                record.temperature_C,
                temperature,
                rtol=1e-9,
                atol=1e-9,
            )
        ]

        if len(group) < 2:
            rows.append(
                {
                    "temperature_C": temperature,
                    "replicate_count": len(group),
                    "relative_complex_rmse": np.nan,
                    "message": "少于 2 条谱，无法自动评价重复性。",
                }
            )
            continue

        low = max(
            float(record.data["frequency_Hz"].min())
            for record in group
        )
        high = min(
            float(record.data["frequency_Hz"].max())
            for record in group
        )

        if low >= high:
            rows.append(
                {
                    "temperature_C": temperature,
                    "replicate_count": len(group),
                    "relative_complex_rmse": np.nan,
                    "message": "重复谱之间没有共同频率区间。",
                }
            )
            continue

        grid = np.linspace(
            np.log10(low),
            np.log10(high),
            int(grid_points),
        )

        curves = [
            _interpolate_complex_log_frequency(
                record.data,
                grid,
            )
            for record in group
        ]

        reference = curves[0]
        reference_scale = np.maximum(
            np.abs(reference),
            max(
                float(np.median(np.abs(reference))) * 1e-6,
                1e-12,
            ),
        )

        errors = []

        for curve in curves[1:]:
            normalized = (
                np.abs(curve - reference)
                / reference_scale
            )

            errors.append(
                float(
                    np.sqrt(
                        np.mean(
                            normalized ** 2
                        )
                    )
                )
            )

        rows.append(
            {
                "temperature_C": temperature,
                "replicate_count": len(group),
                "relative_complex_rmse": float(np.mean(errors)),
                "message": "以第 1 条谱为参考的复阻抗重复性指标。",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Equivalent circuit fitting
# ---------------------------------------------------------------------------

def _parallel_r_cpe(
    omega: np.ndarray,
    resistance_ohm: float,
    q: float,
    n: float,
) -> np.ndarray:
    jw = 1j * omega

    admittance = (
        1.0 / resistance_ohm
        + q * np.power(jw, n)
    )

    return 1.0 / admittance


def _warburg_semi_infinite(
    omega: np.ndarray,
    sigma: float,
) -> np.ndarray:
    jw = 1j * omega
    return sigma / np.sqrt(jw)


def circuit_impedance(
    frequency_Hz: np.ndarray,
    circuit: str,
    parameters: dict[str, float],
) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(frequency_Hz, dtype=float)

    rs = float(parameters["Rs_ohm"])
    impedance = np.full_like(
        omega,
        rs,
        dtype=complex,
    )

    if circuit == CIRCUIT_ONE_ARC:
        impedance = (
            impedance
            + _parallel_r_cpe(
                omega,
                float(parameters["R1_ohm"]),
                float(parameters["Q1_S_s_n"]),
                float(parameters["n1"]),
            )
        )

    elif circuit == CIRCUIT_TWO_ARC:
        impedance = (
            impedance
            + _parallel_r_cpe(
                omega,
                float(parameters["R1_ohm"]),
                float(parameters["Q1_S_s_n"]),
                float(parameters["n1"]),
            )
            + _parallel_r_cpe(
                omega,
                float(parameters["R2_ohm"]),
                float(parameters["Q2_S_s_n"]),
                float(parameters["n2"]),
            )
        )

    elif circuit == CIRCUIT_ONE_ARC_W:
        impedance = (
            impedance
            + _parallel_r_cpe(
                omega,
                float(parameters["R1_ohm"]),
                float(parameters["Q1_S_s_n"]),
                float(parameters["n1"]),
            )
            + _warburg_semi_infinite(
                omega,
                float(parameters["sigma_W"]),
            )
        )

    else:
        raise ValueError(f"不支持的等效电路：{circuit}")

    return impedance


def _estimate_rs(
    frequency: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
) -> float:
    order = np.argsort(frequency)[::-1]
    frequency = frequency[order]
    z_real = z_real[order]
    z_imag = z_imag[order]

    # Search for high-frequency Z'' sign crossing.
    for i in range(min(len(frequency) - 1, max(3, len(frequency) // 3))):
        y1 = z_imag[i]
        y2 = z_imag[i + 1]

        if y1 == 0:
            return max(float(z_real[i]), 1e-12)

        if y1 * y2 < 0:
            fraction = abs(y1) / (abs(y1) + abs(y2))
            x = (
                z_real[i]
                + fraction
                * (
                    z_real[i + 1]
                    - z_real[i]
                )
            )
            return max(float(x), 1e-12)

    high_count = min(
        max(3, len(frequency) // 8),
        12,
    )

    indices = np.argsort(
        np.abs(z_imag[:high_count])
    )

    return max(
        float(
            z_real[:high_count][indices[0]]
        ),
        1e-12,
    )


def _characteristic_frequency(
    frequency: np.ndarray,
    z_imag: np.ndarray,
) -> float:
    capacitive = -z_imag

    if np.nanmax(capacitive) <= 0:
        return float(np.sqrt(np.min(frequency) * np.max(frequency)))

    index = int(np.nanargmax(capacitive))
    return float(frequency[index])


def _q_from_peak(
    resistance: float,
    frequency_peak: float,
    n: float = 0.9,
) -> float:
    resistance = max(float(resistance), 1e-9)
    omega = 2.0 * np.pi * max(float(frequency_peak), 1e-12)

    q = 1.0 / (
        resistance
        * omega ** float(n)
    )

    return float(
        np.clip(
            q,
            1e-12,
            1e6,
        )
    )


def initial_circuit_parameters(
    df: pd.DataFrame,
    circuit: str,
) -> dict[str, float]:
    frequency = df["frequency_Hz"].to_numpy(dtype=float)
    z_real = df["z_real_ohm"].to_numpy(dtype=float)
    z_imag = df["z_imag_ohm"].to_numpy(dtype=float)

    rs = _estimate_rs(
        frequency,
        z_real,
        z_imag,
    )

    span = max(
        float(np.nanmax(z_real) - rs),
        float(np.ptp(z_real)) * 0.30,
        1e-6,
    )

    f_peak = _characteristic_frequency(
        frequency,
        z_imag,
    )

    if circuit == CIRCUIT_ONE_ARC:
        r1 = span
        n1 = 0.90
        q1 = _q_from_peak(r1, f_peak, n1)

        return {
            "Rs_ohm": rs,
            "R1_ohm": r1,
            "Q1_S_s_n": q1,
            "n1": n1,
        }

    if circuit == CIRCUIT_TWO_ARC:
        r1 = max(span * 0.45, 1e-6)
        r2 = max(span * 0.55, 1e-6)
        n1 = 0.90
        n2 = 0.85

        q1 = _q_from_peak(
            r1,
            min(
                max(f_peak * 5.0, np.min(frequency)),
                np.max(frequency),
            ),
            n1,
        )

        q2 = _q_from_peak(
            r2,
            min(
                max(f_peak / 5.0, np.min(frequency)),
                np.max(frequency),
            ),
            n2,
        )

        return {
            "Rs_ohm": rs,
            "R1_ohm": r1,
            "Q1_S_s_n": q1,
            "n1": n1,
            "R2_ohm": r2,
            "Q2_S_s_n": q2,
            "n2": n2,
        }

    if circuit == CIRCUIT_ONE_ARC_W:
        r1 = span
        n1 = 0.90
        q1 = _q_from_peak(r1, f_peak, n1)

        omega_low = (
            2.0
            * np.pi
            * max(
                float(np.min(frequency)),
                1e-12,
            )
        )

        low_imag = abs(
            float(
                z_imag[
                    np.argmin(frequency)
                ]
            )
        )

        sigma = max(
            low_imag
            * np.sqrt(
                2.0 * omega_low
            ),
            1e-9,
        )

        return {
            "Rs_ohm": rs,
            "R1_ohm": r1,
            "Q1_S_s_n": q1,
            "n1": n1,
            "sigma_W": sigma,
        }

    raise ValueError(f"不支持的等效电路：{circuit}")


def _parameter_order(circuit: str) -> list[str]:
    if circuit == CIRCUIT_ONE_ARC:
        return [
            "Rs_ohm",
            "R1_ohm",
            "Q1_S_s_n",
            "n1",
        ]

    if circuit == CIRCUIT_TWO_ARC:
        return [
            "Rs_ohm",
            "R1_ohm",
            "Q1_S_s_n",
            "n1",
            "R2_ohm",
            "Q2_S_s_n",
            "n2",
        ]

    if circuit == CIRCUIT_ONE_ARC_W:
        return [
            "Rs_ohm",
            "R1_ohm",
            "Q1_S_s_n",
            "n1",
            "sigma_W",
        ]

    raise ValueError(f"不支持的等效电路：{circuit}")


def _parameters_to_vector(
    circuit: str,
    parameters: dict[str, float],
) -> np.ndarray:
    vector = []

    for name in _parameter_order(circuit):
        value = float(parameters[name])

        if name.startswith("n"):
            vector.append(value)
        else:
            vector.append(np.log10(max(value, 1e-20)))

    return np.asarray(vector, dtype=float)


def _vector_to_parameters(
    circuit: str,
    vector: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for name, value in zip(
        _parameter_order(circuit),
        vector,
    ):
        if name.startswith("n"):
            result[name] = float(value)
        else:
            result[name] = float(10.0 ** value)

    return result


def _fit_bounds(
    circuit: str,
) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []

    for name in _parameter_order(circuit):
        if name.startswith("n"):
            lower.append(0.30)
            upper.append(1.00)
        else:
            lower.append(-14.0)
            upper.append(14.0)

    return (
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


def _prepare_fit_data(
    df: pd.DataFrame,
    exclude_inductive_points: bool,
) -> pd.DataFrame:
    fit_df = df[
        [
            "frequency_Hz",
            "z_real_ohm",
            "z_imag_ohm",
        ]
    ].copy()

    finite = (
        np.isfinite(fit_df["frequency_Hz"])
        & np.isfinite(fit_df["z_real_ohm"])
        & np.isfinite(fit_df["z_imag_ohm"])
        & (fit_df["frequency_Hz"] > 0)
    )

    fit_df = fit_df.loc[finite].copy()

    if exclude_inductive_points:
        capacitive = fit_df[
            fit_df["z_imag_ohm"] <= 0
        ].copy()

        if len(capacitive) >= 6:
            fit_df = capacitive

    fit_df = (
        fit_df.sort_values(
            "frequency_Hz",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(fit_df) < 6:
        raise ValueError("用于等效电路拟合的数据点不足 6 个。")

    return fit_df


def fit_eis_record(
    record: TemperatureEISRecord,
    circuit: str = CIRCUIT_TWO_ARC,
    weighting: str = "1/|Z|",
    exclude_inductive_points: bool = True,
) -> EISTemperatureFitResult:
    fit_df = _prepare_fit_data(
        record.data,
        exclude_inductive_points=exclude_inductive_points,
    )

    frequency = fit_df["frequency_Hz"].to_numpy(dtype=float)
    z_exp = (
        fit_df["z_real_ohm"].to_numpy(dtype=float)
        + 1j
        * fit_df["z_imag_ohm"].to_numpy(dtype=float)
    )

    initial = initial_circuit_parameters(
        fit_df,
        circuit,
    )

    x0 = _parameters_to_vector(
        circuit,
        initial,
    )

    lower, upper = _fit_bounds(circuit)

    if weighting == "1/|Z|":
        scale = np.maximum(
            np.abs(z_exp),
            max(
                float(np.nanmedian(np.abs(z_exp))) * 1e-6,
                1e-12,
            ),
        )
    elif weighting == "None":
        scale = np.ones_like(
            np.abs(z_exp),
            dtype=float,
        )
    else:
        raise ValueError("weighting 必须为 '1/|Z|' 或 'None'。")

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = _vector_to_parameters(
            circuit,
            vector,
        )

        z_fit = circuit_impedance(
            frequency,
            circuit,
            parameters,
        )

        real_residual = (
            z_fit.real
            - z_exp.real
        ) / scale

        imag_residual = (
            z_fit.imag
            - z_exp.imag
        ) / scale

        return np.concatenate(
            [
                real_residual,
                imag_residual,
            ]
        )

    optimization = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=50000,
    )

    parameters = _vector_to_parameters(
        circuit,
        optimization.x,
    )

    z_fit = circuit_impedance(
        frequency,
        circuit,
        parameters,
    )

    complex_residual = z_fit - z_exp

    rmse_ohm = float(
        np.sqrt(
            np.mean(
                np.abs(complex_residual) ** 2
            )
        )
    )

    relative_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.abs(complex_residual)
                    / np.maximum(
                        np.abs(z_exp),
                        1e-12,
                    )
                ) ** 2
            )
        )
    )

    fitted_data = pd.DataFrame(
        {
            "frequency_Hz": frequency,
            "z_real_exp_ohm": z_exp.real,
            "z_imag_exp_ohm": z_exp.imag,
            "z_real_fit_ohm": z_fit.real,
            "z_imag_fit_ohm": z_fit.imag,
            "z_modulus_fit_ohm": np.abs(z_fit),
            "phase_fit_deg": np.degrees(np.angle(z_fit)),
        }
    )

    return EISTemperatureFitResult(
        record_id=record.record_id,
        filename=record.filename,
        temperature_C=record.temperature_C,
        replicate_index=record.replicate_index,
        circuit=circuit,
        success=bool(optimization.success),
        message=str(optimization.message),
        parameters=parameters,
        initial_parameters=initial,
        rmse_ohm=rmse_ohm,
        relative_rmse=relative_rmse,
        n_points_used=len(fit_df),
        weighting=weighting,
        excluded_inductive_points=bool(exclude_inductive_points),
        fitted_data=fitted_data,
    )


def fit_temperature_eis_dataset(
    records: list[TemperatureEISRecord],
    circuit: str = CIRCUIT_TWO_ARC,
    weighting: str = "1/|Z|",
    exclude_inductive_points: bool = True,
) -> list[EISTemperatureFitResult]:
    if circuit not in SUPPORTED_TEMPERATURE_EIS_CIRCUITS:
        raise ValueError(f"不支持的等效电路：{circuit}")

    return [
        fit_eis_record(
            record,
            circuit=circuit,
            weighting=weighting,
            exclude_inductive_points=exclude_inductive_points,
        )
        for record in records
    ]


def fit_results_table(
    results: list[EISTemperatureFitResult],
) -> pd.DataFrame:
    rows = []

    for result in results:
        row = {
            "temperature_C": result.temperature_C,
            "replicate": result.replicate_index,
            "filename": result.filename,
            "circuit": result.circuit,
            "success": result.success,
            "rmse_ohm": result.rmse_ohm,
            "relative_rmse": result.relative_rmse,
            "n_points_used": result.n_points_used,
        }

        row.update(result.parameters)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_temperature_parameters(
    results: list[EISTemperatureFitResult],
) -> pd.DataFrame:
    table = fit_results_table(results)

    if table.empty:
        return table

    numeric_candidates = [
        column
        for column in table.columns
        if column
        not in {
            "temperature_C",
            "replicate",
            "filename",
            "circuit",
            "success",
        }
        and pd.api.types.is_numeric_dtype(table[column])
    ]

    rows = []

    for temperature, group in table.groupby("temperature_C"):
        row: dict[str, Any] = {
            "temperature_C": float(temperature),
            "n_replicates": int(len(group)),
        }

        for column in numeric_candidates:
            values = group[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]

            if len(finite) == 0:
                continue

            row[f"{column}_mean"] = float(np.mean(finite))
            row[f"{column}_sd"] = (
                float(np.std(finite, ddof=1))
                if len(finite) >= 2
                else np.nan
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("temperature_C")
        .reset_index(drop=True)
    )


def arrhenius_resistance_candidates(
    circuit: str,
) -> list[str]:
    if circuit == CIRCUIT_ONE_ARC:
        return ["R1_ohm", "Rs_ohm"]

    if circuit == CIRCUIT_TWO_ARC:
        return [
            "R1_ohm",
            "R2_ohm",
            "Rs_ohm",
        ]

    if circuit == CIRCUIT_ONE_ARC_W:
        return [
            "R1_ohm",
            "Rs_ohm",
        ]

    return []


# ---------------------------------------------------------------------------
# Arrhenius analysis
# ---------------------------------------------------------------------------

def arrhenius_from_resistance(
    fit_results: list[EISTemperatureFitResult],
    parameter_name: str,
    process_label: str = "",
) -> ArrheniusResult:
    rows = []

    for result in fit_results:
        if not result.success:
            continue

        value = result.parameters.get(parameter_name)

        if value is None:
            continue

        value = float(value)

        if not np.isfinite(value) or value <= 0:
            continue

        rows.append(
            {
                "temperature_C": float(result.temperature_C),
                "temperature_K": float(result.temperature_C + 273.15),
                "replicate": int(result.replicate_index),
                "resistance_ohm": value,
                "ln_inverse_resistance": float(np.log(1.0 / value)),
            }
        )

    raw = pd.DataFrame(rows)

    if raw.empty:
        raise ValueError(f"没有可用于 Arrhenius 分析的 {parameter_name} 数据。")

    grouped_rows = []

    for temperature, group in raw.groupby("temperature_C"):
        resistances = group["resistance_ohm"].to_numpy(dtype=float)
        mean_r = float(np.mean(resistances))
        sd_r = (
            float(np.std(resistances, ddof=1))
            if len(resistances) >= 2
            else np.nan
        )

        temperature_K = float(temperature + 273.15)
        x = 1000.0 / temperature_K
        y = float(np.log(1.0 / mean_r))

        if np.isfinite(sd_r) and mean_r > 0:
            # First-order uncertainty propagation:
            # y = -ln(R), sigma_y ≈ sigma_R / R.
            y_sd = float(sd_r / mean_r)
        else:
            y_sd = np.nan

        grouped_rows.append(
            {
                "temperature_C": float(temperature),
                "temperature_K": temperature_K,
                "x_1000_over_T_K_inv": x,
                "resistance_mean_ohm": mean_r,
                "resistance_sd_ohm": sd_r,
                "ln_inverse_resistance": y,
                "ln_inverse_resistance_sd": y_sd,
                "n_replicates": int(len(group)),
            }
        )

    table = (
        pd.DataFrame(grouped_rows)
        .sort_values("temperature_C")
        .reset_index(drop=True)
    )

    if len(table) < 3:
        raise ValueError("Arrhenius 线性回归至少建议 3 个不同温度点。")

    regression = linregress(
        table["x_1000_over_T_K_inv"].to_numpy(dtype=float),
        table["ln_inverse_resistance"].to_numpy(dtype=float),
    )

    # y = ln(1/R) = b - Ea/(R*T)
    # x = 1000/T, so slope = -Ea/(1000*R).
    # Therefore Ea [kJ/mol] = -slope * R [J/mol/K].
    activation_energy_kJ_mol = (
        -float(regression.slope)
        * R_GAS_J_MOL_K
    )

    pre_exponential_inverse_resistance = float(
        np.exp(regression.intercept)
    )

    table["linear_fit"] = (
        regression.intercept
        + regression.slope
        * table["x_1000_over_T_K_inv"]
    )

    return ArrheniusResult(
        parameter_name=parameter_name,
        process_label=process_label or parameter_name,
        n_temperatures=len(table),
        slope=float(regression.slope),
        intercept=float(regression.intercept),
        r2=float(regression.rvalue ** 2),
        activation_energy_kJ_mol=float(activation_energy_kJ_mol),
        pre_exponential_inverse_resistance=pre_exponential_inverse_resistance,
        table=table,
    )


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def combined_fitted_curve_table(
    results: list[EISTemperatureFitResult],
) -> pd.DataFrame:
    frames = []

    for result in results:
        frame = result.fitted_data.copy()
        frame.insert(0, "temperature_C", result.temperature_C)
        frame.insert(1, "replicate", result.replicate_index)
        frame.insert(2, "filename", result.filename)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )
