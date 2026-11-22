
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress, pearsonr, spearmanr

from core.eis_temperature_analysis import (
    TemperatureEISRecord,
    run_kk_checks,
    kk_results_table,
)


FINAL_CIRCUIT = "L + Rs + CPE"

CANDIDATE_CIRCUITS = (
    "L + Rs + CPE",
    "L + Rs + (Rpar || CPE)",
    "L + Rs + TLM(CPE)",
    "L + Rs + RQ1 + RQ2",
)


@dataclass
class FinalEISFitResult:
    record_id: str
    temperature_C: float
    replicate_index: int
    filename: str
    parameters: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    weighted_rmse: float
    complex_rmse_ohm: float
    boundary_hit: bool
    jacobian_condition_number: float
    fitted_data: pd.DataFrame


@dataclass
class EISModelSelectionResult:
    candidate_detail: pd.DataFrame
    candidate_summary: pd.DataFrame
    selected_circuit: str
    rationale: str


def _coth(z: np.ndarray) -> np.ndarray:
    return 1.0 / np.tanh(z)


def _z_cpe(
    omega: np.ndarray,
    q: float,
    n: float,
) -> np.ndarray:
    return 1.0 / (
        q
        * np.power(
            1j * omega,
            n,
        )
    )


def _z_rq(
    omega: np.ndarray,
    resistance: float,
    q: float,
    n: float,
) -> np.ndarray:
    return 1.0 / (
        1.0 / resistance
        + q
        * np.power(
            1j * omega,
            n,
        )
    )


def _z_tlm_cpe(
    omega: np.ndarray,
    pore_resistance: float,
    q: float,
    n: float,
) -> np.ndarray:
    """
    Finite-length blocking transmission-line analogue with a CPE
    as the distributed interfacial admittance.

    This is used only as a candidate model. It is not automatically
    interpreted as a literal pore geometry.
    """
    y = q * np.power(
        1j * omega,
        n,
    )

    x = np.sqrt(
        pore_resistance
        * y
    )

    return (
        np.sqrt(
            pore_resistance
            / y
        )
        * _coth(
            x
        )
    )


def _candidate_specs() -> dict[str, list[tuple[str, str, float, float]]]:
    return {
        "L + Rs + CPE": [
            ("Rs_Ohm", "log", 1e-4, 10.0),
            ("L_H", "log", 1e-9, 1e-3),
            ("Q_S_s_n", "log", 1e-8, 1e3),
            ("n", "lin", 0.2, 1.0),
        ],

        "L + Rs + (Rpar || CPE)": [
            ("Rs_Ohm", "log", 1e-4, 10.0),
            ("L_H", "log", 1e-9, 1e-3),
            ("Rpar_Ohm", "log", 1e-2, 1e8),
            ("Q_S_s_n", "log", 1e-8, 1e3),
            ("n", "lin", 0.2, 1.0),
        ],

        "L + Rs + TLM(CPE)": [
            ("Rs_Ohm", "log", 1e-4, 10.0),
            ("L_H", "log", 1e-9, 1e-3),
            ("Rpore_Ohm", "log", 1e-4, 1e6),
            ("Q_S_s_n", "log", 1e-8, 1e3),
            ("n", "lin", 0.2, 1.0),
        ],

        "L + Rs + RQ1 + RQ2": [
            ("Rs_Ohm", "log", 1e-4, 10.0),
            ("L_H", "log", 1e-9, 1e-3),
            ("R1_Ohm", "log", 1e-4, 1e7),
            ("Q1_S_s_n", "log", 1e-10, 1e3),
            ("n1", "lin", 0.2, 1.0),
            ("R2_Ohm", "log", 1e-4, 1e9),
            ("Q2_S_s_n", "log", 1e-10, 1e3),
            ("n2", "lin", 0.2, 1.0),
        ],
    }


def _impedance(
    circuit: str,
    frequency_Hz: np.ndarray,
    p: dict[str, float],
) -> np.ndarray:
    omega = (
        2.0
        * np.pi
        * np.asarray(
            frequency_Hz,
            dtype=float,
        )
    )

    z = (
        float(
            p["Rs_Ohm"]
        )
        + 1j
        * omega
        * float(
            p["L_H"]
        )
    )

    if circuit == "L + Rs + CPE":
        return (
            z
            + _z_cpe(
                omega,
                float(
                    p[
                        "Q_S_s_n"
                    ]
                ),
                float(
                    p[
                        "n"
                    ]
                ),
            )
        )

    if circuit == "L + Rs + (Rpar || CPE)":
        return (
            z
            + _z_rq(
                omega,
                float(
                    p[
                        "Rpar_Ohm"
                    ]
                ),
                float(
                    p[
                        "Q_S_s_n"
                    ]
                ),
                float(
                    p[
                        "n"
                    ]
                ),
            )
        )

    if circuit == "L + Rs + TLM(CPE)":
        return (
            z
            + _z_tlm_cpe(
                omega,
                float(
                    p[
                        "Rpore_Ohm"
                    ]
                ),
                float(
                    p[
                        "Q_S_s_n"
                    ]
                ),
                float(
                    p[
                        "n"
                    ]
                ),
            )
        )

    if circuit == "L + Rs + RQ1 + RQ2":
        return (
            z
            + _z_rq(
                omega,
                float(
                    p[
                        "R1_Ohm"
                    ]
                ),
                float(
                    p[
                        "Q1_S_s_n"
                    ]
                ),
                float(
                    p[
                        "n1"
                    ]
                ),
            )
            + _z_rq(
                omega,
                float(
                    p[
                        "R2_Ohm"
                    ]
                ),
                float(
                    p[
                        "Q2_S_s_n"
                    ]
                ),
                float(
                    p[
                        "n2"
                    ]
                ),
            )
        )

    raise ValueError(
        f"Unsupported circuit: {circuit}"
    )


def _theta_bounds(
    spec: list[tuple[str, str, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    low = []
    high = []

    for _, kind, lo, hi in spec:
        if kind == "log":
            low.append(
                np.log10(
                    lo
                )
            )

            high.append(
                np.log10(
                    hi
                )
            )

        else:
            low.append(
                lo
            )

            high.append(
                hi
            )

    return (
        np.asarray(
            low,
            dtype=float,
        ),
        np.asarray(
            high,
            dtype=float,
        ),
    )


def _pack(
    p: dict[str, float],
    spec: list[tuple[str, str, float, float]],
) -> np.ndarray:
    values = []

    for name, kind, lo, hi in spec:
        value = float(
            p[
                name
            ]
        )

        value = float(
            np.clip(
                value,
                lo,
                hi,
            )
        )

        if kind == "log":
            values.append(
                np.log10(
                    value
                )
            )

        else:
            values.append(
                value
            )

    return np.asarray(
        values,
        dtype=float,
    )


def _unpack(
    theta: np.ndarray,
    spec: list[tuple[str, str, float, float]],
) -> dict[str, float]:
    p: dict[str, float] = {}

    for value, (
        name,
        kind,
        _,
        _,
    ) in zip(
        theta,
        spec,
    ):
        if kind == "log":
            p[
                name
            ] = float(
                10.0
                ** value
            )

        else:
            p[
                name
            ] = float(
                value
            )

    return p


def _initial_guess(
    circuit: str,
    record: TemperatureEISRecord,
) -> dict[str, float]:
    df = record.data

    frequency = (
        df[
            "frequency_Hz"
        ]
        .to_numpy(
            dtype=float
        )
    )

    z_real = (
        df[
            "z_real_ohm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    z_imag = (
        df[
            "z_imag_ohm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    modulus = (
        np.sqrt(
            z_real ** 2
            + z_imag ** 2
        )
    )

    zero_index = int(
        np.argmin(
            np.abs(
                z_imag
            )
        )
    )

    rs = max(
        float(
            z_real[
                zero_index
            ]
        ),
        1e-4,
    )

    high_index = int(
        np.argmax(
            frequency
        )
    )

    l = max(
        float(
            z_imag[
                high_index
            ]
        )
        /
        (
            2.0
            * np.pi
            * float(
                frequency[
                    high_index
                ]
            )
        ),
        1e-9,
    )

    low_index = int(
        np.argmin(
            frequency
        )
    )

    low_phase = float(
        np.degrees(
            np.arctan2(
                z_imag[
                    low_index
                ],
                z_real[
                    low_index
                ],
            )
        )
    )

    n = float(
        np.clip(
            abs(
                low_phase
            )
            / 90.0,
            0.45,
            0.90,
        )
    )

    omega_low = (
        2.0
        * np.pi
        * float(
            frequency[
                low_index
            ]
        )
    )

    q = max(
        1.0
        /
        (
            max(
                float(
                    modulus[
                        low_index
                    ]
                ),
                1e-9,
            )
            * omega_low
            ** n
        ),
        1e-8,
    )

    if circuit == "L + Rs + CPE":
        return {
            "Rs_Ohm":
                rs,

            "L_H":
                l,

            "Q_S_s_n":
                q,

            "n":
                n,
        }

    if circuit == "L + Rs + (Rpar || CPE)":
        return {
            "Rs_Ohm":
                rs,

            "L_H":
                l,

            "Rpar_Ohm":
                max(
                    float(
                        modulus[
                            low_index
                        ]
                    )
                    * 3.0,
                    1000.0,
                ),

            "Q_S_s_n":
                q,

            "n":
                n,
        }

    if circuit == "L + Rs + TLM(CPE)":
        return {
            "Rs_Ohm":
                rs,

            "L_H":
                l,

            "Rpore_Ohm":
                10.0,

            "Q_S_s_n":
                q,

            "n":
                n,
        }

    if circuit == "L + Rs + RQ1 + RQ2":
        return {
            "Rs_Ohm":
                rs,

            "L_H":
                l,

            "R1_Ohm":
                1.0,

            "Q1_S_s_n":
                1e-3,

            "n1":
                0.8,

            "R2_Ohm":
                max(
                    float(
                        modulus[
                            low_index
                        ]
                    )
                    * 3.0,
                    1000.0,
                ),

            "Q2_S_s_n":
                q,

            "n2":
                n,
        }

    raise ValueError(
        circuit
    )


def _fit_candidate(
    record: TemperatureEISRecord,
    circuit: str,
    n_starts: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    spec = (
        _candidate_specs()[
            circuit
        ]
    )

    lower, upper = (
        _theta_bounds(
            spec
        )
    )

    x0 = _pack(
        _initial_guess(
            circuit,
            record,
        ),
        spec,
    )

    df = record.data

    frequency = (
        df[
            "frequency_Hz"
        ]
        .to_numpy(
            dtype=float
        )
    )

    z_exp = (
        df[
            "z_real_ohm"
        ]
        .to_numpy(
            dtype=float
        )
        + 1j
        * df[
            "z_imag_ohm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    scale = np.maximum(
        np.abs(
            z_exp
        ),
        1e-12,
    )

    def residual(
        theta: np.ndarray,
    ) -> np.ndarray:
        p = _unpack(
            theta,
            spec,
        )

        try:
            z_fit = _impedance(
                circuit,
                frequency,
                p,
            )

            r = (
                z_fit
                - z_exp
            ) / scale

            out = np.concatenate(
                [
                    r.real,
                    r.imag,
                ]
            )

            if not np.all(
                np.isfinite(
                    out
                )
            ):
                raise FloatingPointError

            return out

        except Exception:
            return np.full(
                2
                * len(
                    z_exp
                ),
                1e6,
                dtype=float,
            )

    rng = np.random.default_rng(
        int(
            random_state
        )
    )

    starts = [
        x0
    ]

    for _ in range(
        max(
            1,
            int(
                n_starts
            )
        )
        - 1
    ):
        start = (
            lower
            + (
                upper
                - lower
            )
            * rng.random(
                len(
                    lower
                )
            )
        )

        # Keep the instrumental Rs/L terms near the data-derived values.
        start[
            0
        ] = np.clip(
            x0[
                0
            ]
            + rng.normal(
                0.0,
                0.25,
            ),
            lower[
                0
            ],
            upper[
                0
            ],
        )

        start[
            1
        ] = np.clip(
            x0[
                1
            ]
            + rng.normal(
                0.0,
                0.35,
            ),
            lower[
                1
            ],
            upper[
                1
            ],
        )

        starts.append(
            start
        )

    best = None

    for start in starts:
        optimization = (
            least_squares(
                residual,
                start,
                bounds=(
                    lower,
                    upper,
                ),
                method=
                    "trf",
                x_scale=
                    "jac",
                max_nfev=
                    6000,
                xtol=
                    1e-10,
                ftol=
                    1e-10,
                gtol=
                    1e-10,
            )
        )

        rss = float(
            np.sum(
                optimization.fun
                ** 2
            )
        )

        if (
            best is None
            or rss
            < best[
                0
            ]
        ):
            best = (
                rss,
                optimization,
            )

    assert best is not None

    rss, optimization = best

    p = _unpack(
        optimization.x,
        spec,
    )

    z_fit = _impedance(
        circuit,
        frequency,
        p,
    )

    relative_residual = (
        z_fit
        - z_exp
    ) / scale

    weighted_rmse = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        relative_residual.real,
                        relative_residual.imag,
                    ]
                )
                ** 2
            )
        )
    )

    complex_rmse = float(
        np.sqrt(
            np.mean(
                np.abs(
                    z_fit
                    - z_exp
                )
                ** 2
            )
        )
    )

    n_observations = (
        2
        * len(
            z_exp
        )
    )

    n_parameters = len(
        spec
    )

    bic = float(
        n_observations
        * np.log(
            max(
                rss
                / n_observations,
                1e-30,
            )
        )
        + n_parameters
        * np.log(
            n_observations
        )
    )

    fraction = (
        (
            optimization.x
            - lower
        )
        /
        (
            upper
            - lower
        )
    )

    boundary_hit = bool(
        np.any(
            (
                fraction
                < 0.01
            )
            |
            (
                fraction
                > 0.99
            )
        )
    )

    try:
        condition_number = float(
            np.linalg.cond(
                optimization.jac.T
                @ optimization.jac
            )
        )

    except Exception:
        condition_number = np.inf

    return {
        "parameters":
            p,

        "weighted_rmse":
            weighted_rmse,

        "complex_rmse_ohm":
            complex_rmse,

        "bic":
            bic,

        "boundary_hit":
            boundary_hit,

        "jacobian_condition_number":
            condition_number,

        "optimization":
            optimization,

        "fitted_data":
            pd.DataFrame(
                {
                    "frequency_Hz":
                        frequency,

                    "z_real_exp_ohm":
                        z_exp.real,

                    "z_imag_exp_ohm":
                        z_exp.imag,

                    "z_real_fit_ohm":
                        z_fit.real,

                    "z_imag_fit_ohm":
                        z_fit.imag,

                    "relative_residual_real":
                        relative_residual.real,

                    "relative_residual_imag":
                        relative_residual.imag,

                    "relative_residual_magnitude":
                        np.abs(
                            relative_residual
                        ),
                }
            ),
    }


def screen_candidate_models(
    records: list[TemperatureEISRecord],
    n_starts: int = 10,
) -> EISModelSelectionResult:
    rows: list[dict[str, Any]] = []

    for record in records:
        for index, circuit in enumerate(
            CANDIDATE_CIRCUITS
        ):
            fit = _fit_candidate(
                record,
                circuit,
                n_starts=
                    n_starts,
                random_state=
                    int(
                        round(
                            record.temperature_C
                            * 100
                        )
                    )
                    + index
                    * 97,
            )

            row = {
                "temperature_C":
                    float(
                        record.temperature_C
                    ),

                "replicate":
                    int(
                        record.replicate_index
                    ),

                "filename":
                    record.filename,

                "circuit":
                    circuit,

                "weighted_RMSE_pct":
                    float(
                        fit[
                            "weighted_rmse"
                        ]
                        * 100.0
                    ),

                "complex_RMSE_Ohm":
                    float(
                        fit[
                            "complex_rmse_ohm"
                        ]
                    ),

                "BIC":
                    float(
                        fit[
                            "bic"
                        ]
                    ),

                "boundary_hit":
                    bool(
                        fit[
                            "boundary_hit"
                        ]
                    ),

                "jacobian_condition_number":
                    float(
                        fit[
                            "jacobian_condition_number"
                        ]
                    ),
            }

            row.update(
                fit[
                    "parameters"
                ]
            )

            rows.append(
                row
            )

    detail = pd.DataFrame(
        rows
    )

    summary = (
        detail.groupby(
            "circuit",
            as_index=False,
        )
        .agg(
            n_spectra=(
                "temperature_C",
                "count",
            ),

            mean_weighted_RMSE_pct=(
                "weighted_RMSE_pct",
                "mean",
            ),

            median_weighted_RMSE_pct=(
                "weighted_RMSE_pct",
                "median",
            ),

            mean_BIC=(
                "BIC",
                "mean",
            ),

            boundary_fraction=(
                "boundary_hit",
                "mean",
            ),

            median_jacobian_condition_number=(
                "jacobian_condition_number",
                "median",
            ),
        )
        .sort_values(
            "mean_weighted_RMSE_pct"
        )
        .reset_index(
            drop=True
        )
    )

    rationale = (
        "复杂双 RQ 模型的残差最低，但在部分温度出现参数触边，"
        "并且条件数/特征频率表明多个过程难以唯一辨识；"
        "TLM(CPE) 与 Rpar||CPE 也在部分温度退化到边界。"
        "因此最终定量温度趋势采用最简、所有温度均可辨识的 L + Rs + CPE；"
        "复杂模型只作为“谱中存在多时间尺度”的描述性证据，不把 R1/R2 强行命名为 Rct。"
    )

    return EISModelSelectionResult(
        candidate_detail=
            detail,

        candidate_summary=
            summary,

        selected_circuit=
            FINAL_CIRCUIT,

        rationale=
            rationale,
    )


def fit_final_circuit(
    record: TemperatureEISRecord,
) -> FinalEISFitResult:
    fit = _fit_candidate(
        record,
        FINAL_CIRCUIT,
        n_starts=
            10,
        random_state=
            int(
                round(
                    record.temperature_C
                    * 1000
                )
            ),
    )

    optimization = (
        fit[
            "optimization"
        ]
    )

    spec = (
        _candidate_specs()[
            FINAL_CIRCUIT
        ]
    )

    lower, upper = (
        _theta_bounds(
            spec
        )
    )

    dof = max(
        len(
            optimization.fun
        )
        - len(
            optimization.x
        ),
        1,
    )

    sigma2 = float(
        np.sum(
            optimization.fun
            ** 2
        )
        / dof
    )

    jtj = (
        optimization.jac.T
        @ optimization.jac
    )

    covariance = (
        np.linalg.pinv(
            jtj
        )
        * sigma2
    )

    standard_error = np.sqrt(
        np.maximum(
            np.diag(
                covariance
            ),
            0.0,
        )
    )

    confidence_intervals: dict[
        str,
        tuple[
            float,
            float,
        ],
    ] = {}

    for index, (
        name,
        kind,
        _,
        _,
    ) in enumerate(
        spec
    ):
        if kind == "log":
            low = float(
                10.0
                ** (
                    optimization.x[
                        index
                    ]
                    - 1.96
                    * standard_error[
                        index
                    ]
                )
            )

            high = float(
                10.0
                ** (
                    optimization.x[
                        index
                    ]
                    + 1.96
                    * standard_error[
                        index
                    ]
                )
            )

        else:
            low = float(
                optimization.x[
                    index
                ]
                - 1.96
                * standard_error[
                    index
                ]
            )

            high = float(
                optimization.x[
                    index
                ]
                + 1.96
                * standard_error[
                    index
                ]
            )

        confidence_intervals[
            name
        ] = (
            low,
            high,
        )

    return FinalEISFitResult(
        record_id=
            record.record_id,

        temperature_C=
            float(
                record.temperature_C
            ),

        replicate_index=
            int(
                record.replicate_index
            ),

        filename=
            record.filename,

        parameters=
            fit[
                "parameters"
            ],

        confidence_intervals=
            confidence_intervals,

        weighted_rmse=
            float(
                fit[
                    "weighted_rmse"
                ]
            ),

        complex_rmse_ohm=
            float(
                fit[
                    "complex_rmse_ohm"
                ]
            ),

        boundary_hit=
            bool(
                fit[
                    "boundary_hit"
                ]
            ),

        jacobian_condition_number=
            float(
                fit[
                    "jacobian_condition_number"
                ]
            ),

        fitted_data=
            fit[
                "fitted_data"
            ],
    )


def fit_final_dataset(
    records: list[TemperatureEISRecord],
) -> list[FinalEISFitResult]:
    return [
        fit_final_circuit(
            record
        )
        for record in records
    ]


def final_fit_table(
    results: list[FinalEISFitResult],
) -> pd.DataFrame:
    rows = []

    for result in results:
        p = result.parameters
        ci = (
            result.confidence_intervals
        )

        q = float(
            p[
                "Q_S_s_n"
            ]
        )

        n = float(
            p[
                "n"
            ]
        )

        row = {
            "temperature_C":
                result.temperature_C,

            "replicate":
                result.replicate_index,

            "filename":
                result.filename,

            "Rs_Ohm":
                float(
                    p[
                        "Rs_Ohm"
                    ]
                ),

            "Rs_CI95_low":
                float(
                    ci[
                        "Rs_Ohm"
                    ][
                        0
                    ]
                ),

            "Rs_CI95_high":
                float(
                    ci[
                        "Rs_Ohm"
                    ][
                        1
                    ]
                ),

            "L_uH":
                float(
                    p[
                        "L_H"
                    ]
                    * 1e6
                ),

            "Q_S_s_n":
                q,

            "Q_CI95_low":
                float(
                    ci[
                        "Q_S_s_n"
                    ][
                        0
                    ]
                ),

            "Q_CI95_high":
                float(
                    ci[
                        "Q_S_s_n"
                    ][
                        1
                    ]
                ),

            "n":
                n,

            "n_CI95_low":
                float(
                    ci[
                        "n"
                    ][
                        0
                    ]
                ),

            "n_CI95_high":
                float(
                    ci[
                        "n"
                    ][
                        1
                    ]
                ),

            # Frequency-referenced, phenomenological effective CPE magnitude.
            "Ceff_mag_1Hz_F":
                float(
                    q
                    * (
                        2.0
                        * np.pi
                        * 1.0
                    )
                    ** (
                        n
                        - 1.0
                    )
                ),

            "Ceff_mag_0p1Hz_F":
                float(
                    q
                    * (
                        2.0
                        * np.pi
                        * 0.1
                    )
                    ** (
                        n
                        - 1.0
                    )
                ),

            "weighted_RMSE_pct":
                float(
                    result.weighted_rmse
                    * 100.0
                ),

            "complex_RMSE_Ohm":
                float(
                    result.complex_rmse_ohm
                ),

            "boundary_hit":
                bool(
                    result.boundary_hit
                ),

            "jacobian_condition_number":
                float(
                    result.jacobian_condition_number
                ),
        }

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "temperature_C",
                "replicate",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def combined_final_fit_curves(
    results: list[FinalEISFitResult],
) -> pd.DataFrame:
    frames = []

    for result in results:
        frame = (
            result.fitted_data
            .copy()
        )

        frame.insert(
            0,
            "temperature_C",
            result.temperature_C,
        )

        frame.insert(
            1,
            "replicate",
            result.replicate_index,
        )

        frame.insert(
            2,
            "filename",
            result.filename,
        )

        frames.append(
            frame
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def metadata_table(
    records: list[TemperatureEISRecord],
) -> pd.DataFrame:
    rows = []

    for record in records:
        metadata = record.metadata

        def get_float(
            key: str,
        ) -> float:
            try:
                return float(
                    metadata.get(
                        key,
                        np.nan,
                    )
                )
            except Exception:
                return np.nan

        rows.append(
            {
                "temperature_C":
                    float(
                        record.temperature_C
                    ),

                "replicate":
                    int(
                        record.replicate_index
                    ),

                "filename":
                    record.filename,

                "Init_E_V":
                    get_float(
                        "Init E (V)"
                    ),

                "Amplitude_V":
                    get_float(
                        "Amplitude (V)"
                    ),

                "Quiet_Time_s":
                    get_float(
                        "Quiet Time (sec)"
                    ),

                "n_points":
                    int(
                        len(
                            record.data
                        )
                    ),

                "f_max_Hz":
                    float(
                        record.data[
                            "frequency_Hz"
                        ]
                        .max()
                    ),

                "f_min_Hz":
                    float(
                        record.data[
                            "frequency_Hz"
                        ]
                        .min()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "temperature_C",
                "replicate",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def direct_high_frequency_intercept(
    records: list[TemperatureEISRecord],
) -> pd.DataFrame:
    rows = []

    for record in records:
        df = record.data

        frequency = (
            df[
                "frequency_Hz"
            ]
            .to_numpy(
                dtype=float
            )
        )

        z_real = (
            df[
                "z_real_ohm"
            ]
            .to_numpy(
                dtype=float
            )
        )

        z_imag = (
            df[
                "z_imag_ohm"
            ]
            .to_numpy(
                dtype=float
            )
        )

        crossing = None

        for index in range(
            len(
                z_imag
            )
            - 1
        ):
            y1 = float(
                z_imag[
                    index
                ]
            )

            y2 = float(
                z_imag[
                    index
                    + 1
                ]
            )

            if (
                y1 == 0.0
                or y1
                * y2
                < 0.0
            ):
                fraction = (
                    (
                        0.0
                        - y1
                    )
                    /
                    (
                        y2
                        - y1
                    )
                )

                log_frequency = (
                    np.log10(
                        frequency[
                            index
                        ]
                    )
                    + fraction
                    * (
                        np.log10(
                            frequency[
                                index
                                + 1
                            ]
                        )
                        - np.log10(
                            frequency[
                                index
                            ]
                        )
                    )
                )

                z_cross = (
                    z_real[
                        index
                    ]
                    + fraction
                    * (
                        z_real[
                            index
                            + 1
                        ]
                        - z_real[
                            index
                        ]
                    )
                )

                crossing = (
                    float(
                        10.0
                        ** log_frequency
                    ),
                    float(
                        z_cross
                    ),
                )

                break

        rows.append(
            {
                "temperature_C":
                    float(
                        record.temperature_C
                    ),

                "replicate":
                    int(
                        record.replicate_index
                    ),

                "zero_imag_frequency_Hz":
                    (
                        crossing[
                            0
                        ]
                        if crossing
                        is not None
                        else np.nan
                    ),

                "zero_imag_Zreal_Ohm":
                    (
                        crossing[
                            1
                        ]
                        if crossing
                        is not None
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def model_free_relaxation_descriptors(
    records: list[TemperatureEISRecord],
) -> pd.DataFrame:
    rows = []

    for record in records:
        df = record.data

        frequency = (
            df[
                "frequency_Hz"
            ]
            .to_numpy(
                dtype=float
            )
        )

        omega = (
            2.0
            * np.pi
            * frequency
        )

        z_real = (
            df[
                "z_real_ohm"
            ]
            .to_numpy(
                dtype=float
            )
        )

        z_imag = (
            df[
                "z_imag_ohm"
            ]
            .to_numpy(
                dtype=float
            )
        )

        z_square = (
            z_real ** 2
            + z_imag ** 2
        )

        c_real = (
            -z_imag
            /
            (
                omega
                * z_square
            )
        )

        c_imag = (
            z_real
            /
            (
                omega
                * z_square
            )
        )

        capacitive = (
            z_imag
            < 0
        )

        if np.any(
            capacitive
        ):
            frequencies = (
                frequency[
                    capacitive
                ]
            )

            c_imag_cap = (
                c_imag[
                    capacitive
                ]
            )

            peak_index = int(
                np.argmax(
                    c_imag_cap
                )
            )

            peak_frequency = float(
                frequencies[
                    peak_index
                ]
            )

            peak_at_lower_boundary = bool(
                np.isclose(
                    peak_frequency,
                    float(
                        np.min(
                            frequency
                        )
                    ),
                    rtol=
                        1e-6,
                    atol=
                        0.0,
                )
            )

        else:
            peak_frequency = np.nan
            peak_at_lower_boundary = False

        rows.append(
            {
                "temperature_C":
                    float(
                        record.temperature_C
                    ),

                "replicate":
                    int(
                        record.replicate_index
                    ),

                "C_imag_peak_frequency_Hz":
                    peak_frequency,

                "C_imag_peak_at_lower_frequency_boundary":
                    peak_at_lower_boundary,

                "low_frequency_phase_deg":
                    float(
                        df.iloc[
                            -1
                        ][
                            "phase_deg"
                        ]
                    ),

                "C_real_at_lowest_frequency_F":
                    float(
                        c_real[
                            -1
                        ]
                    ),

                "C_imag_at_lowest_frequency_F":
                    float(
                        c_imag[
                            -1
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def arrhenius_rs_diagnostic(
    fit_table: pd.DataFrame,
) -> dict[str, Any]:
    grouped = (
        fit_table.groupby(
            "temperature_C",
            as_index=False,
        )
        .agg(
            Rs_Ohm=(
                "Rs_Ohm",
                "mean",
            )
        )
        .sort_values(
            "temperature_C"
        )
    )

    if len(
        grouped
    ) < 3:
        raise ValueError(
            "Arrhenius diagnostic requires at least 3 temperatures."
        )

    temperature_K = (
        grouped[
            "temperature_C"
        ]
        .to_numpy(
            dtype=float
        )
        + 273.15
    )

    x = (
        1.0
        / temperature_K
    )

    y = np.log(
        1.0
        /
        grouped[
            "Rs_Ohm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    regression = linregress(
        x,
        y,
    )

    gas_constant = 8.31446261815324

    apparent_ea_kJ_mol = float(
        -regression.slope
        * gas_constant
        / 1000.0
    )

    table = grouped.copy()

    table[
        "temperature_K"
    ] = temperature_K

    table[
        "inverse_T_K_inv"
    ] = x

    table[
        "ln_inverse_Rs"
    ] = y

    table[
        "linear_fit"
    ] = (
        regression.intercept
        + regression.slope
        * x
    )

    return {
        "apparent_Ea_kJ_mol":
            apparent_ea_kJ_mol,

        "r2":
            float(
                regression.rvalue
                ** 2
            ),

        "p_value":
            float(
                regression.pvalue
            ),

        "slope":
            float(
                regression.slope
            ),

        "intercept":
            float(
                regression.intercept
            ),

        "table":
            table,
    }


def gcd_temperature_summary(
    gcd_records: list[Any],
) -> pd.DataFrame:
    rows = []

    for record in gcd_records:
        # Incomplete GCD metadata must never crash independent EIS fitting.
        # No missing experimental current or duration is imputed.
        try:
            values = (
                getattr(record, "temperature_C", None),
                getattr(record, "current_value", None),
                getattr(record, "charge_duration_s", None),
                getattr(record, "discharge_duration_s", None),
            )
            if any(value is None for value in values):
                continue
            temperature, current, charge_s, discharge_s = map(float, values)
            if not all(np.isfinite(v) for v in (temperature, current, charge_s, discharge_s)):
                continue
            if current <= 0 or charge_s <= 0 or discharge_s <= 0:
                continue
        except (TypeError, ValueError, OverflowError):
            continue

        rows.append(
            {
                "temperature_C": temperature,
                "current_density_A_g": current,
                "charge_q_C_g": current * charge_s,
                "discharge_q_C_g": current * discharge_s,
                "charge_duration_s": charge_s,
                "discharge_duration_s": discharge_s,
            }
        )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(
        rows
    )

    return (
        table.groupby(
            "temperature_C",
            as_index=False,
        )
        .agg(
            mean_charge_q_C_g=(
                "charge_q_C_g",
                "mean",
            ),

            mean_discharge_q_C_g=(
                "discharge_q_C_g",
                "mean",
            ),

            mean_charge_duration_s=(
                "charge_duration_s",
                "mean",
            ),

            mean_discharge_duration_s=(
                "discharge_duration_s",
                "mean",
            ),
        )
        .sort_values(
            "temperature_C"
        )
        .reset_index(
            drop=True
        )
    )


def link_eis_to_gcd(
    fit_table: pd.DataFrame,
    gcd_summary: pd.DataFrame,
) -> dict[str, Any]:
    if gcd_summary.empty:
        raise ValueError(
            "No GCD summary is available."
        )

    eis = (
        fit_table.groupby(
            "temperature_C",
            as_index=False,
        )
        .agg(
            Rs_Ohm=(
                "Rs_Ohm",
                "mean",
            ),

            Q_S_s_n=(
                "Q_S_s_n",
                "mean",
            ),

            n=(
                "n",
                "mean",
            ),
        )
    )

    merged = (
        eis.merge(
            gcd_summary,
            on=
                "temperature_C",
            how=
                "inner",
        )
        .sort_values(
            "temperature_C"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        merged
    ) < 3:
        raise ValueError(
            "At least 3 common temperatures are required."
        )

    x = (
        merged[
            "Rs_Ohm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        merged[
            "mean_discharge_q_C_g"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pearson = pearsonr(
        x,
        y,
    )

    spearman = spearmanr(
        x,
        y,
    )

    return {
        "table":
            merged,

        "pearson_r":
            float(
                pearson.statistic
            ),

        "pearson_p":
            float(
                pearson.pvalue
            ),

        "spearman_rho":
            float(
                spearman.statistic
            ),

        "spearman_p":
            float(
                spearman.pvalue
            ),
    }


def run_final_kk_screen(
    records: list[TemperatureEISRecord],
) -> pd.DataFrame:
    return kk_results_table(
        run_kk_checks(
            records,
            screening_threshold=
                0.05,
        )
    )
