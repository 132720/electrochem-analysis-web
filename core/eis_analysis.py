import numpy as np
import pandas as pd

from scipy.optimize import least_squares


# =========================================================
# Circuit names
# =========================================================

CIRCUIT_RC = "Rs + (Rct || Cdl)"

CIRCUIT_CPE = "Rs + (Rct || CPE)"

CIRCUIT_CPE_W = "Rs + (Rct || CPE) + W"


SUPPORTED_CIRCUITS = [
    CIRCUIT_CPE,
    CIRCUIT_RC,
    CIRCUIT_CPE_W,
]


# =========================================================
# Basic impedance elements
# =========================================================

def _z_capacitor(
    omega,
    capacitance_F,
):
    """
    Ideal capacitor:

        Zc = 1 / (j*w*C)
    """

    return (
        1.0
        /
        (
            1j
            * omega
            * capacitance_F
        )
    )


def _z_cpe(
    omega,
    Q,
    n,
):
    """
    Constant phase element:

        Z_CPE = 1 / [Q * (j*w)^n]

    Q unit:
        S * s^n
    """

    return (
        1.0
        /
        (
            Q
            * (
                1j
                * omega
            ) ** n
        )
    )


def _parallel(
    z1,
    z2,
):
    """
    Parallel impedance.
    """

    return (
        1.0
        /
        (
            1.0 / z1
            +
            1.0 / z2
        )
    )


def _z_warburg(
    omega,
    sigma,
):
    """
    Semi-infinite Warburg:

        Zw = sigma / sqrt(j*w)
    """

    return (
        sigma
        /
        np.sqrt(
            1j
            * omega
        )
    )


# =========================================================
# Full equivalent circuits
# =========================================================

def calculate_circuit_impedance(
    frequency_Hz,
    parameters,
    circuit,
):
    """
    Calculate complex impedance from
    equivalent-circuit parameters.
    """

    frequency_Hz = np.asarray(
        frequency_Hz,
        dtype=float,
    )

    omega = (
        2.0
        * np.pi
        * frequency_Hz
    )

    Rs = parameters[
        "Rs"
    ]

    Rct = parameters[
        "Rct"
    ]

    # -----------------------------------------------------
    # Rs + (Rct || Cdl)
    # -----------------------------------------------------

    if circuit == CIRCUIT_RC:

        z_cap = _z_capacitor(
            omega,
            parameters[
                "Cdl"
            ],
        )

        z_parallel = _parallel(
            Rct,
            z_cap,
        )

        return (
            Rs
            + z_parallel
        )

    # -----------------------------------------------------
    # Rs + (Rct || CPE)
    # -----------------------------------------------------

    if circuit == CIRCUIT_CPE:

        z_cpe = _z_cpe(
            omega,
            parameters[
                "Q"
            ],
            parameters[
                "n"
            ],
        )

        z_parallel = _parallel(
            Rct,
            z_cpe,
        )

        return (
            Rs
            + z_parallel
        )

    # -----------------------------------------------------
    # Rs + (Rct || CPE) + W
    # -----------------------------------------------------

    if circuit == CIRCUIT_CPE_W:

        z_cpe = _z_cpe(
            omega,
            parameters[
                "Q"
            ],
            parameters[
                "n"
            ],
        )

        z_parallel = _parallel(
            Rct,
            z_cpe,
        )

        z_w = _z_warburg(
            omega,
            parameters[
                "sigma"
            ],
        )

        return (
            Rs
            + z_parallel
            + z_w
        )

    raise ValueError(
        f"Unsupported circuit: {circuit}"
    )


# =========================================================
# Rs estimation
# =========================================================

def estimate_rs_intercept(
    df,
):
    """
    Estimate high-frequency Rs.

    If Z'' crosses zero, interpolate the
    real impedance at the crossing.

    Otherwise use the point with the smallest
    |Z''| in the high-frequency region.
    """

    frequency = df[
        "frequency_Hz"
    ].to_numpy(
        dtype=float
    )

    z_real = df[
        "z_real_ohm"
    ].to_numpy(
        dtype=float
    )

    z_imag = df[
        "z_imag_ohm"
    ].to_numpy(
        dtype=float
    )

    order = np.argsort(
        frequency
    )[::-1]

    frequency = frequency[
        order
    ]

    z_real = z_real[
        order
    ]

    z_imag = z_imag[
        order
    ]

    # -----------------------------------------------------
    # Search for high-frequency Z'' zero crossing
    # -----------------------------------------------------

    for i in range(
        len(
            frequency
        ) - 1
    ):

        if z_imag[i] == 0:

            return float(
                z_real[i]
            )

        if (
            z_imag[i]
            * z_imag[
                i + 1
            ]
            < 0
        ):

            fraction = (
                abs(
                    z_imag[i]
                )
                /
                (
                    abs(
                        z_imag[i]
                    )
                    +
                    abs(
                        z_imag[
                            i + 1
                        ]
                    )
                )
            )

            rs = (
                z_real[i]
                +
                fraction
                * (
                    z_real[
                        i + 1
                    ]
                    -
                    z_real[i]
                )
            )

            return float(
                rs
            )

    # -----------------------------------------------------
    # No crossing detected
    # -----------------------------------------------------

    high_frequency_count = max(
        3,
        min(
            10,
            len(
                frequency
            ) // 3,
        ),
    )

    candidate_indices = np.arange(
        high_frequency_count
    )

    best = candidate_indices[
        np.argmin(
            np.abs(
                z_imag[
                    candidate_indices
                ]
            )
        )
    ]

    return float(
        z_real[
            best
        ]
    )


# =========================================================
# Initial Rct estimate
# =========================================================

def estimate_rct_initial(
    df,
    rs_estimate=None,
):
    """
    Rough Rct estimate used ONLY as the
    nonlinear-fitting initial value.

    This is not treated as final fitted Rct.
    """

    if rs_estimate is None:

        rs_estimate = (
            estimate_rs_intercept(
                df
            )
        )

    z_real = df[
        "z_real_ohm"
    ].to_numpy(
        dtype=float
    )

    spread = float(
        np.ptp(
            z_real
        )
    )

    estimate = max(
        float(
            np.max(
                z_real
            )
        )
        -
        rs_estimate,

        spread
        * 0.30,

        1e-6,
    )

    return float(
        estimate
    )


# =========================================================
# Prepare fitting data
# =========================================================

def prepare_eis_fit_data(
    df,
    exclude_inductive=True,
    frequency_min_Hz=None,
    frequency_max_Hz=None,
):
    """
    Prepare selected EIS data for fitting.

    Raw input df is NEVER modified.
    """

    fit_df = (
        df.copy()
    )

    # -----------------------------------------------------
    # Remove invalid points
    # -----------------------------------------------------

    fit_df = fit_df[
        np.isfinite(
            fit_df[
                "frequency_Hz"
            ]
        )
        &
        np.isfinite(
            fit_df[
                "z_real_ohm"
            ]
        )
        &
        np.isfinite(
            fit_df[
                "z_imag_ohm"
            ]
        )
        &
        (
            fit_df[
                "frequency_Hz"
            ]
            > 0
        )
    ]

    # -----------------------------------------------------
    # Frequency range
    # -----------------------------------------------------

    if (
        frequency_min_Hz
        is not None
    ):

        fit_df = fit_df[
            fit_df[
                "frequency_Hz"
            ]
            >= float(
                frequency_min_Hz
            )
        ]

    if (
        frequency_max_Hz
        is not None
    ):

        fit_df = fit_df[
            fit_df[
                "frequency_Hz"
            ]
            <= float(
                frequency_max_Hz
            )
        ]

    # -----------------------------------------------------
    # Exclude inductive points
    #
    # Raw CHI sign is retained:
    # capacitive region usually Z'' < 0.
    # -----------------------------------------------------

    inductive_exclusion_applied = False

    if exclude_inductive:

        capacitive_df = fit_df[
            fit_df[
                "z_imag_ohm"
            ]
            <= 0
        ]

        # Do not leave too few points
        if len(
            capacitive_df
        ) >= 6:

            fit_df = (
                capacitive_df
            )

            inductive_exclusion_applied = True

    fit_df = (
        fit_df
        .sort_values(
            "frequency_Hz",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        fit_df
    ) < 6:

        raise ValueError(
            "Too few EIS points remain for fitting."
        )

    return (
        fit_df,
        inductive_exclusion_applied,
    )


# =========================================================
# Initial guesses
# =========================================================

def estimate_initial_parameters(
    df,
    circuit,
):
    """
    Automatically generate starting parameters
    for least-squares fitting.
    """

    frequency = df[
        "frequency_Hz"
    ].to_numpy(
        dtype=float
    )

    z_real = df[
        "z_real_ohm"
    ].to_numpy(
        dtype=float
    )

    z_imag = df[
        "z_imag_ohm"
    ].to_numpy(
        dtype=float
    )

    Rs = max(
        estimate_rs_intercept(
            df
        ),
        1e-12,
    )

    Rct = max(
        estimate_rct_initial(
            df,
            Rs,
        ),
        1e-12,
    )

    # -----------------------------------------------------
    # Estimate characteristic frequency
    # from the maximum capacitive -Z''
    # -----------------------------------------------------

    capacitive_mask = (
        z_imag < 0
    )

    if np.any(
        capacitive_mask
    ):

        frequencies_cap = (
            frequency[
                capacitive_mask
            ]
        )

        imag_cap = (
            z_imag[
                capacitive_mask
            ]
        )

        peak_index = np.argmax(
            -imag_cap
        )

        f_peak = float(
            frequencies_cap[
                peak_index
            ]
        )

    else:

        f_peak = float(
            np.sqrt(
                np.min(
                    frequency
                )
                *
                np.max(
                    frequency
                )
            )
        )

    f_peak = max(
        f_peak,
        1e-12,
    )

    capacitance_guess = (
        1.0
        /
        (
            2.0
            * np.pi
            * f_peak
            * Rct
        )
    )

    capacitance_guess = max(
        float(
            capacitance_guess
        ),
        1e-12,
    )

    # -----------------------------------------------------
    # RC
    # -----------------------------------------------------

    if circuit == CIRCUIT_RC:

        return {
            "Rs":
                Rs,

            "Rct":
                Rct,

            "Cdl":
                capacitance_guess,
        }

    # -----------------------------------------------------
    # CPE
    # -----------------------------------------------------

    if circuit == CIRCUIT_CPE:

        return {
            "Rs":
                Rs,

            "Rct":
                Rct,

            "Q":
                capacitance_guess,

            "n":
                0.90,
        }

    # -----------------------------------------------------
    # CPE + Warburg
    # -----------------------------------------------------

    if circuit == CIRCUIT_CPE_W:

        lowest_frequency_index = int(
            np.argmin(
                frequency
            )
        )

        z_low = (
            z_real[
                lowest_frequency_index
            ]
            +
            1j
            * z_imag[
                lowest_frequency_index
            ]
        )

        omega_low = (
            2.0
            * np.pi
            * frequency[
                lowest_frequency_index
            ]
        )

        remaining_impedance = max(
            abs(
                z_low
                -
                (
                    Rs
                    + Rct
                )
            ),
            1e-8,
        )

        sigma = max(
            float(
                remaining_impedance
                * np.sqrt(
                    omega_low
                )
            ),
            1e-8,
        )

        return {
            "Rs":
                Rs,

            "Rct":
                Rct,

            "Q":
                capacitance_guess,

            "n":
                0.90,

            "sigma":
                sigma,
        }

    raise ValueError(
        f"Unsupported circuit: {circuit}"
    )


# =========================================================
# Optimization transform
# =========================================================

def _encode_parameters(
    parameters,
    circuit,
):
    """
    Positive parameters are optimized in log10
    space for numerical stability.
    """

    if circuit == CIRCUIT_RC:

        return np.array(
            [
                np.log10(
                    parameters[
                        "Rs"
                    ]
                ),

                np.log10(
                    parameters[
                        "Rct"
                    ]
                ),

                np.log10(
                    parameters[
                        "Cdl"
                    ]
                ),
            ],
            dtype=float,
        )

    if circuit == CIRCUIT_CPE:

        return np.array(
            [
                np.log10(
                    parameters[
                        "Rs"
                    ]
                ),

                np.log10(
                    parameters[
                        "Rct"
                    ]
                ),

                np.log10(
                    parameters[
                        "Q"
                    ]
                ),

                parameters[
                    "n"
                ],
            ],
            dtype=float,
        )

    if circuit == CIRCUIT_CPE_W:

        return np.array(
            [
                np.log10(
                    parameters[
                        "Rs"
                    ]
                ),

                np.log10(
                    parameters[
                        "Rct"
                    ]
                ),

                np.log10(
                    parameters[
                        "Q"
                    ]
                ),

                parameters[
                    "n"
                ],

                np.log10(
                    parameters[
                        "sigma"
                    ]
                ),
            ],
            dtype=float,
        )

    raise ValueError(
        "Unsupported circuit."
    )


def _decode_parameters(
    x,
    circuit,
):

    if circuit == CIRCUIT_RC:

        return {
            "Rs":
                float(
                    10 ** x[0]
                ),

            "Rct":
                float(
                    10 ** x[1]
                ),

            "Cdl":
                float(
                    10 ** x[2]
                ),
        }

    if circuit == CIRCUIT_CPE:

        return {
            "Rs":
                float(
                    10 ** x[0]
                ),

            "Rct":
                float(
                    10 ** x[1]
                ),

            "Q":
                float(
                    10 ** x[2]
                ),

            "n":
                float(
                    x[3]
                ),
        }

    if circuit == CIRCUIT_CPE_W:

        return {
            "Rs":
                float(
                    10 ** x[0]
                ),

            "Rct":
                float(
                    10 ** x[1]
                ),

            "Q":
                float(
                    10 ** x[2]
                ),

            "n":
                float(
                    x[3]
                ),

            "sigma":
                float(
                    10 ** x[4]
                ),
        }

    raise ValueError(
        "Unsupported circuit."
    )


def _optimization_bounds(
    circuit,
):

    # log10 bounds are intentionally broad.

    if circuit == CIRCUIT_RC:

        lower = np.array(
            [
                -12,
                -12,
                -15,
            ]
        )

        upper = np.array(
            [
                12,
                12,
                3,
            ]
        )

        return (
            lower,
            upper,
        )

    if circuit == CIRCUIT_CPE:

        lower = np.array(
            [
                -12,
                -12,
                -15,
                0.30,
            ]
        )

        upper = np.array(
            [
                12,
                12,
                3,
                1.00,
            ]
        )

        return (
            lower,
            upper,
        )

    if circuit == CIRCUIT_CPE_W:

        lower = np.array(
            [
                -12,
                -12,
                -15,
                0.30,
                -12,
            ]
        )

        upper = np.array(
            [
                12,
                12,
                3,
                1.00,
                12,
            ]
        )

        return (
            lower,
            upper,
        )

    raise ValueError(
        "Unsupported circuit."
    )


# =========================================================
# Equivalent-circuit fitting
# =========================================================

def fit_eis_equivalent_circuit(
    df,
    circuit=CIRCUIT_CPE,
    weighting="1/|Z|",
    exclude_inductive=True,
    frequency_min_Hz=None,
    frequency_max_Hz=None,
):
    """
    Fit complex EIS data using scipy least_squares.

    Residual vector:

        [Re(Zfit - Zexp), Im(Zfit - Zexp)]

    Optional weighting:

        None
        1/|Z|
    """

    if circuit not in SUPPORTED_CIRCUITS:

        raise ValueError(
            "Unsupported equivalent circuit."
        )

    fit_df, exclusion_applied = (
        prepare_eis_fit_data(
            df=df,

            exclude_inductive=
                exclude_inductive,

            frequency_min_Hz=
                frequency_min_Hz,

            frequency_max_Hz=
                frequency_max_Hz,
        )
    )

    frequency = fit_df[
        "frequency_Hz"
    ].to_numpy(
        dtype=float
    )

    z_experimental = (
        fit_df[
            "z_real_ohm"
        ].to_numpy(
            dtype=float
        )
        +
        1j
        * fit_df[
            "z_imag_ohm"
        ].to_numpy(
            dtype=float
        )
    )

    initial_parameters = (
        estimate_initial_parameters(
            fit_df,
            circuit,
        )
    )

    x0 = _encode_parameters(
        initial_parameters,
        circuit,
    )

    lower, upper = (
        _optimization_bounds(
            circuit
        )
    )

    # -----------------------------------------------------
    # Weighting
    # -----------------------------------------------------

    if weighting == "None":

        weights = np.ones(
            len(
                frequency
            ),
            dtype=float,
        )

    elif weighting == "1/|Z|":

        magnitude = np.maximum(
            np.abs(
                z_experimental
            ),
            1e-12,
        )

        weights = (
            1.0
            / magnitude
        )

    else:

        raise ValueError(
            "Unsupported weighting method."
        )

    # -----------------------------------------------------
    # Complex residual
    # -----------------------------------------------------

    def residual_function(
        x,
    ):

        parameters = (
            _decode_parameters(
                x,
                circuit,
            )
        )

        z_model = (
            calculate_circuit_impedance(
                frequency,
                parameters,
                circuit,
            )
        )

        difference = (
            z_model
            -
            z_experimental
        )

        residual_real = (
            difference.real
            * weights
        )

        residual_imag = (
            difference.imag
            * weights
        )

        return np.concatenate(
            [
                residual_real,
                residual_imag,
            ]
        )

    result = least_squares(
        residual_function,

        x0,

        bounds=(
            lower,
            upper,
        ),

        method="trf",

        x_scale="jac",

        max_nfev=50000,
    )

    fitted_parameters = (
        _decode_parameters(
            result.x,
            circuit,
        )
    )

    z_fitted = (
        calculate_circuit_impedance(
            frequency,
            fitted_parameters,
            circuit,
        )
    )

    difference = (
        z_fitted
        -
        z_experimental
    )

    # -----------------------------------------------------
    # Fit quality
    # -----------------------------------------------------

    rmse_ohm = float(
        np.sqrt(
            np.mean(
                np.abs(
                    difference
                ) ** 2
            )
        )
    )

    relative_rmse_pct = float(
        100.0
        * np.sqrt(
            np.mean(
                (
                    np.abs(
                        difference
                    )
                    /
                    np.maximum(
                        np.abs(
                            z_experimental
                        ),
                        1e-12,
                    )
                ) ** 2
            )
        )
    )

    weighted_residual_mse = float(
        np.mean(
            residual_function(
                result.x
            ) ** 2
        )
    )

    # -----------------------------------------------------
    # Fitted dataframe
    # -----------------------------------------------------

    fitted_df = pd.DataFrame(
        {
            "frequency_Hz":
                frequency,

            "z_real_fit_ohm":
                z_fitted.real,

            "z_imag_fit_ohm":
                z_fitted.imag,

            "z_modulus_fit_ohm":
                np.abs(
                    z_fitted
                ),

            "phase_fit_deg":
                np.angle(
                    z_fitted,
                    deg=True,
                ),
        }
    )

    return {
        "success":
            bool(
                result.success
            ),

        "message":
            str(
                result.message
            ),

        "circuit":
            circuit,

        "parameters":
            fitted_parameters,

        "initial_parameters":
            initial_parameters,

        "fit_data":
            fit_df,

        "fitted_data":
            fitted_df,

        "number_of_points":
            int(
                len(
                    fit_df
                )
            ),

        "rmse_ohm":
            rmse_ohm,

        "relative_rmse_pct":
            relative_rmse_pct,

        "weighted_residual_mse":
            weighted_residual_mse,

        "weighting":
            weighting,

        "exclude_inductive_requested":
            bool(
                exclude_inductive
            ),

        "inductive_exclusion_applied":
            bool(
                exclusion_applied
            ),

        "frequency_min_Hz":
            float(
                np.min(
                    frequency
                )
            ),

        "frequency_max_Hz":
            float(
                np.max(
                    frequency
                )
            ),

        "rs_intercept_estimate_ohm":
            float(
                estimate_rs_intercept(
                    fit_df
                )
            ),

        "rct_initial_estimate_ohm":
            float(
                estimate_rct_initial(
                    fit_df
                )
            ),
    }


# =========================================================
# Circuit formula text
# =========================================================

def get_circuit_equation(
    circuit,
):

    if circuit == CIRCUIT_RC:

        return (
            "Z = Rs + 1 / "
            "(1/Rct + jωCdl)"
        )

    if circuit == CIRCUIT_CPE:

        return (
            "Z = Rs + 1 / "
            "(1/Rct + Q(jω)^n)"
        )

    if circuit == CIRCUIT_CPE_W:

        return (
            "Z = Rs + 1 / "
            "(1/Rct + Q(jω)^n) "
            "+ σ/√(jω)"
        )

    return ""
