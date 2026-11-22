import numpy as np
import pandas as pd


# =========================================================
# Internal helpers
# =========================================================

def _fill_zero_direction(direction):
    direction = direction.copy()

    nonzero = np.where(direction != 0)[0]

    if len(nonzero) == 0:
        raise ValueError(
            "Potential does not change enough to detect GCD segments."
        )

    first = nonzero[0]

    direction[:first] = direction[first]

    for i in range(first + 1, len(direction)):
        if direction[i] == 0:
            direction[i] = direction[i - 1]

    return direction


def _direction_runs(direction):
    """
    Convert direction array into run-length information.
    """

    if len(direction) == 0:
        return []

    starts = np.r_[
        0,
        np.where(
            direction[1:] != direction[:-1]
        )[0] + 1
    ]

    ends = np.r_[
        starts[1:],
        len(direction)
    ]

    runs = []

    for start, end in zip(starts, ends):
        runs.append(
            {
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "direction": int(direction[start]),
            }
        )

    return runs


def _remove_short_direction_runs(
    direction,
    min_run_points=8,
    expected_segments=None,
):
    """
    Remove tiny direction reversals caused by noise.
    """

    cleaned = direction.copy()

    for _ in range(30):

        runs = _direction_runs(cleaned)

        short_runs = [
            (index, run)
            for index, run in enumerate(runs)
            if run["length"] < min_run_points
        ]

        if not short_runs:
            break

        changed = False

        for run_index, run in short_runs:

            runs = _direction_runs(cleaned)

            if run_index >= len(runs):
                continue

            run = runs[run_index]

            if len(runs) == 1:
                break

            left = (
                runs[run_index - 1]
                if run_index > 0
                else None
            )

            right = (
                runs[run_index + 1]
                if run_index < len(runs) - 1
                else None
            )

            if (
                left is not None
                and right is not None
                and left["direction"] == right["direction"]
            ):
                replacement = left["direction"]

            elif left is None:
                replacement = right["direction"]

            elif right is None:
                replacement = left["direction"]

            else:
                replacement = (
                    left["direction"]
                    if left["length"] >= right["length"]
                    else right["direction"]
                )

            cleaned[
                run["start"]:
                run["end"]
            ] = replacement

            changed = True

            break

        if not changed:
            break

    # -----------------------------------------------------
    # If CHI tells us the expected number of segments,
    # remove remaining very small excess runs.
    # -----------------------------------------------------

    if (
        expected_segments is not None
        and expected_segments > 0
    ):

        for _ in range(30):

            runs = _direction_runs(cleaned)

            if len(runs) <= expected_segments:
                break

            shortest_index = int(
                np.argmin(
                    [
                        run["length"]
                        for run in runs
                    ]
                )
            )

            run = runs[shortest_index]

            left = (
                runs[shortest_index - 1]
                if shortest_index > 0
                else None
            )

            right = (
                runs[shortest_index + 1]
                if shortest_index < len(runs) - 1
                else None
            )

            if left is None:
                replacement = right["direction"]

            elif right is None:
                replacement = left["direction"]

            elif left["length"] >= right["length"]:
                replacement = left["direction"]

            else:
                replacement = right["direction"]

            cleaned[
                run["start"]:
                run["end"]
            ] = replacement

    return cleaned


# =========================================================
# GCD segment detection
# =========================================================

def detect_gcd_segments(
    df,
    expected_segments=None,
    smoothing_window=7,
    min_run_points=8,
):
    """
    Automatically identify GCD charge/discharge segments.

    Increasing potential:
        charge

    Decreasing potential:
        discharge
    """

    required = {
        "time_s",
        "potential_V",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "GCD data must contain time_s and potential_V."
        )

    if len(df) < 5:
        raise ValueError(
            "Too few GCD data points."
        )

    potential = df[
        "potential_V"
    ].to_numpy(dtype=float)

    time = df[
        "time_s"
    ].to_numpy(dtype=float)

    # -----------------------------------------------------
    # Mild median smoothing ONLY for segment detection.
    # Raw data is not modified.
    # -----------------------------------------------------

    smoothing_window = max(
        3,
        int(smoothing_window)
    )

    if smoothing_window % 2 == 0:
        smoothing_window += 1

    smoothed = (
        pd.Series(potential)
        .rolling(
            window=smoothing_window,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy()
    )

    diff = np.diff(smoothed)

    typical_step = np.median(
        np.abs(diff[
            np.isfinite(diff)
        ])
    )

    tolerance = max(
        typical_step * 0.01,
        1e-12,
    )

    direction = np.zeros(
        len(diff),
        dtype=int,
    )

    direction[
        diff > tolerance
    ] = 1

    direction[
        diff < -tolerance
    ] = -1

    direction = _fill_zero_direction(
        direction
    )

    direction = _remove_short_direction_runs(
        direction,
        min_run_points=min_run_points,
        expected_segments=expected_segments,
    )

    runs = _direction_runs(
        direction
    )

    segments = []

    for index, run in enumerate(runs):

        # Direction run is based on differences.
        # A run from diff[start:end] corresponds
        # to data points start:end inclusive.
        start_index = run[
            "start"
        ]

        end_index = run[
            "end"
        ]

        start_index = max(
            0,
            start_index,
        )

        end_index = min(
            len(df) - 1,
            end_index,
        )

        segment_type = (
            "charge"
            if run["direction"] > 0
            else "discharge"
        )

        segments.append(
            {
                "segment":
                    index + 1,

                "type":
                    segment_type,

                "direction":
                    int(
                        run["direction"]
                    ),

                "start_index":
                    int(
                        start_index
                    ),

                "end_index":
                    int(
                        end_index
                    ),

                "start_time_s":
                    float(
                        time[
                            start_index
                        ]
                    ),

                "end_time_s":
                    float(
                        time[
                            end_index
                        ]
                    ),

                "duration_s":
                    float(
                        time[
                            end_index
                        ]
                        -
                        time[
                            start_index
                        ]
                    ),

                "start_potential_V":
                    float(
                        potential[
                            start_index
                        ]
                    ),

                "end_potential_V":
                    float(
                        potential[
                            end_index
                        ]
                    ),

                "point_count":
                    int(
                        end_index
                        - start_index
                        + 1
                    ),
            }
        )

    return segments


# =========================================================
# Complete cycles
# =========================================================

def build_gcd_cycles(
    df,
    expected_segments=None,
):
    """
    Pair consecutive charge + discharge segments
    into complete GCD cycles.
    """

    segments = detect_gcd_segments(
        df,
        expected_segments=expected_segments,
    )

    cycles = []

    cycle_number = 1
    i = 0

    while i < len(segments) - 1:

        first = segments[i]
        second = segments[i + 1]

        types = {
            first["type"],
            second["type"],
        }

        if types == {
            "charge",
            "discharge",
        }:

            charge_segment = (
                first
                if first["type"] == "charge"
                else second
            )

            discharge_segment = (
                first
                if first["type"] == "discharge"
                else second
            )

            cycles.append(
                {
                    "cycle":
                        cycle_number,

                    "start_index":
                        int(
                            min(
                                first[
                                    "start_index"
                                ],
                                second[
                                    "start_index"
                                ],
                            )
                        ),

                    "end_index":
                        int(
                            max(
                                first[
                                    "end_index"
                                ],
                                second[
                                    "end_index"
                                ],
                            )
                        ),

                    "segments":
                        [
                            first[
                                "segment"
                            ],
                            second[
                                "segment"
                            ],
                        ],

                    "charge_segment":
                        charge_segment[
                            "segment"
                        ],

                    "discharge_segment":
                        discharge_segment[
                            "segment"
                        ],

                    "charge_info":
                        charge_segment,

                    "discharge_info":
                        discharge_segment,
                }
            )

            cycle_number += 1
            i += 2

        else:
            i += 1

    return cycles


# =========================================================
# Cycle extraction
# =========================================================

def get_gcd_cycle(
    df,
    cycle_number,
    expected_segments=None,
):
    cycles = build_gcd_cycles(
        df,
        expected_segments=expected_segments,
    )

    selected = next(
        (
            cycle
            for cycle in cycles
            if cycle["cycle"]
            == cycle_number
        ),
        None,
    )

    if selected is None:
        raise ValueError(
            f"Cycle {cycle_number} was not found."
        )

    cycle_df = (
        df.iloc[
            selected["start_index"]:
            selected["end_index"] + 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        cycle_df,
        selected,
    )


def get_last_complete_gcd_cycle(
    df,
    expected_segments=None,
):
    cycles = build_gcd_cycles(
        df,
        expected_segments=expected_segments,
    )

    if not cycles:
        raise ValueError(
            "No complete GCD cycle detected."
        )

    selected = cycles[-1]

    cycle_df = (
        df.iloc[
            selected["start_index"]:
            selected["end_index"] + 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        cycle_df,
        selected,
    )


# =========================================================
# Discharge extraction
# =========================================================

def get_discharge_segment(
    df,
    cycle_info,
):
    """
    Extract raw discharge portion from a cycle.
    """

    info = cycle_info[
        "discharge_info"
    ]

    discharge_df = (
        df.iloc[
            info["start_index"]:
            info["end_index"] + 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    return discharge_df


# =========================================================
# Discharge analysis
# =========================================================

def analyze_gcd_discharge(
    discharge_df,
    ir_drop_ratio_threshold=3.0,
):
    """
    Analyze one discharge segment.

    IR drop is estimated from the first potential
    step when that step is substantially larger
    than the normal discharge step.

    Raw data remains unchanged.
    """

    if len(discharge_df) < 4:
        raise ValueError(
            "Too few discharge data points."
        )

    time = discharge_df[
        "time_s"
    ].to_numpy(dtype=float)

    potential = discharge_df[
        "potential_V"
    ].to_numpy(dtype=float)

    d_v = np.abs(
        np.diff(potential)
    )

    first_step = float(
        d_v[0]
    )

    regular_steps = d_v[
        1:min(
            len(d_v),
            30
        )
    ]

    regular_steps = regular_steps[
        np.isfinite(
            regular_steps
        )
        & (
            regular_steps > 0
        )
    ]

    if len(regular_steps) > 0:

        typical_step = float(
            np.median(
                regular_steps
            )
        )

    else:

        typical_step = 0.0

    ir_drop_detected = False
    ir_drop_V = 0.0
    analysis_start_index = 0

    if (
        typical_step > 0
        and
        first_step
        >= (
            typical_step
            * ir_drop_ratio_threshold
        )
    ):

        ir_drop_detected = True
        ir_drop_V = first_step

        # Exclude the instantaneous drop
        # from capacitance ΔV and Δt.
        analysis_start_index = 1

    if (
        analysis_start_index
        >= len(discharge_df) - 1
    ):
        analysis_start_index = 0

    start_time = float(
        time[
            analysis_start_index
        ]
    )

    end_time = float(
        time[-1]
    )

    start_potential = float(
        potential[
            analysis_start_index
        ]
    )

    end_potential = float(
        potential[-1]
    )

    discharge_time = (
        end_time
        - start_time
    )

    voltage_window = abs(
        start_potential
        - end_potential
    )

    # -----------------------------------------------------
    # Linear regression of discharge branch
    # Useful as a quality indicator.
    # -----------------------------------------------------

    fit_time = (
        time[
            analysis_start_index:
        ]
        -
        time[
            analysis_start_index
        ]
    )

    fit_potential = potential[
        analysis_start_index:
    ]

    if len(fit_time) >= 3:

        coefficients = np.polyfit(
            fit_time,
            fit_potential,
            1,
        )

        predicted = np.polyval(
            coefficients,
            fit_time,
        )

        ss_res = np.sum(
            (
                fit_potential
                - predicted
            ) ** 2
        )

        ss_tot = np.sum(
            (
                fit_potential
                - np.mean(
                    fit_potential
                )
            ) ** 2
        )

        linearity_r2 = (
            1.0
            - ss_res / ss_tot
            if ss_tot > 0
            else 1.0
        )

        discharge_slope_V_s = float(
            coefficients[0]
        )

    else:

        linearity_r2 = None
        discharge_slope_V_s = None

    return {
        "analysis_start_index":
            int(
                analysis_start_index
            ),

        "ir_drop_detected":
            bool(
                ir_drop_detected
            ),

        "ir_drop_V":
            float(
                ir_drop_V
            ),

        "discharge_start_time_s":
            start_time,

        "discharge_end_time_s":
            end_time,

        "discharge_time_s":
            float(
                discharge_time
            ),

        "discharge_start_potential_V":
            start_potential,

        "discharge_end_potential_V":
            end_potential,

        "discharge_voltage_window_V":
            float(
                voltage_window
            ),

        "discharge_slope_V_s":
            discharge_slope_V_s,

        "linearity_r2":
            (
                float(
                    linearity_r2
                )
                if linearity_r2
                is not None
                else None
            ),
    }


# =========================================================
# Default current from CHI metadata
# =========================================================

def get_default_discharge_current_A(
    metadata,
):
    """
    For conventional GCD, decreasing potential
    is usually associated with cathodic current.

    Magnitude is returned.
    """

    cathodic = metadata.get(
        "cathodic_current_A"
    )

    anodic = metadata.get(
        "anodic_current_A"
    )

    if (
        cathodic is not None
        and abs(
            float(cathodic)
        ) > 0
    ):
        return abs(
            float(cathodic)
        )

    if (
        anodic is not None
        and abs(
            float(anodic)
        ) > 0
    ):
        return abs(
            float(anodic)
        )

    return None


# =========================================================
# Full GCD calculation
# =========================================================

def calculate_gcd_parameters(
    discharge_df,
    discharge_current_A,
    active_mass_mg=None,
    capacitance_factor=1.0,
):
    """
    Calculate GCD parameters from the discharge branch.

    Standard gravimetric capacitance:

        Csp = I * Δt / (m * ΔV)

    If capacitance_factor = 4:
        a symmetric-cell single-electrode equivalent
        capacitance is additionally represented.

    Energy density and power density are calculated
    using the k=1 device/mass-basis capacitance:

        E = C_device * ΔV² / 7.2

        P = 3600 * E / Δt
    """

    if discharge_current_A is None:
        raise ValueError(
            "Discharge current is required."
        )

    discharge_current_A = abs(
        float(
            discharge_current_A
        )
    )

    if discharge_current_A <= 0:
        raise ValueError(
            "Discharge current must be > 0."
        )

    analysis = analyze_gcd_discharge(
        discharge_df
    )

    delta_t = analysis[
        "discharge_time_s"
    ]

    delta_v = analysis[
        "discharge_voltage_window_V"
    ]

    if delta_t <= 0:
        raise ValueError(
            "Invalid discharge time."
        )

    if delta_v <= 0:
        raise ValueError(
            "Invalid discharge voltage window."
        )

    device_csp = None
    displayed_csp = None
    energy_density = None
    power_density = None
    current_density = None

    if (
        active_mass_mg is not None
        and active_mass_mg > 0
    ):

        mass_g = (
            float(
                active_mass_mg
            )
            / 1000.0
        )

        current_density = (
            discharge_current_A
            / mass_g
        )

        # k = 1 device / entered mass basis
        device_csp = (
            discharge_current_A
            * delta_t
            /
            (
                mass_g
                * delta_v
            )
        )

        displayed_csp = (
            capacitance_factor
            * device_csp
        )

        # Energy/power always retain the
        # k=1 device/mass-basis result.
        energy_density = (
            device_csp
            * delta_v ** 2
            / 7.2
        )

        power_density = (
            3600.0
            * energy_density
            / delta_t
        )

    result = {
        "discharge_current_A":
            float(
                discharge_current_A
            ),

        "active_mass_mg":
            (
                float(
                    active_mass_mg
                )
                if active_mass_mg
                is not None
                else None
            ),

        "current_density_A_g":
            (
                float(
                    current_density
                )
                if current_density
                is not None
                else None
            ),

        "device_specific_capacitance_F_g":
            (
                float(
                    device_csp
                )
                if device_csp
                is not None
                else None
            ),

        "specific_capacitance_F_g":
            (
                float(
                    displayed_csp
                )
                if displayed_csp
                is not None
                else None
            ),

        "capacitance_factor":
            float(
                capacitance_factor
            ),

        "energy_density_Wh_kg":
            (
                float(
                    energy_density
                )
                if energy_density
                is not None
                else None
            ),

        "power_density_W_kg":
            (
                float(
                    power_density
                )
                if power_density
                is not None
                else None
            ),
    }

    result.update(
        analysis
    )

    return result
