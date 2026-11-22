import numpy as np
import pandas as pd


# =========================================================
# CV segment detection
# =========================================================

def detect_cv_segments(df, tolerance=1e-12):
    """
    Detect CV scan segments according to the change
    in potential direction.

    Parameters
    ----------
    df : pandas.DataFrame
        Required columns:
        potential_V
        current_A

    tolerance : float
        Small potential change regarded as zero.

    Returns
    -------
    list[dict]
    """

    if "potential_V" not in df.columns:
        raise ValueError(
            "Column 'potential_V' was not found."
        )

    potential = df["potential_V"].to_numpy(
        dtype=float
    )

    if len(potential) < 3:
        raise ValueError(
            "Too few CV data points."
        )

    # -----------------------------------------------------
    # Potential difference
    # -----------------------------------------------------

    diff = np.diff(potential)

    direction = np.zeros(
        len(diff),
        dtype=int
    )

    direction[diff > tolerance] = 1
    direction[diff < -tolerance] = -1

    # -----------------------------------------------------
    # Fill zero-direction points
    # -----------------------------------------------------

    nonzero_indices = np.where(
        direction != 0
    )[0]

    if len(nonzero_indices) == 0:
        raise ValueError(
            "Potential does not change."
        )

    first_nonzero = nonzero_indices[0]

    direction[:first_nonzero] = (
        direction[first_nonzero]
    )

    for i in range(
        first_nonzero + 1,
        len(direction)
    ):

        if direction[i] == 0:
            direction[i] = direction[i - 1]

    # -----------------------------------------------------
    # Detect turning points
    # -----------------------------------------------------

    change_points = np.where(
        direction[1:]
        != direction[:-1]
    )[0] + 1

    boundaries = (
        [0]
        + change_points.tolist()
        + [len(df) - 1]
    )

    segments = []

    # -----------------------------------------------------
    # Create segment information
    # -----------------------------------------------------

    for i in range(
        len(boundaries) - 1
    ):

        start_index = boundaries[i]
        end_index = boundaries[i + 1]

        start_potential = potential[
            start_index
        ]

        end_potential = potential[
            end_index
        ]

        delta_v = (
            end_potential
            - start_potential
        )

        if delta_v > tolerance:
            scan_direction = "forward"

        elif delta_v < -tolerance:
            scan_direction = "reverse"

        else:
            scan_direction = "unknown"

        segments.append(
            {
                "segment": i + 1,

                "start_index":
                    int(start_index),

                "end_index":
                    int(end_index),

                "direction":
                    scan_direction,

                "start_potential_V":
                    float(start_potential),

                "end_potential_V":
                    float(end_potential),

                "point_count":
                    int(
                        end_index
                        - start_index
                        + 1
                    )
            }
        )

    return segments


# =========================================================
# Build complete CV cycles
# =========================================================

def build_cv_cycles(df):
    """
    Combine every two consecutive scan segments
    into one complete CV cycle.
    """

    segments = detect_cv_segments(
        df
    )

    cycles = []

    cycle_number = 1

    for i in range(
        0,
        len(segments) - 1,
        2
    ):

        first_segment = segments[i]
        second_segment = segments[i + 1]

        # A complete cycle should contain
        # opposite scan directions
        if (
            first_segment["direction"]
            == second_segment["direction"]
        ):
            continue

        cycles.append(
            {
                "cycle":
                    cycle_number,

                "start_index":
                    first_segment[
                        "start_index"
                    ],

                "end_index":
                    second_segment[
                        "end_index"
                    ],

                "segments":
                    [
                        first_segment[
                            "segment"
                        ],

                        second_segment[
                            "segment"
                        ]
                    ],

                "start_potential_V":
                    first_segment[
                        "start_potential_V"
                    ],

                "turning_potential_V":
                    first_segment[
                        "end_potential_V"
                    ],

                "end_potential_V":
                    second_segment[
                        "end_potential_V"
                    ],
            }
        )

        cycle_number += 1

    return cycles


# =========================================================
# Get a specified cycle
# =========================================================

def get_cv_cycle(
    df,
    cycle_number
):
    """
    Return one complete CV cycle.
    """

    cycles = build_cv_cycles(
        df
    )

    selected = None

    for cycle in cycles:

        if (
            cycle["cycle"]
            == cycle_number
        ):

            selected = cycle
            break

    if selected is None:

        raise ValueError(
            f"Cycle {cycle_number} "
            f"was not found."
        )

    cycle_df = (
        df.iloc[
            selected["start_index"]:
            selected["end_index"] + 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    return cycle_df


# =========================================================
# Get last complete cycle
# =========================================================

def get_last_complete_cycle(df):
    """
    Return the last complete CV cycle.
    """

    cycles = build_cv_cycles(
        df
    )

    if not cycles:

        raise ValueError(
            "No complete CV cycle detected."
        )

    last_cycle = cycles[-1]

    cycle_df = (
        df.iloc[
            last_cycle["start_index"]:
            last_cycle["end_index"] + 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        cycle_df,
        last_cycle
    )


# =========================================================
# CV path integral
# =========================================================

def calculate_cv_integral(df):
    """
    Calculate:

        Integral |I| dV

    along the actual CV acquisition path.

    Using |dV| prevents cancellation between
    forward and reverse scans.

    Unit:
        A * V
    """

    potential = df[
        "potential_V"
    ].to_numpy(
        dtype=float
    )

    current = df[
        "current_A"
    ].to_numpy(
        dtype=float
    )

    if len(potential) < 2:
        raise ValueError(
            "Too few data points "
            "for CV integration."
        )

    delta_v = np.abs(
        np.diff(potential)
    )

    average_current = (
        np.abs(current[:-1])
        + np.abs(current[1:])
    ) / 2

    integral = np.sum(
        average_current
        * delta_v
    )

    return float(integral)


# =========================================================
# CV peak detection
# =========================================================

def detect_cv_peaks(df):
    """
    Detect the maximum and minimum current within
    one complete CV cycle.

    For a conventional redox CV:

    maximum current -> oxidation/anodic peak
    minimum current -> reduction/cathodic peak

    Returns
    -------
    dict
    """

    if len(df) < 3:
        raise ValueError(
            "Too few points for peak detection."
        )

    current = df[
        "current_A"
    ].to_numpy(
        dtype=float
    )

    potential = df[
        "potential_V"
    ].to_numpy(
        dtype=float
    )

    oxidation_index = int(
        np.argmax(current)
    )

    reduction_index = int(
        np.argmin(current)
    )

    oxidation_potential = float(
        potential[
            oxidation_index
        ]
    )

    oxidation_current = float(
        current[
            oxidation_index
        ]
    )

    reduction_potential = float(
        potential[
            reduction_index
        ]
    )

    reduction_current = float(
        current[
            reduction_index
        ]
    )

    peak_separation = abs(
        oxidation_potential
        - reduction_potential
    )

    return {
        "oxidation_peak_potential_V":
            oxidation_potential,

        "oxidation_peak_current_A":
            oxidation_current,

        "reduction_peak_potential_V":
            reduction_potential,

        "reduction_peak_current_A":
            reduction_current,

        "peak_separation_V":
            float(peak_separation)
    }


# =========================================================
# CV specific capacitance
# =========================================================

def calculate_cv_specific_capacitance(
    df,
    scan_rate_V_s,
    active_mass_mg,
    capacitance_factor=1.0
):
    """
    Calculate gravimetric specific capacitance
    from one COMPLETE CV cycle.

    Formula:

        Csp =
        k * Integral(|I| dV)
        --------------------
        2 * m * scan_rate * DeltaV

    where:

        Csp       : F/g
        I         : A
        V         : V
        m         : g
        scan_rate : V/s
        k         : calculation convention factor

    capacitance_factor = 1
        standard calculation

    capacitance_factor = 4
        optional symmetric-cell conversion
        to single-electrode equivalent
    """

    if scan_rate_V_s is None:
        raise ValueError(
            "Scan rate was not found."
        )

    if scan_rate_V_s <= 0:
        raise ValueError(
            "Scan rate must be > 0."
        )

    if active_mass_mg is None:
        raise ValueError(
            "Active mass is required."
        )

    if active_mass_mg <= 0:
        raise ValueError(
            "Active mass must be > 0 mg."
        )

    potential = df[
        "potential_V"
    ].to_numpy(
        dtype=float
    )

    voltage_window = (
        np.max(potential)
        - np.min(potential)
    )

    if voltage_window <= 0:
        raise ValueError(
            "Invalid voltage window."
        )

    # mg -> g
    mass_g = (
        active_mass_mg
        / 1000.0
    )

    integral = calculate_cv_integral(
        df
    )

    capacitance = (
        capacitance_factor
        * integral
        /
        (
            2.0
            * mass_g
            * scan_rate_V_s
            * voltage_window
        )
    )

    return float(capacitance)


# =========================================================
# Full CV parameter calculation
# =========================================================

def calculate_cv_parameters(
    df,
    scan_rate_V_s,
    active_mass_mg=None,
    capacitance_factor=1.0
):
    """
    Calculate all currently supported CV parameters.

    active_mass_mg is optional.

    If mass is omitted:
        peak and integral results are still returned,
        but specific capacitance is None.
    """

    if len(df) < 3:
        raise ValueError(
            "Too few CV data points."
        )

    potential = df[
        "potential_V"
    ].to_numpy(
        dtype=float
    )

    current = df[
        "current_A"
    ].to_numpy(
        dtype=float
    )

    voltage_min = float(
        np.min(potential)
    )

    voltage_max = float(
        np.max(potential)
    )

    voltage_window = (
        voltage_max
        - voltage_min
    )

    integral = calculate_cv_integral(
        df
    )

    peaks = detect_cv_peaks(
        df
    )

    # -----------------------------------------------------
    # Specific capacitance
    # -----------------------------------------------------

    specific_capacitance = None

    if (
        active_mass_mg is not None
        and active_mass_mg > 0
    ):

        specific_capacitance = (
            calculate_cv_specific_capacitance(
                df=df,
                scan_rate_V_s=
                    scan_rate_V_s,

                active_mass_mg=
                    active_mass_mg,

                capacitance_factor=
                    capacitance_factor
            )
        )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    result = {
        "voltage_min_V":
            voltage_min,

        "voltage_max_V":
            voltage_max,

        "voltage_window_V":
            float(
                voltage_window
            ),

        "current_min_A":
            float(
                np.min(current)
            ),

        "current_max_A":
            float(
                np.max(current)
            ),

        "integral_abs_I_dV_A_V":
            integral,

        "scan_rate_V_s":
            float(
                scan_rate_V_s
            ),

        "specific_capacitance_F_g":
            specific_capacitance,
    }

    result.update(
        peaks
    )

    return result
