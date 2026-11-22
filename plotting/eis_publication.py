import numpy as np
import matplotlib.pyplot as plt

from .publication import (
    _create_figure,
    _apply_axis_style,
    _apply_axis_view,
    _finalize,
    LEGEND_LOCATION_MAP,
    LINE_STYLE_MAP,
)

from core.eis_analysis import (
    CIRCUIT_RC,
    CIRCUIT_CPE,
    CIRCUIT_CPE_W,
)


# =========================================================
# Experimental curve helper
# =========================================================

def _plot_experimental(
    ax,
    x,
    y,
    style,
    experimental_style="Points + line",
    color="#1F77B4",
    label="Experimental",
    line_style="Solid",
):
    """
    Draw experimental EIS data.
    """

    marker_size = style.get(
        "marker_size",
        4,
    )

    line_width = style.get(
        "line_width",
        1.2,
    )

    mpl_line_style = LINE_STYLE_MAP.get(
        line_style,
        "-",
    )

    # -----------------------------------------------------
    # Points only
    # -----------------------------------------------------

    if experimental_style == "Points":

        ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            color=color,
            label=label,
        )

    # -----------------------------------------------------
    # Line only
    # -----------------------------------------------------

    elif experimental_style == "Line":

        ax.plot(
            x,
            y,
            linestyle=mpl_line_style,
            linewidth=line_width,
            color=color,
            label=label,
        )

    # -----------------------------------------------------
    # Points + line
    # -----------------------------------------------------

    else:

        ax.plot(
            x,
            y,
            linestyle=mpl_line_style,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            color=color,
            label=label,
        )


# =========================================================
# Single Nyquist
# =========================================================

def plot_eis_nyquist(
    experimental_df,
    style,
    fit_result=None,
    experimental_style="Points + line",
    experimental_color="#1F77B4",
    show_fitted_curve=False,
    fit_color="#D62728",
    show_legend=False,
    legend_location="Upper right",
):
    """
    Publication-quality Nyquist plot.

    Fitted curve is hidden by default.
    """

    fig, ax = _create_figure(
        style
    )

    _plot_experimental(
        ax=ax,

        x=experimental_df[
            "z_real_ohm"
        ],

        y=-experimental_df[
            "z_imag_ohm"
        ],

        style=style,

        experimental_style=
            experimental_style,

        color=
            experimental_color,

        label=
            "Experimental",
    )

    # -----------------------------------------------------
    # Optional fitted curve
    # -----------------------------------------------------

    if (
        show_fitted_curve
        and
        fit_result is not None
    ):

        fitted = fit_result[
            "fitted_data"
        ]

        ax.plot(
            fitted[
                "z_real_fit_ohm"
            ],

            -fitted[
                "z_imag_fit_ohm"
            ],

            color=
                fit_color,

            linewidth=
                style.get(
                    "line_width",
                    1.5,
                ),

            linestyle="-",

            label=
                "Fitted",
        )

    ax.set_xlabel(
        "Z' (Ω)"
    )

    ax.set_ylabel(
        "-Z'' (Ω)"
    )

    _apply_axis_style(
        ax,
        style,
    )

    _apply_axis_view(
        ax,
        style,
    )

    if (
        style.get(
            "nyquist_aspect",
            "Equal X/Y scale",
        )
        ==
        "Equal X/Y scale"
    ):

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    else:

        ax.set_aspect(
            "auto"
        )

    if show_legend:

        ax.legend(
            loc=
                LEGEND_LOCATION_MAP.get(
                    legend_location,
                    "upper right",
                ),

            frameon=False,
        )

    return _finalize(
        fig,
        ax,
        style,
    )


# =========================================================
# Multiple Nyquist curves
# =========================================================

def plot_eis_multi_nyquist(
    curves,
    style,
    experimental_style="Points + line",
    show_legend=True,
    legend_location="Upper right",
):
    """
    Overlay multiple EIS Nyquist curves.

    curves:
        [
            {
                "df": dataframe,
                "label": "20 °C",
                "color": "#1F77B4",
                "line_style": "Solid",
                "temperature_C": 20.0,
            }
        ]
    """

    fig, ax = _create_figure(
        style
    )

    for curve in curves:

        df = curve[
            "df"
        ]

        _plot_experimental(
            ax=ax,

            x=df[
                "z_real_ohm"
            ],

            y=-df[
                "z_imag_ohm"
            ],

            style=style,

            experimental_style=
                experimental_style,

            color=
                curve.get(
                    "color",
                    "#1F77B4",
                ),

            label=
                curve.get(
                    "label",
                    "",
                ),

            line_style=
                curve.get(
                    "line_style",
                    "Solid",
                ),
        )

    ax.set_xlabel(
        "Z' (Ω)"
    )

    ax.set_ylabel(
        "-Z'' (Ω)"
    )

    _apply_axis_style(
        ax,
        style,
    )

    _apply_axis_view(
        ax,
        style,
    )

    if (
        style.get(
            "nyquist_aspect",
            "Equal X/Y scale",
        )
        ==
        "Equal X/Y scale"
    ):

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    else:

        ax.set_aspect(
            "auto"
        )

    if show_legend:

        ax.legend(
            loc=
                LEGEND_LOCATION_MAP.get(
                    legend_location,
                    "upper right",
                ),

            frameon=False,
        )

    return _finalize(
        fig,
        ax,
        style,
    )


# =========================================================
# Bode magnitude
# =========================================================

def plot_eis_bode_magnitude(
    experimental_df,
    style,
    fit_result=None,
    experimental_style="Points + line",
    experimental_color="#1F77B4",
    show_fitted_curve=False,
    fit_color="#D62728",
    show_legend=False,
    legend_location="Upper right",
):
    """
    Publication-quality Bode |Z|.
    """

    fig, ax = _create_figure(
        style
    )

    _plot_experimental(
        ax=ax,

        x=experimental_df[
            "frequency_Hz"
        ],

        y=experimental_df[
            "z_modulus_ohm"
        ],

        style=style,

        experimental_style=
            experimental_style,

        color=
            experimental_color,

        label=
            "Experimental",
    )

    if (
        show_fitted_curve
        and
        fit_result is not None
    ):

        fitted = fit_result[
            "fitted_data"
        ]

        ax.plot(
            fitted[
                "frequency_Hz"
            ],

            fitted[
                "z_modulus_fit_ohm"
            ],

            color=
                fit_color,

            linewidth=
                style.get(
                    "line_width",
                    1.5,
                ),

            linestyle="-",

            label=
                "Fitted",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "|Z| (Ω)"
    )

    _apply_axis_style(
        ax,
        style,
        x_log=True,
        y_log=True,
    )

    _apply_axis_view(
        ax,
        style,
        x_log=True,
        y_log=True,
    )

    if show_legend:

        ax.legend(
            loc=
                LEGEND_LOCATION_MAP.get(
                    legend_location,
                    "upper right",
                ),

            frameon=False,
        )

    return _finalize(
        fig,
        ax,
        style,
    )


# =========================================================
# Bode phase
# =========================================================

def plot_eis_bode_phase(
    experimental_df,
    style,
    fit_result=None,
    experimental_style="Points + line",
    experimental_color="#1F77B4",
    show_fitted_curve=False,
    fit_color="#D62728",
    show_legend=False,
    legend_location="Upper right",
):
    """
    Publication-quality Bode phase.
    """

    fig, ax = _create_figure(
        style
    )

    _plot_experimental(
        ax=ax,

        x=experimental_df[
            "frequency_Hz"
        ],

        y=experimental_df[
            "phase_deg"
        ],

        style=style,

        experimental_style=
            experimental_style,

        color=
            experimental_color,

        label=
            "Experimental",
    )

    if (
        show_fitted_curve
        and
        fit_result is not None
    ):

        fitted = fit_result[
            "fitted_data"
        ]

        ax.plot(
            fitted[
                "frequency_Hz"
            ],

            fitted[
                "phase_fit_deg"
            ],

            color=
                fit_color,

            linewidth=
                style.get(
                    "line_width",
                    1.5,
                ),

            linestyle="-",

            label=
                "Fitted",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Phase (°)"
    )

    _apply_axis_style(
        ax,
        style,
        x_log=True,
        y_log=False,
    )

    _apply_axis_view(
        ax,
        style,
        x_log=True,
        y_log=False,
    )

    if show_legend:

        ax.legend(
            loc=
                LEGEND_LOCATION_MAP.get(
                    legend_location,
                    "upper right",
                ),

            frameon=False,
        )

    return _finalize(
        fig,
        ax,
        style,
    )


# =========================================================
# Circuit helpers
# =========================================================

def _draw_wire(
    ax,
    x1,
    y1,
    x2,
    y2,
):

    ax.plot(
        [x1, x2],
        [y1, y2],
        color="black",
        linewidth=1.25,
    )


def _draw_node(
    ax,
    x,
    y,
):

    ax.plot(
        x,
        y,
        marker="o",
        color="black",
        markersize=3.5,
    )


def _draw_resistor(
    ax,
    x1,
    x2,
    y,
    label,
):

    lead = (
        x2 - x1
    ) * 0.15

    _draw_wire(
        ax,
        x1,
        y,
        x1 + lead,
        y,
    )

    _draw_wire(
        ax,
        x2 - lead,
        y,
        x2,
        y,
    )

    left = (
        x1 + lead
    )

    right = (
        x2 - lead
    )

    amplitude = 0.045

    count = 8

    xs = np.linspace(
        left,
        right,
        count + 1,
    )

    ys = []

    for i in range(
        count + 1
    ):

        if (
            i == 0
            or
            i == count
        ):

            ys.append(
                y
            )

        elif i % 2 == 1:

            ys.append(
                y + amplitude
            )

        else:

            ys.append(
                y - amplitude
            )

    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=1.25,
    )

    ax.text(
        (
            x1 + x2
        ) / 2,

        y + 0.10,

        label,

        ha="center",
        va="bottom",
        fontsize=10,
    )


def _draw_capacitor(
    ax,
    x1,
    x2,
    y,
    label,
    cpe=False,
):

    center = (
        x1 + x2
    ) / 2

    spacing = 0.025
    height = 0.12

    _draw_wire(
        ax,
        x1,
        y,
        center - spacing,
        y,
    )

    _draw_wire(
        ax,
        center + spacing,
        y,
        x2,
        y,
    )

    ax.plot(
        [
            center - spacing,
            center - spacing,
        ],

        [
            y - height / 2,
            y + height / 2,
        ],

        color="black",
        linewidth=1.4,
    )

    ax.plot(
        [
            center + spacing,
            center + spacing,
        ],

        [
            y - height / 2,
            y + height / 2,
        ],

        color="black",
        linewidth=1.4,
    )

    if cpe:

        ax.plot(
            [
                center - 0.045,
                center + 0.045,
            ],

            [
                y - 0.075,
                y + 0.075,
            ],

            color="black",
            linewidth=0.9,
        )

    ax.text(
        center,
        y - 0.11,
        label,
        ha="center",
        va="top",
        fontsize=10,
    )


def _draw_warburg(
    ax,
    x1,
    x2,
    y,
):

    _draw_wire(
        ax,
        x1,
        y,
        x1 + 0.02,
        y,
    )

    _draw_wire(
        ax,
        x2 - 0.02,
        y,
        x2,
        y,
    )

    left = (
        x1 + 0.02
    )

    right = (
        x2 - 0.02
    )

    xs = np.linspace(
        left,
        right,
        7,
    )

    ys = (
        y
        +
        np.array(
            [
                0,
                0.04,
                -0.04,
                0.04,
                -0.04,
                0.04,
                0,
            ]
        )
    )

    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=1.25,
    )

    ax.text(
        (
            x1 + x2
        ) / 2,

        y + 0.10,

        r"$W$",

        ha="center",
        fontsize=10,
    )


def _format_parameter(
    value,
):

    value = float(
        value
    )

    if (
        abs(
            value
        ) >= 1000
        or
        (
            abs(
                value
            ) < 0.001
            and
            value != 0
        )
    ):

        return (
            f"{value:.2e}"
        )

    return (
        f"{value:.3g}"
    )


# =========================================================
# Equivalent circuit
# =========================================================

def plot_equivalent_circuit(
    circuit,
    fit_result=None,
    show_values=False,
    width_mm=110,
    height_mm=42,
    font_family="Arial",
):

    style = {
        "width_mm":
            width_mm,

        "height_mm":
            height_mm,

        "font_family":
            font_family,

        "tick_font_size":
            9,

        "label_font_size":
            9,

        "legend_font_size":
            8,
    }

    fig, ax = _create_figure(
        style
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.axis(
        "off"
    )

    # -----------------------------------------------------
    # Rs
    # -----------------------------------------------------

    _draw_wire(
        ax,
        0.03,
        0.50,
        0.08,
        0.50,
    )

    _draw_resistor(
        ax,
        0.08,
        0.31,
        0.50,
        r"$R_s$",
    )

    _draw_wire(
        ax,
        0.31,
        0.50,
        0.39,
        0.50,
    )

    # -----------------------------------------------------
    # Parallel branch
    # -----------------------------------------------------

    x_left = 0.39
    x_right = 0.72

    y_top = 0.72
    y_bottom = 0.28

    _draw_node(
        ax,
        x_left,
        0.50,
    )

    _draw_node(
        ax,
        x_right,
        0.50,
    )

    _draw_wire(
        ax,
        x_left,
        y_bottom,
        x_left,
        y_top,
    )

    _draw_wire(
        ax,
        x_right,
        y_bottom,
        x_right,
        y_top,
    )

    _draw_resistor(
        ax,
        x_left,
        x_right,
        y_top,
        r"$R_{ct}$",
    )

    if circuit == CIRCUIT_RC:

        _draw_capacitor(
            ax,
            x_left,
            x_right,
            y_bottom,
            r"$C_{dl}$",
            cpe=False,
        )

    else:

        _draw_capacitor(
            ax,
            x_left,
            x_right,
            y_bottom,
            r"CPE $(Q,n)$",
            cpe=True,
        )

    # -----------------------------------------------------
    # Warburg
    # -----------------------------------------------------

    if circuit == CIRCUIT_CPE_W:

        _draw_wire(
            ax,
            x_right,
            0.50,
            0.77,
            0.50,
        )

        _draw_warburg(
            ax,
            0.77,
            0.94,
            0.50,
        )

        _draw_wire(
            ax,
            0.94,
            0.50,
            0.98,
            0.50,
        )

    else:

        _draw_wire(
            ax,
            x_right,
            0.50,
            0.98,
            0.50,
        )

    # -----------------------------------------------------
    # Optional fitted values
    # -----------------------------------------------------

    if (
        show_values
        and
        fit_result is not None
    ):

        parameters = fit_result[
            "parameters"
        ]

        lines = [
            (
                r"$R_s$ = "
                +
                _format_parameter(
                    parameters[
                        "Rs"
                    ]
                )
                +
                r" $\Omega$"
            ),

            (
                r"$R_{ct}$ = "
                +
                _format_parameter(
                    parameters[
                        "Rct"
                    ]
                )
                +
                r" $\Omega$"
            ),
        ]

        if circuit == CIRCUIT_RC:

            lines.append(
                (
                    r"$C_{dl}$ = "
                    +
                    _format_parameter(
                        parameters[
                            "Cdl"
                        ]
                    )
                    +
                    " F"
                )
            )

        else:

            lines.append(
                (
                    r"$Q$ = "
                    +
                    _format_parameter(
                        parameters[
                            "Q"
                        ]
                    )
                )
            )

            lines.append(
                (
                    r"$n$ = "
                    +
                    _format_parameter(
                        parameters[
                            "n"
                        ]
                    )
                )
            )

        if circuit == CIRCUIT_CPE_W:

            lines.append(
                (
                    r"$\sigma$ = "
                    +
                    _format_parameter(
                        parameters[
                            "sigma"
                        ]
                    )
                )
            )

        ax.text(
            0.50,
            0.04,
            "   ".join(
                lines
            ),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.tight_layout(
        pad=0.2
    )

    return fig
