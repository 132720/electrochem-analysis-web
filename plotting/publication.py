import io

import matplotlib.pyplot as plt

from matplotlib.ticker import (
    AutoMinorLocator,
    MultipleLocator,
)


CURRENT_UNIT_SCALE = {
    "A": 1.0,
    "mA": 1000.0,
    "μA": 1_000_000.0,
}


LINE_STYLE_MAP = {
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
}


LEGEND_LOCATION_MAP = {
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Lower right": "lower right",
    "Lower left": "lower left",
    "Center right": "center right",
    "Center left": "center left",
    "Upper center": "upper center",
    "Lower center": "lower center",
    "Best": "best",
}


# =========================================================
# Base figure
# =========================================================

def _create_figure(style):

    width_inch = (
        style["width_mm"]
        / 25.4
    )

    height_inch = (
        style["height_mm"]
        / 25.4
    )

    rc = {
        "font.family":
            style.get(
                "font_family",
                "Arial"
            ),

        "font.size":
            style.get(
                "tick_font_size",
                8
            ),

        "axes.labelsize":
            style.get(
                "label_font_size",
                9
            ),

        "xtick.labelsize":
            style.get(
                "tick_font_size",
                8
            ),

        "ytick.labelsize":
            style.get(
                "tick_font_size",
                8
            ),

        "legend.fontsize":
            style.get(
                "legend_font_size",
                8
            ),

        "axes.unicode_minus":
            False,

        "svg.fonttype":
            "none",

        "pdf.fonttype":
            42,
    }

    with plt.rc_context(rc):

        fig, ax = plt.subplots(
            figsize=(
                width_inch,
                height_inch
            ),
            dpi=120
        )

    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )

    return fig, ax


# =========================================================
# Axis appearance
# =========================================================

def _apply_axis_style(
    ax,
    style,
    x_log=False,
    y_log=False,
):

    axis_width = style.get(
        "axis_line_width",
        0.8
    )

    four_sided = style.get(
        "four_sided_frame",
        True
    )

    direction = style.get(
        "tick_direction",
        "in"
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            axis_width
        )

    ax.spines[
        "top"
    ].set_visible(
        four_sided
    )

    ax.spines[
        "right"
    ].set_visible(
        four_sided
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction=direction,
        length=4,
        width=axis_width,
        top=four_sided,
        right=four_sided,
    )

    if style.get(
        "minor_ticks",
        True
    ):

        ax.minorticks_on()

        subdivisions = int(
            style.get(
                "minor_subdivisions",
                2
            )
        )

        if not x_log:
            ax.xaxis.set_minor_locator(
                AutoMinorLocator(
                    subdivisions
                )
            )

        if not y_log:
            ax.yaxis.set_minor_locator(
                AutoMinorLocator(
                    subdivisions
                )
            )

        ax.tick_params(
            axis="both",
            which="minor",
            direction=direction,
            length=2,
            width=axis_width * 0.8,
            top=four_sided,
            right=four_sided,
        )

    else:

        ax.minorticks_off()

    if style.get(
        "grid",
        False
    ):

        ax.grid(
            True,
            which="major",
            linewidth=0.5,
            alpha=0.25,
        )

    else:

        ax.grid(
            False
        )


# =========================================================
# View controls
# =========================================================

def _apply_axis_view(
    ax,
    style,
    x_log=False,
    y_log=False,
):

    if (
        style.get(
            "x_axis_mode",
            "Auto"
        )
        == "Manual"
    ):

        x_min = float(
            style["x_min"]
        )

        x_max = float(
            style["x_max"]
        )

        if x_min >= x_max:
            raise ValueError(
                "X min must be smaller than X max."
            )

        if (
            x_log
            and x_min <= 0
        ):
            raise ValueError(
                "Log X axis requires X min > 0."
            )

        ax.set_xlim(
            x_min,
            x_max
        )

    else:

        ax.margins(
            x=(
                style.get(
                    "x_padding_pct",
                    5.0
                )
                / 100.0
            )
        )

    if (
        style.get(
            "y_axis_mode",
            "Auto"
        )
        == "Manual"
    ):

        y_min = float(
            style["y_min"]
        )

        y_max = float(
            style["y_max"]
        )

        if y_min >= y_max:
            raise ValueError(
                "Y min must be smaller than Y max."
            )

        if (
            y_log
            and y_min <= 0
        ):
            raise ValueError(
                "Log Y axis requires Y min > 0."
            )

        ax.set_ylim(
            y_min,
            y_max
        )

    else:

        ax.margins(
            y=(
                style.get(
                    "y_padding_pct",
                    8.0
                )
                / 100.0
            )
        )

    x_interval = style.get(
        "x_major_interval"
    )

    y_interval = style.get(
        "y_major_interval"
    )

    if (
        x_interval is not None
        and not x_log
    ):

        ax.xaxis.set_major_locator(
            MultipleLocator(
                float(
                    x_interval
                )
            )
        )

    if (
        y_interval is not None
        and not y_log
    ):

        ax.yaxis.set_major_locator(
            MultipleLocator(
                float(
                    y_interval
                )
            )
        )


# =========================================================
# Final formatting
# =========================================================

def _finalize(
    fig,
    ax,
    style,
):

    font = style.get(
        "font_family",
        "Arial"
    )

    label_size = style.get(
        "label_font_size",
        9
    )

    tick_size = style.get(
        "tick_font_size",
        8
    )

    ax.xaxis.label.set_fontfamily(
        font
    )

    ax.yaxis.label.set_fontfamily(
        font
    )

    ax.xaxis.label.set_fontsize(
        label_size
    )

    ax.yaxis.label.set_fontsize(
        label_size
    )

    for label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):

        label.set_fontfamily(
            font
        )

        label.set_fontsize(
            tick_size
        )

    fig.tight_layout(
        pad=0.8
    )

    return fig


# =========================================================
# CV
# =========================================================

def plot_cv_publication(
    df,
    style,
    current_unit="mA",
    line_color="#1F77B4",
    line_style="Solid",
    cv_parameters=None,
    show_peak_markers=False,
):

    scale = CURRENT_UNIT_SCALE[
        current_unit
    ]

    fig, ax = _create_figure(
        style
    )

    ax.plot(
        df[
            "potential_V"
        ],
        (
            df[
                "current_A"
            ]
            * scale
        ),
        color=line_color,
        linestyle=LINE_STYLE_MAP[
            line_style
        ],
        linewidth=style.get(
            "line_width",
            1.5
        ),
    )

    if (
        show_peak_markers
        and cv_parameters is not None
    ):

        marker_size = style.get(
            "marker_size",
            4
        )

        ax.plot(
            cv_parameters[
                "oxidation_peak_potential_V"
            ],
            (
                cv_parameters[
                    "oxidation_peak_current_A"
                ]
                * scale
            ),
            marker="o",
            linestyle="None",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=line_color,
        )

        ax.plot(
            cv_parameters[
                "reduction_peak_potential_V"
            ],
            (
                cv_parameters[
                    "reduction_peak_current_A"
                ]
                * scale
            ),
            marker="s",
            linestyle="None",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=line_color,
        )

    ax.set_xlabel(
        "Potential (V)"
    )

    ax.set_ylabel(
        f"Current ({current_unit})"
    )

    _apply_axis_style(
        ax,
        style
    )

    _apply_axis_view(
        ax,
        style
    )

    return _finalize(
        fig,
        ax,
        style
    )


def plot_cv_multi_publication(
    curves,
    style,
    current_unit="mA",
    show_legend=True,
    legend_location="Upper right",
):

    scale = CURRENT_UNIT_SCALE[
        current_unit
    ]

    fig, ax = _create_figure(
        style
    )

    for curve in curves:

        ax.plot(
            curve["df"][
                "potential_V"
            ],
            (
                curve["df"][
                    "current_A"
                ]
                * scale
            ),
            label=curve[
                "label"
            ],
            color=curve[
                "color"
            ],
            linestyle=LINE_STYLE_MAP[
                curve.get(
                    "line_style",
                    "Solid"
                )
            ],
            linewidth=style.get(
                "line_width",
                1.5
            ),
        )

    ax.set_xlabel(
        "Potential (V)"
    )

    ax.set_ylabel(
        f"Current ({current_unit})"
    )

    _apply_axis_style(
        ax,
        style
    )

    _apply_axis_view(
        ax,
        style
    )

    if show_legend:

        ax.legend(
            loc=LEGEND_LOCATION_MAP.get(
                legend_location,
                "upper right"
            ),
            frameon=False,
        )

    return _finalize(
        fig,
        ax,
        style
    )


# =========================================================
# GCD
# =========================================================

def plot_gcd_publication(
    df,
    style,
    line_color="#1F77B4",
    line_style="Solid",
):

    fig, ax = _create_figure(
        style
    )

    ax.plot(
        df[
            "time_s"
        ],
        df[
            "potential_V"
        ],
        color=line_color,
        linestyle=LINE_STYLE_MAP[
            line_style
        ],
        linewidth=style.get(
            "line_width",
            1.5
        ),
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Potential (V)"
    )

    _apply_axis_style(
        ax,
        style
    )

    _apply_axis_view(
        ax,
        style
    )

    return _finalize(
        fig,
        ax,
        style
    )


def plot_gcd_multi_publication(
    curves,
    style,
    show_legend=True,
    legend_location="Upper right",
):

    fig, ax = _create_figure(
        style
    )

    for curve in curves:

        ax.plot(
            curve["df"][
                "time_s"
            ],
            curve["df"][
                "potential_V"
            ],
            label=curve[
                "label"
            ],
            color=curve[
                "color"
            ],
            linestyle=LINE_STYLE_MAP[
                curve.get(
                    "line_style",
                    "Solid"
                )
            ],
            linewidth=style.get(
                "line_width",
                1.5
            ),
        )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Potential (V)"
    )

    _apply_axis_style(
        ax,
        style
    )

    _apply_axis_view(
        ax,
        style
    )

    if show_legend:

        ax.legend(
            loc=LEGEND_LOCATION_MAP.get(
                legend_location,
                "upper right"
            ),
            frameon=False,
        )

    return _finalize(
        fig,
        ax,
        style
    )


# =========================================================
# EIS
# =========================================================

def plot_nyquist_publication(
    df,
    style,
    plot_mode="Points + line",
    line_color="#1F77B4",
):

    fig, ax = _create_figure(
        style
    )

    x = df[
        "z_real_ohm"
    ]

    y = -df[
        "z_imag_ohm"
    ]

    kwargs = {
        "color":
            line_color,

        "linewidth":
            style.get(
                "line_width",
                1.5
            ),
    }

    if plot_mode == "Points":

        kwargs.update(
            {
                "linestyle":
                    "None",

                "marker":
                    "o",

                "markersize":
                    style.get(
                        "marker_size",
                        4
                    ),

                "markerfacecolor":
                    "white",
            }
        )

    elif plot_mode == "Points + line":

        kwargs.update(
            {
                "marker":
                    "o",

                "markersize":
                    style.get(
                        "marker_size",
                        4
                    ),

                "markerfacecolor":
                    "white",
            }
        )

    ax.plot(
        x,
        y,
        **kwargs
    )

    ax.set_xlabel(
        "Z' (Ω)"
    )

    ax.set_ylabel(
        "-Z'' (Ω)"
    )

    _apply_axis_style(
        ax,
        style
    )

    _apply_axis_view(
        ax,
        style
    )

    if (
        style.get(
            "nyquist_aspect"
        )
        == "Equal X/Y scale"
    ):

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

    return _finalize(
        fig,
        ax,
        style
    )


def plot_bode_modulus_publication(
    df,
    style,
    line_color="#1F77B4",
):

    fig, ax = _create_figure(
        style
    )

    ax.plot(
        df[
            "frequency_Hz"
        ],
        df[
            "z_modulus_ohm"
        ],
        color=line_color,
        linewidth=style.get(
            "line_width",
            1.5
        ),
        marker="o",
        markersize=style.get(
            "marker_size",
            4
        ),
        markerfacecolor="white",
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

    return _finalize(
        fig,
        ax,
        style
    )


def plot_bode_phase_publication(
    df,
    style,
    line_color="#1F77B4",
):

    fig, ax = _create_figure(
        style
    )

    ax.plot(
        df[
            "frequency_Hz"
        ],
        df[
            "phase_deg"
        ],
        color=line_color,
        linewidth=style.get(
            "line_width",
            1.5
        ),
        marker="o",
        markersize=style.get(
            "marker_size",
            4
        ),
        markerfacecolor="white",
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
    )

    _apply_axis_view(
        ax,
        style,
        x_log=True,
    )

    return _finalize(
        fig,
        ax,
        style
    )


# =========================================================
# Export
# =========================================================

def figure_to_bytes(
    fig,
    file_format,
    dpi=600,
):

    file_format = (
        file_format.lower()
    )

    if file_format not in {
        "pdf",
        "svg",
        "png",
        "tiff",
    }:

        raise ValueError(
            "Unsupported export format."
        )

    buffer = io.BytesIO()

    kwargs = {
        "format":
            file_format,

        "facecolor":
            "white",

        "edgecolor":
            "white",
    }

    if file_format in {
        "png",
        "tiff",
    }:

        kwargs[
            "dpi"
        ] = dpi

    fig.savefig(
        buffer,
        **kwargs
    )

    buffer.seek(0)

    return buffer.getvalue()
