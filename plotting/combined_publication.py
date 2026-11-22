import matplotlib.pyplot as plt

from matplotlib.ticker import (
    AutoMinorLocator,
)


LINE_STYLE_MAP = {

    "Solid":
        "-",

    "Dashed":
        "--",

    "Dotted":
        ":",

    "Dash-dot":
        "-.",
}


# =========================================================
# Axis text
# =========================================================

def _axis_text(
    style,
    english,
    chinese,
):

    if (
        style.get(
            "figure_language"
        )
        ==
        "Chinese"
    ):

        return chinese

    return english


# =========================================================
# Axis style
# =========================================================

def _style_axis(
    ax,
    style,
):

    font = (
        style.get(
            "font_family",
            "Arial",
        )
    )

    label_size = (
        style.get(
            "label_font_size",
            9,
        )
    )

    tick_size = (
        style.get(
            "tick_font_size",
            8,
        )
    )

    axis_width = (
        style.get(
            "axis_line_width",
            0.8,
        )
    )

    ax.tick_params(
        which="both",
        direction=
            style.get(
                "tick_direction",
                "in",
            ),
        labelsize=
            tick_size,
        width=
            axis_width,
        top=
            style.get(
                "four_sided_frame",
                True,
            ),
        right=
            style.get(
                "four_sided_frame",
                True,
            ),
    )

    for label in (
        ax.get_xticklabels()
        +
        ax.get_yticklabels()
    ):

        label.set_fontfamily(
            font
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

    for spine in (
        ax.spines.values()
    ):

        spine.set_linewidth(
            axis_width
        )

        spine.set_color(
            "black"
        )

    if style.get(
        "minor_ticks",
        True,
    ):

        ax.xaxis.set_minor_locator(
            AutoMinorLocator(
                2
            )
        )

        ax.yaxis.set_minor_locator(
            AutoMinorLocator(
                2
            )
        )

    ax.grid(
        style.get(
            "grid",
            False,
        ),
        linewidth=0.4,
        alpha=0.35,
    )


# =========================================================
# Combined figure
# =========================================================

def plot_combined_electrochemistry(
    cv_df,
    gcd_df,
    eis_df,
    style,
    *,
    cv_color="#1F77B4",
    gcd_color="#D62728",
    eis_color="#2CA02C",
    cv_line_style="Solid",
    gcd_line_style="Solid",
    eis_line_style="Solid",
    current_unit="mA",
):

    width_in = (
        float(
            style.get(
                "width_mm",
                180.0,
            )
        )
        /
        25.4
    )

    height_in = (
        float(
            style.get(
                "height_mm",
                78.0,
            )
        )
        /
        25.4
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            width_in,
            height_in,
        ),
        facecolor="white",
    )

    font = (
        style.get(
            "font_family",
            "Arial",
        )
    )

    line_width = (
        style.get(
            "line_width",
            1.2,
        )
    )

    marker_size = (
        style.get(
            "marker_size",
            4.0,
        )
    )

    panel_size = max(
        8,
        int(
            style.get(
                "label_font_size",
                9,
            )
        ),
    )

    current_scale = {

        "A":
            1.0,

        "mA":
            1e3,

        "μA":
            1e6,

    }[
        current_unit
    ]


    # =====================================================
    # (a) CV
    # =====================================================

    ax = (
        axes[0]
    )

    ax.plot(
        cv_df[
            "potential_V"
        ],

        cv_df[
            "current_A"
        ]
        *
        current_scale,

        color=
            cv_color,

        linewidth=
            line_width,

        linestyle=
            LINE_STYLE_MAP.get(
                cv_line_style,
                "-",
            ),
    )

    ax.set_xlabel(
        _axis_text(
            style,
            "Potential (V)",
            "电位（V）",
        )
    )

    ax.set_ylabel(
        _axis_text(
            style,
            f"Current ({current_unit})",
            f"电流（{current_unit}）",
        )
    )

    _style_axis(
        ax,
        style,
    )


    # =====================================================
    # (b) GCD
    # =====================================================

    ax = (
        axes[1]
    )

    time = (
        gcd_df[
            "time_s"
        ].to_numpy(
            dtype=float
        )
    )

    if len(
        time
    ):

        time = (
            time
            -
            time[0]
        )

    ax.plot(
        time,

        gcd_df[
            "potential_V"
        ],

        color=
            gcd_color,

        linewidth=
            line_width,

        linestyle=
            LINE_STYLE_MAP.get(
                gcd_line_style,
                "-",
            ),
    )

    ax.set_xlabel(
        _axis_text(
            style,
            "Time (s)",
            "时间（s）",
        )
    )

    ax.set_ylabel(
        _axis_text(
            style,
            "Potential (V)",
            "电位（V）",
        )
    )

    _style_axis(
        ax,
        style,
    )


    # =====================================================
    # (c) EIS
    # =====================================================

    ax = (
        axes[2]
    )

    ax.plot(
        eis_df[
            "z_real_ohm"
        ],

        -eis_df[
            "z_imag_ohm"
        ],

        color=
            eis_color,

        linewidth=
            line_width,

        linestyle=
            LINE_STYLE_MAP.get(
                eis_line_style,
                "-",
            ),

        marker=
            "o",

        markersize=
            marker_size,

        markerfacecolor=
            "white",

        markeredgecolor=
            eis_color,

        markeredgewidth=
            0.8,
    )

    ax.set_xlabel(
        _axis_text(
            style,
            "Z' (Ω)",
            "阻抗实部 Z′（Ω）",
        )
    )

    ax.set_ylabel(
        _axis_text(
            style,
            "-Z'' (Ω)",
            "阻抗虚部 −Z″（Ω）",
        )
    )

    _style_axis(
        ax,
        style,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )


    # =====================================================
    # Panel labels
    # =====================================================

    for label, ax in zip(
        [
            "a",
            "b",
            "c",
        ],
        axes,
    ):

        ax.text(
            -0.17,
            1.04,
            label,

            transform=
                ax.transAxes,

            fontsize=
                panel_size,

            fontweight=
                "bold",

            fontstyle=
                "normal",

            fontfamily=
                font,

            va=
                "bottom",

            ha=
                "left",
        )


    # =====================================================
    # Physical canvas
    # =====================================================

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.22,
        top=0.92,
        wspace=0.42,
    )

    return fig
