from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from plotting.publication import (
    figure_to_bytes,
)

from ui.publication_presets import (
    PUBLICATION_PRESETS,
    DPI_OPTIONS,
)


# =========================================================
# Plotly
# =========================================================

PLOTLY_CONFIG = {

    "scrollZoom":
        False,

    "displayModeBar":
        True,

    "displaylogo":
        False,

    "responsive":
        True,

    "modeBarButtonsToRemove": [
        "zoomIn2d",
        "zoomOut2d",
        "select2d",
        "lasso2d",
    ],
}


# =========================================================
# Default colors
# =========================================================

DEFAULT_COLORS = [
    "#1F77B4",
    "#D62728",
    "#2CA02C",
    "#9467BD",
    "#FF7F0E",
    "#17BECF",
    "#8C564B",
    "#E377C2",
]


# =========================================================
# Line style
# =========================================================

LINE_STYLE_OPTIONS = [
    "Solid",
    "Dashed",
    "Dotted",
    "Dash-dot",
]


LINE_STYLE_CN = {

    "Solid":
        "实线",

    "Dashed":
        "虚线",

    "Dotted":
        "点线",

    "Dash-dot":
        "点划线",
}


# =========================================================
# Legend
# =========================================================

LEGEND_LOCATIONS = [
    "Upper right",
    "Upper left",
    "Lower right",
    "Lower left",
    "Best",
]


LEGEND_LOCATION_CN = {

    "Upper right":
        "右上",

    "Upper left":
        "左上",

    "Lower right":
        "右下",

    "Lower left":
        "左下",

    "Best":
        "自动最佳位置",
}


# =========================================================
# Metadata labels
# =========================================================

METADATA_LABELS = {

    "initial_potential_V":
        "初始电位（V）",

    "high_potential_V":
        "最高电位（V）",

    "low_potential_V":
        "最低电位（V）",

    "scan_rate_V_s":
        "扫描速率（V/s）",

    "segment_count":
        "程序段数",

    "sample_interval_V":
        "采样电位间隔（V）",

    "quiet_time_s":
        "静置时间（s）",

    "cathodic_current_A":
        "阴极电流（A）",

    "anodic_current_A":
        "阳极电流（A）",

    "data_storage_interval_s":
        "数据存储间隔（s）",

    "high_potential_limit_V":
        "高电位限制（V）",

    "low_potential_limit_V":
        "低电位限制（V）",

    "cathodic_time_s":
        "阴极时间（s）",

    "anodic_time_s":
        "阳极时间（s）",

    "high_frequency_Hz":
        "最高频率（Hz）",

    "low_frequency_Hz":
        "最低频率（Hz）",

    "amplitude_V":
        "交流扰动幅值（V）",
}


# =========================================================
# Result labels
# =========================================================

RESULT_LABELS = {

    "specific_capacitance_F_g":
        "比电容（Csp，F/g）",

    "device_specific_capacitance_F_g":
        "器件质量基准比电容（F/g）",

    "energy_density_Wh_kg":
        "能量密度（E，Wh/kg）",

    "power_density_W_kg":
        "功率密度（P，W/kg）",

    "discharge_time_s":
        "放电时间（Δt，s）",

    "discharge_voltage_window_V":
        "放电电压窗口（ΔV，V）",

    "ir_drop_V":
        "初始电压降（IR drop，V）",

    "slope_V_s":
        "放电斜率（V/s）",

    "r_squared":
        "线性拟合优度（R²）",

    "rmse_ohm":
        "均方根误差（RMSE，Ω）",

    "relative_rmse_pct":
        "相对均方根误差（%）",

    "number_of_points":
        "拟合数据点数",

    "integral_A_V":
        "电流-电位积分（A·V）",

    "absolute_integral_A_V":
        "绝对电流-电位积分（A·V）",

    "potential_window_V":
        "电位窗口（ΔV，V）",

    "oxidation_peak_current_A":
        "氧化峰电流（A）",

    "oxidation_peak_potential_V":
        "氧化峰电位（V）",

    "reduction_peak_current_A":
        "还原峰电流（A）",

    "reduction_peak_potential_V":
        "还原峰电位（V）",
}


# =========================================================
# Technique
# =========================================================

def technique_cn(
    technique: str,
) -> str:

    return {

        "CV":
            "循环伏安法（CV）",

        "GCD":
            "恒流充放电（GCD）",

        "EIS":
            "电化学阻抗谱（EIS）",

    }.get(
        technique,
        technique,
    )


# =========================================================
# Range helpers
# =========================================================

def safe_range(
    values,
    padding=0.05,
):

    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(
            arr
        )
    ]

    if len(
        arr
    ) == 0:

        return (
            0.0,
            1.0,
        )

    low = float(
        np.min(
            arr
        )
    )

    high = float(
        np.max(
            arr
        )
    )

    span = (
        high
        -
        low
    )

    if span <= 0:

        span = max(
            abs(
                high
            ),
            1.0,
        )

    extra = (
        span
        *
        padding
    )

    return (
        low - extra,
        high + extra,
    )


def positive_range(
    values,
):

    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(
            arr
        )
        &
        (
            arr > 0
        )
    ]

    if len(
        arr
    ) == 0:

        return (
            1e-3,
            1.0,
        )

    return (
        float(
            np.min(
                arr
            )
        ),
        float(
            np.max(
                arr
            )
        ),
    )


# =========================================================
# Current unit
# =========================================================

def current_scale(
    unit: str,
) -> float:

    return {

        "A":
            1.0,

        "mA":
            1e3,

        "μA":
            1e6,

    }[
        unit
    ]


# =========================================================
# Normalize GCD time
# =========================================================

def normalize_time(
    df,
):

    out = (
        df.copy()
    )

    if len(
        out
    ):

        out[
            "time_s"
        ] = (
            out[
                "time_s"
            ]
            -
            out[
                "time_s"
            ].iloc[0]
        )

    return out


# =========================================================
# Metadata UI
# =========================================================

def show_metadata_cn(
    ec_data,
):

    if not ec_data.metadata:

        return

    rows = []

    for key, value in (
        ec_data.metadata.items()
    ):

        rows.append(
            {
                "参数":
                    METADATA_LABELS.get(
                        key,
                        "实验参数",
                    ),

                "数值":
                    value,
            }
        )

    with st.expander(
        "实验信息"
    ):

        st.dataframe(
            pd.DataFrame(
                rows
            ),
            width="stretch",
            hide_index=True,
        )


# =========================================================
# Result UI
# =========================================================

def show_result_dict_cn(
    result: dict,
):

    rows = []

    for key, label in (
        RESULT_LABELS.items()
    ):

        if (
            key in result
            and
            result[key]
            is not None
        ):

            value = (
                result[
                    key
                ]
            )

            if isinstance(
                value,
                float,
            ):

                value = (
                    f"{value:.6g}"
                )

            rows.append(
                {
                    "参数":
                        label,

                    "数值":
                        value,
                }
            )

    if rows:

        st.dataframe(
            pd.DataFrame(
                rows
            ),
            width="stretch",
            hide_index=True,
        )


# =========================================================
# Line style UI
# =========================================================

def select_line_style(
    label: str,
    key: str,
    index: int = 0,
):

    return st.selectbox(
        label,
        LINE_STYLE_OPTIONS,
        index=index,
        format_func=lambda x:
            LINE_STYLE_CN[x],
        key=key,
    )


# =========================================================
# Legend UI
# =========================================================

def legend_controls(
    prefix: str,
    default_show=True,
):

    show = st.checkbox(
        "显示图例",
        value=default_show,
        key=
            f"{prefix}_show_legend",
    )

    location = st.selectbox(
        "图例位置",
        LEGEND_LOCATIONS,
        format_func=lambda x:
            LEGEND_LOCATION_CN[x],
        key=
            f"{prefix}_legend_location",
    )

    return (
        show,
        location,
    )


# =========================================================
# Publication controls
# =========================================================

def publication_controls(
    prefix: str,
    x_values=None,
    y_values=None,
    *,
    x_log=False,
    y_log=False,
    nyquist=False,
    combined=False,
):

    st.subheader(
        "出版级绘图设置"
    )

    # =====================================================
    # Preset
    # =====================================================

    preset_name = st.selectbox(
        "出版预设",
        (
            list(
                PUBLICATION_PRESETS.keys()
            )
            +
            [
                "自定义"
            ]
        ),
        key=
            f"{prefix}_preset_name",
    )

    if preset_name == "自定义":

        preset = (
            PUBLICATION_PRESETS[
                "通用 SCI 推荐"
            ].copy()
        )

        c1, c2 = (
            st.columns(
                2
            )
        )

        width_mm = (
            c1.number_input(
                "图宽（mm）",
                40.0,
                300.0,
                90.0,
                key=
                    f"{prefix}_custom_width",
            )
        )

        height_mm = (
            c2.number_input(
                "图高（mm）",
                40.0,
                300.0,
                70.0,
                key=
                    f"{prefix}_custom_height",
            )
        )

    else:

        preset = (
            PUBLICATION_PRESETS[
                preset_name
            ]
        )

        if (
            preset.get(
                "note",
                ""
            ).strip()
        ):

            st.caption(
                preset[
                    "note"
                ]
            )

        size_label = st.selectbox(
            "版面尺寸",

            [
                item[0]
                for item
                in preset[
                    "sizes"
                ]
            ],

            key=
                f"{prefix}_{preset_name}_size",
        )

        selected = next(
            item
            for item
            in preset[
                "sizes"
            ]
            if
            item[0]
            ==
            size_label
        )

        width_mm = (
            selected[1]
        )

        height_mm = (
            selected[2]
        )

        if combined:

            height_mm = max(
                height_mm,
                (
                    62.0
                    if
                    width_mm < 120
                    else
                    78.0
                ),
            )


    # =====================================================
    # Axis range
    # =====================================================

    x_mode = (
        "Auto"
    )

    y_mode = (
        "Auto"
    )

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    x_padding = 5.0
    y_padding = 8.0

    if (
        x_values is not None
        and
        y_values is not None
    ):

        with st.expander(
            "坐标轴范围",
            expanded=True,
        ):

            c1, c2 = (
                st.columns(
                    2
                )
            )

            x_mode = c1.selectbox(
                "X 轴范围",
                [
                    "Auto",
                    "Manual",
                ],
                format_func=lambda x:
                    (
                        "自动"
                        if x == "Auto"
                        else
                        "手动"
                    ),
                key=
                    f"{prefix}_x_mode",
            )

            y_mode = c2.selectbox(
                "Y 轴范围",
                [
                    "Auto",
                    "Manual",
                ],
                format_func=lambda x:
                    (
                        "自动"
                        if x == "Auto"
                        else
                        "手动"
                    ),
                key=
                    f"{prefix}_y_mode",
            )

            if x_log:

                dx0, dx1 = (
                    positive_range(
                        x_values
                    )
                )

            else:

                dx0, dx1 = (
                    safe_range(
                        x_values,
                        0.05,
                    )
                )

            if y_log:

                dy0, dy1 = (
                    positive_range(
                        y_values
                    )
                )

            else:

                dy0, dy1 = (
                    safe_range(
                        y_values,
                        0.08,
                    )
                )

            if x_mode == "Manual":

                c1, c2 = (
                    st.columns(
                        2
                    )
                )

                x_min = (
                    c1.number_input(
                        "X 轴最小值",
                        value=
                            float(
                                dx0
                            ),
                        format=
                            "%.8g",
                        key=
                            f"{prefix}_xmin",
                    )
                )

                x_max = (
                    c2.number_input(
                        "X 轴最大值",
                        value=
                            float(
                                dx1
                            ),
                        format=
                            "%.8g",
                        key=
                            f"{prefix}_xmax",
                    )
                )

            if y_mode == "Manual":

                c1, c2 = (
                    st.columns(
                        2
                    )
                )

                y_min = (
                    c1.number_input(
                        "Y 轴最小值",
                        value=
                            float(
                                dy0
                            ),
                        format=
                            "%.8g",
                        key=
                            f"{prefix}_ymin",
                    )
                )

                y_max = (
                    c2.number_input(
                        "Y 轴最大值",
                        value=
                            float(
                                dy1
                            ),
                        format=
                            "%.8g",
                        key=
                            f"{prefix}_ymax",
                    )
                )

            c1, c2 = (
                st.columns(
                    2
                )
            )

            x_padding = (
                c1.number_input(
                    "X 轴留白（%）",
                    0.0,
                    50.0,
                    5.0,
                    disabled=
                        (
                            x_mode
                            ==
                            "Manual"
                        ),
                    key=
                        f"{prefix}_xpad",
                )
            )

            y_padding = (
                c2.number_input(
                    "Y 轴留白（%）",
                    0.0,
                    50.0,
                    8.0,
                    disabled=
                        (
                            y_mode
                            ==
                            "Manual"
                        ),
                    key=
                        f"{prefix}_ypad",
                )
            )


    # =====================================================
    # Font / lines
    # =====================================================

    with st.expander(
        "字体、线条与刻度"
    ):

        default_font = (
            preset[
                "font_family"
            ]
        )

        fonts = [
            "Arial",
            "Times New Roman",
            "DejaVu Sans",
        ]

        if default_font not in fonts:

            default_font = (
                "Arial"
            )

        font_family = st.selectbox(
            "字体",
            fonts,
            index=
                fonts.index(
                    default_font
                ),
            key=
                f"{prefix}_{preset_name}_font",
        )

        c1, c2, c3 = (
            st.columns(
                3
            )
        )

        label_font_size = (
            c1.number_input(
                "坐标轴标题字号（pt）",
                5,
                20,
                int(
                    preset[
                        "label_font_size"
                    ]
                ),
                key=
                    f"{prefix}_{preset_name}_label_font",
            )
        )

        tick_font_size = (
            c2.number_input(
                "刻度字号（pt）",
                5,
                20,
                int(
                    preset[
                        "tick_font_size"
                    ]
                ),
                key=
                    f"{prefix}_{preset_name}_tick_font",
            )
        )

        legend_font_size = (
            c3.number_input(
                "图例字号（pt）",
                5,
                20,
                int(
                    preset[
                        "legend_font_size"
                    ]
                ),
                key=
                    f"{prefix}_{preset_name}_legend_font",
            )
        )

        c1, c2, c3 = (
            st.columns(
                3
            )
        )

        line_width = (
            c1.number_input(
                "曲线宽度（pt）",
                0.2,
                5.0,
                float(
                    preset[
                        "line_width"
                    ]
                ),
                0.1,
                key=
                    f"{prefix}_{preset_name}_line_width",
            )
        )

        axis_line_width = (
            c2.number_input(
                "坐标轴宽度（pt）",
                0.2,
                3.0,
                float(
                    preset[
                        "axis_line_width"
                    ]
                ),
                0.1,
                key=
                    f"{prefix}_{preset_name}_axis_width",
            )
        )

        marker_size = (
            c3.number_input(
                "数据点大小（pt）",
                1.0,
                12.0,
                float(
                    preset[
                        "marker_size"
                    ]
                ),
                0.5,
                key=
                    f"{prefix}_{preset_name}_marker",
            )
        )

        c1, c2, c3 = (
            st.columns(
                3
            )
        )

        minor_ticks = (
            c1.checkbox(
                "显示次刻度",
                True,
                key=
                    f"{prefix}_minor_ticks",
            )
        )

        four_sided = (
            c2.checkbox(
                "四边框",
                True,
                key=
                    f"{prefix}_four_sided",
            )
        )

        grid = (
            c3.checkbox(
                "显示网格",
                False,
                key=
                    f"{prefix}_grid",
            )
        )


    # =====================================================
    # Nyquist
    # =====================================================

    nyquist_aspect = (
        "Free aspect"
    )

    if nyquist:

        nyquist_aspect = (
            st.selectbox(
                "Nyquist 坐标比例",
                [
                    "Equal X/Y scale",
                    "Free aspect",
                ],
                format_func=lambda x:
                    (
                        "X/Y 等比例（推荐）"
                        if
                        x
                        ==
                        "Equal X/Y scale"
                        else
                        "自由比例"
                    ),
                key=
                    f"{prefix}_aspect",
            )
        )


    # =====================================================
    # DPI
    # =====================================================

    default_dpi = int(
        preset[
            "dpi"
        ]
    )

    if default_dpi in DPI_OPTIONS:

        dpi_index = (
            DPI_OPTIONS.index(
                default_dpi
            )
        )

    else:

        dpi_index = (
            len(
                DPI_OPTIONS
            )
            -
            1
        )

    dpi = st.selectbox(
        "位图导出分辨率（dpi）",
        DPI_OPTIONS,
        index=dpi_index,
        key=
            f"{prefix}_{preset_name}_dpi",
    )


    # =====================================================
    # Final style
    # =====================================================

    style = {

        "width_mm":
            float(
                width_mm
            ),

        "height_mm":
            float(
                height_mm
            ),

        "font_family":
            font_family,

        "label_font_size":
            int(
                label_font_size
            ),

        "tick_font_size":
            int(
                tick_font_size
            ),

        "legend_font_size":
            int(
                legend_font_size
            ),

        "line_width":
            float(
                line_width
            ),

        "axis_line_width":
            float(
                axis_line_width
            ),

        "marker_size":
            float(
                marker_size
            ),

        "tick_direction":
            "in",

        "four_sided_frame":
            bool(
                four_sided
            ),

        "minor_ticks":
            bool(
                minor_ticks
            ),

        "minor_subdivisions":
            2,

        "grid":
            bool(
                grid
            ),

        "x_axis_mode":
            x_mode,

        "y_axis_mode":
            y_mode,

        "x_min":
            x_min,

        "x_max":
            x_max,

        "y_min":
            y_min,

        "y_max":
            y_max,

        "x_padding_pct":
            float(
                x_padding
            ),

        "y_padding_pct":
            float(
                y_padding
            ),

        "x_major_interval":
            None,

        "y_major_interval":
            None,

        "nyquist_aspect":
            nyquist_aspect,

        "figure_language":
            "English",

        "publication_preset":
            preset_name,
    }

    return (
        style,
        int(
            dpi
        ),
    )


# =========================================================
# Export
# =========================================================

def export_buttons(
    fig,
    filename: str,
    dpi: int,
    prefix: str,
):

    filename = (
        Path(
            filename
        ).stem
    )

    c1, c2 = (
        st.columns(
            2
        )
    )

    c1.download_button(
        "导出 PDF（矢量）",
        figure_to_bytes(
            fig,
            "pdf",
            dpi,
        ),
        file_name=
            f"{filename}.pdf",
        key=
            f"{prefix}_pdf",
        width="stretch",
    )

    c2.download_button(
        "导出 SVG（矢量）",
        figure_to_bytes(
            fig,
            "svg",
            dpi,
        ),
        file_name=
            f"{filename}.svg",
        key=
            f"{prefix}_svg",
        width="stretch",
    )

    c1, c2 = (
        st.columns(
            2
        )
    )

    c1.download_button(
        f"导出 PNG（{dpi} dpi）",
        figure_to_bytes(
            fig,
            "png",
            dpi,
        ),
        file_name=
            f"{filename}.png",
        key=
            f"{prefix}_png",
        width="stretch",
    )

    c2.download_button(
        f"导出 TIFF（{dpi} dpi）",
        figure_to_bytes(
            fig,
            "tiff",
            dpi,
        ),
        file_name=
            f"{filename}.tiff",
        key=
            f"{prefix}_tiff",
        width="stretch",
    )
