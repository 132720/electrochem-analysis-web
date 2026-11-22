from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import (
    read_chi_txt,
)

from core.eis_analysis import (
    CIRCUIT_RC,
    CIRCUIT_CPE,
    CIRCUIT_CPE_W,
    SUPPORTED_CIRCUITS,
    estimate_rs_intercept,
    estimate_rct_initial,
    prepare_eis_fit_data,
    estimate_initial_parameters,
    fit_eis_equivalent_circuit,
    get_circuit_equation,
)

from core.experiment_grouping import (
    infer_temperature_from_filename,
)

from plotting.eis_publication import (
    plot_eis_nyquist,
    plot_eis_multi_nyquist,
    plot_eis_bode_magnitude,
    plot_eis_bode_phase,
    plot_equivalent_circuit,
)

from ui.common import (
    DEFAULT_COLORS,
    PLOTLY_CONFIG,
    export_buttons,
    legend_controls,
    publication_controls,
    select_line_style,
    show_metadata_cn,
)


# =========================================================
# Circuit labels
# =========================================================

CIRCUIT_LABELS = {

    CIRCUIT_RC:
        (
            "串联欧姆电阻 + "
            "电荷转移电阻/双电层电容并联"
            "（Rs + Rct∥Cdl）"
        ),

    CIRCUIT_CPE:
        (
            "串联欧姆电阻 + "
            "电荷转移电阻/恒相位元件并联"
            "（Rs + Rct∥CPE）"
        ),

    CIRCUIT_CPE_W:
        (
            "串联欧姆电阻 + "
            "电荷转移电阻/恒相位元件并联 + "
            "沃堡扩散（Warburg）"
            "（Rs + Rct∥CPE + W）"
        ),
}


STYLE_LABELS = {

    "Points":
        "仅散点",

    "Points + line":
        "散点 + 连线",

    "Line":
        "仅连线",
}


# =========================================================
# Interactive single EIS
# =========================================================

def _plot_eis_interactive(
    df,
    fit_result=None,
    experimental_style="Points + line",
    show_fit=False,
):

    mode = {

        "Points":
            "markers",

        "Line":
            "lines",

    }.get(
        experimental_style,
        "lines+markers",
    )

    fig = (
        go.Figure()
    )

    fig.add_trace(
        go.Scatter(
            x=
                df[
                    "z_real_ohm"
                ],

            y=
                -df[
                    "z_imag_ohm"
                ],

            mode=
                mode,

            name=
                "实验数据",

            marker=dict(
                size=7,
            ),

            line=dict(
                width=2,
            ),
        )
    )

    if (
        show_fit
        and
        fit_result
        is not None
    ):

        fitted = (
            fit_result[
                "fitted_data"
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=
                    fitted[
                        "z_real_fit_ohm"
                    ],

                y=
                    -fitted[
                        "z_imag_fit_ohm"
                    ],

                mode=
                    "lines",

                name=
                    "拟合曲线",

                line=dict(
                    width=2,
                ),
            )
        )

    fig.update_layout(
        template=
            "simple_white",

        xaxis_title=
            "阻抗实部 Z′（Ω）",

        yaxis_title=
            "阻抗虚部 −Z″（Ω）",

        hovermode=
            "closest",

        dragmode=
            "zoom",
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# =========================================================
# Interactive multiple EIS
# =========================================================



def _plot_multi_eis_interactive(
    curves,
    style="Points + line",
    show_fit=False,
):
    """
    Interactive Nyquist overlay.

    Raw Z'' sign is preserved in storage. Display uses -Z''.
    Fitted curves are optional and are never required for the raw-data plot.
    """

    mode = {
        "Points":
            "markers",
        "Line":
            "lines",
    }.get(
        style,
        "lines+markers",
    )

    dash_map = {
        "Solid":
            "solid",
        "Dashed":
            "dash",
        "Dotted":
            "dot",
        "Dash-dot":
            "dashdot",
    }

    fig = go.Figure()

    for curve in curves:
        fig.add_trace(
            go.Scatter(
                x=
                    curve[
                        "df"
                    ][
                        "z_real_ohm"
                    ],
                y=
                    -curve[
                        "df"
                    ][
                        "z_imag_ohm"
                    ],
                mode=
                    mode,
                name=
                    curve[
                        "label"
                    ],
                marker=dict(
                    size=7,
                    color=
                        curve[
                            "color"
                        ],
                ),
                line=dict(
                    width=2,
                    color=
                        curve[
                            "color"
                        ],
                    dash=
                        dash_map.get(
                            curve.get(
                                "line_style",
                                "Solid",
                            ),
                            "solid",
                        ),
                ),
            )
        )

        fit_result = curve.get(
            "fit_result"
        )

        if (
            show_fit
            and fit_result
            is not None
        ):
            fitted_df = (
                fit_result[
                    "fitted_data"
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=
                        fitted_df[
                            "z_real_fit_ohm"
                        ],
                    y=
                        -fitted_df[
                            "z_imag_fit_ohm"
                        ],
                    mode=
                        "lines",
                    name=
                        f"{curve['label']} fit",
                    line=dict(
                        width=2,
                        color=
                            curve[
                                "color"
                            ],
                        dash=
                            "dash",
                    ),
                    opacity=0.85,
                )
            )

    fig.update_layout(
        template=
            "simple_white",
        xaxis_title=
            "阻抗实部 Z′（Ω）",
        yaxis_title=
            "阻抗虚部 −Z″（Ω）",
        hovermode=
            "closest",
        dragmode=
            "zoom",
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def _show_fit_results(
    result,
):

    p = (
        result[
            "parameters"
        ]
    )

    st.subheader(
        "等效电路拟合结果"
    )

    c1, c2 = (
        st.columns(
            2
        )
    )

    c1.metric(
        "串联欧姆电阻（Rs）",
        f"{p['Rs']:.6g} Ω",
    )

    c2.metric(
        "电荷转移电阻（Rct）",
        f"{p['Rct']:.6g} Ω",
    )

    if (
        result[
            "circuit"
        ]
        ==
        CIRCUIT_RC
    ):

        st.metric(
            "双电层电容（Cdl）",
            f"{p['Cdl']:.6e} F",
        )

    else:

        c1, c2 = (
            st.columns(
                2
            )
        )

        c1.metric(
            "恒相位元件系数（Q）",
            f"{p['Q']:.6e} S·sⁿ",
        )

        c2.metric(
            "恒相位元件指数（n）",
            f"{p['n']:.5f}",
        )

    if (
        result[
            "circuit"
        ]
        ==
        CIRCUIT_CPE_W
    ):

        st.metric(
            "Warburg 扩散系数（σ）",
            f"{p['sigma']:.6e}",
        )

    c1, c2, c3 = (
        st.columns(
            3
        )
    )

    c1.metric(
        "均方根误差（RMSE）",
        f"{result['rmse_ohm']:.6g} Ω",
    )

    c2.metric(
        "相对均方根误差",
        f"{result['relative_rmse_pct']:.3f} %",
    )

    c3.metric(
        "拟合数据点数",
        result[
            "number_of_points"
        ],
    )


# =========================================================
# Single EIS
# =========================================================

def _single_eis(
    ec_data,
):

    show_metadata_cn(
        ec_data
    )

    rs = (
        estimate_rs_intercept(
            ec_data.data
        )
    )

    rct0 = (
        estimate_rct_initial(
            ec_data.data,
            rs,
        )
    )

    c1, c2 = (
        st.columns(
            2
        )
    )

    c1.metric(
        "自动估算串联欧姆电阻（Rs）",
        f"{rs:.6g} Ω",
    )

    c2.metric(
        "初始电荷转移电阻估计（Rct）",
        f"{rct0:.6g} Ω",
    )

    (
        fit_tab,
        interactive_tab,
        pub_tab,
    ) = st.tabs(
        [
            "等效电路拟合",
            "交互式图形",
            "出版级绘图",
        ]
    )

    circuit = (
        CIRCUIT_CPE
    )

    weighting = (
        "1/|Z|"
    )

    exclude_inductive = (
        True
    )

    fmin = None
    fmax = None

    current_signature = (
        None
    )


    # =====================================================
    # Fitting
    # =====================================================

    with fit_tab:

        circuit = st.selectbox(
            "等效电路模型",
            SUPPORTED_CIRCUITS,
            format_func=lambda x:
                CIRCUIT_LABELS.get(
                    x,
                    x,
                ),
            key=
                "eis_circuit",
        )

        st.caption(
            "模型表达式"
        )

        st.code(
            get_circuit_equation(
                circuit
            ),
            language=None,
        )

        c1, c2 = (
            st.columns(
                2
            )
        )

        exclude_inductive = (
            c1.checkbox(
                "排除感抗区数据（Z″ > 0）",
                True,
                key=
                    "eis_exclude",
            )
        )

        weighting = (
            c2.selectbox(
                "残差加权方式",

                [
                    "1/|Z|",
                    "None",
                ],

                format_func=lambda x:
                    (
                        "按阻抗模值倒数加权（1/|Z|）"
                        if
                        x == "1/|Z|"
                        else
                        "不加权"
                    ),

                key=
                    "eis_weight",
            )
        )

        manual_range = (
            st.checkbox(
                "手动设置拟合频率范围",
                False,
                key=
                    "eis_manual_range",
            )
        )

        if manual_range:

            c1, c2 = (
                st.columns(
                    2
                )
            )

            fmin = (
                c1.number_input(
                    "最低拟合频率（Hz）",
                    min_value=1e-12,
                    value=
                        float(
                            ec_data.data[
                                "frequency_Hz"
                            ].min()
                        ),
                    format=
                        "%.8g",
                    key=
                        "eis_fmin",
                )
            )

            fmax = (
                c2.number_input(
                    "最高拟合频率（Hz）",
                    min_value=1e-12,
                    value=
                        float(
                            ec_data.data[
                                "frequency_Hz"
                            ].max()
                        ),
                    format=
                        "%.8g",
                    key=
                        "eis_fmax",
                )
            )

        try:

            (
                fit_preview,
                _,
            ) = (
                prepare_eis_fit_data(
                    ec_data.data,

                    exclude_inductive=
                        exclude_inductive,

                    frequency_min_Hz=
                        fmin,

                    frequency_max_Hz=
                        fmax,
                )
            )

            initial = (
                estimate_initial_parameters(
                    fit_preview,
                    circuit,
                )
            )

            with st.expander(
                "自动初始参数"
            ):

                rows = []

                label_map = {

                    "Rs":
                        "串联欧姆电阻（Rs）",

                    "Rct":
                        "电荷转移电阻（Rct）",

                    "Cdl":
                        "双电层电容（Cdl）",

                    "Q":
                        "恒相位元件系数（Q）",

                    "n":
                        "恒相位元件指数（n）",

                    "sigma":
                        "Warburg 扩散系数（σ）",
                }

                for key, value in (
                    initial.items()
                ):

                    rows.append(
                        {
                            "参数":
                                label_map.get(
                                    key,
                                    key,
                                ),

                            "初始值":
                                value,
                        }
                    )

                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                )

        except Exception as exc:

            st.warning(
                f"拟合数据预处理提示：{exc}"
            )

        current_signature = (
            ec_data.filename,
            circuit,
            weighting,
            exclude_inductive,
            fmin,
            fmax,
        )

        if st.button(
            "开始等效电路拟合",
            type="primary",
            key=
                "eis_fit_button",
        ):

            try:

                result = (
                    fit_eis_equivalent_circuit(
                        df=
                            ec_data.data,

                        circuit=
                            circuit,

                        weighting=
                            weighting,

                        exclude_inductive=
                            exclude_inductive,

                        frequency_min_Hz=
                            fmin,

                        frequency_max_Hz=
                            fmax,
                    )
                )

                st.session_state[
                    "eis_fit_result_cn"
                ] = result

                st.session_state[
                    "eis_fit_signature_cn"
                ] = current_signature

                st.success(
                    "等效电路拟合完成。"
                )

            except Exception as exc:

                st.error(
                    "等效电路拟合失败。"
                )

                st.exception(
                    exc
                )

        fit_result = None

        if (
            st.session_state.get(
                "eis_fit_signature_cn"
            )
            ==
            current_signature
        ):

            fit_result = (
                st.session_state.get(
                    "eis_fit_result_cn"
                )
            )

        if fit_result:

            _show_fit_results(
                fit_result
            )


    # =====================================================
    # Strict fit signature
    # =====================================================

    fit_result = None

    if (
        current_signature
        is not None
        and
        st.session_state.get(
            "eis_fit_signature_cn"
        )
        ==
        current_signature
    ):

        fit_result = (
            st.session_state.get(
                "eis_fit_result_cn"
            )
        )


    # =====================================================
    # Interactive
    # =====================================================

    with interactive_tab:

        c1, c2 = (
            st.columns(
                2
            )
        )

        exp_style = (
            c1.selectbox(
                "实验曲线样式",
                list(
                    STYLE_LABELS.keys()
                ),
                index=1,
                format_func=lambda x:
                    STYLE_LABELS[x],
                key=
                    "eis_interactive_style",
            )
        )

        show_fit = (
            c2.checkbox(
                "显示拟合曲线",
                False,
                disabled=
                    (
                        fit_result
                        is None
                    ),
                key=
                    "eis_interactive_fit",
            )
        )

        st.plotly_chart(
            _plot_eis_interactive(
                ec_data.data,
                fit_result,
                exp_style,
                show_fit,
            ),
            width="stretch",
            config=
                PLOTLY_CONFIG,
        )


    # =====================================================
    # Publication
    # =====================================================

    with pub_tab:

        figure_type = st.radio(
            "图形类型",

            [
                "Nyquist",
                "Bode |Z|",
                "Bode Phase",
                "Equivalent Circuit",
            ],

            horizontal=True,

            format_func=lambda x: {

                "Nyquist":
                    "奈奎斯特图（Nyquist）",

                "Bode |Z|":
                    "波特阻抗模值图（Bode |Z|）",

                "Bode Phase":
                    "波特相位图（Bode Phase）",

                "Equivalent Circuit":
                    "等效电路图",

            }[
                x
            ],

            key=
                "eis_pub_type",
        )

        controls, preview = (
            st.columns(
                [
                    0.38,
                    0.62,
                ],
                gap="large",
            )
        )


        # =================================================
        # Equivalent Circuit
        # =================================================

        if (
            figure_type
            ==
            "Equivalent Circuit"
        ):

            with controls:

                style, dpi = (
                    publication_controls(
                        "eis_circuit_pub",
                        combined=True,
                    )
                )

                show_values = (
                    st.checkbox(
                        "显示拟合参数数值",
                        False,
                        disabled=
                            (
                                fit_result
                                is None
                            ),
                        key=
                            "eis_circuit_values",
                    )
                )

                selected_circuit = (
                    fit_result[
                        "circuit"
                    ]
                    if
                    fit_result
                    else
                    circuit
                )

            with preview:

                fig = (
                    plot_equivalent_circuit(
                        selected_circuit,
                        fit_result,
                        show_values,
                        style[
                            "width_mm"
                        ],
                        style[
                            "height_mm"
                        ],
                        style[
                            "font_family"
                        ],
                    )
                )

                st.pyplot(
                    fig,
                    width="stretch",
                )

                export_buttons(
                    fig,
                    (
                        f"{Path(ec_data.filename).stem}"
                        "_等效电路"
                    ),
                    dpi,
                    "eis_circuit_export",
                )

                plt.close(
                    fig
                )

            return


        # =================================================
        # Other EIS figures
        # =================================================

        with controls:

            if (
                figure_type
                ==
                "Nyquist"
            ):

                style, dpi = (
                    publication_controls(
                        "eis_nyquist_pub",

                        ec_data.data[
                            "z_real_ohm"
                        ],

                        -ec_data.data[
                            "z_imag_ohm"
                        ],

                        nyquist=True,
                    )
                )

            elif (
                figure_type
                ==
                "Bode |Z|"
            ):

                style, dpi = (
                    publication_controls(
                        "eis_bode_z_pub",

                        ec_data.data[
                            "frequency_Hz"
                        ],

                        ec_data.data[
                            "z_modulus_ohm"
                        ],

                        x_log=True,
                        y_log=True,
                    )
                )

            else:

                style, dpi = (
                    publication_controls(
                        "eis_bode_phase_pub",

                        ec_data.data[
                            "frequency_Hz"
                        ],

                        ec_data.data[
                            "phase_deg"
                        ],

                        x_log=True,
                    )
                )

            exp_style = (
                st.selectbox(
                    "实验曲线样式",
                    list(
                        STYLE_LABELS.keys()
                    ),
                    index=1,
                    format_func=lambda x:
                        STYLE_LABELS[x],
                    key=
                        f"{figure_type}_pub_style",
                )
            )

            exp_color = (
                st.color_picker(
                    "实验曲线颜色",
                    DEFAULT_COLORS[0],
                    key=
                        f"{figure_type}_pub_color",
                )
            )

            show_fit = (
                st.checkbox(
                    "显示拟合曲线",
                    False,
                    disabled=
                        (
                            fit_result
                            is None
                        ),
                    key=
                        f"{figure_type}_pub_fit",
                )
            )

            if show_fit:

                fit_color = (
                    st.color_picker(
                        "拟合曲线颜色",
                        "#D62728",
                        key=
                            f"{figure_type}_fit_color",
                    )
                )

            else:

                fit_color = (
                    "#D62728"
                )

            (
                show_legend,
                legend_location,
            ) = (
                legend_controls(
                    f"eis_{figure_type}",
                    False,
                )
            )

        with preview:

            if (
                figure_type
                ==
                "Nyquist"
            ):

                fig = (
                    plot_eis_nyquist(
                        ec_data.data,
                        style,
                        fit_result,
                        exp_style,
                        exp_color,
                        show_fit,
                        fit_color,
                        show_legend,
                        legend_location,
                    )
                )

                suffix = (
                    "Nyquist"
                )

            elif (
                figure_type
                ==
                "Bode |Z|"
            ):

                fig = (
                    plot_eis_bode_magnitude(
                        ec_data.data,
                        style,
                        fit_result,
                        exp_style,
                        exp_color,
                        show_fit,
                        fit_color,
                        show_legend,
                        legend_location,
                    )
                )

                suffix = (
                    "Bode_Z"
                )

            else:

                fig = (
                    plot_eis_bode_phase(
                        ec_data.data,
                        style,
                        fit_result,
                        exp_style,
                        exp_color,
                        show_fit,
                        fit_color,
                        show_legend,
                        legend_location,
                    )
                )

                suffix = (
                    "Bode_Phase"
                )

            st.pyplot(
                fig,
                width="stretch",
            )

            export_buttons(
                fig,
                (
                    f"{Path(ec_data.filename).stem}_"
                    f"{suffix}"
                ),
                dpi,
                f"eis_{suffix}_export",
            )

            plt.close(
                fig
            )


# =========================================================
# Multiple EIS
# =========================================================



def _multi_eis(
    datasets,
):
    """
    Multi-file EIS workflow with true batch fitting.

    Supports:
    - shared / individual / mixed fit parameters;
    - common or per-file temperature;
    - per-file labels, colors and line styles;
    - batch equivalent-circuit fitting;
    - per-file fit summary and CSV export;
    - optional interactive fit overlays;
    - publication raw Nyquist overlay.
    """

    st.info(
        "多文件 EIS 模式现在支持批量等效电路拟合。"
        "若比较同一体系的不同温度，通常建议优先使用相同的等效电路拓扑，"
        "再比较拟合参数随温度的变化。"
    )

    st.subheader(
        "1. 批量拟合参数"
    )

    mode = st.radio(
        "参数输入方式",
        [
            "所有文件使用相同参数",
            "每个文件单独设置",
            "混合模式（逐项决定）",
        ],
        horizontal=True,
        key=
            "eis_multi_parameter_mode",
    )

    if mode == "所有文件使用相同参数":
        scopes = {
            "circuit": "统一",
            "weighting": "统一",
            "exclude": "统一",
            "range": "统一",
            "temperature": "统一",
        }
    elif mode == "每个文件单独设置":
        scopes = {
            "circuit": "逐文件",
            "weighting": "逐文件",
            "exclude": "逐文件",
            "range": "逐文件",
            "temperature": "逐文件",
        }
    else:
        st.caption(
            "混合模式可保持同一等效电路，同时让不同文件使用不同频率范围或温度。"
        )

        c1, c2, c3, c4, c5 = st.columns(
            5
        )

        scopes = {
            "circuit":
                c1.selectbox(
                    "等效电路",
                    ["统一", "逐文件"],
                    key=
                        "eis_scope_circuit",
                ),
            "weighting":
                c2.selectbox(
                    "残差权重",
                    ["统一", "逐文件"],
                    key=
                        "eis_scope_weighting",
                ),
            "exclude":
                c3.selectbox(
                    "感抗排除",
                    ["统一", "逐文件"],
                    key=
                        "eis_scope_exclude",
                ),
            "range":
                c4.selectbox(
                    "拟合频率范围",
                    ["统一", "逐文件"],
                    key=
                        "eis_scope_range",
                ),
            "temperature":
                c5.selectbox(
                    "温度",
                    ["统一", "逐文件"],
                    key=
                        "eis_scope_temperature",
                ),
        }

    common = {}

    # Common frequency intersection is safest when one range is shared.
    file_mins = [
        float(
            data.data[
                "frequency_Hz"
            ].min()
        )
        for data
        in datasets
    ]

    file_maxs = [
        float(
            data.data[
                "frequency_Hz"
            ].max()
        )
        for data
        in datasets
    ]

    intersection_min = max(
        file_mins
    )
    intersection_max = min(
        file_maxs
    )

    if (
        intersection_min
        >= intersection_max
    ):
        intersection_min = min(
            file_mins
        )
        intersection_max = max(
            file_maxs
        )

    with st.expander(
        "统一拟合参数",
        expanded=(
            "统一"
            in scopes.values()
        ),
    ):
        if scopes[
            "circuit"
        ] == "统一":
            common[
                "circuit"
            ] = st.selectbox(
                "统一等效电路模型",
                SUPPORTED_CIRCUITS,
                index=
                    SUPPORTED_CIRCUITS.index(
                        CIRCUIT_CPE
                    ),
                format_func=lambda x:
                    CIRCUIT_LABELS.get(
                        x,
                        x,
                    ),
                key=
                    "eis_multi_common_circuit",
            )

            st.code(
                get_circuit_equation(
                    common[
                        "circuit"
                    ]
                ),
                language=None,
            )

        if scopes[
            "weighting"
        ] == "统一":
            common[
                "weighting"
            ] = st.selectbox(
                "统一残差加权",
                [
                    "1/|Z|",
                    "None",
                ],
                format_func=lambda x:
                    (
                        "按阻抗模值倒数加权（1/|Z|）"
                        if x == "1/|Z|"
                        else "不加权"
                    ),
                key=
                    "eis_multi_common_weighting",
            )

        if scopes[
            "exclude"
        ] == "统一":
            common[
                "exclude_inductive"
            ] = st.checkbox(
                "统一排除感抗区数据（Z″ > 0）",
                True,
                key=
                    "eis_multi_common_exclude",
            )

        if scopes[
            "range"
        ] == "统一":
            common[
                "manual_range"
            ] = st.checkbox(
                "所有文件使用同一手动拟合频率范围",
                False,
                key=
                    "eis_multi_common_manual_range",
            )

            if common[
                "manual_range"
            ]:
                c1, c2 = st.columns(
                    2
                )

                common[
                    "fmin"
                ] = c1.number_input(
                    "统一最低拟合频率（Hz）",
                    min_value=1e-12,
                    value=
                        float(
                            intersection_min
                        ),
                    format=
                        "%.8g",
                    key=
                        "eis_multi_common_fmin",
                )

                common[
                    "fmax"
                ] = c2.number_input(
                    "统一最高拟合频率（Hz）",
                    min_value=1e-12,
                    value=
                        float(
                            intersection_max
                        ),
                    format=
                        "%.8g",
                    key=
                        "eis_multi_common_fmax",
                )
            else:
                common[
                    "fmin"
                ] = None
                common[
                    "fmax"
                ] = None

        if scopes[
            "temperature"
        ] == "统一":
            common[
                "temperature_text"
            ] = st.text_input(
                "统一温度（°C，可选）",
                "",
                key=
                    "eis_multi_common_temperature",
            )

    st.subheader(
        "2. 文件参数"
    )

    curves = []
    config_rows = []

    for index, data in enumerate(
        datasets
    ):
        name = Path(
            data.filename
        ).stem

        guessed = (
            infer_temperature_from_filename(
                data.filename
            )
        )

        key_prefix = (
            f"eis_multi_file_{index}"
        )

        with st.expander(
            (
                f"{index + 1}. "
                f"{data.filename}"
            ),
            expanded=(
                index < 3
            ),
        ):
            # ---------------------------------------------
            # Temperature
            # ---------------------------------------------
            if scopes[
                "temperature"
            ] == "统一":
                temp_text = common.get(
                    "temperature_text",
                    "",
                )
                st.caption(
                    "温度使用统一设置。"
                )
            else:
                temp_text = st.text_input(
                    "温度（°C，可选）",
                    (
                        ""
                        if guessed
                        is None
                        else f"{guessed:g}"
                    ),
                    key=
                        f"{key_prefix}_temperature",
                )

            try:
                temperature = (
                    float(
                        temp_text
                    )
                    if str(
                        temp_text
                    ).strip()
                    else None
                )
            except ValueError:
                temperature = None
                st.warning(
                    "温度请输入数字。"
                )

            # ---------------------------------------------
            # Label
            # ---------------------------------------------
            label_mode = st.selectbox(
                "图例来源",
                [
                    "Temperature",
                    "Filename",
                    "Custom",
                ],
                index=
                    (
                        0
                        if temperature
                        is not None
                        else 1
                    ),
                format_func=lambda x: {
                    "Temperature":
                        "温度",
                    "Filename":
                        "文件名",
                    "Custom":
                        "自定义",
                }[
                    x
                ],
                key=
                    f"{key_prefix}_label_mode",
            )

            if (
                label_mode
                ==
                "Temperature"
                and temperature
                is not None
            ):
                label = (
                    f"{temperature:g} °C"
                )
            elif (
                label_mode
                ==
                "Custom"
            ):
                label = st.text_input(
                    "自定义图例名称",
                    name,
                    key=
                        f"{key_prefix}_custom_label",
                )
            else:
                label = name

            # ---------------------------------------------
            # Circuit
            # ---------------------------------------------
            if scopes[
                "circuit"
            ] == "统一":
                circuit = common.get(
                    "circuit",
                    CIRCUIT_CPE,
                )
                st.caption(
                    "等效电路："
                    f"{CIRCUIT_LABELS.get(circuit, circuit)}（统一）"
                )
            else:
                circuit = st.selectbox(
                    "等效电路模型",
                    SUPPORTED_CIRCUITS,
                    index=
                        SUPPORTED_CIRCUITS.index(
                            CIRCUIT_CPE
                        ),
                    format_func=lambda x:
                        CIRCUIT_LABELS.get(
                            x,
                            x,
                        ),
                    key=
                        f"{key_prefix}_circuit",
                )

                st.code(
                    get_circuit_equation(
                        circuit
                    ),
                    language=None,
                )

            # ---------------------------------------------
            # Weighting
            # ---------------------------------------------
            if scopes[
                "weighting"
            ] == "统一":
                weighting = common.get(
                    "weighting",
                    "1/|Z|",
                )
                st.caption(
                    f"残差权重：{weighting}（统一）"
                )
            else:
                weighting = st.selectbox(
                    "残差加权方式",
                    [
                        "1/|Z|",
                        "None",
                    ],
                    format_func=lambda x:
                        (
                            "按阻抗模值倒数加权（1/|Z|）"
                            if x == "1/|Z|"
                            else "不加权"
                        ),
                    key=
                        f"{key_prefix}_weighting",
                )

            # ---------------------------------------------
            # Inductive exclusion
            # ---------------------------------------------
            if scopes[
                "exclude"
            ] == "统一":
                exclude_inductive = bool(
                    common.get(
                        "exclude_inductive",
                        True,
                    )
                )
                st.caption(
                    "感抗区处理："
                    + (
                        "排除 Z″ > 0（统一）"
                        if exclude_inductive
                        else "保留全部数据（统一）"
                    )
                )
            else:
                exclude_inductive = st.checkbox(
                    "排除感抗区数据（Z″ > 0）",
                    True,
                    key=
                        f"{key_prefix}_exclude",
                )

            # ---------------------------------------------
            # Frequency range
            # ---------------------------------------------
            if scopes[
                "range"
            ] == "统一":
                fmin = common.get(
                    "fmin"
                )
                fmax = common.get(
                    "fmax"
                )

                if fmin is None:
                    st.caption(
                        "拟合频率范围：使用该文件全部有效频率（统一策略）"
                    )
                else:
                    st.caption(
                        "拟合频率范围："
                        f"{fmin:g}–{fmax:g} Hz（统一）"
                    )
            else:
                manual_range = st.checkbox(
                    "手动设置拟合频率范围",
                    False,
                    key=
                        f"{key_prefix}_manual_range",
                )

                if manual_range:
                    c1, c2 = st.columns(
                        2
                    )

                    fmin = c1.number_input(
                        "最低拟合频率（Hz）",
                        min_value=1e-12,
                        value=
                            float(
                                data.data[
                                    "frequency_Hz"
                                ].min()
                            ),
                        format=
                            "%.8g",
                        key=
                            f"{key_prefix}_fmin",
                    )

                    fmax = c2.number_input(
                        "最高拟合频率（Hz）",
                        min_value=1e-12,
                        value=
                            float(
                                data.data[
                                    "frequency_Hz"
                                ].max()
                            ),
                        format=
                            "%.8g",
                        key=
                            f"{key_prefix}_fmax",
                    )
                else:
                    fmin = None
                    fmax = None

            # ---------------------------------------------
            # Style
            # ---------------------------------------------
            color = st.color_picker(
                "曲线颜色",
                DEFAULT_COLORS[
                    index
                    %
                    len(
                        DEFAULT_COLORS
                    )
                ],
                key=
                    f"{key_prefix}_color",
            )

            line_style = (
                select_line_style(
                    "曲线线型",
                    f"{key_prefix}_line",
                )
            )

            try:
                rs_estimate = (
                    estimate_rs_intercept(
                        data.data
                    )
                )
            except Exception:
                rs_estimate = None

            try:
                rct_estimate = (
                    estimate_rct_initial(
                        data.data,
                        rs_estimate,
                    )
                )
            except Exception:
                rct_estimate = None

            curves.append(
                {
                    "index":
                        index,
                    "data_obj":
                        data,
                    "df":
                        data.data,
                    "filename":
                        data.filename,
                    "label":
                        label,
                    "color":
                        color,
                    "line_style":
                        line_style,
                    "temperature_C":
                        temperature,
                    "circuit":
                        circuit,
                    "weighting":
                        weighting,
                    "exclude_inductive":
                        exclude_inductive,
                    "fmin":
                        fmin,
                    "fmax":
                        fmax,
                    "rs_estimate":
                        rs_estimate,
                    "rct_estimate":
                        rct_estimate,
                }
            )

            config_rows.append(
                {
                    "文件":
                        data.filename,
                    "图例":
                        label,
                    "温度_C":
                        temperature,
                    "等效电路":
                        circuit,
                    "残差权重":
                        weighting,
                    "排除感抗区":
                        exclude_inductive,
                    "最低拟合频率_Hz":
                        fmin,
                    "最高拟合频率_Hz":
                        fmax,
                    "Rs初始估计_ohm":
                        rs_estimate,
                    "Rct初始估计_ohm":
                        rct_estimate,
                }
            )

    st.dataframe(
        pd.DataFrame(
            config_rows
        ),
        width="stretch",
        hide_index=True,
    )

    # -----------------------------------------------------
    # Batch fit
    # -----------------------------------------------------
    st.subheader(
        "3. 批量等效电路拟合"
    )

    signature = tuple(
        (
            curve[
                "filename"
            ],
            curve[
                "circuit"
            ],
            curve[
                "weighting"
            ],
            bool(
                curve[
                    "exclude_inductive"
                ]
            ),
            (
                None
                if curve[
                    "fmin"
                ]
                is None
                else float(
                    curve[
                        "fmin"
                    ]
                )
            ),
            (
                None
                if curve[
                    "fmax"
                ]
                is None
                else float(
                    curve[
                        "fmax"
                    ]
                )
            ),
        )
        for curve
        in curves
    )

    if st.button(
        "开始批量拟合所有 EIS 文件",
        type="primary",
        key=
            "eis_multi_fit_button",
    ):
        fit_results = {}
        fit_errors = {}

        progress = st.progress(
            0.0
        )

        for position, curve in enumerate(
            curves
        ):
            try:
                result = (
                    fit_eis_equivalent_circuit(
                        df=
                            curve[
                                "df"
                            ],
                        circuit=
                            curve[
                                "circuit"
                            ],
                        weighting=
                            curve[
                                "weighting"
                            ],
                        exclude_inductive=
                            curve[
                                "exclude_inductive"
                            ],
                        frequency_min_Hz=
                            curve[
                                "fmin"
                            ],
                        frequency_max_Hz=
                            curve[
                                "fmax"
                            ],
                    )
                )

                fit_results[
                    curve[
                        "index"
                    ]
                ] = result

            except Exception as exc:
                fit_errors[
                    curve[
                        "index"
                    ]
                ] = str(
                    exc
                )

            progress.progress(
                (
                    position
                    + 1
                )
                /
                len(
                    curves
                )
            )

        progress.empty()

        st.session_state[
            "eis_multi_fit_state"
        ] = {
            "signature":
                signature,
            "results":
                fit_results,
            "errors":
                fit_errors,
        }

        st.success(
            "批量拟合完成。"
        )

    fit_state = st.session_state.get(
        "eis_multi_fit_state"
    )

    if (
        fit_state
        is not None
        and fit_state.get(
            "signature"
        )
        ==
        signature
    ):
        fit_results = fit_state.get(
            "results",
            {},
        )
        fit_errors = fit_state.get(
            "errors",
            {},
        )
    else:
        fit_results = {}
        fit_errors = {}

    summary_rows = []

    for curve in curves:
        index = curve[
            "index"
        ]

        row = {
            "文件":
                curve[
                    "filename"
                ],
            "图例":
                curve[
                    "label"
                ],
            "温度_C":
                curve[
                    "temperature_C"
                ],
            "等效电路":
                curve[
                    "circuit"
                ],
            "Rs初始估计_ohm":
                curve[
                    "rs_estimate"
                ],
            "Rct初始估计_ohm":
                curve[
                    "rct_estimate"
                ],
        }

        result = fit_results.get(
            index
        )

        curve[
            "fit_result"
        ] = result

        if result is not None:
            parameters = result[
                "parameters"
            ]

            for key, value in (
                parameters.items()
            ):
                row[
                    key
                ] = value

            row[
                "RMSE_ohm"
            ] = result[
                "rmse_ohm"
            ]

            row[
                "Relative_RMSE_pct"
            ] = result[
                "relative_rmse_pct"
            ]

            row[
                "拟合点数"
            ] = result[
                "number_of_points"
            ]

            row[
                "状态"
            ] = (
                "拟合完成"
                if result[
                    "success"
                ]
                else "拟合器未正常收敛"
            )

        elif index in fit_errors:
            row[
                "状态"
            ] = (
                "拟合失败："
                + fit_errors[
                    index
                ]
            )
        else:
            row[
                "状态"
            ] = "尚未执行批量拟合"

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.dataframe(
        summary_df,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "下载 EIS 批量拟合结果（CSV）",
        data=
            summary_df.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),
        file_name=
            "EIS_batch_fit_results.csv",
        mime=
            "text/csv",
        key=
            "eis_multi_result_download",
    )

    if fit_results:
        with st.expander(
            "查看各文件详细拟合参数",
            expanded=False,
        ):
            for curve in curves:
                result = fit_results.get(
                    curve[
                        "index"
                    ]
                )

                if result is None:
                    continue

                st.markdown(
                    f"### {curve['filename']}"
                )

                _show_fit_results(
                    result
                )

                st.divider()

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    st.subheader(
        "4. 多文件 EIS 比较"
    )

    tab1, tab2 = st.tabs(
        [
            "交互式叠图",
            "出版级叠图",
        ]
    )

    with tab1:
        exp_style = st.selectbox(
            "实验曲线样式",
            list(
                STYLE_LABELS.keys()
            ),
            index=1,
            format_func=lambda x:
                STYLE_LABELS[
                    x
                ],
            key=
                "eis_multi_interactive_style",
        )

        show_fit = st.checkbox(
            "显示批量拟合曲线",
            value=False,
            disabled=(
                not bool(
                    fit_results
                )
            ),
            key=
                "eis_multi_show_fit",
        )

        st.plotly_chart(
            _plot_multi_eis_interactive(
                curves,
                exp_style,
                show_fit=
                    show_fit,
            ),
            width="stretch",
            config=
                PLOTLY_CONFIG,
        )

    with tab2:
        st.caption(
            "出版级多文件 Nyquist 默认只绘制实验数据；"
            "拟合参数和拟合优度请使用上方批量结果表。"
        )

        controls, preview = (
            st.columns(
                [
                    0.38,
                    0.62,
                ],
                gap="large",
            )
        )

        with controls:
            all_x = np.concatenate(
                [
                    curve[
                        "df"
                    ][
                        "z_real_ohm"
                    ].to_numpy()
                    for curve
                    in curves
                ]
            )

            all_y = np.concatenate(
                [
                    -curve[
                        "df"
                    ][
                        "z_imag_ohm"
                    ].to_numpy()
                    for curve
                    in curves
                ]
            )

            style, dpi = (
                publication_controls(
                    "eis_multi_pub",
                    all_x,
                    all_y,
                    nyquist=True,
                )
            )

            exp_style = st.selectbox(
                "实验曲线样式",
                list(
                    STYLE_LABELS.keys()
                ),
                index=1,
                format_func=lambda x:
                    STYLE_LABELS[
                        x
                    ],
                key=
                    "eis_multi_pub_style",
            )

            (
                show_legend,
                legend_location,
            ) = legend_controls(
                "eis_multi",
                True,
            )

        with preview:
            fig = (
                plot_eis_multi_nyquist(
                    curves,
                    style,
                    exp_style,
                    show_legend,
                    legend_location,
                )
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            export_buttons(
                fig,
                "EIS_Nyquist_多文件叠图",
                dpi,
                "eis_multi_export",
            )

            plt.close(
                fig
            )



def render_eis_screen():
    st.header(
        "电化学阻抗谱分析（EIS）"
    )

    st.caption(
        "支持一次上传多个 EIS TXT 文件。"
        "单文件保留原有等效电路拟合与出版绘图；"
        "多文件支持统一/独立拟合参数、批量拟合、温度设置、"
        "结果汇总、CSV 导出与 Nyquist 叠图。"
    )

    uploaded = st.file_uploader(
        "上传一个或多个电化学阻抗谱 TXT 文件",
        type=[
            "txt"
        ],
        accept_multiple_files=True,
        key=
            "eis_uploader",
    )

    if not uploaded:
        st.info(
            "请上传电化学阻抗谱（EIS）TXT 文件。"
        )
        return

    datasets = []
    read_errors = []

    for file in uploaded:
        try:
            data = read_chi_txt(
                file
            )

            if data.technique != "EIS":
                read_errors.append(
                    (
                        file.name,
                        "文件不是 EIS 类型",
                    )
                )
                continue

            datasets.append(
                data
            )

        except Exception as exc:
            read_errors.append(
                (
                    file.name,
                    str(
                        exc
                    ),
                )
            )

    if read_errors:
        with st.expander(
            "部分文件未载入",
            expanded=True,
        ):
            for filename, message in read_errors:
                st.warning(
                    f"{filename}：{message}"
                )

    if not datasets:
        st.error(
            "没有成功载入可分析的 EIS 文件。"
        )
        return

    st.success(
        f"已成功载入 {len(datasets)} 个 EIS 文件。"
    )

    if len(
        datasets
    ) == 1:
        _single_eis(
            datasets[0]
        )
    else:
        _multi_eis(
            datasets
        )


