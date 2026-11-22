from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import (
    read_chi_txt,
)

from core.gcd_analysis import (
    detect_gcd_segments,
    build_gcd_cycles,
    get_gcd_cycle,
    get_last_complete_gcd_cycle,
    get_discharge_segment,
    get_default_discharge_current_A,
    calculate_gcd_parameters,
)

from plotting.publication import (
    plot_gcd_publication,
    plot_gcd_multi_publication,
)

from ui.common import (
    DEFAULT_COLORS,
    PLOTLY_CONFIG,
    export_buttons,
    legend_controls,
    normalize_time,
    publication_controls,
    select_line_style,
    show_metadata_cn,
    show_result_dict_cn,
)


# =========================================================
# Interactive GCD
# =========================================================

def _interactive_gcd(
    curves,
):

    fig = (
        go.Figure()
    )

    for curve in curves:

        fig.add_trace(
            go.Scatter(
                x=
                    curve[
                        "df"
                    ][
                        "time_s"
                    ],

                y=
                    curve[
                        "df"
                    ][
                        "potential_V"
                    ],

                mode=
                    "lines",

                name=
                    curve[
                        "label"
                    ],

                line=dict(
                    color=
                        curve[
                            "color"
                        ],

                    width=2,
                ),
            )
        )

    fig.update_layout(
        template=
            "simple_white",

        xaxis_title=
            "时间（s）",

        yaxis_title=
            "电位（V）",

        hovermode=
            "closest",

        dragmode=
            "zoom",
    )

    return fig


# =========================================================
# Cycle selection
# =========================================================

def _select_gcd_data(
    ec_data,
    expected,
    key_prefix,
):

    cycles = (
        build_gcd_cycles(
            ec_data.data,
            expected,
        )
    )

    if not cycles:

        st.error(
            "未检测到完整的恒流充放电循环。"
        )

        return (
            None,
            None,
        )

    options = (
        [
            "最后一个完整循环"
        ]
        +
        [
            f"第 {item['cycle']} 个循环"
            for item
            in cycles
        ]
        +
        [
            "全部循环"
        ]
    )

    selected = st.selectbox(
        "显示循环",
        options,
        key=
            f"{key_prefix}_cycle",
    )

    if (
        selected
        ==
        "最后一个完整循环"
    ):

        return (
            get_last_complete_gcd_cycle(
                ec_data.data,
                expected,
            )
        )

    if (
        selected
        ==
        "全部循环"
    ):

        return (
            ec_data.data,
            None,
        )

    cycle_number = int(
        selected.split()[1]
    )

    return (
        get_gcd_cycle(
            ec_data.data,
            cycle_number,
            expected,
        )
    )


# =========================================================
# Single GCD
# =========================================================

def _single_gcd(
    ec_data,
):

    show_metadata_cn(
        ec_data
    )

    expected = (
        ec_data.metadata.get(
            "segment_count"
        )
    )

    segments = (
        detect_gcd_segments(
            ec_data.data,
            expected,
        )
    )

    cycles = (
        build_gcd_cycles(
            ec_data.data,
            expected,
        )
    )

    c1, c2 = (
        st.columns(
            2
        )
    )

    c1.metric(
        "检测到的充放电段数",
        len(
            segments
        ),
    )

    c2.metric(
        "完整循环数",
        len(
            cycles
        ),
    )

    cycle_df, cycle_info = (
        _select_gcd_data(
            ec_data,
            expected,
            "gcd_single",
        )
    )

    if cycle_df is None:

        return

    gcd_result = None


    # =====================================================
    # Calculation
    # =====================================================

    if cycle_info is not None:

        discharge_df = (
            get_discharge_segment(
                ec_data.data,
                cycle_info,
            )
        )

        with st.expander(
            "参数计算",
            expanded=True,
        ):

            default_current = (
                get_default_discharge_current_A(
                    ec_data.metadata
                )
            )

            if default_current is not None:

                st.metric(
                    "自动读取放电电流",
                    (
                        f"{default_current * 1000:g} mA"
                    ),
                )

            current_A = (
                default_current
            )

            if st.checkbox(
                "手动指定放电电流",
                False,
                key=
                    "gcd_manual_current",
            ):

                current_mA = (
                    st.number_input(
                        "放电电流（mA）",
                        min_value=0.0,
                        value=
                            float(
                                (
                                    default_current
                                    or
                                    0.0004
                                )
                                *
                                1000
                            ),
                        step=0.01,
                        key=
                            "gcd_current_mA",
                    )
                )

                current_A = (
                    current_mA
                    /
                    1000.0
                )

            mass = (
                st.number_input(
                    "活性物质质量（mg）",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=
                        "gcd_mass",
                )
            )

            basis = st.selectbox(
                "计算基准",

                [
                    "标准质量基准（k = 1）",
                    "对称器件单电极等效（k = 4）",
                ],

                key=
                    "gcd_basis",
            )

            factor = (
                1.0
                if
                basis.startswith(
                    "标准"
                )
                else
                4.0
            )

            if current_A is None:

                st.warning(
                    "未读取到放电电流，"
                    "请手动指定后再计算。"
                )

            elif st.button(
                "计算恒流充放电参数",
                type="primary",
                key=
                    "gcd_calc",
            ):

                gcd_result = (
                    calculate_gcd_parameters(
                        discharge_df,
                        current_A,
                        (
                            mass
                            if mass > 0
                            else None
                        ),
                        factor,
                    )
                )

                st.session_state[
                    "gcd_result_cn"
                ] = gcd_result

            gcd_result = (
                st.session_state.get(
                    "gcd_result_cn"
                )
            )

            if gcd_result:

                show_result_dict_cn(
                    gcd_result
                )


    # =====================================================
    # Plot data
    # =====================================================

    plot_df = (
        normalize_time(
            cycle_df
        )
    )

    tab1, tab2 = st.tabs(
        [
            "交互式图形",
            "出版级绘图",
        ]
    )


    # =====================================================
    # Interactive
    # =====================================================

    with tab1:

        fig = (
            _interactive_gcd(
                [
                    {
                        "df":
                            plot_df,

                        "label":
                            Path(
                                ec_data.filename
                            ).stem,

                        "color":
                            DEFAULT_COLORS[0],
                    }
                ]
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config=
                PLOTLY_CONFIG,
        )


    # =====================================================
    # Publication
    # =====================================================

    with tab2:

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

            style, dpi = (
                publication_controls(
                    "gcd_pub",

                    plot_df[
                        "time_s"
                    ],

                    plot_df[
                        "potential_V"
                    ],
                )
            )

            color = (
                st.color_picker(
                    "曲线颜色",
                    DEFAULT_COLORS[0],
                    key=
                        "gcd_pub_color",
                )
            )

            line_style = (
                select_line_style(
                    "曲线线型",
                    "gcd_pub_line",
                )
            )

        with preview:

            fig = (
                plot_gcd_publication(
                    plot_df,
                    style,
                    color,
                    line_style,
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
                    "_GCD"
                ),
                dpi,
                "gcd_export",
            )

            plt.close(
                fig
            )


# =========================================================
# Multiple GCD
# =========================================================



def _multi_gcd(
    datasets,
):
    """
    True multi-file GCD analysis.

    Supports shared / individual / mixed parameters, per-file cycle
    selection, current override, active mass, calculation basis,
    temperature, batch parameter calculation and summary export.
    """

    st.info(
        "多文件 GCD 模式默认叠加每个文件的最后一个完整循环，"
        "并将每条曲线时间起点归一到 0 s。"
        "现在可对不同文件分别输入质量、电流、温度和计算基准。"
    )

    st.subheader(
        "1. 批量参数设置"
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
            "gcd_multi_parameter_mode",
    )

    if mode == "所有文件使用相同参数":
        scopes = {
            "mass": "统一",
            "basis": "统一",
            "current": "统一",
            "temperature": "统一",
        }
    elif mode == "每个文件单独设置":
        scopes = {
            "mass": "逐文件",
            "basis": "逐文件",
            "current": "逐文件",
            "temperature": "逐文件",
        }
    else:
        st.caption(
            "混合模式可以让某些参数统一、某些参数逐文件输入。"
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        scopes = {
            "mass":
                c1.selectbox(
                    "活性质量",
                    [
                        "统一",
                        "逐文件",
                    ],
                    key=
                        "gcd_scope_mass",
                ),

            "basis":
                c2.selectbox(
                    "计算基准",
                    [
                        "统一",
                        "逐文件",
                    ],
                    key=
                        "gcd_scope_basis",
                ),

            "current":
                c3.selectbox(
                    "放电电流",
                    [
                        "统一",
                        "逐文件",
                    ],
                    key=
                        "gcd_scope_current",
                ),

            "temperature":
                c4.selectbox(
                    "温度",
                    [
                        "统一",
                        "逐文件",
                    ],
                    key=
                        "gcd_scope_temperature",
                ),
        }

    common = {}

    with st.expander(
        "统一参数",
        expanded=(
            "统一"
            in scopes.values()
        ),
    ):
        if scopes["mass"] == "统一":
            common[
                "mass_mg"
            ] = st.number_input(
                "统一活性物质质量（mg）",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=
                    "gcd_multi_common_mass",
            )

        if scopes["basis"] == "统一":
            common[
                "basis"
            ] = st.selectbox(
                "统一计算基准",
                [
                    "标准质量基准（k = 1）",
                    "对称器件单电极等效（k = 4）",
                ],
                key=
                    "gcd_multi_common_basis",
            )

        if scopes["current"] == "统一":
            common[
                "current_source"
            ] = st.selectbox(
                "放电电流来源",
                [
                    "使用每个文件自动读取值",
                    "所有文件使用同一个手动值",
                ],
                key=
                    "gcd_multi_common_current_source",
            )

            if (
                common[
                    "current_source"
                ]
                ==
                "所有文件使用同一个手动值"
            ):
                detected = [
                    get_default_discharge_current_A(
                        data.metadata
                    )
                    for data
                    in datasets
                ]

                detected = [
                    value
                    for value
                    in detected
                    if value
                    is not None
                ]

                default_mA = (
                    float(
                        detected[0]
                    )
                    * 1000.0
                    if detected
                    else 0.4
                )

                common[
                    "current_A"
                ] = (
                    st.number_input(
                        "统一放电电流（mA）",
                        min_value=1e-12,
                        value=
                            default_mA,
                        format=
                            "%.8g",
                        key=
                            "gcd_multi_common_current_mA",
                    )
                    /
                    1000.0
                )

        if scopes[
            "temperature"
        ] == "统一":
            common[
                "temperature_text"
            ] = st.text_input(
                "统一温度（°C，可选）",
                "",
                key=
                    "gcd_multi_common_temperature",
            )

    cycle_mode = st.radio(
        "循环选择",
        [
            "每个文件使用最后一个完整循环",
            "每个文件单独选择循环",
        ],
        horizontal=True,
        key=
            "gcd_multi_cycle_mode",
    )

    prepared = []
    config_rows = []

    st.subheader(
        "2. 文件参数"
    )

    for index, data in enumerate(
        datasets
    ):
        expected = (
            data.metadata.get(
                "segment_count"
            )
        )

        cycles = build_gcd_cycles(
            data.data,
            expected,
        )

        key_prefix = (
            f"gcd_multi_file_{index}"
        )

        stem = Path(
            data.filename
        ).stem

        with st.expander(
            f"{index + 1}. {data.filename}",
            expanded=(
                index < 3
            ),
        ):
            c1, c2 = st.columns(
                2
            )

            label = c1.text_input(
                "图例名称",
                stem,
                key=
                    f"{key_prefix}_label",
            )

            if scopes[
                "temperature"
            ] == "统一":
                temp_text = common.get(
                    "temperature_text",
                    "",
                )
                c2.caption(
                    "温度使用统一设置。"
                )
            else:
                temp_text = c2.text_input(
                    "温度（°C，可选）",
                    "",
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

            if scopes[
                "mass"
            ] == "统一":
                mass = float(
                    common.get(
                        "mass_mg",
                        0.0,
                    )
                )
                st.caption(
                    f"活性质量：{mass:g} mg（统一）"
                )
            else:
                mass = st.number_input(
                    "活性物质质量（mg）",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=
                        f"{key_prefix}_mass",
                )

            if scopes[
                "basis"
            ] == "统一":
                basis = common.get(
                    "basis",
                    "标准质量基准（k = 1）",
                )
                st.caption(
                    f"计算基准：{basis}（统一）"
                )
            else:
                basis = st.selectbox(
                    "计算基准",
                    [
                        "标准质量基准（k = 1）",
                        "对称器件单电极等效（k = 4）",
                    ],
                    key=
                        f"{key_prefix}_basis",
                )

            factor = (
                1.0
                if basis.startswith(
                    "标准"
                )
                else 4.0
            )

            auto_current = (
                get_default_discharge_current_A(
                    data.metadata
                )
            )

            if scopes[
                "current"
            ] == "统一":
                if (
                    common.get(
                        "current_source"
                    )
                    ==
                    "所有文件使用同一个手动值"
                ):
                    current_A = float(
                        common[
                            "current_A"
                        ]
                    )

                    st.caption(
                        "放电电流："
                        f"{current_A * 1000:g} mA（统一手动值）"
                    )
                else:
                    current_A = (
                        float(
                            auto_current
                        )
                        if auto_current
                        is not None
                        else None
                    )

                    if current_A is None:
                        st.warning(
                            "该文件未自动读取到放电电流。"
                            "请切换到逐文件电流，或使用统一手动值。"
                        )
                    else:
                        st.caption(
                            "自动读取放电电流："
                            f"{current_A * 1000:g} mA"
                        )

            else:
                manual_current = st.checkbox(
                    "手动覆盖放电电流",
                    value=(
                        auto_current
                        is None
                    ),
                    key=
                        f"{key_prefix}_manual_current",
                )

                if manual_current:
                    current_A = (
                        st.number_input(
                            "放电电流（mA）",
                            min_value=1e-12,
                            value=
                                float(
                                    auto_current
                                    if auto_current
                                    is not None
                                    else 0.0004
                                )
                                * 1000.0,
                            format=
                                "%.8g",
                            key=
                                f"{key_prefix}_current_mA",
                        )
                        /
                        1000.0
                    )
                else:
                    current_A = (
                        float(
                            auto_current
                        )
                        if auto_current
                        is not None
                        else None
                    )

                    if current_A is not None:
                        st.caption(
                            "自动读取放电电流："
                            f"{current_A * 1000:g} mA"
                        )

            if not cycles:
                st.warning(
                    "未检测到完整 GCD 循环；"
                    "该文件仅用于原始曲线叠图。"
                )

                cycle_df = (
                    data.data
                )
                cycle_info = None
                cycle_number = None
                cycle_label = (
                    "未检测到完整循环"
                )

            else:
                if (
                    cycle_mode
                    ==
                    "每个文件使用最后一个完整循环"
                ):
                    cycle_df, cycle_info = (
                        get_last_complete_gcd_cycle(
                            data.data,
                            expected,
                        )
                    )
                else:
                    options = [
                        item[
                            "cycle"
                        ]
                        for item
                        in cycles
                    ]

                    selected_cycle = (
                        st.selectbox(
                            "分析循环",
                            options,
                            index=
                                len(
                                    options
                                )
                                - 1,
                            format_func=
                                lambda x:
                                    f"第 {x} 个循环",
                            key=
                                f"{key_prefix}_cycle",
                        )
                    )

                    cycle_df, cycle_info = (
                        get_gcd_cycle(
                            data.data,
                            selected_cycle,
                            expected,
                        )
                    )

                cycle_number = (
                    cycle_info.get(
                        "cycle"
                    )
                    if cycle_info
                    is not None
                    else None
                )

                cycle_label = (
                    f"第 {cycle_number} 个循环"
                    if cycle_number
                    is not None
                    else "完整循环"
                )

            plot_df = normalize_time(
                cycle_df
            )

            discharge_df = None

            if cycle_info is not None:
                try:
                    discharge_df = (
                        get_discharge_segment(
                            data.data,
                            cycle_info,
                        )
                    )
                except Exception as exc:
                    st.warning(
                        f"放电段提取失败：{exc}"
                    )

            prepared.append(
                {
                    "index":
                        index,
                    "data":
                        data,
                    "plot_df":
                        plot_df,
                    "cycle_info":
                        cycle_info,
                    "cycle_number":
                        cycle_number,
                    "discharge_df":
                        discharge_df,
                    "label":
                        label,
                    "temperature_C":
                        temperature,
                    "mass_mg":
                        float(
                            mass
                        ),
                    "basis":
                        basis,
                    "factor":
                        factor,
                    "current_A":
                        current_A,
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
                    "循环":
                        cycle_label,
                    "放电电流_mA":
                        (
                            current_A
                            * 1000.0
                            if current_A
                            is not None
                            else None
                        ),
                    "活性质量_mg":
                        (
                            float(
                                mass
                            )
                            if mass > 0
                            else None
                        ),
                    "计算因子_k":
                        factor,
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
    # Batch calculation
    # -----------------------------------------------------
    st.subheader(
        "3. 批量参数计算"
    )

    result_rows = []

    for item in prepared:
        row = {
            "文件":
                item[
                    "data"
                ].filename,
            "图例":
                item[
                    "label"
                ],
            "温度_C":
                item[
                    "temperature_C"
                ],
            "循环":
                item[
                    "cycle_number"
                ],
            "放电电流_mA":
                (
                    item[
                        "current_A"
                    ]
                    * 1000.0
                    if item[
                        "current_A"
                    ]
                    is not None
                    else None
                ),
            "活性质量_mg":
                (
                    item[
                        "mass_mg"
                    ]
                    if item[
                        "mass_mg"
                    ] > 0
                    else None
                ),
            "计算因子_k":
                item[
                    "factor"
                ],
        }

        if item[
            "discharge_df"
        ] is None:
            row[
                "状态"
            ] = "无可分析放电段"

        elif item[
            "current_A"
        ] is None:
            row[
                "状态"
            ] = "缺少放电电流"

        else:
            try:
                result = (
                    calculate_gcd_parameters(
                        item[
                            "discharge_df"
                        ],
                        item[
                            "current_A"
                        ],
                        (
                            item[
                                "mass_mg"
                            ]
                            if item[
                                "mass_mg"
                            ] > 0
                            else None
                        ),
                        item[
                            "factor"
                        ],
                    )
                )

                item[
                    "result"
                ] = result

                row.update(
                    result
                )

                row[
                    "状态"
                ] = "计算完成"

            except Exception as exc:
                row[
                    "状态"
                ] = f"计算失败：{exc}"

        result_rows.append(
            row
        )

    result_df = pd.DataFrame(
        result_rows
    )

    st.dataframe(
        result_df,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "下载 GCD 批量分析结果（CSV）",
        data=
            result_df.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),
        file_name=
            "GCD_batch_analysis.csv",
        mime=
            "text/csv",
        key=
            "gcd_multi_result_download",
    )

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    st.subheader(
        "4. 多文件曲线比较"
    )

    tab1, tab2 = st.tabs(
        [
            "交互式叠图",
            "出版级叠图",
        ]
    )

    with tab1:
        curves = [
            {
                "df":
                    item[
                        "plot_df"
                    ],
                "label":
                    item[
                        "label"
                    ],
                "color":
                    DEFAULT_COLORS[
                        item[
                            "index"
                        ]
                        %
                        len(
                            DEFAULT_COLORS
                        )
                    ],
            }
            for item
            in prepared
        ]

        st.plotly_chart(
            _interactive_gcd(
                curves
            ),
            width="stretch",
            config=
                PLOTLY_CONFIG,
        )

    with tab2:
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
                    item[
                        "plot_df"
                    ][
                        "time_s"
                    ].to_numpy()
                    for item
                    in prepared
                ]
            )

            all_y = np.concatenate(
                [
                    item[
                        "plot_df"
                    ][
                        "potential_V"
                    ].to_numpy()
                    for item
                    in prepared
                ]
            )

            style, dpi = (
                publication_controls(
                    "gcd_multi_pub",
                    all_x,
                    all_y,
                )
            )

            (
                show_legend,
                legend_location,
            ) = legend_controls(
                "gcd_multi",
                True,
            )

            curves = []

            with st.expander(
                "各文件曲线样式",
                expanded=True,
            ):
                for item in prepared:
                    index = item[
                        "index"
                    ]

                    st.markdown(
                        f"**{item['data'].filename}**"
                    )

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
                            f"gcd_multi_pub_color_{index}",
                    )

                    line_style = (
                        select_line_style(
                            "曲线线型",
                            f"gcd_multi_pub_line_{index}",
                        )
                    )

                    curves.append(
                        {
                            "df":
                                item[
                                    "plot_df"
                                ],
                            "label":
                                item[
                                    "label"
                                ],
                            "color":
                                color,
                            "line_style":
                                line_style,
                        }
                    )

        with preview:
            fig = (
                plot_gcd_multi_publication(
                    curves,
                    style,
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
                "GCD_多文件叠图",
                dpi,
                "gcd_multi_export",
            )

            plt.close(
                fig
            )



def render_gcd_screen():
    st.header(
        "恒流充放电分析（GCD）"
    )

    st.caption(
        "支持一次上传多个 CHI660E GCD TXT 文件。"
        "单文件保留原有功能；多文件支持统一参数、逐文件参数、"
        "混合参数、批量计算、汇总导出和曲线叠加。"
    )

    uploaded = st.file_uploader(
        "上传一个或多个恒流充放电 TXT 文件",
        type=[
            "txt"
        ],
        accept_multiple_files=True,
        key=
            "gcd_uploader",
    )

    if not uploaded:
        st.info(
            "请上传恒流充放电（GCD）TXT 文件。"
        )
        return

    datasets = []
    read_errors = []

    for file in uploaded:
        try:
            data = read_chi_txt(
                file
            )

            if data.technique != "GCD":
                read_errors.append(
                    (
                        file.name,
                        "文件不是 GCD 类型",
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
            "没有成功载入可分析的 GCD 文件。"
        )
        return

    st.success(
        f"已成功载入 {len(datasets)} 个 GCD 文件。"
    )

    if len(
        datasets
    ) == 1:
        _single_gcd(
            datasets[0]
        )
    else:
        _multi_gcd(
            datasets
        )


