from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import (
    read_chi_txt,
)

from core.cv_analysis import (
    build_cv_cycles,
    get_cv_cycle,
    get_last_complete_cycle,
    calculate_cv_parameters,
)

from plotting.publication import (
    plot_cv_publication,
    plot_cv_multi_publication,
)

from ui.common import (
    DEFAULT_COLORS,
    PLOTLY_CONFIG,
    current_scale,
    export_buttons,
    legend_controls,
    publication_controls,
    select_line_style,
    show_metadata_cn,
    show_result_dict_cn,
)


# =========================================================
# Interactive CV
# =========================================================

def _interactive_cv(
    curves,
    unit="mA",
):

    scale = (
        current_scale(
            unit
        )
    )

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
                        "potential_V"
                    ],

                y=
                    curve[
                        "df"
                    ][
                        "current_A"
                    ]
                    *
                    scale,

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
            "电位（V）",

        yaxis_title=
            f"电流（{unit}）",

        hovermode=
            "closest",

        dragmode=
            "zoom",
    )

    return fig


# =========================================================
# Cycle selection
# =========================================================

def _select_cv_data(
    ec_data,
    key_prefix,
):

    cycles = (
        build_cv_cycles(
            ec_data.data
        )
    )

    if not cycles:

        st.error(
            "未检测到完整的循环伏安循环。"
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

        df, info = (
            get_last_complete_cycle(
                ec_data.data
            )
        )

        return (
            df,
            info.get(
                "cycle"
            ),
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
        get_cv_cycle(
            ec_data.data,
            cycle_number,
        ),
        cycle_number,
    )


# =========================================================
# Single CV
# =========================================================

def _single_cv(
    ec_data,
):

    show_metadata_cn(
        ec_data
    )

    plot_df, cycle_number = (
        _select_cv_data(
            ec_data,
            "cv_single",
        )
    )

    if plot_df is None:

        return

    cv_result = None

    # =====================================================
    # Calculation
    # =====================================================

    if cycle_number is not None:

        with st.expander(
            "参数计算",
            expanded=True,
        ):

            scan_rate = (
                ec_data.metadata.get(
                    "scan_rate_V_s"
                )
            )

            if scan_rate is not None:

                st.metric(
                    "扫描速率",
                    f"{scan_rate:g} V/s",
                )

            mass = (
                st.number_input(
                    "活性物质质量（mg）",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=
                        "cv_mass",
                )
            )

            basis = st.selectbox(
                "计算基准",

                [
                    "标准质量基准（k = 1）",
                    "对称器件单电极等效（k = 4）",
                ],

                key=
                    "cv_basis",
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

            if scan_rate is None:

                st.warning(
                    "文件中未读取到扫描速率，"
                    "无法计算比电容。"
                )

            elif st.button(
                "计算循环伏安参数",
                type="primary",
                key=
                    "cv_calc",
            ):

                cv_result = (
                    calculate_cv_parameters(
                        plot_df,
                        scan_rate,
                        (
                            mass
                            if mass > 0
                            else None
                        ),
                        factor,
                    )
                )

                st.session_state[
                    "cv_result_cn"
                ] = cv_result

            cv_result = (
                st.session_state.get(
                    "cv_result_cn"
                )
            )

            if cv_result:

                show_result_dict_cn(
                    cv_result
                )


    # =====================================================
    # Tabs
    # =====================================================

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

        unit = st.selectbox(
            "电流显示单位",
            [
                "A",
                "mA",
                "μA",
            ],
            index=1,
            key=
                "cv_interactive_unit",
        )

        fig = _interactive_cv(
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
            ],
            unit,
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

            unit = st.selectbox(
                "导出图电流单位",

                [
                    "A",
                    "mA",
                    "μA",
                ],

                index=1,

                key=
                    "cv_pub_unit",
            )

            scale = (
                current_scale(
                    unit
                )
            )

            style, dpi = (
                publication_controls(
                    "cv_pub",

                    plot_df[
                        "potential_V"
                    ],

                    plot_df[
                        "current_A"
                    ]
                    *
                    scale,
                )
            )

            color = (
                st.color_picker(
                    "曲线颜色",
                    DEFAULT_COLORS[0],
                    key=
                        "cv_pub_color",
                )
            )

            line_style = (
                select_line_style(
                    "曲线线型",
                    "cv_pub_line",
                )
            )

        with preview:

            fig = (
                plot_cv_publication(
                    plot_df,
                    style,
                    unit,
                    color,
                    line_style,
                    cv_result,
                    False,
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
                    "_CV"
                ),
                dpi,
                "cv_export",
            )

            plt.close(
                fig
            )


# =========================================================
# Multiple CV
# =========================================================



def _multi_cv(
    datasets,
):
    """
    True multi-file CV analysis.

    Supports:
    - shared parameters for all files;
    - independent parameters for every file;
    - mixed mode where each parameter can be shared or independent;
    - per-file cycle selection;
    - per-file scan-rate override, active mass, calculation basis,
      temperature and legend;
    - batch calculation + summary CSV;
    - interactive / publication overlays.
    """

    st.info(
        "多文件 CV 模式已启用批量参数分析。"
        "默认使用每个文件的最后一个完整循环；"
        "可选择统一参数、逐文件参数或混合参数。"
    )

    # -----------------------------------------------------
    # Parameter scope
    # -----------------------------------------------------
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
        key="cv_multi_parameter_mode",
    )

    if mode == "所有文件使用相同参数":
        scopes = {
            "mass": "统一",
            "basis": "统一",
            "scan": "统一",
            "temperature": "统一",
        }
    elif mode == "每个文件单独设置":
        scopes = {
            "mass": "逐文件",
            "basis": "逐文件",
            "scan": "逐文件",
            "temperature": "逐文件",
        }
    else:
        st.caption(
            "混合模式：例如活性质量逐文件输入，但计算基准统一设置。"
        )
        c1, c2, c3, c4 = st.columns(4)

        scopes = {
            "mass":
                c1.selectbox(
                    "活性质量",
                    ["统一", "逐文件"],
                    key="cv_scope_mass",
                ),
            "basis":
                c2.selectbox(
                    "计算基准",
                    ["统一", "逐文件"],
                    key="cv_scope_basis",
                ),
            "scan":
                c3.selectbox(
                    "扫描速率",
                    ["统一", "逐文件"],
                    key="cv_scope_scan",
                ),
            "temperature":
                c4.selectbox(
                    "温度",
                    ["统一", "逐文件"],
                    key="cv_scope_temperature",
                ),
        }

    st.caption(
        "活性质量仅用于质量归一化比电容；"
        "如果只比较原始 CV 曲线，可以保持 0 mg。"
    )

    common = {}

    with st.expander(
        "统一参数",
        expanded=(
            "统一"
            in scopes.values()
        ),
    ):
        if scopes["mass"] == "统一":
            common["mass_mg"] = st.number_input(
                "统一活性物质质量（mg）",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key="cv_multi_common_mass",
            )

        if scopes["basis"] == "统一":
            common["basis"] = st.selectbox(
                "统一计算基准",
                [
                    "标准质量基准（k = 1）",
                    "对称器件单电极等效（k = 4）",
                ],
                key="cv_multi_common_basis",
            )

        if scopes["scan"] == "统一":
            common["scan_source"] = st.selectbox(
                "扫描速率来源",
                [
                    "使用每个文件自动读取值",
                    "所有文件使用同一个手动值",
                ],
                key="cv_multi_common_scan_source",
            )

            if (
                common["scan_source"]
                ==
                "所有文件使用同一个手动值"
            ):
                detected = [
                    data.metadata.get(
                        "scan_rate_V_s"
                    )
                    for data in datasets
                    if data.metadata.get(
                        "scan_rate_V_s"
                    )
                    is not None
                ]

                default_scan = (
                    float(detected[0])
                    if detected
                    else 0.01
                )

                common["scan_rate_V_s"] = st.number_input(
                    "统一扫描速率（V/s）",
                    min_value=1e-12,
                    value=default_scan,
                    format="%.8g",
                    key="cv_multi_common_scan_rate",
                )

        if scopes["temperature"] == "统一":
            common["temperature_text"] = st.text_input(
                "统一温度（°C，可选）",
                "",
                key="cv_multi_common_temperature",
            )

    cycle_mode = st.radio(
        "循环选择",
        [
            "每个文件使用最后一个完整循环",
            "每个文件单独选择循环",
        ],
        horizontal=True,
        key="cv_multi_cycle_mode",
    )

    prepared = []
    config_rows = []

    # -----------------------------------------------------
    # Per-file configuration
    # -----------------------------------------------------
    st.subheader(
        "2. 文件参数"
    )

    for index, data in enumerate(
        datasets
    ):
        stem = Path(
            data.filename
        ).stem

        key_prefix = (
            f"cv_multi_file_{index}"
        )

        cycles = build_cv_cycles(
            data.data
        )

        with st.expander(
            f"{index + 1}. {data.filename}",
            expanded=(
                index < 3
            ),
        ):
            c1, c2 = st.columns(2)

            label = c1.text_input(
                "图例名称",
                stem,
                key=
                    f"{key_prefix}_label",
            )

            if scopes["temperature"] == "统一":
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

            if scopes["mass"] == "统一":
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

            if scopes["basis"] == "统一":
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

            auto_scan = data.metadata.get(
                "scan_rate_V_s"
            )

            if scopes["scan"] == "统一":
                if (
                    common.get(
                        "scan_source"
                    )
                    ==
                    "所有文件使用同一个手动值"
                ):
                    scan_rate = float(
                        common[
                            "scan_rate_V_s"
                        ]
                    )
                    st.caption(
                        f"扫描速率：{scan_rate:g} V/s（统一手动值）"
                    )
                else:
                    scan_rate = (
                        float(
                            auto_scan
                        )
                        if auto_scan
                        is not None
                        else None
                    )
                    if scan_rate is None:
                        st.warning(
                            "该文件未自动读取到扫描速率。"
                            "请切换到逐文件扫描速率或使用统一手动值。"
                        )
                    else:
                        st.caption(
                            f"扫描速率：{scan_rate:g} V/s（文件自动读取）"
                        )
            else:
                manual_scan = st.checkbox(
                    "手动覆盖扫描速率",
                    value=(
                        auto_scan
                        is None
                    ),
                    key=
                        f"{key_prefix}_manual_scan",
                )

                if manual_scan:
                    scan_rate = st.number_input(
                        "扫描速率（V/s）",
                        min_value=1e-12,
                        value=
                            float(
                                auto_scan
                                if auto_scan
                                is not None
                                else 0.01
                            ),
                        format="%.8g",
                        key=
                            f"{key_prefix}_scan_rate",
                    )
                else:
                    scan_rate = (
                        float(
                            auto_scan
                        )
                        if auto_scan
                        is not None
                        else None
                    )
                    if scan_rate is not None:
                        st.caption(
                            f"自动读取扫描速率：{scan_rate:g} V/s"
                        )

            if not cycles:
                st.warning(
                    "未检测到完整 CV 循环；"
                    "该文件仅用于原始曲线叠图，不进行循环参数计算。"
                )
                plot_df = (
                    data.data
                )
                cycle_number = None
                cycle_label = "未检测到完整循环"
            else:
                if (
                    cycle_mode
                    ==
                    "每个文件使用最后一个完整循环"
                ):
                    plot_df, info = (
                        get_last_complete_cycle(
                            data.data
                        )
                    )
                    cycle_number = info.get(
                        "cycle"
                    )
                else:
                    options = [
                        item[
                            "cycle"
                        ]
                        for item
                        in cycles
                    ]

                    cycle_number = st.selectbox(
                        "分析循环",
                        options,
                        index=
                            len(
                                options
                            )
                            - 1,
                        format_func=lambda x:
                            f"第 {x} 个循环",
                        key=
                            f"{key_prefix}_cycle",
                    )

                    plot_df = get_cv_cycle(
                        data.data,
                        cycle_number,
                    )

                cycle_label = (
                    f"第 {cycle_number} 个循环"
                )

            prepared.append(
                {
                    "index":
                        index,
                    "data":
                        data,
                    "df":
                        plot_df,
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
                    "scan_rate_V_s":
                        scan_rate,
                    "cycle_number":
                        cycle_number,
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
                    "扫描速率_V_s":
                        scan_rate,
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
    # Batch calculations
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
            "扫描速率_V_s":
                item[
                    "scan_rate_V_s"
                ],
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

        if (
            item[
                "cycle_number"
            ]
            is None
        ):
            row[
                "状态"
            ] = "无完整循环，仅绘图"

        elif (
            item[
                "scan_rate_V_s"
            ]
            is None
        ):
            row[
                "状态"
            ] = "缺少扫描速率"

        else:
            try:
                result = calculate_cv_parameters(
                    item[
                        "df"
                    ],
                    item[
                        "scan_rate_V_s"
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
        "下载 CV 批量分析结果（CSV）",
        data=
            result_df.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            ),
        file_name=
            "CV_batch_analysis.csv",
        mime=
            "text/csv",
        key=
            "cv_multi_result_download",
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
        unit = st.selectbox(
            "电流显示单位",
            [
                "A",
                "mA",
                "μA",
            ],
            index=1,
            key=
                "cv_multi_interactive_unit",
        )

        curves = []

        for item in prepared:
            index = item[
                "index"
            ]

            curves.append(
                {
                    "df":
                        item[
                            "df"
                        ],
                    "label":
                        item[
                            "label"
                        ],
                    "color":
                        DEFAULT_COLORS[
                            index
                            %
                            len(
                                DEFAULT_COLORS
                            )
                        ],
                }
            )

        st.plotly_chart(
            _interactive_cv(
                curves,
                unit,
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
            unit = st.selectbox(
                "导出图电流单位",
                [
                    "A",
                    "mA",
                    "μA",
                ],
                index=1,
                key=
                    "cv_multi_pub_unit",
            )

            scale = current_scale(
                unit
            )

            all_x = np.concatenate(
                [
                    item[
                        "df"
                    ][
                        "potential_V"
                    ].to_numpy()
                    for item
                    in prepared
                ]
            )

            all_y = np.concatenate(
                [
                    item[
                        "df"
                    ][
                        "current_A"
                    ].to_numpy()
                    * scale
                    for item
                    in prepared
                ]
            )

            style, dpi = (
                publication_controls(
                    "cv_multi_pub",
                    all_x,
                    all_y,
                )
            )

            (
                show_legend,
                legend_location,
            ) = legend_controls(
                "cv_multi",
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
                            f"cv_multi_pub_color_{index}",
                    )

                    line_style = select_line_style(
                        "曲线线型",
                        f"cv_multi_pub_line_{index}",
                    )

                    curves.append(
                        {
                            "df":
                                item[
                                    "df"
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
            fig = plot_cv_multi_publication(
                curves,
                style,
                unit,
                show_legend,
                legend_location,
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            export_buttons(
                fig,
                "CV_多文件叠图",
                dpi,
                "cv_multi_export",
            )

            plt.close(
                fig
            )



def render_cv_screen():
    st.header(
        "循环伏安分析（CV）"
    )

    st.caption(
        "支持一次上传多个 CHI660E CV TXT 文件。"
        "单文件保留原有分析；多文件支持统一参数、逐文件参数、"
        "混合参数、批量计算、汇总导出和曲线叠加。"
    )

    uploaded = st.file_uploader(
        "上传一个或多个循环伏安 TXT 文件",
        type=[
            "txt"
        ],
        accept_multiple_files=True,
        key=
            "cv_uploader",
    )

    if not uploaded:
        st.info(
            "请上传循环伏安（CV）TXT 文件。"
        )
        return

    datasets = []
    read_errors = []

    for file in uploaded:
        try:
            data = read_chi_txt(
                file
            )

            if data.technique != "CV":
                read_errors.append(
                    (
                        file.name,
                        "文件不是 CV 类型",
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
            "没有成功载入可分析的 CV 文件。"
        )
        return

    st.success(
        f"已成功载入 {len(datasets)} 个 CV 文件。"
    )

    if len(
        datasets
    ) == 1:
        _single_cv(
            datasets[0]
        )
    else:
        _multi_cv(
            datasets
        )


