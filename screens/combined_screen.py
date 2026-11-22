from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.data_loader import (
    read_chi_txt,
)

from core.cv_analysis import (
    build_cv_cycles,
    get_last_complete_cycle,
)

from core.gcd_analysis import (
    build_gcd_cycles,
    get_last_complete_gcd_cycle,
    get_discharge_segment,
    get_default_discharge_current_A,
    analyze_gcd_discharge,
)

from core.eis_analysis import (
    estimate_rs_intercept,
    estimate_rct_initial,
)

from core.experiment_grouping import (
    infer_group_label,
    group_datasets,
)

from plotting.combined_publication import (
    plot_combined_electrochemistry,
)

from ui.common import (
    DEFAULT_COLORS,
    export_buttons,
    publication_controls,
    select_line_style,
    technique_cn,
)


# =========================================================
# Choose one dataset
# =========================================================

def _choose_dataset(
    items,
    label,
    key,
):

    if not items:

        return None

    if len(
        items
    ) == 1:

        return (
            items[0]
        )

    names = [
        item.filename
        for item
        in items
    ]

    selected = st.selectbox(
        label,
        names,
        key=key,
    )

    return next(
        item
        for item
        in items
        if item.filename
        ==
        selected
    )


# =========================================================
# CV overview
# =========================================================

def _overview_cv(
    data,
):

    if data is None:

        st.info(
            "本实验组没有循环伏安（CV）文件。"
        )

        return

    cycles = (
        build_cv_cycles(
            data.data
        )
    )

    try:

        df, _ = (
            get_last_complete_cycle(
                data.data
            )
        )

    except Exception:

        df = (
            data.data
        )

    st.metric(
        "完整循环数",
        len(
            cycles
        ),
    )

    scan_rate = (
        data.metadata.get(
            "scan_rate_V_s"
        )
    )

    if scan_rate is not None:

        st.metric(
            "扫描速率",
            f"{scan_rate:g} V/s",
        )

    st.metric(
        "电位窗口",
        (
            f"{df['potential_V'].max() - df['potential_V'].min():.6g} V"
        ),
    )

    st.metric(
        "最大电流",
        (
            f"{df['current_A'].max() * 1000:.6g} mA"
        ),
    )

    st.metric(
        "最小电流",
        (
            f"{df['current_A'].min() * 1000:.6g} mA"
        ),
    )


# =========================================================
# GCD overview
# =========================================================

def _overview_gcd(
    data,
):

    if data is None:

        st.info(
            "本实验组没有恒流充放电（GCD）文件。"
        )

        return

    expected = (
        data.metadata.get(
            "segment_count"
        )
    )

    cycles = (
        build_gcd_cycles(
            data.data,
            expected,
        )
    )

    st.metric(
        "完整循环数",
        len(
            cycles
        ),
    )

    current = (
        get_default_discharge_current_A(
            data.metadata
        )
    )

    if current is not None:

        st.metric(
            "自动读取放电电流",
            f"{current * 1000:g} mA",
        )

    try:

        _, info = (
            get_last_complete_gcd_cycle(
                data.data,
                expected,
            )
        )

        discharge = (
            get_discharge_segment(
                data.data,
                info,
            )
        )

        result = (
            analyze_gcd_discharge(
                discharge
            )
        )

        if (
            result.get(
                "discharge_time_s"
            )
            is not None
        ):

            st.metric(
                "放电时间（Δt）",
                (
                    f"{result['discharge_time_s']:.6g} s"
                ),
            )

        if (
            result.get(
                "discharge_voltage_window_V"
            )
            is not None
        ):

            st.metric(
                "放电电压窗口（ΔV）",
                (
                    f"{result['discharge_voltage_window_V']:.6g} V"
                ),
            )

        if (
            result.get(
                "ir_drop_V"
            )
            is not None
        ):

            st.metric(
                "初始电压降（IR drop）",
                (
                    f"{result['ir_drop_V']:.6g} V"
                ),
            )

        r2 = result.get(
            "r_squared",
            result.get(
                "linear_fit_r_squared"
            ),
        )

        if r2 is not None:

            st.metric(
                "放电线性拟合优度（R²）",
                f"{r2:.5f}",
            )

    except Exception as exc:

        st.caption(
            f"自动放电参数提取未完成：{exc}"
        )


# =========================================================
# EIS overview
# =========================================================

def _overview_eis(
    data,
):

    if data is None:

        st.info(
            "本实验组没有电化学阻抗谱（EIS）文件。"
        )

        return

    try:

        rs = (
            estimate_rs_intercept(
                data.data
            )
        )

        rct0 = (
            estimate_rct_initial(
                data.data,
                rs,
            )
        )

        st.metric(
            "串联欧姆电阻自动估计（Rs）",
            f"{rs:.6g} Ω",
        )

        st.metric(
            "电荷转移电阻初始估计（Rct）",
            f"{rct0:.6g} Ω",
        )

        st.metric(
            "最高频率",
            (
                f"{data.data['frequency_Hz'].max():.6g} Hz"
            ),
        )

        st.metric(
            "最低频率",
            (
                f"{data.data['frequency_Hz'].min():.6g} Hz"
            ),
        )

    except Exception as exc:

        st.caption(
            f"EIS 自动估计未完成：{exc}"
        )


# =========================================================
# Main screen
# =========================================================

def render_combined_screen():

    st.header(
        "联合电化学分析"
    )

    st.caption(
        "允许循环伏安（CV）、恒流充放电（GCD）"
        "和电化学阻抗谱（EIS）TXT 文件同时上传，"
        "软件会自动分类并按实验组整理。"
    )

    uploaded = st.file_uploader(
        "混合上传 CV / GCD / EIS TXT 文件",
        type=[
            "txt"
        ],
        accept_multiple_files=True,
        key=
            "combined_uploader",
    )

    if not uploaded:

        st.info(
            "可同时拖入同一样品或不同实验条件下的 "
            "CV、GCD、EIS 文件。"
        )

        return

    try:

        datasets = [
            read_chi_txt(
                file
            )
            for file
            in uploaded
        ]

    except Exception as exc:

        st.error(
            "文件读取或自动识别失败。"
        )

        st.exception(
            exc
        )

        return


    # =====================================================
    # Group assignment
    # =====================================================

    labels = []

    table_rows = []

    with st.expander(
        "文件自动分类与实验组",
        expanded=True,
    ):

        for index, data in enumerate(
            datasets
        ):

            auto_group = (
                infer_group_label(
                    data.filename,
                    data.technique,
                )
            )

            group_label = (
                st.text_input(
                    (
                        f"{Path(data.filename).name}"
                        " → 实验组"
                    ),
                    value=
                        auto_group,
                    key=
                        f"combined_group_{index}",
                )
            )

            final_group = (
                group_label.strip()
                or
                auto_group
            )

            labels.append(
                final_group
            )

            table_rows.append(
                {
                    "文件":
                        data.filename,

                    "自动识别类型":
                        technique_cn(
                            data.technique
                        ),

                    "实验组":
                        final_group,
                }
            )

        st.dataframe(
            pd.DataFrame(
                table_rows
            ),
            width="stretch",
            hide_index=True,
        )


    # =====================================================
    # Grouping
    # =====================================================

    groups = (
        group_datasets(
            datasets,
            labels,
        )
    )

    group_name = st.selectbox(
        "选择要联合分析的实验组",
        list(
            groups.keys()
        ),
        key=
            "combined_group_select",
    )

    group = (
        groups[
            group_name
        ]
    )

    cv_data = (
        _choose_dataset(
            group.get(
                "CV",
                [],
            ),
            "选择循环伏安（CV）文件",
            "combined_cv_file",
        )
    )

    gcd_data = (
        _choose_dataset(
            group.get(
                "GCD",
                [],
            ),
            "选择恒流充放电（GCD）文件",
            "combined_gcd_file",
        )
    )

    eis_data = (
        _choose_dataset(
            group.get(
                "EIS",
                [],
            ),
            "选择电化学阻抗谱（EIS）文件",
            "combined_eis_file",
        )
    )


    # =====================================================
    # Tabs
    # =====================================================

    tab1, tab2 = st.tabs(
        [
            "联合分析概览",
            "SCI 三联图",
        ]
    )


    # =====================================================
    # Overview
    # =====================================================

    with tab1:

        c1, c2, c3 = (
            st.columns(
                3,
                gap="large",
            )
        )

        with c1:

            st.subheader(
                "循环伏安（CV）"
            )

            _overview_cv(
                cv_data
            )

        with c2:

            st.subheader(
                "恒流充放电（GCD）"
            )

            _overview_gcd(
                gcd_data
            )

        with c3:

            st.subheader(
                "电化学阻抗谱（EIS）"
            )

            _overview_eis(
                eis_data
            )

        st.caption(
            "联合概览只进行快速、稳健的自动提取；"
            "需要活性物质质量、计算基准或等效电路模型的参数，"
            "仍在对应的独立分析页面中计算。"
        )


    # =====================================================
    # SCI triptych
    # =====================================================

    with tab2:

        if (
            cv_data is None
            or
            gcd_data is None
            or
            eis_data is None
        ):

            st.warning(
                "生成 CV + GCD + EIS 三联图需要当前实验组"
                "三种文件都存在。"
            )

            return

        try:

            cv_df, _ = (
                get_last_complete_cycle(
                    cv_data.data
                )
            )

        except Exception:

            cv_df = (
                cv_data.data
            )

        try:

            gcd_df, _ = (
                get_last_complete_gcd_cycle(
                    gcd_data.data,
                    gcd_data.metadata.get(
                        "segment_count"
                    ),
                )
            )

        except Exception:

            gcd_df = (
                gcd_data.data
            )

        eis_df = (
            eis_data.data
        )

        controls, preview = (
            st.columns(
                [
                    0.35,
                    0.65,
                ],
                gap="large",
            )
        )


        # =================================================
        # Controls
        # =================================================

        with controls:

            style, dpi = (
                publication_controls(
                    "combined_pub",
                    combined=True,
                )
            )

            current_unit = (
                st.selectbox(
                    "CV 电流单位",
                    [
                        "A",
                        "mA",
                        "μA",
                    ],
                    index=1,
                    key=
                        "combined_current_unit",
                )
            )

            st.markdown(
                "#### 三个分图的曲线样式"
            )

            cv_color = (
                st.color_picker(
                    "CV 曲线颜色",
                    DEFAULT_COLORS[0],
                    key=
                        "combined_cv_color",
                )
            )

            cv_line = (
                select_line_style(
                    "CV 曲线线型",
                    "combined_cv_line",
                )
            )

            gcd_color = (
                st.color_picker(
                    "GCD 曲线颜色",
                    DEFAULT_COLORS[1],
                    key=
                        "combined_gcd_color",
                )
            )

            gcd_line = (
                select_line_style(
                    "GCD 曲线线型",
                    "combined_gcd_line",
                )
            )

            eis_color = (
                st.color_picker(
                    "EIS 曲线颜色",
                    DEFAULT_COLORS[2],
                    key=
                        "combined_eis_color",
                )
            )

            eis_line = (
                select_line_style(
                    "EIS 曲线线型",
                    "combined_eis_line",
                )
            )


        # =================================================
        # Preview
        # =================================================

        with preview:

            fig = (
                plot_combined_electrochemistry(
                    cv_df,
                    gcd_df,
                    eis_df,
                    style,

                    cv_color=
                        cv_color,

                    gcd_color=
                        gcd_color,

                    eis_color=
                        eis_color,

                    cv_line_style=
                        cv_line,

                    gcd_line_style=
                        gcd_line,

                    eis_line_style=
                        eis_line,

                    current_unit=
                        current_unit,
                )
            )

            st.pyplot(
                fig,
                width="stretch",
            )

            export_buttons(
                fig,
                (
                    f"{group_name}"
                    "_CV_GCD_EIS_三联图"
                ),
                dpi,
                "combined_export",
            )

            plt.close(
                fig
            )
