from __future__ import annotations

import hashlib
import io
import traceback
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.eis_flexible_import import (
    list_candidates, parse_candidate, preview_columns, guess_columns,
)
from core.eis_professional import (
    available as professional_available, run_professional, backend_version,
)
from core.analysis_export import build_eis_analysis_zip
from core.eis_paper_figures import (
    MODELS as PAPER_MODELS, fit_many as fit_paper_models,
    parameter_table as paper_parameter_table, make_paper_bundle,
    draw_figure as draw_paper_figure, draw_circuit as draw_paper_circuit,
    draw_parameters as draw_paper_parameters,
    relaxation_summary as paper_relaxation_summary,
)

from core.eis_temperature_final import (
    FINAL_CIRCUIT, run_final_kk_screen, screen_candidate_models,
    fit_final_dataset, final_fit_table, direct_high_frequency_intercept,
    model_free_relaxation_descriptors, combined_final_fit_curves,
)

try:
    from ui.common import PLOTLY_CONFIG
except Exception:
    PLOTLY_CONFIG = {
        "scrollZoom": False, "displayModeBar": True,
        "displaylogo": False, "responsive": True,
    }


def _spectrum_label(record):
    return record.filename


def _general_import_widget():
    """Build EIS spectra without asking for a temperature or fixed ZIP layout.

    Older final-model functions expect a numeric temperature_C grouping key;
    a unique synthetic ID is used *internally only* and is never displayed
    as a measured temperature. An actual measured temperature, if supplied,
    is saved in metadata instead.
    """
    uploaded = st.file_uploader(
        "上传 EIS 文件或 ZIP（无需指定目录结构）",
        type=["txt", "csv", "tsv", "xlsx", "mpt", "dta", "idf", "ids", "zip"],
        accept_multiple_files=True,
        key="eis_general_files_v6",
    )
    if not uploaded:
        st.info("可直接上传多条 EIS 谱线。温度信息为可选项，"
                "缺少温度不影响 Nyquist、Bode、KK 和等效电路分析。")
        return [], pd.DataFrame(), ""

    candidates, scan_errors = list_candidates(uploaded)
    records, errors, digest = [], list(scan_errors), hashlib.sha256()
    display_rows = []
    for candidate in candidates:
        key = candidate.identifier
        temperature_text = (
            "" if candidate.temperature is None else f"{candidate.temperature:g}"
        )
        with st.expander(f"{candidate.path}", expanded=False):
            st.caption("自动识别格式、数据列；无法确认时请手动指定。")
            temp_str = st.text_input(
                "实验温度（°C，可留空）",
                value=temperature_text, key="eis_general_temp_v6_"+key,
            )
            actual_temp = None
            if temp_str.strip():
                try:
                    actual_temp = float(temp_str)
                    if not np.isfinite(actual_temp):
                        raise ValueError("温度不是有效数值")
                except ValueError:
                    st.warning("温度无效，已作为未标注温度处理。")
                    actual_temp = None
            mapping = None
            try:
                preview = preview_columns(candidate)
                if len(preview):
                    st.dataframe(preview.head(5), hide_index=True,
                                 width="stretch")
                    f, r, im, negative = guess_columns(preview.columns)
                    if any(col is None for col in (f, r, im)):
                        st.warning("无法识别全部阻抗列，请手动指定。")
                        fields = [str(col) for col in preview.columns]
                        f = st.selectbox("频率列 (Hz)", fields,
                                         key="eis_general_f_v6_"+key)
                        r = st.selectbox("阻抗实部 Z' (Ω)", fields,
                                         key="eis_general_r_v6_"+key)
                        im = st.selectbox("阻抗虚部列 (Ω)", fields,
                                          key="eis_general_i_v6_"+key)
                        negative = st.checkbox(
                            "所选列存储的是 -Z''（勾选后取负转换为 Z''）",
                            value=False, key="eis_general_sign_v6_"+key,
                        )
                        mapping = (f, r, im, negative)
                    elif negative:
                        st.caption("检测到负虚部 -Z''，导入后统一为 Z''。")
            except Exception:
                # Native instrument parsers handle non-tabular metadata.
                pass
        try:
            # This numeric key is an internal group index, NOT temperature.
            key_number = float(10000 + len(records))
            parsed = parse_candidate(candidate, key_number, 1, mapping)
            for rec in parsed:
                original_name = rec.filename
                existing_names = {item.filename for item in records}
                if rec.filename in existing_names:
                    suffix = 2
                    while f"{original_name} (#{suffix})" in existing_names:
                        suffix += 1
                    rec.filename = f"{original_name} (#{suffix})"
                rec.temperature_C = float(10000 + len(records))
                rec.replicate_index = 1
                rec.metadata["actual_temperature_C"] = actual_temp
                rec.metadata["spectrum_name"] = rec.filename
                rec.metadata["sample_index"] = len(records)+1
                rec.record_id = hashlib.sha256(
                    (candidate.identifier + "|" +
                     str(rec.metadata["sample_index"])).encode("utf-8")
                ).hexdigest()
                records.append(rec)
                display_rows.append({
                    "谱线": rec.filename,
                    "温度 (°C)": actual_temp,
                    "频率点": len(rec.data),
                    "导入来源": candidate.source,
                })
            digest.update(candidate.identifier.encode())
            digest.update(repr(mapping).encode())
            digest.update(str(actual_temp).encode())
        except Exception as exc:
            errors.append({
                "file": candidate.path,
                "error": f"{type(exc).__name__}: {exc}",
            })
    if display_rows:
        st.dataframe(pd.DataFrame(display_rows), hide_index=True,
                     width="stretch")
    if errors:
        with st.expander(f"未导入文件（{len(errors)}）", expanded=not records):
            st.dataframe(pd.DataFrame(errors), hide_index=True,
                         width="stretch")
    if not records:
        st.warning("尚未识别到有效 EIS 阻抗谱。请检查文件、阻抗列及单位。")
    return records, pd.DataFrame(errors), digest.hexdigest()


def _visible_table(frame, records):
    """Replace synthetic grouping IDs with original filenames and real temp."""
    table = frame.copy()
    if "temperature_C" in table.columns:
        ids = {
            float(rec.temperature_C): (rec.filename,
                                       rec.metadata.get("actual_temperature_C"))
            for rec in records
        }
        sample = table["temperature_C"].map(
            lambda x: ids.get(float(x), (f"unknown {x}", None))[0])
        actual_temp = table["temperature_C"].map(
            lambda x: ids.get(float(x), ("", None))[1])
        table.insert(0, "spectrum", sample)
        table.insert(1, "measured_temperature_C", actual_temp)
        table = table.drop(columns="temperature_C")
    return table


def _export_csv(label, table, filename, key):
    st.download_button(
        label, table.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename, mime="text/csv", key=key,
    )


def _plot_nyquist(
    records,
    final_results=None,
):
    fig = go.Figure()

    result_map = {}

    if final_results:
        result_map = {
            result.record_id:
                result
            for result
            in final_results
        }

    for record in records:
        label = record.filename

        if len(
            [
                item
                for item
                in records
                if np.isclose(
                    item.temperature_C,
                    record.temperature_C,
                )
            ]
        ) > 1:
            label += (
                f" · R{record.replicate_index}"
            )

        df = record.data

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
                    "markers",

                name=
                    label
                    + " exp",

                marker={
                    "size":
                        5,
                },
            )
        )

        result = result_map.get(
            record.record_id
        )

        if result is not None:
            fitted = (
                result.fitted_data
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
                        label
                        + " fit",
                )
            )

    fig.update_layout(
        xaxis_title=
            "Z' (Ω)",

        yaxis_title=
            "−Z'' (Ω)",

        template=
            "plotly_white",

        hovermode=
            "closest",
    )

    return fig


def _plot_bode_phase(
    records,
):
    fig = go.Figure()

    for record in records:
        df = record.data

        fig.add_trace(
            go.Scatter(
                x=
                    df[
                        "frequency_Hz"
                    ],

                y=
                    df[
                        "phase_deg"
                    ],

                mode=
                    "lines+markers",

                name=
                    record.filename,
            )
        )

    fig.update_layout(
        xaxis_type=
            "log",

        xaxis_autorange=
            "reversed",

        xaxis_title=
            "Frequency (Hz)",

        yaxis_title=
            "Phase (deg)",

        template=
            "plotly_white",
    )

    return fig


def _plot_bode_modulus(
    records,
):
    fig = go.Figure()

    for record in records:
        df = record.data

        fig.add_trace(
            go.Scatter(
                x=
                    df[
                        "frequency_Hz"
                    ],

                y=
                    df[
                        "z_modulus_ohm"
                    ],

                mode=
                    "lines+markers",

                name=
                    record.filename,
            )
        )

    fig.update_layout(
        xaxis_type=
            "log",

        yaxis_type=
            "log",

        xaxis_autorange=
            "reversed",

        xaxis_title=
            "Frequency (Hz)",

        yaxis_title=
            "|Z| (Ω)",

        template=
            "plotly_white",
    )

    return fig


def _plot_final_residuals(
    final_results,
):
    fig = go.Figure()

    for result in final_results:
        df = result.fitted_data

        fig.add_trace(
            go.Scatter(
                x=
                    df[
                        "frequency_Hz"
                    ],

                y=
                    df[
                        "relative_residual_magnitude"
                    ]
                    * 100.0,

                mode=
                    "lines+markers",

                name=
                    result.filename,
            )
        )

    fig.update_layout(
        xaxis_type=
            "log",

        xaxis_autorange=
            "reversed",

        xaxis_title=
            "Frequency (Hz)",

        yaxis_title=
            "|relative residual| (%)",

        template=
            "plotly_white",
    )

    return fig



def render_temperature_eis_screen():
    st.title("EIS 自动定量分析")
    st.caption("优先输出可复核的阻抗参数与 Origin 数据，曲线仅作预览。"
               "不需要填写温度，也不要求固定文件路径。")
    st.subheader("上传数据")
    records, import_errors, signature = _general_import_widget()
    if not records:
        return

    st.subheader("1. 数据预览")
    with st.expander("查看 Nyquist / Bode 预览曲线", expanded=True):
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(_plot_nyquist(records),width="stretch",
                            config=PLOTLY_CONFIG)
        with c2:
            st.plotly_chart(_plot_bode_phase(records),width="stretch",
                            config=PLOTLY_CONFIG)
        st.caption("图仅用于检查数据方向和质量。后续可用 Origin 自行绘制投稿图。")

    st.subheader("2. 等效电路与自动计算")
    col1,col2=st.columns([3,2])
    with col1:
        paper_model=st.selectbox(
            "选择电路（所有样品同一模型进行比较）",
            list(PAPER_MODELS), index=1,key="eis_analysis_model_v11",
        )
    with col2:
        remove_inductive=st.checkbox(
            "拟合时排除正虚部点",value=True,key="eis_analysis_inductive_v11",
        )
    st.caption("电路拟合得到的是模型参数；缺少可分辨半圆时，不应将 Rct 直接解释为电荷转移电阻。")
    options_signature=hashlib.sha256(
        repr((signature,paper_model,remove_inductive)).encode()
    ).hexdigest()
    if st.button("批量计算 EIS 参数",type="primary",
                 width="stretch",key="eis_analysis_run_v11"):
        with st.spinner("正在拟合各样品…"):
            fits,failed=fit_paper_models(records,paper_model,remove_inductive)
        st.session_state["eis_analysis_payload_v11"]=(options_signature,fits,failed)
    payload=st.session_state.get("eis_analysis_payload_v11")
    if payload and payload[0]==options_signature:
        _,fits,failed=payload
        if not failed.empty:
            st.error(f"{len(failed)} 条谱线分析失败；查看以下具体原因")
            st.dataframe(failed,hide_index=True,width="stretch")
        if fits:
            parameters=paper_parameter_table(fits)
            st.subheader("3. 参数汇总与质量检查")
            st.dataframe(parameters,hide_index=True,width="stretch")
            st.caption(
                "表格包括 Rs、Rct/Rinterface、CPE (Q,n)、Warburg 系数、"
                "复阻抗误差、收敛状态与参数警告。必须根据电路与实验现象解释结果。"
            )
            with st.expander("查看拟合曲线（仅用于核查）",expanded=False):
                from core.eis_paper_figures import draw_figure
                import matplotlib.pyplot as plt
                fig=draw_figure(records,fits,"Nyquist",width=5.6,height=3.6,inset=False)
                st.pyplot(fig,clear_figure=False)
                plt.close(fig)
            with st.expander("可选：复电容与响应时间",expanded=False):
                st.dataframe(pd.DataFrame([
                    paper_relaxation_summary(r) for r in records
                ]),hide_index=True,width="stretch")
                st.caption("表观 C* 不等于质量归一化比电容；仅在 C'' 内部峰明显时报告响应时间。")
            st.subheader("4. 导出 Origin 数据与可追溯结果")
            bundle=build_eis_analysis_zip(records,fits,failed,paper_model,remove_inductive)
            st.download_button(
                "下载 EIS 全部分析结果（ZIP，含 Origin CSV 和质量报告）",
                bundle,file_name="EIS_analysis_origin_data.zip",
                mime="application/zip",width="stretch",
                key="eis_analysis_export_v11",
            )
        elif failed.empty:
            st.warning("本次没有得到有效拟合结果。")

    with st.expander("高级：旧版 KK 与固定电路诊断",expanded=False):
        st.subheader("通用 EIS 拟合与诊断")
        st.caption(
            "当前内置定量模型：L + Rs + CPE；同时提供候选电路筛选。"
            "这是一种内置参考电路，并非适用于所有材料的唯一物理模型。"
            "其他电路可在下面的专业分析中自定义。"
        )
        fit_signature = hashlib.sha256(
            (signature + "|eis_general_v6").encode("utf-8")
        ).hexdigest()
        if st.button(
            "运行 EIS 分析", type="primary",
            width="stretch", key="eis_general_run_v6",
        ):
            stage = "初始计算"
            try:
                with st.spinner("正在进行数据质量筛查与模型拟合…"):
                    stage = "KK 类数值筛查"
                    kk = run_final_kk_screen(records)
                    stage = "候选电路筛选"
                    selection = screen_candidate_models(records)
                    stage = "L + Rs + CPE 拟合"
                    fits = fit_final_dataset(records)
                    final_table = final_fit_table(fits)
                    stage = "高频截距检查"
                    intercept = direct_high_frequency_intercept(records)
                    stage = "低频弛豫描述符"
                    relaxation = model_free_relaxation_descriptors(records)
                st.session_state["eis_general_analysis_v6"] = {
                    "signature": fit_signature,
                    "kk": kk, "selection": selection,
                    "fits": fits, "final": final_table,
                    "intercept": intercept, "relaxation": relaxation,
                }
            except Exception as exc:
                st.error(f"EIS 分析失败（{stage}）：{type(exc).__name__}: {exc}")
                st.code(traceback.format_exc(), language="text")

        payload = st.session_state.get("eis_general_analysis_v6")
        if payload and payload.get("signature") == fit_signature:
            kk = _visible_table(payload["kk"], records)
            final_table = _visible_table(payload["final"], records)
            intercept = _visible_table(payload["intercept"], records)
            relaxation = _visible_table(payload["relaxation"], records)
            selection = payload["selection"]
            fits = payload["fits"]

            t1, t2, t3 = st.tabs(["模型拟合", "数据质量", "模型比较"])
            with t1:
                st.plotly_chart(
                    _plot_nyquist(records, fits),
                    width="stretch", config=PLOTLY_CONFIG)
                st.plotly_chart(
                    _plot_final_residuals(fits),
                    width="stretch", config=PLOTLY_CONFIG)
                st.dataframe(final_table, width="stretch", hide_index=True)
                _export_csv("下载拟合参数 CSV", final_table,
                            "EIS_fitted_parameters.csv", "eis_general_fit_csv_v6")
                curves = combined_final_fit_curves(fits)
                curves = _visible_table(curves, records)
                _export_csv("下载拟合曲线 CSV", curves,
                            "EIS_fitted_curves.csv", "eis_general_curves_csv_v6")
                st.caption(
                    "局部拟合参数的 95% CI 不等同于独立重复实验的不确定度；"
                    "CPE 的 Q 也不等于理想电容。"
                )
            with t2:
                st.markdown("**KK 类数值筛查**")
                st.dataframe(kk, width="stretch", hide_index=True)
                st.caption(
                    "KK 类数值筛查是数据质量诊断，不能单凭阈值证明物理 KK 一致性。"
                )
                _export_csv("下载 KK 检查 CSV", kk,
                            "EIS_KK_screening.csv", "eis_general_kk_csv_v6")
                st.markdown("**高频截距与低频描述符**")
                st.dataframe(intercept, width="stretch", hide_index=True)
                st.dataframe(relaxation, width="stretch", hide_index=True)
                _export_csv("下载高频特征 CSV", intercept,
                            "EIS_high_frequency.csv", "eis_general_hf_csv_v6")
                _export_csv("下载低频特征 CSV", relaxation,
                            "EIS_low_frequency.csv", "eis_general_lf_csv_v6")
            with t3:
                st.dataframe(selection.candidate_summary,
                             hide_index=True, width="stretch")
                st.info(selection.rationale)
                detail = _visible_table(selection.candidate_detail, records)
                st.dataframe(detail, hide_index=True, width="stretch")
                _export_csv("下载候选电路比较 CSV", detail,
                            "EIS_circuit_comparison.csv", "eis_general_models_csv_v6")

    st.subheader("可选专业阻抗分析（Z-HIT / DRT）")
    st.caption("支持 Z-HIT、DRT 和 CDC 自定义等效电路（需要 pyimpspec）。")
    with st.expander("Z-HIT / DRT / 自定义等效电路", expanded=False):
        if not professional_available():
            st.warning(
                "尚未检测到专业 EIS 后端 pyimpspec。请在启动本项目的同一个 "
                "Python/conda 环境中运行项目根目录的 install_eis_professional.bat，"
                "或执行 `python -m pip install -r requirements_eis_professional_optional.txt`。"
                "安装完成后重启界面即可使用 Z-HIT、DRT 和自定义等效电路。"
                "基础 EIS 分析不受影响。"
            )
        else:
            st.caption(f"当前专业后端：pyimpspec {backend_version()}")
            selected = st.selectbox(
                "选择谱线", range(len(records)),
                format_func=lambda i: records[i].filename,
                key="eis_general_prof_record_v6",
            )
            record = records[selected]
            operation = st.selectbox(
                "分析类型", ["Z-HIT", "DRT (TR-NNLS)", "自定义等效电路"],
                key="eis_general_prof_op_v6",
            )
            cdc, method, weight, drt_lambda = "", "auto", "auto", -1.0
            if operation == "自定义等效电路":
                cdc = st.text_input("CDC 表达式", "R(RQ)",
                                    key="eis_general_cdc_v6")
                cc1, cc2 = st.columns(2)
                with cc1:
                    method = st.selectbox(
                        "优化方法", ["auto", "leastsq", "least_squares"],
                        key="eis_general_method_v6",
                    )
                with cc2:
                    weight = st.selectbox(
                        "拟合权重",
                        ["auto", "modulus", "proportional", "unity", "boukamp"],
                        key="eis_general_weight_v6",
                    )
            elif operation == "DRT (TR-NNLS)":
                drt_lambda = st.number_input(
                    "正则化 λ（-1：自动）", value=-1.0, format="%.5g",
                    key="eis_general_lambda_v6",
                )
            setting_sig = hashlib.sha256(
                repr((signature, record.filename, record.record_id,
                      operation, cdc, method, weight, drt_lambda)).encode()
            ).hexdigest()
            if st.button("运行专业分析", key="eis_general_prof_run_v6"):
                try:
                    with st.spinner("专业 EIS 后端分析中…"):
                        summary, points, extra = run_professional(
                            record, operation=operation, cdc=cdc,
                            method=method, weight=weight,
                            drt_lambda=drt_lambda,
                        )
                    st.session_state["eis_general_prof_result_v6"] = (
                        setting_sig, summary, points, extra,
                    )
                except Exception as exc:
                    st.error(f"专业分析失败：{type(exc).__name__}: {exc}")
            result = st.session_state.get("eis_general_prof_result_v6")
            if result and result[0] == setting_sig:
                _, summary, points, extra = result
                st.dataframe(summary, hide_index=True, width="stretch")
                if extra is not None:
                    st.dataframe(extra, hide_index=True, width="stretch")
                if {"z_real_ohm", "z_imag_ohm"}.issubset(points.columns):
                    fig = go.Figure()
                    fig.add_scatter(
                        x=record.data["z_real_ohm"],
                        y=-record.data["z_imag_ohm"],
                        mode="markers", name="实验")
                    fig.add_scatter(
                        x=points["z_real_ohm"], y=-points["z_imag_ohm"],
                        mode="lines", name=operation)
                    fig.update_layout(xaxis_title="Z' (Ω)",
                                      yaxis_title="-Z'' (Ω)")
                    st.plotly_chart(fig, width="stretch")
                elif {"tau_s", "gamma"}.issubset(points.columns):
                    fig = go.Figure()
                    fig.add_scatter(x=points["tau_s"], y=points["gamma"],
                                    mode="lines", name="DRT")
                    fig.update_layout(xaxis_type="log",
                                      xaxis_title="Relaxation time τ (s)",
                                      yaxis_title="γ(τ)")
                    st.plotly_chart(fig, width="stretch")
                st.dataframe(points, hide_index=True, width="stretch")
                _export_csv(
                    "下载专业分析结果 CSV", points,
                    "EIS_professional_result.csv", "eis_general_prof_csv_v6",
                )
                st.caption(
                    "Z-HIT/DRT/电路拟合只提供数值证据，"
                    "不保证电路或弛豫峰存在唯一的物理解释。"
                )
