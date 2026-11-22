"""Technique-specific advanced research analysis, excluding long-cycle workflows."""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.supercap_kinetics import (
    analyse_cv_kinetics, gcd_rate_summary, make_kinetics_zip,
)
from core.literature_methods_v17 import cv_peak_b_fits, eis_compare_fits
from core.cv_parameters_v18 import kinetics_signature
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from screens.madap_dashboard import scientific_axes



def render_extension(technique, items, analysis_signature):
    if technique == "CV":
        st.markdown("#### 多扫描速率动力学：b 值与 Dunn 方法")
        st.caption("同一物理样品至少三种不同扫描速率，"
                   "只在共同电位范围内比较相同扫描方向。")
        choices={f"{i+1}. {x['data'].filename} · {x['config']['scan_rate_V_s']*1000:g} mV/s":i
                 for i,x in enumerate(items)}
        selected=st.multiselect("选择同一样品的 CV 数据",
            list(choices),key="v16_cv_kinetics_selected")
        branch=st.selectbox("扫描方向",["forward","reverse"],
                            format_func=lambda x:"正扫" if x=="forward" else "反扫",
                            key="v16_cv_kinetics_branch")
        verified=st.checkbox("已核对：样品、实验条件和电位窗口可比较",
            key="v16_cv_kinetics_same_sample")
        manual=st.checkbox("限定氧化/还原峰电位范围（建议存在多个氧化还原峰时使用）",
                           key="v17_peak_manual_window")
        windows=None
        if manual and selected:
            chosen_items=[items[choices[x]] for x in selected]
            min_v=min(float(x["plot_df"].potential_V.min()) for x in chosen_items)
            max_v=max(float(x["plot_df"].potential_V.max()) for x in chosen_items)
            if min_v<max_v:
                st.caption(f"当前选择的 CV 电位范围约为 {min_v:.3f}～{max_v:.3f} V")
                lo_a=st.number_input("氧化峰窗口下限 (V)",value=min_v,
                                     format="%.4f",key="v17_anodic_min")
                hi_a=st.number_input("氧化峰窗口上限 (V)",value=max_v,
                                     format="%.4f",key="v17_anodic_max")
                lo_c=st.number_input("还原峰窗口下限 (V)",value=min_v,
                                     format="%.4f",key="v17_cathodic_min")
                hi_c=st.number_input("还原峰窗口上限 (V)",value=max_v,
                                     format="%.4f",key="v17_cathodic_max")
                windows={"anodic":(lo_a,hi_a),"cathodic":(lo_c,hi_c)}
        signature=kinetics_signature(
            analysis_signature,selected,branch,verified,manual,windows)
        if st.button("计算 b 值 / Dunn",disabled=len(selected)<3 or not verified,
                     key="v16_cv_kinetics_run"):
            try:
                subset=[items[choices[x]] for x in selected]
                result=analyse_cv_kinetics(subset,branch)
                # Both directions are separately fitted for the peak-current
                # b values; invalid peak attribution remains user-reviewable.
                peak_error=None
                try:
                    peak_summary,peak_points=cv_peak_b_fits(subset,windows)
                except ValueError as exc:
                    peak_summary=pd.DataFrame()
                    peak_points=pd.DataFrame()
                    peak_error=str(exc)
                st.session_state["v16_cv_kinetics_result"]=(
                    signature,result,peak_summary,peak_points,peak_error)
                # The main Analysis parameters table is rendered above this
                # advanced expander; re-run once so it updates immediately.
                st.rerun()
            except Exception as exc:
                st.session_state.pop("v16_cv_kinetics_result",None)
                st.error(str(exc))
        stored=st.session_state.get("v16_cv_kinetics_result")
        if stored and stored[0]==signature:
            _,r,peak_summary,peak_points,peak_error=stored
            st.markdown("**峰电流 b 值：氧化/还原分支分别拟合**")
            if peak_error:
                st.info("未输出峰电流 b 值："+peak_error+
                        "。Dunn 分析仍可以独立显示。")
            else:
                st.dataframe(peak_summary,hide_index=True,width="stretch")
                st.warning("峰电流是正扫/反扫的候选极值，"
                           "须核对是否属于真正氧化还原峰；"
                           "峰位明显移动或峰在电位边界时应人工复核。")
            st.markdown("**逐电位 b 值与 Dunn 贡献**")
            st.dataframe(pd.DataFrame([r["summary"]]),hide_index=True,width="stretch")
            st.dataframe(r["dunn_by_scan_rate"],hide_index=True,width="stretch")
            _fig=make_subplots(rows=1,cols=2,
                subplot_titles=("Peak current b-value fit",
                                "Dunn contribution vs scan rate"),
                horizontal_spacing=.26)
            for direction in ("anodic","cathodic"):
                if peak_points.empty:
                    break
                part=peak_points[peak_points.Branch==direction]
                _fig.add_trace(go.Scatter(x=part["log scan rate"],
                    y=part["log abs peak current"],mode="markers",
                    name=direction),row=1,col=1)
                _fig.add_trace(go.Scatter(x=part["log scan rate"],
                    y=part["log predicted current"],mode="lines",
                    name=direction+" fit"),row=1,col=1)
            contribution=r["dunn_by_scan_rate"]
            _fig.add_trace(go.Scatter(
                x=contribution["Scan rate (V/s)"]*1000,
                y=contribution["Capacitive contribution (%)"],
                mode="lines+markers",name="Capacitive (%)"),row=1,col=2)
            _fig.update_xaxes(title_text="log scan rate (V/s)",row=1,col=1)
            _fig.update_yaxes(title_text="log |peak current (A)|",row=1,col=1)
            _fig.update_xaxes(title_text="Scan rate (mV/s)",row=1,col=2)
            _fig.update_yaxes(title_text="Capacitive contribution (%)",row=1,col=2)
            scientific_axes(_fig,gutter=True)
            st.plotly_chart(_fig,width="stretch")
            st.markdown("**Dunn 逐电位分量（示例扫描速率）**")
            _curves=r["interpolated_curves"]
            first_rate=float(_curves["Scan rate (V/s)"].min())
            _sub=_curves[_curves["Scan rate (V/s)"]==first_rate]
            _dunn=go.Figure()
            for column,label in [
                ("Current experimental (A)","Measured current"),
                ("Capacitive current (A)","k1 v"),
                ("Diffusive current (A)","k2 sqrt(v)"),
                ("Current reconstructed (A)","Reconstructed"),
            ]:
                _dunn.add_trace(go.Scatter(
                    x=_sub["Potential (V)"],y=_sub[column]*1000,
                    name=label,mode="lines"))
            _dunn.update_xaxes(title_text="Potential (V)")
            _dunn.update_yaxes(title_text="Current (mA)")
            scientific_axes(_dunn,gutter=True)
            st.plotly_chart(_dunn,width="stretch")
            if (_sub["Admissible"].astype(bool)==False).any():
                st.warning("当前扫描速率有电容/扩散分量符号不一致的电位点；"
                           "不能将相应的百分比解释为可靠的储能占比。")
            package=make_kinetics_zip({
                 "CV/peak_b_summary.csv":peak_summary,
                 "CV/peak_b_fit_points.csv":peak_points,
                 "CV/b_vs_potential.csv":r["b_vs_potential"],
                 "CV/dunn_contributions.csv":r["dunn_by_scan_rate"],
                 "CV/interpolated_curves.csv":r["interpolated_curves"],
                 "CV/selected_potential_data.csv":r["selected_potential_data"],
                },r["summary"])
            st.download_button("导出峰电流 b 值 / Dunn 分析 CSV",
                package,file_name="CV_kinetics_origin.zip",
                mime="application/zip",key="v16_cv_kinetics_download")
    elif technique == "GCD":
        st.markdown("#### 倍率性能")
        st.caption("选择同一样品的不同电流密度数据，"
                   "以最低电流密度的比电容作为参考值。不包含长循环分析。")
        choices={f"{i+1}. {x['data'].filename}":i for i,x in enumerate(items)}
        selected=st.multiselect("选择同一样品的 GCD 数据",
                    list(choices),key="v16_gcd_rate_selected")
        verified=st.checkbox("已核对：同一物理样品、质量、电流及电压窗口可比较",
                    key="v16_gcd_rate_confirm")
        signature=hashlib.sha256(repr((analysis_signature,selected,verified)).encode()).hexdigest()
        if st.button("计算倍率保持率",
                     disabled=len(selected)<2 or not verified,
                     key="v16_gcd_rate_run"):
            try:
                frame=gcd_rate_summary([items[choices[x]] for x in selected])
                st.session_state["v16_gcd_rate_result"]=(signature,frame)
            except Exception as exc:
                st.session_state.pop("v16_gcd_rate_result",None)
                st.error(str(exc))
        stored=st.session_state.get("v16_gcd_rate_result")
        if stored and stored[0]==signature:
            frame=stored[1]
            st.dataframe(frame,hide_index=True,width="stretch")
            _fig=make_subplots(rows=1,cols=2,
                subplot_titles=("Specific capacitance vs current density",
                                "Rate retention vs current density"),
                horizontal_spacing=.25)
            _fig.add_trace(go.Scatter(x=frame["Current density (A/g)"],
                y=frame["Specific capacitance (F/g)"],
                mode="lines+markers",name="Specific capacitance"),row=1,col=1)
            _fig.add_trace(go.Scatter(x=frame["Current density (A/g)"],
                y=frame["Rate retention vs lowest current (%)"],
                mode="lines+markers",name="Retention"),row=1,col=2)
            _fig.update_xaxes(title_text="Current density (A/g)")
            _fig.update_yaxes(title_text="Specific capacitance (F/g)",row=1,col=1)
            _fig.update_yaxes(title_text="Rate retention (%)",row=1,col=2)
            scientific_axes(_fig,gutter=True)
            st.plotly_chart(_fig,width="stretch")
            package=make_kinetics_zip({
                "GCD/rate_capability.csv":frame
                },{"reference":"lowest current density",
                   "basis":"same sample and consistent mass"})
            st.download_button("导出 GCD 倍率结果 CSV",
                package,file_name="GCD_rate_origin.zip",mime="application/zip",
                key="v16_gcd_rate_download")


def render_eis_comparison(items,analysis_signature):
    """Uniform-model EIS parameter comparison, without forcing a fit model."""
    st.markdown("#### 多样品 EIS 拟合参数比较")
    st.caption("仅比较使用同一电路、频率范围、拟合权重与感抗处理条件的结果。")
    choices={f"{i+1}. {x['data'].filename}":i for i,x in enumerate(items)}
    selected=st.multiselect("选择待比较的 EIS 样品",
                list(choices),key="v17_eis_compare_selection")
    signature=hashlib.sha256(repr((analysis_signature,selected)).encode()).hexdigest()
    if st.button("汇总 Rs / Rct / 拟合误差",disabled=len(selected)<2,
                 key="v17_eis_compare_run"):
        try:
            summary=eis_compare_fits([items[choices[k]] for k in selected])
            st.session_state["v17_eis_compare_result"]=(signature,summary)
        except Exception as exc:
            st.session_state.pop("v17_eis_compare_result",None)
            st.error(f"无法直接比较：{exc}")
    stored=st.session_state.get("v17_eis_compare_result")
    if stored and stored[0]==signature:
        frame=stored[1]
        st.dataframe(frame,hide_index=True,width="stretch")
        fig=make_subplots(rows=1,cols=2,
            subplot_titles=("Fitted Rs by sample","Fitted Rct by sample"),
            horizontal_spacing=.28)
        for c,col in [(1,"Rs fit (ohm)"),(2,"Rct fit (ohm)")]:
            fig.add_trace(go.Bar(x=frame.Sample,y=frame[col],
                name=col),row=1,col=c)
            fig.update_yaxes(title_text=col,row=1,col=c)
            fig.update_xaxes(title_text="Sample",row=1,col=c)
        scientific_axes(fig,gutter=True)
        st.plotly_chart(fig,width="stretch")
        st.download_button("下载 EIS 拟合参数比较 CSV",
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="EIS_fitted_parameters_comparison.csv",
            mime="text/csv",key="v17_eis_compare_download")
