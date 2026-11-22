"""Optional, analysis-first CV kinetics and GCD performance UI."""
from __future__ import annotations

import hashlib
import pandas as pd
import streamlit as st

from core.supercap_kinetics import (
    analyse_cv_kinetics, gcd_rate_summary, gcd_cycling_summary,make_kinetics_zip,
)


def _choices(items):
    return {f"{index+1:02d} | {item['data'].filename} | {item['config']['label']}":index
            for index,item in enumerate(items)}


def _status_table(data):
    st.dataframe(data,hide_index=True,width="stretch")


def render_supercap_kinetics(state):
    """Show alongside the existing CV/GCD/EIS results, without altering them."""
    cv_items=state["results"].get("CV",[])
    gcd_items=state["results"].get("GCD",[])
    # v12 changes this digest when files or their calculation settings change.
    # Include it in every v13 cache key to prevent stale cross-file results.
    parent_digest=st.session_state.get("electrochem_result_signature","")
    if not cv_items and not gcd_items:
        return
    st.divider()
    st.subheader("进阶：超级电容器动力学与倍率/循环分析")
    st.caption("仅在输入满足实验前提时计算；重点导出参数与 Origin CSV，不进行论文图排版。"
               "请选择属于同一个物理样品的实验文件。")

    tabs=st.tabs(["CV · b 值 / Dunn 分析",
                  "GCD · 倍率性能","GCD · 循环保持率"])
    with tabs[0]:
        st.markdown("**相同电极、不同扫描速率的完整 CV 曲线**")
        st.caption("至少 3 个不同扫描速率。分别分析正扫/反扫，"
                   "对共同电位区间插值。b 值来源于 log|i(V)|–log v，"
                   "Dunn 使用 i(V)=k₁v+k₂√v。")
        opts=_choices(cv_items)
        selection=st.multiselect("选择同一物理样品的 CV 数据（不选则不计算）",
                    options=list(opts),key="v13_cv_files")
        branch=st.selectbox("扫描分支",["forward","reverse"],
                            format_func=lambda x:"正向扫描（电位增加）" if x=="forward"
                            else "反向扫描（电位降低）",
                            key="v13_cv_branch")
        grid=st.slider("共同电位区间采样点",50,300,120,10,key="v13_cv_points")
        confirmed=st.checkbox("已确认：选中数据来自同一物理样品，"
            "电解液、电位窗口及其他实验条件可比",
            key="v13_cv_confirm")
        chosen=[cv_items[opts[k]] for k in selection]
        sig=hashlib.sha256(repr((
            [(i["data"].filename,i["config"]["scan_rate_V_s"],
              i["config"]["cycle_number"]) for i in chosen],
            branch,grid,confirmed,parent_digest)).encode()).hexdigest()
        if st.button("计算 CV b 值和 Dunn 分析",key="v13_cv_run",
                     type="primary",disabled=not confirmed or len(chosen)<3):
            try:
                with st.spinner("正在对多扫描速率 CV 进行拟合…"):
                    result=analyse_cv_kinetics(chosen,branch,grid)
                    st.session_state["v13_cv_result"]=(sig,result)
            except Exception as exc:
                st.session_state.pop("v13_cv_result",None)
                st.error(f"无法分析：{exc}")
        entry=st.session_state.get("v13_cv_result")
        if entry and entry[0]==sig:
            result=entry[1]
            summary=pd.DataFrame([result["summary"]])
            st.markdown("**拟合摘要**")
            _status_table(summary)
            st.markdown("**不同扫描速率下的 Dunn 贡献**")
            _status_table(result["dunn_by_scan_rate"])
            with st.expander("逐电位 b 值、Dunn 系数及插值曲线"):
                _status_table(result["b_vs_potential"])
                st.caption("出现负的分量贡献、零电流、符号变换或不可可靠拟合时，"
                           "贡献百分比留空，而非强制截断到 0–100%。")
            files={"CV/b_vs_potential.csv":result["b_vs_potential"],
                   "CV/dunn_contributions.csv":result["dunn_by_scan_rate"],
                   "CV/interpolated_currents.csv":result["interpolated_curves"],
                   "CV/selected_potential_log_fit.csv":result["selected_potential_data"]}
            zipdata=make_kinetics_zip(files,result["summary"])
            st.download_button("下载 CV 动力学分析与 Origin 数据",
                data=zipdata,file_name="CV_kinetics_origin_v13.zip",
                mime="application/zip",key="v13_cv_download")
            st.warning("b 与 Dunn 表示扫描速率依赖的动力学贡献，"
                       "不能单独证明储能反应的化学身份；请检查 R²、"
                       "符号一致性、是否存在峰位移动及共同电位区间。")

    with tabs[1]:
        st.markdown("**同一物理样品，不同恒流密度**")
        opts=_choices(gcd_items)
        selection=st.multiselect("选择同一物理样品的 GCD 数据",
                   options=list(opts),key="v13_gcd_rate_files")
        verified=st.checkbox("已确认电极/质量/放电电压窗口和"
                   "计算约定可比，所选文件均来自同一物理样品",
                   key="v13_gcd_rate_confirm")
        chosen=[gcd_items[opts[k]] for k in selection]
        sig=hashlib.sha256(repr([
           (x["data"].filename,x["config"]["current_A"],x["config"]["mass_mg"])
           for x in chosen]+[verified,parent_digest]).encode()).hexdigest()
        if st.button("计算 GCD 倍率保持率",key="v13_gcd_rate_run",
                     disabled=len(chosen)<2 or not verified):
            try:
                frame=gcd_rate_summary(chosen)
                st.session_state["v13_gcd_rate_result"]=(sig,frame)
            except Exception as exc:
                st.session_state.pop("v13_gcd_rate_result",None)
                st.error(f"无法计算倍率性能：{exc}")
        entry=st.session_state.get("v13_gcd_rate_result")
        if entry and entry[0]==sig:
            frame=entry[1]
            _status_table(frame)
            st.download_button("下载 GCD 倍率分析 Origin 数据",
                data=make_kinetics_zip({"GCD/rate_capability.csv":frame},
                {"reference":"lowest current density","basis":"same electrode"}),
                file_name="GCD_rate_capability_v13.zip",
                mime="application/zip",key="v13_gcd_rate_download")
            st.caption("倍率保持率以所选数据中**最低电流密度**对应的比电容为 100%。")

    with tabs[2]:
        st.markdown("**单个连续 GCD 文件中的多个完整循环**")
        if not gcd_items:
            st.info("没有可供分析的 GCD 文件。")
        else:
            opts=_choices(gcd_items)
            selected=st.selectbox("选择连续循环文件",list(opts),
                key="v13_gcd_cycle_file")
            index=opts[selected]
            item=gcd_items[index]
            ref_cycle=st.number_input("参考循环号（默认 1）",
                 min_value=1,value=1,step=1,key="v13_gcd_cycle_ref")
            confirmed=st.checkbox("已确认：记录为同一物理样品的连续循环，"
                   "循环期间电流和质量基准不变",
                   key="v13_gcd_cycle_confirm")
            sig=hashlib.sha256(repr((
                selected,ref_cycle,confirmed,
                item["config"].get("current_A"),item["config"].get("mass_mg"),
                parent_digest
            )).encode()).hexdigest()
            if st.button("逐循环计算容量保持率",key="v13_gcd_cycle_run",
                         disabled=not confirmed):
                try:
                    with st.spinner("逐循环识别放电段和计算参数…"):
                        frame,failures,reference=gcd_cycling_summary(
                            item,int(ref_cycle))
                    st.session_state["v13_gcd_cycle_result"]=(
                        sig,frame,failures,reference)
                except Exception as exc:
                    st.session_state.pop("v13_gcd_cycle_result",None)
                    st.error(f"循环分析失败：{exc}")
            entry=st.session_state.get("v13_gcd_cycle_result")
            if entry and entry[0]==sig:
                _,frame,failures,reference=entry
                st.metric("最后一个有效循环的保持率",
                  f"{frame['Retention vs reference (%)'].iloc[-1]:.2f}%")
                st.caption(f"相对于第 {reference} 个有效循环。"
                    "循环点不是自动推算数据；仅统计已识别的真实放电段。")
                _status_table(frame)
                if not failures.empty:
                    st.warning(f"{len(failures)} 个循环未成功分析，请检查失败原因")
                    with st.expander("无法分析的循环"):
                        _status_table(failures)
                st.download_button("下载 GCD 循环保持率 Origin 数据",
                    data=make_kinetics_zip({
                        "GCD/cycle_retention.csv":frame,
                        "GCD/failed_cycles.csv":failures,
                    },{"reference_cycle":reference,
                       "source":item["data"].filename}),
                    file_name="GCD_cycling_retention_v13.zip",
                    mime="application/zip",key="v13_gcd_cycle_download")
