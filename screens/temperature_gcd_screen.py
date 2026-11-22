"""GCD temperature prediction — paper-stage3.3 sole PCA-GPR scientific workflow."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pca_gpr_paper33 import (
    training_manifest, load_training_records, find_heldout_record,
    fit_and_predict_paper33, predict_pca_gpr_full_curve,
    physical_prediction_as_frame, paper33_heldout_comparison,
)

BUNDLED=Path(__file__).resolve().parents[1]/"data"/"gcd_paper_stage33_training.zip"
PAPER_CURRENTS=(1.0,1.5,2.0,2.5)
PAPER_EVALUATION_CURRENTS=(.5,1.0,1.5,2.0,2.5)
MODEL_VERSION="paper-stage3.3-PCA-GPR-only-v20"


def _get_training_source():
    generation=int(st.session_state.get("stage33_gpr_upload_version",0))
    custom=st.file_uploader(
        "可选：替换为同目录格式的原始 GCD ZIP",
        type=["zip"],key=f"stage33_gpr_upload_{generation}",
        help="默认直接使用你上传 paper-stage3.3 的 testwendu3 原始 GCD 数据")
    if custom is not None:
        raw=custom.getvalue()
        if len(raw)>30_000_000:
            st.error("上传 ZIP 超过 30 MB；请仅包含训练所需的 GCD TXT")
            return None,None
        st.session_state["stage33_gpr_custom_raw"]=raw
        st.session_state["stage33_gpr_custom_name"]=custom.name
    raw=st.session_state.get("stage33_gpr_custom_raw")
    if raw is None:
        return BUNDLED.read_bytes(),"paper-stage3.3 原始 GCD"
    return raw,st.session_state.get("stage33_gpr_custom_name","自定义 GCD ZIP")


def _style(fig,height=430):
    fig.update_layout(template="simple_white",height=height,
        margin=dict(l=85,r=55,t=65,b=75),hovermode="closest",
        paper_bgcolor="white",plot_bgcolor="white",
        legend=dict(orientation="h",yanchor="bottom",y=1.1,x=0),
        font=dict(family="Arial",size=11,color="#304944"))
    fig.update_xaxes(showline=True,mirror=True,linewidth=1.3,
        linecolor="#344c47",ticks="outside",showgrid=False,
        zeroline=False,title_standoff=12,automargin=True)
    fig.update_yaxes(showline=True,mirror=True,linewidth=1.3,
        linecolor="#344c47",ticks="outside",showgrid=False,
        zeroline=False,title_standoff=10,automargin=True)
    return fig


def _single_plot(result):
    pred=result["prediction"]
    actual=result.get("comparison")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pred["time_s"],y=pred["predicted_potential_V"],
        mode="lines",name="PCA-GPR prediction",
        line=dict(width=2.4,color="#277d76")))
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=actual["measured_time_s"],y=actual["measured_potential_V"],
            mode="lines",name="Held-out measurement",
            line=dict(width=1.9,color="#bf7e4b")))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Potential (V)")
    return _style(fig)


def _four_current_plot(result):
    curves=result["paper_curves"]
    fig=go.Figure()
    colors=["#267c72","#b57551","#6a78a5","#958155"]
    for (current,df),color in zip(curves.items(),colors):
        fig.add_trace(go.Scatter(
            x=df["time_s"],y=df["predicted_potential_V"],
            mode="lines",name=f"{current:g} A/g",
            line=dict(color=color,width=2)))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Potential (V)")
    return _style(fig,height=450)


def _zip_export(result):
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        def csv(name,df):
            z.writestr(name,df.to_csv(index=False).encode("utf-8-sig"))
        csv("PCA_GPR_single_prediction_Origin.csv",result["prediction"])
        csv("PCA_GPR_training_inventory.csv",result["training_manifest"])
        csv("PCA_GPR_paper_four_current_periods.csv",result["paper_periods"])
        for current,curve in result["paper_curves"].items():
            csv(f"PCA_GPR_four_currents_{current:g}Ag.csv",curve)
        if result.get("comparison") is not None:
            csv("PCA_GPR_measured_vs_prediction_Origin.csv",result["comparison"])
        if result.get("metrics") is not None:
            csv("PCA_GPR_heldout_metrics.csv",pd.DataFrame([result["metrics"]]))
        config={
            "paper_source":"paper-stage3.3, GCD PCA-GPR, Fig4-5 and Fig4-6",
            "model":"PCA-GPR ONLY",
            "model_version":MODEL_VERSION,
            "data_sha256":result["source_hash"],
            "concentration_mol_L":result["concentration"],
            "target_temperature_C":result["temperature_C"],
            "target_current_A_g":result["current_A_g"],
            "training_temperatures_C":result["training_temperatures"],
            "training_records":result["training_count"],
            "train_rule":"strictly lower temperatures only",
            "model_kernel":"ARD Matern 3/2 + WhiteKernel; GaussianProcessRegressor",
            "PCA_normalized_grid_points":401,
            "PCA_variance_target":0.995,
            "PCA_max_components":10,
            "paper_figure_currents_A_g":list(PAPER_CURRENTS),
            "paper_validation_currents_A_g":list(PAPER_EVALUATION_CURRENTS),
            "predicted_total_period_s":result["predicted_period"],
            "model_period_ci95_s":result["period_ci95"],
            "PCA_components":result["n_components"],
            "PCA_explained_variance":result["explained"],
            "notes":["95% GPR interval is model-based, not experimentally calibrated",
                     "paper Fig4-5/6 use four currents, evaluation uses five",
                     "a withheld actual record is used only to assess the prediction"],
        }
        z.writestr("PCA_GPR_reproducibility.json",
                   json.dumps(config,ensure_ascii=False,indent=2).encode("utf-8"))
    return out.getvalue()


def render_temperature_gcd_screen():
    st.title("GCD 温度预测")
    st.caption("与大论文第四章对应 · 仅 PCA-GPR · GCD 最后完整充放电周期")
    st.caption("采用 paper-stage3.3 的曲线 PCA 降维、温度与 ln(电流密度) "
               "输入、Gaussian Process 回归；图 4-5/4-6 为四电流曲线展示。")

    if st.button("恢复论文项目内置训练数据",key="stage33_gpr_reset"):
        for k in ("stage33_gpr_custom_raw","stage33_gpr_custom_name",
                  "stage33_gpr_prediction_result"):
            st.session_state.pop(k,None)
        st.session_state["stage33_gpr_upload_version"]=int(
            st.session_state.get("stage33_gpr_upload_version",0))+1
        st.rerun()

    raw,label=_get_training_source()
    if raw is None:return
    try:
        manifest=training_manifest(raw)
    except Exception as exc:
        st.error("训练数据无法识别："+str(exc))
        return
    source_hash=hashlib.sha256(raw).hexdigest()
    st.caption(f"数据源：{label}；识别到 {len(manifest)} 条 GCD 原始曲线。")
    with st.expander("查看原始训练数据范围与论文方法边界"):
        summary=manifest.groupby("concentration_mol_L").agg(**{
            "训练曲线数":("filename","count"),
            "实验温度数":("temperature_C","nunique"),
            "最低温度":("temperature_C","min"),
            "最高温度":("temperature_C","max"),
        }).reset_index()
        st.dataframe(summary,hide_index=True,width="stretch")
        st.info("1、2、4 mol/L 分浓度训练。论文 6 mol/L 仅有 30、40 °C，"
                "不足以进行至少两级低温训练后的严格温度留出检验，"
                "因此不提供无依据的高温预测。")

    left,right=st.columns([.36,.64],gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 预测条件")
            eligible=[]
            for c in sorted(manifest.concentration_mol_L.unique()):
                part=manifest[manifest.concentration_mol_L==c]
                if len(part)>=8 and part.temperature_C.nunique()>=3:
                    eligible.append(int(c))
            if not eligible:
                st.error("未找到至少三个温度且具有足够 GCD 的浓度组")
                return
            concentration=st.selectbox(
                "电解液浓度 (mol/L)",eligible,
                key="stage33_gpr_concentration")
            part=manifest[manifest.concentration_mol_L==concentration]
            temps=sorted(float(x) for x in part.temperature_C.unique())
            earliest=temps[2]
            preferred={1:70.,2:60.,4:50.}.get(concentration,earliest)
            default=preferred if preferred>=earliest else earliest
            paper_limit={1:70.,2:60.,4:50.}.get(concentration, max(temps))
            exploration=st.checkbox(
                "允许超出论文正式验证温区（探索性外推）",
                value=False,key="stage33_gpr_allow_exploration",
                help="论文正式有效范围：1 mol/L 至70°C，2 mol/L 至60°C，"
                     "4 mol/L 至50°C。高于此范围时不应使用论文表4-1准确率背书。")
            temperature=st.number_input(
                "目标温度 (°C)",min_value=float(earliest),
                max_value=160. if exploration else float(paper_limit),
                value=float(default),
                step=10.,format="%.1f",
                key=("stage33_gpr_temperature_exploratory"
                     if exploration else "stage33_gpr_temperature_paper"))
            current=st.number_input(
                "目标电流密度 (A/g)",min_value=.01,max_value=50.,
                value=1.,step=.5,format="%.2f",
                key="stage33_gpr_current")
            train_part=part[part.temperature_C<float(temperature)]
            train_temps=sorted(float(x) for x in train_part.temperature_C.unique())
            st.caption("用于训练的温度："+", ".join(f"{x:g} °C" for x in train_temps))
            st.caption(f"用于训练的曲线：{len(train_part)} 条；目标温度数据不参加训练。")
            if temperature>paper_limit:
                st.warning("目标温度超出大论文正式有效分析范围：该预测仅供探索，"
                           "不能直接沿用论文表4-1的模型准确率。")
            if temperature>max(temps):
                st.warning("目标温度也超出原始数据最高实测温度，属于更远的外推。")
            elif temperature in temps:
                st.info("该目标温度有实测记录，但将严格留出以进行独立条件对照。")
            min_curr=float(train_part.current_A_g.min())
            max_curr=float(train_part.current_A_g.max())
            if current<min_curr or current>max_curr:
                st.warning(f"电流密度已超出训练范围 {min_curr:g}–{max_curr:g} A/g。")
            st.caption("论文图 4-5/4-6：1、1.5、2、2.5 A/g；"
                       "模型周期误差评价采用 0.5–2.5 A/g 共五档。")
            signature=hashlib.sha256(repr((
                MODEL_VERSION,source_hash,concentration,temperature,current
            )).encode()).hexdigest()
            if st.button("▶ PCA-GPR 训练并预测",type="primary",
                         width="stretch",key="stage33_gpr_run"):
                st.info("已接收到预测请求，正在按严格低温训练规则执行 PCA-GPR…")
                try:
                    with st.spinner("拟合 PCA 潜在得分及总周期的高斯过程…"):
                        records=load_training_records(
                            raw,manifest,concentration,temperature)
                        model,prediction=fit_and_predict_paper33(
                            records,temperature,current,
                            uncertainty_samples=120)
                        single=physical_prediction_as_frame(prediction)
                        observed=find_heldout_record(
                            raw,manifest,concentration,temperature,current)
                        comparison,metrics=None,None
                        if observed is not None:
                            comparison,metrics=paper33_heldout_comparison(
                                prediction,observed)
                        paper_curves={}
                        paper_period_rows=[]
                        for i in PAPER_CURRENTS:
                            pred_i=predict_pca_gpr_full_curve(
                                model,temperature,i,uncertainty_samples=0)
                            curve_i=physical_prediction_as_frame(pred_i)
                            paper_curves[i]=curve_i
                            paper_period_rows.append({
                                "Current density (A/g)":i,
                                "Predicted total period (s)":
                                    pred_i.predicted_total_period_s})
                        result={
                            "prediction":single,"comparison":comparison,
                            "metrics":metrics,"paper_curves":paper_curves,
                            "paper_periods":pd.DataFrame(paper_period_rows),
                            "training_manifest":train_part,
                            "source_hash":source_hash,
                            "concentration":int(concentration),
                            "temperature_C":float(temperature),
                            "current_A_g":float(current),
                            "predicted_period":prediction.predicted_total_period_s,
                            "period_ci95":(
                                prediction.total_period_ci95_low_s,
                                prediction.total_period_ci95_high_s),
                            "charge_period":prediction.predicted_charge_duration_s,
                            "discharge_period":prediction.predicted_discharge_duration_s,
                            "n_components":int(model.pca.n_components_),
                            "explained":model.explained_variance_ratio_sum,
                            "training_temperatures":train_temps,
                            "training_count":len(records),
                        }
                        st.session_state["stage33_gpr_prediction_result"]=(
                            signature,result)
                    st.success("PCA-GPR 已完成训练和预测。")
                except Exception as exc:
                    st.session_state.pop("stage33_gpr_prediction_result",None)
                    st.error("预测失败："+str(exc))
            saved=st.session_state.get("stage33_gpr_prediction_result")
            if saved and saved[0]!=signature:
                st.info("条件已改变；旧预测不再展示，请重新计算。")
    with right:
        with st.container(border=True):
            st.markdown("#### PCA-GPR 曲线与模型验证")
            saved=st.session_state.get("stage33_gpr_prediction_result")
            if not saved or saved[0]!=signature:
                st.info("输入预测条件并点击左侧「PCA-GPR 训练并预测」。")
                return
            result=saved[1]
            st.plotly_chart(_single_plot(result),width="stretch")
            m1,m2,m3=st.columns(3)
            m1.metric("预测完整周期",f"{result['predicted_period']:.2f} s")
            m2.metric("预测充电时间",f"{result['charge_period']:.2f} s")
            m3.metric("预测放电时间",f"{result['discharge_period']:.2f} s")
            st.caption(f"PCA：{result['n_components']} 个主成分，"
                       f"累计解释方差 {result['explained']*100:.2f}%；"
                       f"模型形式的周期 95% 区间 "
                       f"[{result['period_ci95'][0]:.2f}, "
                       f"{result['period_ci95'][1]:.2f}] s，"
                       "未证明具有外推条件下的频率学覆盖率。")
            if result["metrics"] is not None:
                st.markdown("**目标条件实测留出验证**")
                met=result["metrics"]
                st.dataframe(pd.DataFrame([met]),hide_index=True,width="stretch")
                if abs(met["Period relative error (%)"])>15 or met["Curve RMSE (mV)"]>50:
                    st.warning("该条件下预测误差较大，不能仅凭拟合曲线用于论文结论。")
            else:
                st.caption("无对应实测曲线，不能报告当前条件的实测预测误差。")
            st.markdown("#### 论文四电流曲线（图 4-6 对应展示）")
            st.plotly_chart(_four_current_plot(result),width="stretch")
            st.dataframe(result["paper_periods"],hide_index=True,width="stretch")
            with st.expander("查看单工况预测曲线及导出信息"):
                st.dataframe(result["prediction"],hide_index=True,width="stretch",
                             height=190)
            st.download_button("下载 PCA-GPR 预测、四电流曲线及 Origin CSV",
                data=_zip_export(result),file_name="PCA_GPR_paper_stage33.zip",
                mime="application/zip",width="stretch",
                key="stage33_gpr_download")
