"""MADAP Figure-2-inspired Streamlit visual workspace.

Layout only: existing verified CV/GCD/EIS algorithms remain authoritative.
Four diagnostic panels visually echo MADAP (raw, frequency/derived,
fit/selected, residual/parameters). Never fabricate fits for absent data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from core.cv_parameters_v18 import build_cv_parameters_table

COLORS=("#24688a","#bc8056","#5f8979","#936d90","#a5a057","#5c7f9c")

# Vertical clearance between the four subplot headings and their own
# upper frame edges, in *screen pixels*. Edit this single value to tune
# all CV/GCD/EIS panels together without changing the scientific plots.
SUBPLOT_TITLE_GAP_PX = 16



def set_style():
    st.markdown("""
    <style>
      /* MADAP-inspired soft-green research desk; scoped to this page's content. */
      .block-container {max-width: 1600px; padding-top: 1.2rem;}
      .stApp {background: #f6faf6;}
      div[data-testid="stVerticalBlockBorderWrapper"] {
          background: #ffffff; border: 1px solid #d8e5dc;
          border-radius: 9px;
      }
      h1 {color: #21433c; letter-spacing: -.03em;}
      h2,h3 {color:#30544d;}
      div[data-testid="stMetric"] {background:#f5f8f5;
          padding: .5rem .65rem;border-radius: 6px;border:1px solid #dfeae2}
      div[data-testid="stDataFrame"] {border-radius: 6px;}
      div[data-testid="stButton"] button[kind="primary"] {
          background:#306958; border-color:#306958;}
    </style>""",unsafe_allow_html=True)


def _empty():
    return dict(EIS=[],CV=[],GCD=[])


def _clean(values):
    return np.asarray(values,dtype=float)


def _add(fig,frame,x,y,row,col,name,color,mode="lines",dash=None,markersize=5):
    if frame is None or x not in frame or y not in frame:return
    arrx=_clean(frame[x]);arry=_clean(frame[y])
    ok=np.isfinite(arrx)&np.isfinite(arry)
    if row==1 and col==2 or row==2 and col==2:
        # for frequency plots only when x is frequency
        if "frequency" in x.lower():ok &= (arrx>0)
    if not np.any(ok):return
    fig.add_trace(go.Scatter(x=arrx[ok],y=arry[ok],name=name,
         mode=mode,line=dict(color=color,width=1.75,dash=dash or "solid"),
         marker=dict(size=markersize,color=color),legendgroup=name,
         showlegend=True),row=row,col=col)


def _note(fig,row,col,msg):
    # annotations positioned inside the corresponding subplot domain
    axis=(row-1)*2+col
    axisname="" if axis==1 else str(axis)
    fig.add_annotation(text=msg,x=.5,y=.5,xref="x"+axisname+" domain",
        yref="y"+axisname+" domain",showarrow=False,font=dict(size=12,color="#8a9b94"),
        row=row,col=col)


def scientific_axes(fig,gutter=False):
    """Unified fully framed, grid-free axes and safe y-label positioning.

    Two-column figures reserve a wide mid-page gutter. Title standoff is
    deliberately restrained to keep the right subplot's y label inside
    that gutter without overlapping the left subplot's data frame.
    """
    fig.update_layout(
        template="simple_white",
        height=420 if gutter else 700,
        margin=dict(l=92,r=92,t=118 if not gutter else 70,b=85),
        paper_bgcolor="white",plot_bgcolor="white",
        font=dict(family="Arial",size=11,color="#355b50"))
    fig.update_xaxes(showline=True,mirror=True,linewidth=1.35,
                     linecolor="#334c47",ticks="outside",
                     tickcolor="#334c47",showgrid=False,
                     zeroline=False,title_standoff=12,
                     automargin=True)
    fig.update_yaxes(showline=True,mirror=True,linewidth=1.35,
                     linecolor="#334c47",ticks="outside",
                     tickcolor="#334c47",showgrid=False,
                     zeroline=False,title_standoff=6,
                     automargin=False)
    for name in fig.layout:
        if str(name).startswith("yaxis"):
            axis=fig.layout[name]
            if getattr(axis,"overlaying",None):
                axis.title.standoff=5
                axis.automargin=False
    return fig


def _space_subplot_titles(fig, gap_px=SUBPLOT_TITLE_GAP_PX):
    """Keep subplot titles a fixed pixel distance ABOVE their plot frames.

    `make_subplots` places these headings at the correct domain edges,
    including EIS with an overlaid secondary y-axis. Do not recompute
    coordinates from axis numbers, which differ by technique. Instead,
    retain the authoritative Plotly-generated title positions and apply
    a small positive pixel offset, with bottom anchoring.
    """
    if not (0 <= float(gap_px) <= 48):
        raise ValueError("subplot title clearance must be 0–48 px")
    for annotation in fig.layout.annotations[:4]:
        annotation.yref="paper"
        annotation.yanchor="bottom"
        annotation.yshift=int(gap_px)
        annotation.font=dict(size=12,color="#2c514a")
    return fig


def diagnostic_figure(technique,items=None,raw=None,rate_frame=None):
    items=list(items or [])
    raw=list(raw or [])
    if technique=="EIS":
        titles=("Nyquist · Raw","Bode · |Z| and phase",
                "Nyquist · Circuit fit","Fit residual · %")
    elif technique=="CV":
        titles=("CV · Full curves","CV · Current detail",
                "Peak candidates","Specific capacitance")
    else:
        titles=("GCD · Charge/discharge","Discharge · Selected segment",
                "Specific capacitance","Rate retention")
    fig=make_subplots(rows=2,cols=2,subplot_titles=titles,
         horizontal_spacing=.24,vertical_spacing=.23,
         specs=[[{},{"secondary_y":technique=="EIS"}],[{},{}]])
    specs=items if items else [{"data":d,"plot_df":d.data,"config":{"label":d.filename}}
                                   for d in raw if d.technique==technique]
    for i,item in enumerate(specs[:10]):
        d=item["data"];frame=item.get("plot_df",d.data)
        label=str(item.get("config",{}).get("label",d.filename))
        col=COLORS[i%len(COLORS)]
        if technique=="EIS":
            rawdf=d.data
            if not all(c in rawdf for c in ["frequency_Hz","z_real_ohm","z_imag_ohm"]):
                continue
            fig.add_trace(go.Scatter(x=rawdf.z_real_ohm,y=-rawdf.z_imag_ohm,
                  mode="markers",name=label,marker=dict(size=5,color=col)),
                  row=1,col=1)
            f=_clean(rawdf.frequency_Hz)
            z=_clean(rawdf.z_real_ohm)+1j*_clean(rawdf.z_imag_ohm)
            order=np.argsort(f)
            fig.add_trace(go.Scatter(x=f[order],y=np.abs(z)[order],
                name=label+" |Z|",mode="lines+markers",
                line=dict(color=col,width=1.5),marker=dict(size=3),
                showlegend=False),row=1,col=2,secondary_y=False)
            fig.add_trace(go.Scatter(x=f[order],y=np.angle(z[order],deg=True),
                name=label+" phase",mode="lines",
                line=dict(color=col,width=1.25,dash="dot"),
                showlegend=False),row=1,col=2,secondary_y=True)
            fig.add_trace(go.Scatter(x=rawdf.z_real_ohm,y=-rawdf.z_imag_ohm,
                mode="markers",name=label+" raw",
                marker=dict(size=4,color=col),showlegend=False),row=2,col=1)
            fit=item.get("fit_result")
            if fit:
                fd=fit.get("fitted_data")
                if fd is not None:
                    fig.add_trace(go.Scatter(x=fd.z_real_fit_ohm,
                       y=-fd.z_imag_fit_ohm,name=label+" fitted",mode="lines",
                       line=dict(color=col,width=1.8),showlegend=False),row=2,col=1)
                    if "frequency_Hz" in fd:
                        fr=fit.get("fit_data")
                        if fr is not None and len(fr)==len(fd):
                            diff=np.hypot(fd.z_real_fit_ohm.to_numpy()-fr.z_real_ohm.to_numpy(),
                               fd.z_imag_fit_ohm.to_numpy()-fr.z_imag_ohm.to_numpy())
                            den=np.maximum(np.hypot(fr.z_real_ohm,fr.z_imag_ohm),1e-12)
                            fig.add_trace(go.Scatter(x=fd.frequency_Hz,y=diff/den*100,
                                  mode="lines+markers",name=label+" residual",
                                  line=dict(color=col,width=1.35),
                                  marker=dict(size=3),showlegend=False),
                                  row=2,col=2)
            else:
                # no line shows for unfitted data (explicitly not invented)
                pass
        elif technique=="CV":
            if not all(c in frame for c in ["potential_V","current_A"]):continue
            v=_clean(frame.potential_V);j=_clean(frame.current_A)*1000
            fig.add_trace(go.Scatter(x=v,y=j,mode="lines",name=label,
                line=dict(color=col,width=1.7)),row=1,col=1)
            fig.add_trace(go.Scatter(x=v,y=j,mode="lines",name=label,
                showlegend=False,line=dict(color=col,width=1.4)),row=1,col=2)
            result=item.get("result") or {}
            for key,marker in (("oxidation_peak_potential_V","triangle-up"),
                               ("reduction_peak_potential_V","triangle-down")):
                point=result.get(key)
                if point is not None and np.isfinite(float(point)):
                    ix=int(np.argmin(np.abs(v-float(point))))
                    fig.add_trace(go.Scatter(x=[v[ix]],y=[j[ix]],mode="markers",
                        showlegend=False,marker=dict(symbol=marker,size=9,color=col)),
                        row=2,col=1)
            c=result.get("specific_capacitance_F_g")
            scan=item.get("config",{}).get("scan_rate_V_s")
            if c is not None and scan and np.isfinite(c):
                fig.add_trace(go.Scatter(x=[scan*1000],y=[c],mode="markers",
                      name=label,showlegend=False,marker=dict(size=9,color=col)),
                      row=2,col=2)
        elif technique=="GCD":
            if not all(c in frame for c in ["time_s","potential_V"]):continue
            x=_clean(frame.time_s);y=_clean(frame.potential_V)
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=label,
                line=dict(color=col,width=1.7)),row=1,col=1)
            r=item.get("result",{})
            start=r.get("discharge_start_time_s")
            end=r.get("discharge_end_time_s")
            if start is not None and end is not None:
                try:
                    # The main GCD plot is normalized to cycle start; the
                    # measured discharge must be extracted in raw time space.
                    from core.gcd_analysis import get_gcd_cycle, get_discharge_segment
                    cycle_index=item.get("config",{}).get("cycle_number")
                    _,cycle_info=get_gcd_cycle(
                        d.data,int(cycle_index),d.metadata.get("segment_count"))
                    segment=get_discharge_segment(d.data,cycle_info)
                    dx=segment.time_s.to_numpy(float)-float(segment.time_s.iloc[0])
                    dy=segment.potential_V.to_numpy(float)
                    fig.add_trace(go.Scatter(x=dx,y=dy,
                        mode="lines",name=label,showlegend=False,
                        line=dict(color=col,width=1.7)),row=1,col=2)
                except (ValueError,KeyError,TypeError):
                    pass
            density=r.get("current_density_A_g")
            c=r.get("specific_capacitance_F_g")
            if density is not None and c is not None and np.isfinite(float(density)) and np.isfinite(float(c)):
                fig.add_trace(go.Scatter(x=[density],y=[c],mode="markers",
                    name=label,showlegend=False,marker=dict(size=9,color=col)),
                    row=2,col=1)
            # IR drop remains in the result table and in the discharge
            # analysis; the fourth subplot is reserved for rate retention.
            # It is calculated against the lowest current density in the
            # *same selected physical specimen* and is only informative
            # after the user has selected comparable rate-series files.
            pass

    if technique=="GCD" and rate_frame is not None and not rate_frame.empty:
        # Render retention only from the user's verified same-sample
        # rate-capability analysis, never from arbitrary uploaded GCD files.
        fig.add_trace(go.Scatter(
            x=rate_frame["Current density (A/g)"],
            y=rate_frame["Rate retention vs lowest current (%)"],
            mode="lines+markers",name="Verified rate retention",
            showlegend=False,line=dict(color="#24688a",width=1.8),
            marker=dict(size=7)),row=2,col=2)
    if technique=="EIS":
        fig.update_xaxes(title_text="Z′ (Ω)",row=1,col=1)
        fig.update_yaxes(title_text="−Z″ (Ω)",row=1,col=1)
        fig.update_xaxes(title_text="Frequency (Hz)",type="log",row=1,col=2)
        fig.update_yaxes(title_text="|Z| (Ω)",row=1,col=2,secondary_y=False)
        fig.update_yaxes(title_text="Phase (°)",row=1,col=2,secondary_y=True)
        fig.update_xaxes(title_text="Z′ (Ω)",row=2,col=1)
        fig.update_yaxes(title_text="−Z″ (Ω)",row=2,col=1)
        fig.update_xaxes(title_text="Frequency (Hz)",type="log",row=2,col=2)
        fig.update_yaxes(title_text="Relative residual (%)",row=2,col=2)
    elif technique=="CV":
        for row,col in ((1,1),(1,2),(2,1)):
            fig.update_xaxes(title_text="Potential (V)",row=row,col=col)
            fig.update_yaxes(title_text="Current (mA)",row=row,col=col)
        fig.update_xaxes(title_text="Scan rate (mV/s)",row=2,col=2)
        fig.update_yaxes(title_text="Specific capacitance (F/g)",row=2,col=2)
    else:
        for row,col in ((1,1),(1,2)):
            fig.update_xaxes(title_text="Time (s)",row=row,col=col)
            fig.update_yaxes(title_text="Potential (V)",row=row,col=col)
        fig.update_xaxes(title_text="Current density (A/g)",row=2,col=1)
        fig.update_yaxes(title_text="Specific capacitance (F/g)",row=2,col=1)
        fig.update_xaxes(title_text="Current density (A/g)",row=2,col=2)
        fig.update_yaxes(title_text="Rate retention (%)",row=2,col=2)
    fig.update_layout(
        template="simple_white",height=700,
        margin=dict(l=92,r=92,t=105,b=83),
        paper_bgcolor="white",plot_bgcolor="white",
        legend=dict(orientation="h",yanchor="bottom",y=1.10,
                    xanchor="left",x=0,font=dict(size=10)),
        font=dict(family="Arial",size=10,color="#355b50"),
        hovermode="closest",
    )
    scientific_axes(fig)
    _space_subplot_titles(fig)
    return fig


def show_workspace(state=None,datasets=None,technique="EIS",rate_frame=None,
                   cv_kinetics_entry=None,cv_kinetics_valid=False):
    results=state["results"].get(technique,[]) if state else []
    editable_items=results
    edited=None
    if results:
        with st.expander("可编辑：样品图例和论文备注",expanded=False):
            note_rows=pd.DataFrame({
                "样品":[x["data"].filename for x in results],
                "论文图例":[x["config"].get("label",x["data"].filename) for x in results],
                "备注":["" for _ in results],
            })
            edited=st.data_editor(note_rows,hide_index=True,
                    width="stretch",disabled=["样品"],
                    key=f"madap_annotations_{technique}")
            st.caption("这里编辑的图例实时影响右侧曲线显示，"
                       "不会更改实测电流、电压或拟合参数。")
            st.download_button("下载图例/备注 CSV",
                data=edited.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{technique}_annotations.csv",
                mime="text/csv",key=f"madap_notes_download_{technique}")
        # Display names are independent from the calculated scientific data.
        if edited is not None and len(edited)==len(results):
            editable_items=[]
            for item,(_,row) in zip(results,edited.iterrows()):
                visual=dict(item)
                config=dict(item["config"])
                config["label"]=str(row["论文图例"]).strip() or item["data"].filename
                visual["config"]=config
                editable_items.append(visual)
    fig=diagnostic_figure(technique,editable_items,datasets,
                          rate_frame=rate_frame)
    st.plotly_chart(fig,width="stretch",
        config={"displaylogo":False,"toImageButtonOptions":{"format":"png",
                 "filename":f"{technique}_analysis_preview","scale":2}})
    if state and results:
        st.markdown("**Analysis parameters**")
        if technique=="CV":
            frame=build_cv_parameters_table(
                results,kinetics_entry=cv_kinetics_entry,
                kinetics_signature_valid=cv_kinetics_valid)
            st.dataframe(frame,hide_index=True,width="stretch",
                         height=240)
            st.download_button(
                "导出 CV Analysis parameters（含 b 值 / Dunn）",
                data=frame.to_csv(index=False).encode("utf-8-sig"),
                file_name="CV_Analysis_parameters.csv",
                mime="text/csv",
                key="madap_cv_parameters_download",
            )
            if cv_kinetics_valid:
                st.caption("氧化/还原峰 b 值为所选多扫描速率数据的同组拟合斜率，"
                           "在参与拟合的各行重复显示；Dunn 百分比对应各行的扫描速率。"
                           "不满足电流符号一致性时贡献比例为空。")
            else:
                st.caption("b 值与 Dunn 储能贡献需同一样品至少 3 个不同扫描速率的"
                           "完整 CV。请展开下方「CV 进阶科研分析」确认条件并计算；"
                           "单条 CV 不会虚构动力学结果。")
        else:
            frame=pd.DataFrame([x["row"] for x in results])
            st.dataframe(frame,hide_index=True,width="stretch",height=200)
        st.caption("计算结果为只读，保持与数值分析和 Origin 数据一致。")
    elif not datasets:
        st.info("从左侧上传文件并点击运行后，这里显示 Nyquist / Bode / 拟合 / 残差等分析结果。")
    else:
        st.caption("当前显示原始曲线预览。拟合、残差和数值表需要点击左侧“运行分析”后生成。")
