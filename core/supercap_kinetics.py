"""Supercapacitor multi-scan CV kinetics and multi-cycle GCD summaries.

No automatic pooling across physical samples. Branches are analysed separately.
Interpolated grid and regression uncertainties are exported for independent review.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.cv_analysis import detect_cv_segments
from core.gcd_analysis import build_gcd_cycles, get_gcd_cycle, get_discharge_segment, calculate_gcd_parameters
from core.research_validation import apply_gcd_safety


def _finite_positive(value, label):
    try: f=float(value)
    except (TypeError,ValueError): raise ValueError(f"{label} 必须为正数") from None
    if not np.isfinite(f) or f<=0: raise ValueError(f"{label} 必须为正数")
    return f

def _r2(y, yhat):
    ss=float(np.sum((y-np.mean(y))**2))
    return float(1-np.sum((y-yhat)**2)/ss) if ss>1e-24 else float("nan")

def _select_branch(frame, direction):
    """Select the longest complete monotonic branch for a preselected CV cycle."""
    segments=detect_cv_segments(frame)
    candidates=[x for x in segments if x["direction"]==direction]
    if not candidates:
        raise ValueError(f"没有找到 {direction} 扫描分支")
    segment=max(candidates,key=lambda x:x["point_count"])
    branch=frame.iloc[segment["start_index"]:segment["end_index"]+1]
    branch=branch[["potential_V","current_A"]].copy()
    branch=branch.replace([np.inf,-np.inf],np.nan).dropna()
    branch=branch.sort_values("potential_V")
    branch=branch.groupby("potential_V",as_index=False).mean(numeric_only=True)
    if len(branch)<12:
        raise ValueError("CV 单向扫描点数不足")
    return branch

def _prepare_cv(items, direction, grid_points):
    if len(items)<3:raise ValueError("b 值/Dunn 分析至少需要 3 个不同扫描速率的完整 CV")
    rates=[]
    branches=[]
    filenames=[]
    for item in items:
        rate=_finite_positive(item["config"].get("scan_rate_V_s"),"扫描速率")
        if not item["config"].get("scan_rate_confirmed",True):
            raise ValueError(f"{item['data'].filename} 的扫描速率未经确认")
        if item["config"].get("cycle_number") is None:
            raise ValueError(f"{item['data'].filename} 未选择完整 CV 循环")
        branches.append(_select_branch(item["plot_df"],direction))
        rates.append(rate)
        filenames.append(item["data"].filename)
    rates=np.asarray(rates,float)
    if len(np.unique(np.round(np.log(rates),12)))<3:
        raise ValueError("扫描速率至少需要 3 个不同取值")
    lo=max(float(b.potential_V.iloc[0]) for b in branches)
    hi=min(float(b.potential_V.iloc[-1]) for b in branches)
    if not hi>lo:
        raise ValueError("所选 CV 不存在共同的电位区间")
    widths=[float(b.potential_V.iloc[-1]-b.potential_V.iloc[0]) for b in branches]
    if min(widths)<.9*max(widths):
        raise ValueError("电位窗口差异明显，不能直接进行多扫描速率动力学比较")
    coverage=(hi-lo)/max(widths)
    if coverage<.85:raise ValueError("各扫描速率共同电位区间小于电位窗口的85%")
    # Turning-point acquisition may include both forward/reverse currents
    # at an identical potential. Exclude a 1% margin at both ends instead
    # of introducing zero-current reversal artifacts into signed Dunn fits.
    margin=.01*(hi-lo)
    grid=np.linspace(lo+margin,hi-margin,int(grid_points))
    currents=np.array([np.interp(grid,b.potential_V,b.current_A) for b in branches])
    return rates,grid,currents,filenames,coverage

def analyse_cv_kinetics(items,direction="forward",grid_points=120,
                        target_potential=None):
    """Fit b= d log|i(V)|/d log(scan rate) and Dunn at each potential.

    Dunn: i(V)/sqrt(v) = k1(V)*sqrt(v)+k2(V)
    Fitting signed i on a single scan direction avoids mixing anodic/cathodic.
    Invalid sign-decompositions are flagged and excluded from *percentage*
    rather than silently clipped.
    """
    if direction not in ("forward","reverse"):
        raise ValueError("请选择 forward / reverse 分支")
    rates,voltage,current,names,coverage=_prepare_cv(items,direction,grid_points)
    X=np.vstack([np.sqrt(rates),np.ones(len(rates))]).T
    coefs,_,_,_=np.linalg.lstsq(X,current/np.sqrt(rates)[:,None],rcond=None)
    k1,k2=coefs
    residual=current-(k1[None,:]*rates[:,None]+k2[None,:]*np.sqrt(rates)[:,None])
    rmse=np.sqrt(np.mean(residual**2,axis=0))
    b_values=np.full(len(voltage),np.nan)
    b_r2=np.full(len(voltage),np.nan)
    for j in range(len(voltage)):
        vals=current[:,j]
        if np.any(np.abs(vals)<1e-14) or not (np.all(vals>0) or np.all(vals<0)):
            continue
        x=np.log(rates);y=np.log(np.abs(vals))
        beta=np.polyfit(x,y,1)
        b_values[j]=float(beta[0])
        b_r2[j]=_r2(y,np.polyval(beta,x))
    pointwise=pd.DataFrame({
        "Potential (V)":voltage,"b":b_values,"b fit R2":b_r2,
        "Dunn k1":k1,"Dunn k2":k2,"Dunn current fit RMSE (A)":rmse,
    })
    contributions=[]
    curves=[]
    for ix,rate in enumerate(rates):
        cap=k1*rate
        diff=k2*np.sqrt(rate)
        total=cap+diff
        obs=current[ix]
        # A negative signed contribution indicates a poor mechanistic
        # separation, rather than a legitimate positive storage fraction.
        valid=(cap*total>=0)&(diff*total>=0)&(np.abs(total)>1e-12)
        # report ratio only if every interpolated point is physically admissible
        cap_area=float(np.trapezoid(np.abs(cap),voltage))
        diff_area=float(np.trapezoid(np.abs(diff),voltage))
        frac=cap_area/(cap_area+diff_area)*100 if np.all(valid) and cap_area+diff_area>0 else np.nan
        contributions.append({
           "Sample":names[ix],"Scan rate (V/s)":rate,"Branch":direction,
           "Capacitive contribution (%)":frac,
           "Diffusion contribution (%)":100-frac if np.isfinite(frac) else np.nan,
           "Admissible potential fraction (%)":float(np.mean(valid)*100),
           "Dunn overall RMSE (A)":float(np.sqrt(np.mean(residual[ix]**2))),
           "Quality status":"OK" if np.all(valid) else "SIGNED_COMPONENTS_INCONSISTENT; percentages suppressed",
        })
        curves.append(pd.DataFrame({
           "Sample":names[ix],"Scan rate (V/s)":rate,"Branch":direction,
           "Potential (V)":voltage,"Current experimental (A)":obs,
           "Capacitive current (A)":cap,"Diffusive current (A)":diff,
           "Current reconstructed (A)":total,
           "Dunn residual (A)":residual[ix],"Admissible":valid,
        }))
    if target_potential is None:target_potential=float((voltage[0]+voltage[-1])/2)
    target_potential=float(target_potential)
    if not voltage[0]<=target_potential<=voltage[-1]:
        raise ValueError("目标电位不在共同电位区间内")
    j=int(np.argmin(np.abs(voltage-target_potential)))
    selected=pd.DataFrame({"Scan rate (V/s)":rates,
       "Current at selected V (A)":current[:,j],
       "log scan rate":np.log(rates),
       "log abs current":np.where(np.abs(current[:,j])>0,np.log(np.maximum(np.abs(current[:,j]),1e-300)),np.nan)})
    checks={
        "Branch":direction,"Number of rates":len(rates),
        "Common potential min (V)":float(voltage[0]),
        "Common potential max (V)":float(voltage[-1]),
        "Common window coverage (%)":100*coverage,
        "Selected potential (V)":float(voltage[j]),
        "b at selected potential":float(b_values[j]),
        "b fit R2":float(b_r2[j]),
        "b median valid points":float(np.nanmedian(b_values)) if np.isfinite(b_values).any() else np.nan,
        "Dunn model":"i(V) = k1(V)*v + k2(V)*sqrt(v)",
        "Mechanistic caution":"kinetic allocation, not proof of chemical identity",
    }
    return {"b_vs_potential":pointwise,
            "dunn_by_scan_rate":pd.DataFrame(contributions),
            "interpolated_curves":pd.concat(curves,ignore_index=True),
            "selected_potential_data":selected,
            "summary":checks}

def gcd_rate_summary(items):
    """Compare rates only if each input is the same verified sample/basis."""
    if len(items)<2:raise ValueError("倍率分析至少需要 2 个不同电流密度的 GCD")
    rows=[]
    for item in items:
        row=item["row"];result=item["result"];config=item["config"]
        mass=_finite_positive(config.get("mass_mg"),"活性质量")
        density=_finite_positive(result.get("current_density_A_g"),"电流密度")
        c=_finite_positive(result.get("specific_capacitance_F_g"),"比电容")
        rows.append({"Sample":item["data"].filename,
          "Current density (A/g)":density,"Specific capacitance (F/g)":c,
          "Specific capacity (C/g)":result.get("specific_capacity_C_g"),
          "Discharge time (s)":result.get("discharge_time_s"),
          "Active mass (mg)":mass,
          "Capacitance factor":config["factor"],
          "Device mode":bool(config.get("device_mode",False))})
    df=pd.DataFrame(rows).sort_values("Current density (A/g)")
    if df["Current density (A/g)"].nunique()<2:
        raise ValueError("需要至少两个不同电流密度")
    if df["Capacitance factor"].nunique()!=1 or df["Device mode"].nunique()!=1:
        raise ValueError("计算约定/电极体系不一致，不能进行倍率比较")
    windows=np.array([float(i["result"].get("discharge_voltage_window_V",np.nan))
                      for i in items],dtype=float)
    if (not np.all(np.isfinite(windows)) or np.any(windows<=0)
        or np.ptp(windows)>0.15*max(np.max(windows),1e-12)):
        raise ValueError("放电电压窗口差异超过15%，不建议直接比较倍率保持率")
    m=df["Active mass (mg)"].to_numpy()
    if np.ptp(m)>max(np.mean(m)*.05,1e-7):
        raise ValueError("不同文件活性质量差异超过5%，需确认是同一个样品与质量基准")
    ref=float(df["Specific capacitance (F/g)"].iloc[0])
    df["Rate retention vs lowest current (%)"]=100*df["Specific capacitance (F/g)"]/ref
    return df.reset_index(drop=True)

def gcd_cycling_summary(item,reference_cycle=None,max_cycles=20000):
    """Explicitly measured cycles from one file, same current and mass.

    Uses existing segment detector and GCD equation for every cycle.
    """
    data=item["data"];cfg=item["config"]
    mass=_finite_positive(cfg.get("mass_mg"),"活性质量")
    current=_finite_positive(cfg.get("current_A"),"放电电流")
    if not cfg.get("current_confirmed",True):
        raise ValueError("放电电流未经人工确认")
    cycles=build_gcd_cycles(data.data,data.metadata.get("segment_count"))
    if len(cycles)<2:raise ValueError("至少需要两个已识别完整 GCD 循环")
    if len(cycles)>max_cycles:raise ValueError(f"检测到超过 {max_cycles} 个循环；为保护交互性能请分批导出")
    rows=[];failures=[]
    for cy in cycles:
        index=int(cy["cycle"])
        try:
            # Cycles were detected once above. Reusing the known global
            # indices avoids O(cycles^2) repeated segmentation on long tests.
            info=cy
            discharge=get_discharge_segment(data.data,info)
            result=calculate_gcd_parameters(discharge,current,mass,float(cfg.get("factor",1)))
            capacity=apply_gcd_safety(result,discharge,current,mass,False)
            rows.append({"Cycle":index,"Specific capacitance (F/g)":result["specific_capacitance_F_g"],
              "Specific capacity (C/g)":capacity["specific_capacity_C_g"],
              "Discharge time (s)":result["discharge_time_s"],
              "IR drop (V)":result["ir_drop_V"],
              "Discharge linearity R2":result["linearity_r2"],
              "File":data.filename})
        except Exception as exc:
            failures.append({"Cycle":index,"Reason":str(exc)})
    if len(rows)<2:raise ValueError("有效 GCD 循环不足两个")
    df=pd.DataFrame(rows).sort_values("Cycle")
    if reference_cycle is None: reference_cycle=int(df.Cycle.iloc[0])
    match=df[df.Cycle==int(reference_cycle)]
    if match.empty:raise ValueError("参考循环不在有效循环内")
    start=float(match["Specific capacitance (F/g)"].iloc[0])
    if start<=0:raise ValueError("参考循环电容不为正")
    df["Retention vs reference (%)"]=100*df["Specific capacitance (F/g)"]/start
    return df,pd.DataFrame(failures,columns=["Cycle","Reason"]),int(reference_cycle)

def _json_clean(value):
    if isinstance(value,dict):
        return {str(k):_json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value,(np.integer,np.floating)):
        value=value.item()
    if isinstance(value,float) and not np.isfinite(value):
        return None
    if isinstance(value,(str,int,float,bool)) or value is None:
        return value
    return str(value)


def make_kinetics_zip(files,metadata):
    """Numeric tables only; compatible with Origin, no imposed plot styles."""
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,frame in files.items():
            z.writestr(name,frame.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("analysis_settings.json",json.dumps(_json_clean(metadata),
                 indent=2,ensure_ascii=False,allow_nan=False).encode("utf-8"))
        z.writestr("README.txt",(
            "CV kinetics and GCD rate/cycling numeric results for Origin.\n"
            "Compare only one physical electrode/sample and consistent experimental conditions.\n"
            "CV kinetic allocation requires multiple scan rates and may be nonidentifiable.\n"
            "GCD cycling retention depends on chosen reference cycle and current conditions.\n"
        ).encode("utf-8"))
    return out.getvalue()
