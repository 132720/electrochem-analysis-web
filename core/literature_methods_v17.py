"""Literature-aligned analyses for supercapacitor CV/GCD/EIS (excluding cycling).

Peak-current b: log|I_peak| = log(a)+b log(v).
Dunn calculations use the existing signed-current per-potential model;
this module adds auditable peak picks and consistent EIS cross-sample checks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.cv_analysis import detect_cv_segments

def _branch(data,positive):
    segments=detect_cv_segments(data)
    want="forward" if positive else "reverse"
    candidates=[s for s in segments if s["direction"]==want]
    if not candidates:
        raise ValueError(f"缺少 {want} 扫描方向")
    seg=max(candidates,key=lambda s:s["point_count"])
    return data.iloc[seg["start_index"]:seg["end_index"]+1].copy()

def _validate_items(items):
    if len(items)<3:raise ValueError("至少需要3个不同扫描速率")
    rates=np.array([float(x["config"]["scan_rate_V_s"]) for x in items])
    if not np.all(np.isfinite(rates)) or np.any(rates<=0):
        raise ValueError("扫描速率必须全部为正")
    if len(np.unique(np.round(np.log(rates),10)))<3:
        raise ValueError("至少需要3个不同扫描速率")
    for x in items:
        if x["config"].get("scan_rate_confirmed") is False:
            raise ValueError(f"{x['data'].filename} 扫描速率未经确认")
        if x["config"].get("cycle_number") is None:
            raise ValueError(f"{x['data'].filename} 未选择完整 CV 循环")
    return rates

def cv_peak_b_fits(items,windows=None):
    """Fit peak-current b separately for anodic/cathodic branches.

    The reported values are *peak candidates* from extrema of the correctly
    directed branch, not automatically chemically assigned redox peaks.
    windows: optional {anodic:(Vmin,Vmax), cathodic:(Vmin,Vmax)}.
    """
    rates=_validate_items(items)
    windows=windows or {}
    rows=[]
    for x,rate in zip(items,rates):
        for mode,positive in [("anodic",True),("cathodic",False)]:
            branch=_branch(x["plot_df"],positive)
            potential=branch.potential_V.to_numpy(float)
            current=branch.current_A.to_numpy(float)
            valid=np.isfinite(potential)&np.isfinite(current)
            bounds=windows.get(mode)
            if bounds is not None:
                lo,hi=map(float,bounds)
                if not lo<hi:raise ValueError(f"{mode} 峰电位窗口需满足最小值<最大值")
                valid &= (potential>=lo)&(potential<=hi)
            if np.count_nonzero(valid)<5:
                raise ValueError(f"{mode} 峰窗口内点数不足：{x['data'].filename}")
            idx=np.where(valid)[0][np.argmax(current[valid]) if positive
                                    else np.argmin(current[valid])]
            peak=float(current[idx])
            if mode=="anodic" and peak<=0:
                raise ValueError("正扫区域内未找到正向峰电流，不能计算氧化峰b值")
            if mode=="cathodic" and peak>=0:
                raise ValueError("反扫区域内未找到负向峰电流，不能计算还原峰b值")
            rows.append({"Sample":x["data"].filename,"Branch":mode,
                         "Scan rate (V/s)":float(rate),
                         "Peak potential (V)":float(potential[idx]),
                         "Peak current (A)":peak,
                         "log scan rate":float(np.log(rate)),
                         "log abs peak current":float(np.log(abs(peak))),
                         "Peak at selected window boundary":bool(
                             bounds is not None and min(
                                abs(potential[idx]-bounds[0]),
                                abs(potential[idx]-bounds[1]))<
                                .01*(bounds[1]-bounds[0]))})
    peaks=pd.DataFrame(rows)
    models=[]
    fitrows=[]
    for mode in ("anodic","cathodic"):
        part=peaks[peaks.Branch==mode].copy()
        x=part["log scan rate"].to_numpy(float)
        y=part["log abs peak current"].to_numpy(float)
        slope,intercept=np.polyfit(x,y,1)
        prediction=slope*x+intercept
        denom=np.sum((y-y.mean())**2)
        r2=float(1-np.sum((y-prediction)**2)/denom) if denom>1e-22 else np.nan
        if len(part)>2:
            sigma2=np.sum((y-prediction)**2)/(len(part)-2)
            uncertainty=float(np.sqrt(sigma2/np.sum((x-x.mean())**2)))
        else:uncertainty=np.nan
        span=float(part["Peak potential (V)"].max()-part["Peak potential (V)"].min())
        models.append({"Branch":mode,"b":float(slope),"log a":float(intercept),
                       "R2":r2,"b standard error":uncertainty,
                       "Peak potential shift (V)":span,
                       "Peak selection":"directed branch extremum; verify redox attribution",
                       "Fit points":int(len(part))})
        part["log predicted current"]=prediction
        part["log residual"]=y-prediction
        fitrows.append(part)
    return pd.DataFrame(models),pd.concat(fitrows,ignore_index=True)

def eis_compare_fits(items):
    """Compare only successful EIS fits with matching model and fit conditions."""
    if len(items)<2:raise ValueError("至少需要两个已拟合EIS样品")
    circuits=set();settings=set();rows=[]
    for item in items:
        fit=item.get("fit_result")
        if fit is None:
            raise ValueError(f"{item['data'].filename} 未运行等效电路拟合")
        if not fit.get("success",False):
            raise ValueError(f"{item['data'].filename} 拟合未收敛")
        circuits.add(fit["circuit"])
        cfg=item["config"]
        settings.add((fit["weighting"],bool(fit["exclude_inductive_requested"]),
                      cfg.get("use_range",False),
                      cfg.get("frequency_min_Hz") if cfg.get("use_range") else None,
                      cfg.get("frequency_max_Hz") if cfg.get("use_range") else None))
        params=fit["parameters"]
        rows.append({
            "Sample":item["data"].filename,"Circuit":fit["circuit"],
            "Rs fit (ohm)":params.get("Rs"),"Rct fit (ohm)":params.get("Rct"),
            "Q":params.get("Q"),"n":params.get("n"),
            "Cdl (F)":params.get("Cdl"),"Warburg sigma":params.get("sigma"),
            "Relative RMSE (%)":fit.get("relative_rmse_pct"),
            "RMSE (ohm)":fit.get("rmse_ohm"),"F min (Hz)":fit.get("frequency_min_Hz"),
            "F max (Hz)":fit.get("frequency_max_Hz"),
            "Fit points":fit.get("number_of_points"),
            "Fitting weighting":fit["weighting"],
            "Excluded inductive points":fit.get("inductive_exclusion_applied"),
        })
    if len(circuits)!=1:raise ValueError("等效电路不一致：请统一拟合模型再比较Rs/Rct")
    if len(settings)!=1:
        raise ValueError("频率范围、权重或感抗剔除设置不一致，不能直接对比")
    return pd.DataFrame(rows).sort_values("Rct fit (ohm)").reset_index(drop=True)
