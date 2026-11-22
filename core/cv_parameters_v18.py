"""Scientific CV Analysis parameters table with conditional group kinetics.

Single CV trace is insufficient for peak-current b and Dunn allocation.
Summaries are filled ONLY for members of a user-verified, successfully
analysed multi-scan series, against an unchanged set of analysis settings.
"""

from __future__ import annotations

import hashlib
import math
import pandas as pd


BASE_ORDER=(
    "文件名","类型","图例","循环","扫描速率 (V/s)",
    "电位窗口 (V)","比电容 (F/g)","峰间距 (V)",
    "最大电流位置 (V; 峰需确认)",
    "最小电流位置 (V; 峰需确认)",
)
KINETIC_ORDER=(
    "氧化峰 b（同组拟合）",
    "还原峰 b（同组拟合）",
    "氧化峰 b 拟合 R²",
    "还原峰 b 拟合 R²",
    "逐电位 b 中位数（所选分支）",
    "Dunn 电容控制贡献 (%)",
    "Dunn 扩散控制贡献 (%)",
    "Dunn 电流拟合 RMSE (A)",
    "动力学拟合状态",
)


def kinetics_signature(analysis_signature, selected, branch, verified,
                       manual, windows):
    """Mirror the scientific-analysis widget signature with stable inputs."""
    return hashlib.sha256(
        repr((analysis_signature,selected,branch,verified,manual,windows))
        .encode("utf-8")).hexdigest()


def _first(series, field):
    if series is None or series.empty or field not in series.columns:
        return None
    value=series.iloc[0][field]
    try:
        if not math.isfinite(float(value)):
            return None
        return float(value)
    except (TypeError,ValueError):
        return None


def build_cv_parameters_table(items, kinetics_entry=None,
                              kinetics_signature_valid=False):
    """Build an Analysis parameters table, never assign inferred numbers.

    The b slopes belong to a *group* of scan rates, not each individual
    sweep. The repeated b values in selected rows are explicitly marked
    as group fits. Dunn contribution is row-specific by scan rate.
    """
    frame=pd.DataFrame([dict(x["row"]) for x in items])
    if frame.empty:
        return frame
    for col in KINETIC_ORDER[:-1]:
        frame[col]=float("nan")
    frame[KINETIC_ORDER[-1]]="未计算：需同一样品至少3个不同扫描速率的CV并运行b/Dunn"
    if kinetics_signature_valid and kinetics_entry is not None:
        _,analysis,peak_summary,_,peak_error=kinetics_entry
        peak_lookup={}
        if peak_error is None and peak_summary is not None and not peak_summary.empty:
            for mode in ("anodic","cathodic"):
                row=peak_summary[peak_summary["Branch"]==mode]
                if not row.empty:
                    peak_lookup[mode]=row
        contributions=analysis.get("dunn_by_scan_rate")
        overview=analysis.get("summary",{})
        if isinstance(contributions,pd.DataFrame) and not contributions.empty:
            chosen={(str(x["Sample"]),round(float(x["Scan rate (V/s)"]),12)):
                    x for _,x in contributions.iterrows()}
            for i,item in enumerate(items):
                name=str(item["data"].filename)
                rate=round(float(item["config"]["scan_rate_V_s"]),12)
                record=chosen.get((name,rate))
                if record is None:continue
                frame.at[i,"动力学拟合状态"]=str(record.get("Quality status",""))
                frame.at[i,"Dunn 电容控制贡献 (%)"]=record.get(
                    "Capacitive contribution (%)")
                frame.at[i,"Dunn 扩散控制贡献 (%)"]=record.get(
                    "Diffusion contribution (%)")
                frame.at[i,"Dunn 电流拟合 RMSE (A)"]=record.get(
                    "Dunn overall RMSE (A)")
                frame.at[i,"逐电位 b 中位数（所选分支）"]=overview.get(
                    "b median valid points")
                if "anodic" in peak_lookup:
                    frame.at[i,"氧化峰 b（同组拟合）"]=_first(peak_lookup["anodic"],"b")
                    frame.at[i,"氧化峰 b 拟合 R²"]=_first(peak_lookup["anodic"],"R2")
                if "cathodic" in peak_lookup:
                    frame.at[i,"还原峰 b（同组拟合）"]=_first(peak_lookup["cathodic"],"b")
                    frame.at[i,"还原峰 b 拟合 R²"]=_first(peak_lookup["cathodic"],"R2")
                if peak_error:
                    frame.at[i,"动力学拟合状态"] += f"; 峰电流b值不可用：{peak_error}"
    order=[c for c in BASE_ORDER if c in frame.columns]
    extras=[c for c in frame.columns if c not in order and c not in KINETIC_ORDER]
    return frame[order+extras+list(KINETIC_ORDER)]
