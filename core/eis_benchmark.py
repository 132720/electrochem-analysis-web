"""Reproducible EIS model comparison, without claiming equivalence to DearEIS.

Both curves are scored on exactly the observed measurement frequencies.
Different circuits / weights are flagged as NOT parameter-comparable.
"""
from __future__ import annotations

import io
import json
import platform
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd

METRIC_RULE = "RMSE of complex residual / abs(Z_experiment), expressed as percent"

def _clean_curve(frame, real, imag):
    tab=frame[["frequency_Hz", real, imag]].copy()
    tab.columns=["frequency_Hz", "zr", "zi"]
    for col in tab.columns:
        tab[col]=pd.to_numeric(tab[col],errors="coerce")
    tab=tab.replace([np.inf,-np.inf],np.nan).dropna()
    tab=tab[tab["frequency_Hz"]>0].drop_duplicates("frequency_Hz")
    return tab.sort_values("frequency_Hz").reset_index(drop=True)

def matched_metrics(record, predicted, *, model_name, real="z_real_ohm", imag="z_imag_ohm"):
    """Match frequencies, no interpolation; require >=6 exact common f samples."""
    actual=_clean_curve(record.data,"z_real_ohm","z_imag_ohm")
    fitted=_clean_curve(predicted,real,imag)
    merged=actual.merge(fitted,on="frequency_Hz",suffixes=("_obs","_fit"))
    if len(merged)<6:
        raise ValueError("至少需要 6 个严格相同的频率采样点才能比较；未执行插值。")
    observed=merged["zr_obs"].to_numpy()+1j*merged["zi_obs"].to_numpy()
    calculated=merged["zr_fit"].to_numpy()+1j*merged["zi_fit"].to_numpy()
    resid=calculated-observed
    scale=np.maximum(abs(observed),1e-12)
    relative=abs(resid)/scale
    metrics={
       "method":model_name,
       "frequency_matched_count":len(merged),
       "frequency_coverage_pct":100*len(merged)/len(actual),
       "complex_RMSE_ohm":float(np.sqrt(np.mean(abs(resid)**2))),
       "relative_RMSE_pct":float(100*np.sqrt(np.mean(relative**2))),
       "relative_MAE_pct":float(100*np.mean(relative)),
       "max_relative_error_pct":float(100*np.max(relative)),
    }
    detail=pd.DataFrame({
        "frequency_Hz":merged["frequency_Hz"],
        "Zreal_observed_ohm":observed.real,
        "Zimag_observed_ohm":observed.imag,
        "Zreal_fit_ohm":calculated.real,
        "Zimag_fit_ohm":calculated.imag,
        "residual_real_ohm":resid.real,
        "residual_imag_ohm":resid.imag,
        "relative_residual_pct":100*relative,
        "method":model_name,
    })
    return metrics,detail

def compare_with_paper(record, professional_points, operation, cdc="",
                       method="auto", weight="auto"):
    """Independent recomputation of the v3.3-paper final fit on same record."""
    from core.eis_temperature_final import fit_final_dataset
    paper_result=fit_final_dataset([record])[0]
    m_paper, d_paper=matched_metrics(
        record,paper_result.fitted_data,model_name="paper_v3.3_L_Rs_CPE",
        real="z_real_fit_ohm",imag="z_imag_fit_ohm")
    m_pro,d_pro=matched_metrics(record,professional_points,
        model_name="pyimpspec_"+operation)
    comparison=pd.DataFrame([m_paper,m_pro])
    comparable=(operation=="自定义等效电路" and
                cdc.strip() in {"L-R-Q","L-RQ","L+Rs+CPE"})
    # CDC syntax/parameter mapping cannot be assumed even if a name resembles it.
    comparison["direct_parameter_equivalence_verified"]=False
    comparison["interpretation"]=(
        "相同原始谱线和频点上的预测误差比较；"
        "拟合电路/约束/权重可能不同，不能据此判断算法优劣。"
    )
    return comparison,pd.concat([d_paper,d_pro],ignore_index=True),paper_result

def make_comparison_bundle(comparison,detail,record,operation,cdc,
                           method,weight,backend_version,optional_tables=None):
    """Standardized CSV + JSON manifest; source data preserved, not overwritten."""
    raw=record.data.copy()
    manifest={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "platform_python":platform.python_version(),
        "backend":"pyimpspec (the analysis library also used by DearEIS)",
        "backend_version":backend_version,
        "reference":"electrochem_app paper v3.3 L + Rs + CPE",
        "file":record.filename,
        "temperature_C":float(record.temperature_C),
        "operation":operation,"cdc":cdc,"method":method,"weight":weight,
        "metric_definition":METRIC_RULE,
        "frequency_match":"inner join on exact Hz, no interpolation",
        "direct_deareis_gui_benchmark_performed":False,
        "note":"Comparisons are not proof of mechanistic or parameter equivalence.",
    }
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,frame in [("01_original_EIS.csv",raw),
                           ("02_fit_metric_summary.csv",comparison),
                           ("03_pointwise_residuals.csv",detail)]:
            z.writestr(name,frame.to_csv(index=False).encode("utf-8-sig"))
        for name,frame in (optional_tables or {}).items():
            if isinstance(frame,pd.DataFrame):
                z.writestr(name,frame.to_csv(index=False).encode("utf-8-sig"))
        z.writestr("04_reproducibility_manifest.json",
                   json.dumps(manifest,ensure_ascii=False,indent=2).encode("utf-8"))
    return buffer.getvalue()
