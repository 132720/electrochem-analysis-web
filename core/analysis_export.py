"""Auditable, Origin-friendly exports for CV/GCD/EIS batch results.

Do not infer electrochemical mechanism from a fitted model alone.
All outputs are numeric CSV and structured metadata; figures are previews.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
from core.research_validation import FORMULAS

def _safe(name):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")[:80] or "sample"

def _scalar(v):
    if isinstance(v,(int,float,np.integer,np.floating,str,bool)) or v is None:
        if isinstance(v,(np.integer,np.floating)):
            v=v.item()
        if isinstance(v,float) and not np.isfinite(v):
            return None
        return v
    return repr(v)

def _csv(z,path,df):
    z.writestr(path,df.to_csv(index=False).encode("utf-8-sig"))

def quality_flags(kind,item):
    """Transparent flags only; no arbitrary experimental 'pass' claims."""
    data=item["data"]
    cfg=item["config"]
    flags=[]
    raw=data.data
    if len(raw)<8:flags.append("DATA_TOO_SHORT")
    if raw.isna().any().any():flags.append("SOURCE_HAS_MISSING_VALUES")
    if kind=="CV":
        if data.metadata.get("scan_rate_V_s") is None:
            flags.append("SCAN_RATE_MISSING_IN_SOURCE_VERIFY_MANUAL_SETTING")
        if not cfg.get("scan_rate_confirmed",
                       data.metadata.get("scan_rate_V_s") is not None):
            flags.append("SCAN_RATE_NOT_CONFIRMED")
        if cfg.get("mass_mg",0)<=0:
            flags.append("MASS_MISSING_NO_SPECIFIC_CAPACITANCE")
        if cfg.get("cycle_number") is None:
            flags.append("NO_DETECTED_COMPLETE_CYCLE")
    elif kind=="GCD":
        if cfg.get("mass_mg",0)<=0:
            flags.append("MASS_MISSING_NO_SPECIFIC_CAPACITANCE")
        if data.metadata.get("segment_count") is None:
            flags.append("SEGMENT_COUNT_NOT_IN_METADATA")
        if not cfg.get("device_mode",False):
            flags.append("DEVICE_ENERGY_POWER_NOT_APPLICABLE_UNCONFIRMED")
        if cfg.get("factor",1)!=1:
            flags.append("CAPACITANCE_K4_CONVERSION_REQUIRES_BASIS_REVIEW")
        if "current_A" not in cfg or not cfg["current_A"]:
            flags.append("DISCHARGE_CURRENT_UNAVAILABLE")
        if not cfg.get("current_confirmed",True):
            flags.append("DISCHARGE_CURRENT_NOT_CONFIRMED")
    elif kind=="EIS":
        fit=item.get("fit_result")
        if fit is None:
            flags.append("NO_CIRCUIT_FIT")
        elif float(fit.get("relative_rmse_pct",0))>10:
            flags.append("LARGE_FIT_RELATIVE_RMSE_GT_10_PERCENT")
        if (raw.z_imag_ohm>0).any():
            flags.append("INDUCTIVE_POINTS_PRESENT_CHECK_EXCLUSION")
    return flags

def build_batch_analysis_zip(state):
    """Package standardized data, results, source hashes and user settings."""
    out=io.BytesIO()
    rows, inventory, warnings_list=[],[],[]
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        serial=0
        for kind in ("CV","GCD","EIS"):
            items=state["results"].get(kind,[])
            if items:
                _csv(z,f"{kind}/summary.csv",pd.DataFrame([it["row"] for it in items]))
            for item in items:
                serial+=1
                sample=f"{serial:03d}_{_safe(item['data'].filename)}"
                data=item["data"];cfg=item["config"]
                meta=data.metadata
                raw=data.data.copy()
                processed=item["plot_df"].copy()
                _csv(z,f"{kind}/origin/{sample}_raw.csv",raw)
                _csv(z,f"{kind}/origin/{sample}_processed.csv",processed)
                fitted=item.get("fit_result")
                if kind=="EIS" and fitted is not None:
                    curve=fitted.get("fitted_data")
                    if isinstance(curve,pd.DataFrame):
                        _csv(z,f"{kind}/origin/{sample}_fitted.csv",curve)
                flags=quality_flags(kind,item)
                src_hash=hashlib.sha256(raw.to_csv(index=False).encode("utf-8")).hexdigest()
                entry={
                    "technique":kind,"filename":data.filename,"source_data_sha256":src_hash,
                    "selected_label":cfg.get("label"),"points":len(raw),
                    "configuration":{k:_scalar(v) for k,v in cfg.items()},
                    "source_metadata":{str(k):_scalar(v) for k,v in meta.items()},
                    "quality_flags":flags,
                    "scientific_warnings":item.get("analysis_warnings",[]),
                    "formula_references":FORMULAS,
                    "result":{str(k):_scalar(v) for k,v in item["result"].items()
                              if not isinstance(v,(dict,list,tuple,pd.DataFrame))},
                }
                z.writestr(f"{kind}/audit/{sample}_analysis.json",
                           json.dumps(entry,indent=2,ensure_ascii=False,allow_nan=False,
                            default=str).encode("utf-8"))
                inventory.append({
                  "sample_id":sample,"technique":kind,
                  "filename":data.filename,"records":len(raw),
                  "quality_flags":"; ".join(flags),
                  "source_data_sha256":src_hash,
                })
                if flags:warnings_list.append({"sample":sample,"warnings":"; ".join(flags)})
                rows.append(item["row"])
        _csv(z,"00_master_summary.csv",pd.DataFrame(rows))
        _csv(z,"00_file_inventory.csv",pd.DataFrame(inventory))
        _csv(z,"00_quality_flags.csv",pd.DataFrame(warnings_list,
                                  columns=["sample","warnings"]))
        errors=state.get("errors",[])
        _csv(z,"00_failed_files.csv",pd.DataFrame(
            [{"error":str(e)} for e in errors],columns=["error"]))
        z.writestr("00_formulas.json",
            json.dumps(FORMULAS,ensure_ascii=False,indent=2).encode("utf-8"))
        z.writestr("README.txt",(
           "Electrochemical CV/GCD/EIS numeric analysis export.\n"
           "Import CSV files from CV/GCD/EIS/origin into Origin or other plotting software.\n"
           "Audit JSON includes user settings, parsed source metadata, selected cycles, and quality flags.\n"
           "SHA256 is computed over canonical parsed source columns, not uploaded byte-for-byte.\n"
           "Quality flags are review prompts, NOT an assertion that an experiment is valid or invalid.\n"
           "Do not interpret Rct as a physical charge-transfer resistance without circuit justification.\n"
           "CV/GCD normalized values require valid active mass, electrode topology and correct formulas.\n"
           f"Exported UTC: {datetime.now(timezone.utc).isoformat()}\n"
        ).encode("utf-8"))
    return out.getvalue()

def build_eis_analysis_zip(records,fits,failures,model,exclude_inductive):
    """Lightweight, no figures: results, original/fitted Origin tables and metadata."""
    from core.eis_paper_figures import parameter_table
    out=io.BytesIO()
    params=parameter_table(fits) if fits else pd.DataFrame()
    lookup={x.label:x for x in fits}
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        _csv(z,"01_EIS_parameter_summary.csv",params)
        if failures is not None:
            _csv(z,"02_failed_samples.csv",failures)
        details=[]
        for i,rec in enumerate(records):
            label=rec.filename
            sample=f"{i+1:03d}_{_safe(label)}"
            _csv(z,f"origin/{sample}_raw.csv",rec.data)
            fit=lookup.get(label)
            if fit:
                _csv(z,f"origin/{sample}_fit.csv",fit.curve)
                flags=list(fit.warnings)
                if not fit.success: flags.insert(0,"NOT_CONVERGED")
                details.append({"Sample":label,"Circuit":model,
                    "Relative RMSE (%)":fit.relative_rmse_pct,
                    "Warnings":"; ".join(flags)})
            else:
                details.append({"Sample":label,"Circuit":model,
                    "Relative RMSE (%)":np.nan,"Warnings":"FIT_FAILED_OR_NOT_SELECTED"})
        _csv(z,"03_quality_review.csv",pd.DataFrame(details))
        z.writestr("analysis_conditions.json",
            json.dumps({"formula":FORMULAS["EIS_fit"],"circuit":model,"exclude_inductive_points":exclude_inductive,
                        "objective":"relative weighted complex least squares",
                        "note":"Rct parameter is model dependent and can be unidentifiable.",
                        "source":"original uploaded EIS frequency, real and imaginary impedance"},
                        indent=2).encode("utf-8"))
        z.writestr("README.txt",(
            "EIS automatic analysis result, no publication figures.\n"
            "The origin folder has raw frequency/impedance and fitted data for Origin plotting.\n"
            "See parameter summary and quality review before interpreting results.\n"
            "No specific physical mechanism is guaranteed by numerical convergence.\n"
        ).encode())
    return out.getvalue()
