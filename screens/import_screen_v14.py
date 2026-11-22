"""Streamlit UI for auditable file import; CHI files bypass tabular remapping."""
from __future__ import annotations

import hashlib
import re

import streamlit as st

from core.universal_import_v14 import (
    TARGET, auto_map_columns, make_dataset, preview_blob, unpack_uploads,
    excel_sheet_names, MAX_ITEMS,
)

UNITS = {
    "potential_V": {"V":1.,"mV":.001},
    "current_A": {"A":1.,"mA":.001,"uA":1e-6},
    "time_s": {"s":1.,"ms":.001,"min":60.},
    "frequency_Hz": {"Hz":1.,"kHz":1000.},
    "z_real_ohm": {"ohm":1.,"kOhm":1000.},
    "z_imag_ohm": {"ohm":1.,"kOhm":1000.},
}
LABELS={
    "potential_V":"电位 (V)","current_A":"电流 (A)","time_s":"时间 (s)",
    "frequency_Hz":"频率 (Hz)","z_real_ohm":"阻抗实部 Z′ (Ω)",
    "z_imag_ohm":"阻抗虚部 Z″ (Ω)",
}

def import_multiformat(uploaded_files, context="batch"):
    blobs,errors=unpack_uploads(uploaded_files)
    datasets=[]
    digest=hashlib.sha256()
    used_names=set()
    if not blobs and not errors:
        st.info("未找到受支持的数据文件。")
    for index,blob in enumerate(blobs):
        digest.update(blob.name.encode("utf-8",errors="replace"))
        digest.update(blob.raw)
        key=context+"_"+hashlib.sha256((blob.name+":"+str(index)+":"+
                    hashlib.sha256(blob.raw).hexdigest()).encode()).hexdigest()[:15]
        try:
            sheet=None
            if blob.name.lower().endswith(".xlsx"):
                sheets=excel_sheet_names(blob)
                if len(sheets)>1:
                    sheet=st.selectbox(
                        f"选择 Excel 工作表 · {blob.name}",sheets,
                        key=f"v14_excel_sheet_{key}")
            preview=preview_blob(blob,sheet_name=sheet)
            if sheet is not None:
                digest.update(repr((blob.name,sheet)).encode())
        except Exception as exc:
            errors.append(f"{blob.name}：文件解析失败：{exc}")
            continue
        if preview["chi"] is not None:
            d=preview["chi"]
            d.metadata["import_format"]="CHI TXT"
            # CHI parser already mapped native units.
            name=d.filename
            if name in used_names:
                name=f"{name} [{index+1}]"
                d.filename=name
            used_names.add(name)
            datasets.append(d)
            continue
        columns=list(preview["raw"].columns)
        guess=preview["technique"]
        with st.expander(f"检查列名与单位 · {blob.name}",expanded=(guess is None)):
            st.caption("CSV/Excel 的列名、单位和阻抗虚部符号请逐项检查。"
                       "数据不规范时不会自动丢行或修复。")
            kind=st.selectbox(
                "数据类型",["CV","GCD","EIS"],
                index=["CV","GCD","EIS"].index(guess) if guess else 0,
                key=f"v14_kind_{key}")
            suggested=preview["suggested"]
            mapping={}
            for target in TARGET[kind]:
                entry=suggested.get(target)
                source=entry[0] if entry is not None else None
                with st.container():
                    c1,c2,c3=st.columns([3,1.6,1.3])
                    with c1:
                        col=st.selectbox(LABELS[target],["请选择"]+columns,
                           index=columns.index(source)+1 if source in columns else 0,
                           key=f"v14_col_{target}_{key}")
                    scale_options=UNITS[target]
                    default_unit=list(scale_options)[0]
                    if entry:
                        candidate=abs(float(entry[1]))
                        for text,factor in scale_options.items():
                            if abs(candidate-factor)<max(1e-14,factor*1e-8):
                                default_unit=text
                                break
                    with c2:
                        unit=st.selectbox("原始单位",list(scale_options),
                             index=list(scale_options).index(default_unit),
                             key=f"v14_unit_{target}_{key}")
                    with c3:
                        if target=="z_imag_ohm":
                            sign=st.selectbox("原始符号",
                                ["Z″（直接使用）","−Z″（取反）"],
                                index=1 if entry is not None and entry[2]==-1 else 0,
                                key=f"v14_sign_{target}_{key}")
                            multiplier=-1 if sign.startswith("−") else 1
                        else:
                            st.caption("保持符号")
                            multiplier=1
                    if col!="请选择":
                        mapping[target]=(col,scale_options[unit],multiplier)
            st.dataframe(preview["raw"].head(5),hide_index=True,
                         width="stretch")
            confirmed=st.checkbox(
                "我已经核对数据类型、列映射、原始单位及阻抗虚部符号",
                key=f"v14_confirm_{key}")
            digest.update(repr((blob.name,kind,mapping,confirmed)).encode("utf-8"))
            if not confirmed:
                st.info("尚未确认映射，此文件不进入计算；不会静默使用未经核对的单位。")
                continue
            try:
                d=make_dataset(blob,preview,kind,mapping)
                if d.filename in used_names:
                    d.filename=f"{d.filename} [{index+1}]"
                used_names.add(d.filename)
                datasets.append(d)
            except Exception as exc:
                errors.append(f"{blob.name}：列映射/数值校验失败：{exc}")
    digest.update(repr(sorted((d.filename,d.technique) for d in datasets)).encode())
    return datasets,errors,digest.hexdigest(),len(blobs)
