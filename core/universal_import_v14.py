"""Auditable CHI TXT / tabular CSV / Excel / ZIP import for CV/GCD/EIS.

Do not silently change signs, units, or discard malformed rows. The GUI presents
manual mapping for non-standard column names and requires explicit tech choice
when automatic identification is ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import io
import re
import zipfile

import numpy as np
import pandas as pd

from core.data_loader import ElectrochemicalData, read_chi_txt

MAX_FILE_BYTES = 24 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 120 * 1024 * 1024
MAX_ITEMS = 120
SUPPORTED = {".txt",".csv",".tsv",".xlsx"}
TARGET = {
    "CV":("potential_V","current_A"),
    "GCD":("time_s","potential_V"),
    "EIS":("frequency_Hz","z_real_ohm","z_imag_ohm"),
}

@dataclass
class InputBlob:
    name:str
    raw:bytes
    archive:str|None=None

class VirtualUpload:
    def __init__(self,name,raw):
        self.name=name
        self.raw=raw
    def getvalue(self):
        return self.raw

def unpack_uploads(uploads):
    blobs=[]
    errors=[]
    expanded_size=0
    for uploaded in uploads:
        name=getattr(uploaded,"name","upload")
        try:
            raw=uploaded.getvalue()
            if len(raw)>MAX_FILE_BYTES and not name.lower().endswith(".zip"):
                raise ValueError("单文件超过 24MB 限制")
            if name.lower().endswith(".zip"):
                if len(raw)>MAX_FILE_BYTES:
                    raise ValueError("ZIP 压缩包超过 24MB，请分批上传")
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    infos=z.infolist()
                    if len(infos)>MAX_ITEMS:
                        raise ValueError("ZIP 内条目数量超过限制")
                    if sum(i.file_size for i in infos)>MAX_ZIP_TOTAL_BYTES:
                        raise ValueError("ZIP 解压后累计体积超过 120MB")
                    for info in infos:
                        if info.is_dir():continue
                        path=PurePosixPath(info.filename.replace("\\","/"))
                        if ".." in path.parts or path.is_absolute():
                            errors.append(f"{name}:{info.filename}：危险压缩包路径被拒绝")
                            continue
                        if path.suffix.lower() not in SUPPORTED:
                            continue
                        # ZIP documentation is not an electrochemical TXT dataset.
                        if path.name.lower() in ("readme.txt","license.txt","notes.txt"):
                            continue
                        if info.file_size>MAX_FILE_BYTES:
                            errors.append(f"{name}:{info.filename}：单文件超过24MB")
                            continue
                        # individual entry is capped before read
                        content=z.read(info)
                        expanded_size+=len(content)
                        blobs.append(InputBlob(f"{name}::{str(path)}",content,archive=name))
            else:
                ext=PurePosixPath(name).suffix.lower()
                if ext not in SUPPORTED:
                    raise ValueError(f"暂不支持此文件类型：{ext}")
                blobs.append(InputBlob(name,raw))
                expanded_size+=len(raw)
            if len(blobs)>MAX_ITEMS:
                raise ValueError("上传文件总数超过 120 个；请分批上传")
            if expanded_size>MAX_ZIP_TOTAL_BYTES:
                raise ValueError("累计文件体积超过120MB")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return blobs,errors

def _decode(raw):
    for enc in ("utf-8-sig","gb18030","cp1252"):
        try:return raw.decode(enc)
        except UnicodeDecodeError:continue
    raise ValueError("未知文本编码")

def _dataframe(blob,sheet_name=None):
    ext=PurePosixPath(blob.name.split("::")[-1]).suffix.lower()
    if ext==".xlsx":
        wb=pd.ExcelFile(io.BytesIO(blob.raw))
        sheets=wb.sheet_names
        if not sheets:raise ValueError("Excel 不包含可读取工作表")
        chosen=sheet_name if sheet_name in sheets else sheets[0]
        frame=pd.read_excel(wb,sheet_name=chosen,header=0)
        sheet=chosen
    else:
        text=_decode(blob.raw)
        # CHI TXT is treated separately; do not infer units from metadata lines.
        # sep=None detects commas, tabs and semicolons with Python CSV sniffer.
        frame=pd.read_csv(io.StringIO(text),sep=None,engine="python",
                          comment="#",on_bad_lines="error")
        sheet=None
    if len(frame)<3:raise ValueError("有效数值行不足 3 行")
    if len(frame)>1_000_000:raise ValueError("数据超过100万行")
    frame.columns=[str(x).strip() for x in frame.columns]
    if len(set(frame.columns))!=len(frame.columns):
        raise ValueError("数据列名重复，请先修正")
    return frame,sheet

def excel_sheet_names(blob):
    if PurePosixPath(blob.name.split("::")[-1]).suffix.lower()!=".xlsx":
        return []
    return pd.ExcelFile(io.BytesIO(blob.raw)).sheet_names

def _canonical(col):
    return re.sub(r"[^a-z0-9]+","",str(col).lower().replace("μ","u").replace("µ","u").replace("ω","ohm").replace("Ω","ohm"))

def _units(col):
    value=str(col).lower().replace("μ","u").replace("µ","u")
    # Avoid matching column name text alone (e.g. "Time") as a unit.
    if re.search(r"(?i)(?:[\(\[/ _])ms(?:[\)\] /]|$)",value):
        return "ms"
    if re.search(r"(?i)(?:[\(\[/ _])mv(?:[\)\] /]|$)",value):
        return "mV"
    if re.search(r"(?i)(?:[\(\[/ _])ua(?:[\)\] /]|$)",value):
        return "uA"
    if re.search(r"(?i)(?:[\(\[/ _])ma(?:[\)\] /]|$)",value):
        return "mA"
    if re.search(r"(?i)(?:[\(\[/ _])min(?:[\)\] /]|$)",value):
        return "min"
    if re.search(r"(?i)(?:[\(\[/ _])khz(?:[\)\] /]|$)",value):
        return "kHz"
    if re.search(r"(?i)(?:[\(\[/ _])kohm(?:[\)\] /]|$)",value):
        return "kOhm"
    return None

def guess_column(col):
    """Return (internal type, scale, sign) based on explicit header tokens."""
    raw=str(col).lower().replace("μ","u").replace("µ","u")
    normal=_canonical(col)
    # Important: exported -Z'' is NOT original Z''. Convert back with -1.
    negative_imag=bool(re.search(r"^\s*[-−]\s*z\s*(?:''|[\"”]|imag|im)",raw))
    imag_token=bool(re.search(r"z\s*(?:''|[\"”])",raw))
    real_token=bool(re.search(r"z\s*['’]",raw)) and not imag_token
    if ("freq" in normal or normal in ("f","frequencyhz")):
        return "frequency_Hz",1e3 if _units(col)=="kHz" else 1,1
    if normal in ("time","times","timesec","timems","timemin","t","seconds",
                  "elapsedtime","elapsedtimes","timeinseconds"):
        return "time_s",(.001 if _units(col)=="ms" else
                         60 if _units(col)=="min" else 1),1
    if ("potential" in normal or normal in ("voltage","ewe","ev","ewev","potentialv",
                                           "voltagemv","potentialmv","ewevsref")):
        return "potential_V",.001 if _units(col)=="mV" else 1,1
    if ("current" in normal or normal in ("ia","ima","im","currenta","currentma","currentua","iua")):
        return "current_A",(.001 if _units(col)=="mA" else
                            1e-6 if _units(col)=="uA" else 1),1
    if (negative_imag or imag_token or "zimag" in normal or "imaginary" in normal
        or normal in ("zimohm","zim","zohmimag")):
        return "z_imag_ohm",1e3 if _units(col)=="kOhm" else 1,-1 if negative_imag else 1
    if ((real_token or "zreal" in normal or "realz" in normal or "zprime" in normal
         or normal in ("zohm","zohmreal","zreohm","zre"))):
        return "z_real_ohm",1e3 if _units(col)=="kOhm" else 1,1
    return None,None,None

def auto_map_columns(columns):
    mapped={}
    for col in columns:
        target,scale,sign=guess_column(col)
        if target is not None:
            if target in mapped:
                # Multiple candidates for same target are ambiguous, require manual mapping.
                mapped[target]=None
            else:
                mapped[target]=(col,scale,sign)
    return mapped

def identify_technique(columns):
    mapping=auto_map_columns(columns)
    possibilities=[key for key,required in TARGET.items()
                   if all(mapping.get(c) is not None for c in required)]
    if len(possibilities)==1:
        return possibilities[0]
    return None

def preview_blob(blob,sheet_name=None):
    """CHI TXT -> canonical; generic -> raw table + mapping suggestions."""
    name=blob.name
    ext=PurePosixPath(name.split("::")[-1]).suffix.lower()
    if ext==".txt":
        try:
            chi=read_chi_txt(VirtualUpload(name,blob.raw))
            return {"chi":chi,"raw":None,"sheet":None,
                    "suggested":{k:(k,1,1) for k in chi.data.columns},
                    "technique":chi.technique}
        except Exception:
            # Non-CHI numeric TXT: allow generic CSV/TSV parsing below.
            pass
    frame,sheet=_dataframe(blob,sheet_name=sheet_name)
    suggested=auto_map_columns(frame.columns)
    return {"chi":None,"raw":frame,"sheet":sheet,
            "suggested":suggested,"technique":identify_technique(frame.columns)}

def _numeric_column(frame,name,scale=1,sign=1):
    column=pd.to_numeric(frame[name],errors="coerce")
    if column.isna().any():
        raise ValueError(f"{name} 有 {column.isna().sum()} 行不能转为数值；未擅自丢弃行")
    arr=column.to_numpy(float)*float(scale)*float(sign)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 Inf/NaN，未擅自修复")
    return arr

def make_dataset(blob,preview,technique,mapping,metadata=None):
    if technique not in TARGET:raise ValueError("请选择 CV、GCD 或 EIS")
    if preview["chi"] is not None:
        chi=preview["chi"]
        # Original CHI parser is authoritative for CHI TXT.
        return chi
    original=preview["raw"]
    if original is None:raise ValueError("缺少表格数据")
    data={}
    used=set()
    for target in TARGET[technique]:
        if target not in mapping or mapping[target] is None:
            raise ValueError(f"缺少所需数据列 {target}")
        col,scale,sign=mapping[target]
        if col in used:raise ValueError(f"数据列 {col} 被重复映射")
        if col not in original:raise ValueError(f"数据列 {col} 不存在")
        used.add(col)
        data[target]=_numeric_column(original,col,scale,sign)
    standardized=pd.DataFrame(data)
    if technique=="EIS":
        if np.any(standardized.frequency_Hz<=0):
            raise ValueError("EIS 频率必须大于零")
    if technique=="GCD":
        if not np.all(np.diff(standardized.time_s.to_numpy())>0):
            raise ValueError("GCD 时间必须严格递增；检查时间列与单位")
    if technique=="CV":
        if standardized.potential_V.nunique()<3:
            raise ValueError("CV 电位点不足")
    info={
        "import_format":PurePosixPath(blob.name.split("::")[-1]).suffix.lower(),
        "source_sheet":preview["sheet"],
        "column_mapping":{target:{"original_column":src,"factor_to_SI":factor,
                                   "sign_multiplier":sgn}
                          for target,(src,factor,sgn) in mapping.items()
                          if target in TARGET[technique]},
        "source_rows":len(original),
        "source_columns":list(original.columns),
    }
    if metadata:
        info.update(metadata)
    return ElectrochemicalData(technique=technique,filename=blob.name,
                               metadata=info,data=standardized)
