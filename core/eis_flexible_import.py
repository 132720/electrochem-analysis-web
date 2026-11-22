"""Path-independent, non-destructive EIS intake for electrochem_app.

Source adapters: CHI TXT, delimited TXT/CSV/TSV, XLSX, and (if installed)
pyimpspec's instrument parsers (e.g. .mpt/.dta). Never guess temperature.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import PurePosixPath, Path
import io
import re
import uuid
import tempfile
import zipfile
import hashlib
from typing import Optional
import numpy as np
import pandas as pd

from core.eis_temperature_analysis import TemperatureEISRecord, parse_temperature_eis_upload

SUPPORTED = {".txt", ".csv", ".tsv", ".xlsx", ".mpt", ".dta", ".idf", ".ids"}
ZIP_MEMBER_LIMIT = 500
MAX_MEMBER_BYTES = 30 * 1024 * 1024
MAX_TOTAL_BYTES = 160 * 1024 * 1024

@dataclass
class Candidate:
    identifier: str
    path: str
    raw: bytes
    temperature: Optional[float]
    source: str
    format: str
    error: str = ""

class MemoryUpload:
    def __init__(self, name, raw):
        self.name = name
        self._raw = raw
    def getvalue(self):
        return self._raw

def _decode(raw):
    for codec in ("utf-8-sig", "gb18030", "cp1252", "latin1"):
        try:
            return raw.decode(codec)
        except UnicodeError:
            continue
    raise ValueError("无法解码文本")

def temperature_from_path(name, raw=b""):
    path = PurePosixPath(name.replace("\\", "/"))
    parts = path.parts
    # (1) the classic parent of EIS directory: 25/EIS/a.txt
    for i in range(len(parts) - 2):
        if parts[i+1].casefold() == "eis":
            try:
                t = float(parts[i].rstrip("℃°Cc"))
                if -100 < t < 300: return t, "目录温度"
            except (TypeError, ValueError): pass
    # (2) file name with a clearly marked temperature suffix
    stem = path.stem
    match = re.search(r"(?<![\d.])(-?\d{1,3}(?:\.\d+)?)\s*(?:°\s*C|℃|C)(?![A-Za-z])", stem, re.I)
    if not match:
        match = re.search(r"(?:temp|temperature|温度)[_\s=:-]*(-?\d{1,3}(?:\.\d+)?)", stem, re.I)
    if match:
        v=float(match.group(1))
        if -100<v<300:return v,"文件名温度"
    # (3) metadata explicitly names temperature
    if path.suffix.lower() in {".txt",".csv",".tsv",".mpt",".dta"}:
        top=_decode(raw[:8192])
        match=re.search(r"(?im)^\s*(?:sample\s*)?(?:temperature|temp|温度)\s*(?:\(?\s*°?C\s*\)?)?\s*[:=]\s*(-?\d{1,3}(?:\.\d+)?)",top)
        if match:
            v=float(match.group(1))
            if -100<v<300:return v,"文件元数据"
    return None, "需人工输入"

def list_candidates(files):
    """Return candidates and archive errors. Zip contents are never executed/extracted."""
    result, errors=[],[]
    total=0
    for item in files:
        name=item.name
        raw=item.getvalue()
        ext=PurePosixPath(name).suffix.lower()
        members=[]
        if ext==".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    infos=[i for i in z.infolist() if not i.is_dir()]
                    if len(infos)>ZIP_MEMBER_LIMIT:
                        raise ValueError("ZIP 文件数量超出上限")
                    for info in infos:
                        suffix=PurePosixPath(info.filename).suffix.lower()
                        if suffix not in SUPPORTED:continue
                        if info.file_size>MAX_MEMBER_BYTES:
                            errors.append({"file":info.filename,"error":"单文件超出30MB上限"});continue
                        total+=info.file_size
                        if total>MAX_TOTAL_BYTES:raise ValueError("总解压数据超出160MB")
                        members.append((info.filename,z.read(info)))
            except Exception as exc:
                errors.append({"file":name,"error":str(exc)})
                continue
        elif ext in SUPPORTED:
            members=[(name,raw)]
        else:
            errors.append({"file":name,"error":"格式不在支持范围内"})
        for inner, data in members:
            path=PurePosixPath(inner.replace("\\\\","/"))
            parts=[part.casefold() for part in path.parts]
            stem=path.stem.casefold()
            # EIS-only import: files clearly belonging to CV/GCD are excluded,
            # even if they happen to have three numeric columns.
            if any(part in {"cv","gcd"} for part in parts[:-1]) or re.search(
                    r"(?:^|[_.-])(cv|gcd)(?:$|[_.-])",stem):
                continue
            if path.suffix.lower() in {".txt",".csv",".tsv"}:
                head=_decode(data[:4096]).casefold()
                if "cyclic voltammetry" in head or "chronopotentiometry" in head:
                    continue
            if len(data)>MAX_MEMBER_BYTES:
                errors.append({"file":inner,"error":"单文件超出30MB上限"});continue
            temp,source=temperature_from_path(inner,data)
            identifier=hashlib.sha256((name+"::"+inner).encode()+data).hexdigest()[:16]
            result.append(Candidate(identifier,inner,data,temp,source,PurePosixPath(inner).suffix.lower()))
    return result,errors

def _normal_name(s):
    s=str(s).casefold().replace("ω","ohm").replace("Ω","ohm")
    s=s.replace("−","-").replace("’","'").replace("″",'"').replace("''",'"')
    return re.sub(r"[\s_\[\](){}]", "",s)

def guess_columns(columns):
    f=real=imag=None
    minus_imag=False
    for col in columns:
        n=_normal_name(col)
        if f is None and (n.startswith(("freq","frequency","f/hz","f(Hz)")) or n in ("f","hz","frequencyhz")):f=col
        if real is None and (n.startswith(("z'", "zre","zreal","re(z", "real")) or n in ("re","zrealohm")):real=col
        if imag is None and (n.startswith(("z\"", "zim","zimag","im(z","imag")) or n in ("im","imaginary")):
            imag=col;minus_imag=n.startswith(("-z","-im")) or "negative" in n
        if imag is None and (n.startswith(("-z\"", "-zim","-im")) or n in ("-z''","-zimag")):
            imag=col;minus_imag=True
    return f,real,imag,minus_imag

def _frame_to_eis(df, columns=None):
    if columns:
        f,r,im,negative=columns
    else:
        f,r,im,negative=guess_columns(df.columns)
    if f is None or r is None or im is None:
        raise ValueError("未确定频率、Z实部、Z虚部列；请在下方手动指定列映射")
    freq=pd.to_numeric(df[f],errors="coerce")
    zr=pd.to_numeric(df[r],errors="coerce")
    zi=pd.to_numeric(df[im],errors="coerce")*(-1 if negative else 1)
    clean=pd.DataFrame({"frequency_Hz":freq,"z_real_ohm":zr,"z_imag_ohm":zi})
    clean=clean.replace([np.inf,-np.inf],np.nan).dropna()
    clean=clean.loc[clean.frequency_Hz>0]
    clean=clean.drop_duplicates("frequency_Hz").sort_values("frequency_Hz",ascending=False).reset_index(drop=True)
    if len(clean)<6:raise ValueError("有效频率点少于6个")
    clean["z_modulus_ohm"]=np.hypot(clean.z_real_ohm,clean.z_imag_ohm)
    clean["phase_deg"]=np.degrees(np.arctan2(clean.z_imag_ohm,clean.z_real_ohm))
    return clean

def _read_table(candidate):
    if candidate.format ==".xlsx":
        return pd.read_excel(io.BytesIO(candidate.raw))
    text=_decode(candidate.raw)
    # Locate actual header after possible comment/instrument metadata lines.
    for i,line in enumerate(text.splitlines()[:120]):
        cols=re.split(r"[\t,;]+",line.strip())
        if len(cols)<3:
            cols=re.split(r"\s{2,}",line.strip())
        if len(cols)<3:continue
        f,r,im,_=guess_columns(cols)
        if f is None or r is None or im is None:continue
        sep=r"\t|,|;" if ("\t" in line or "," in line or ";" in line) else r"\s+"
        frame=pd.read_csv(io.StringIO("\n".join(text.splitlines()[i:])),sep=sep,engine="python",on_bad_lines="skip")
        frame.columns=[str(x).strip() for x in frame.columns]
        return frame
    # Fallback common tables with header on first row. No anonymous 3-column guesses.
    try:return pd.read_csv(io.StringIO(text),sep=None,engine="python")
    except Exception as exc:raise ValueError(f"无法识别表格：{exc}")

def preview_columns(candidate):
    """Only for mapping UI; errors handled per file."""
    return _read_table(candidate)

def _pyimpspec_records(candidate, temperature, replicate):
    try:import pyimpspec as pyi
    except ImportError:
        raise ValueError("此仪器格式需安装可选依赖：pip install pyimpspec")
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/(PurePosixPath(candidate.path).name)
        path.write_bytes(candidate.raw)
        datasets=pyi.parse_data(path)
    records=[]
    for k,ds in enumerate(datasets):
        z=np.asarray(ds.get_impedances(),complex)
        freq=np.asarray(ds.get_frequencies(),float)
        frame=_frame_to_eis(pd.DataFrame({"Frequency":freq,"Zreal":z.real,"Zimag":z.imag}))
        records.append(TemperatureEISRecord(str(uuid.uuid4()),
           candidate.path+(f" [spectrum {k+1}]" if len(datasets)>1 else ""),
           float(temperature),replicate+k,{"importer":"pyimpspec","source":candidate.source},frame))
    return records

def parse_candidate(candidate, temperature, replicate, mapping=None):
    if temperature is None or not np.isfinite(float(temperature)):
        raise ValueError("必须提供有效温度；系统不会默认假设25°C")
    if candidate.format==".txt" and mapping is None:
        try:
            return [parse_temperature_eis_upload(
                MemoryUpload(candidate.path,candidate.raw),float(temperature),replicate)]
        except ValueError:pass
    if candidate.format in {".mpt",".dta",".ids",".idf"}:
        return _pyimpspec_records(candidate,temperature,replicate)
    frame=_frame_to_eis(_read_table(candidate),mapping)
    return [TemperatureEISRecord(str(uuid.uuid4()),candidate.path,float(temperature),
               int(replicate),{"importer":"flexible","source":candidate.source},frame)]
