from dataclasses import dataclass
from pathlib import Path
import io
import re

import pandas as pd


# =========================================================
# 统一的数据对象
# =========================================================

@dataclass
class ElectrochemicalData:
    """
    所有 CV / GCD / EIS 数据最终都转换成这个统一结构。
    """

    technique: str
    filename: str
    metadata: dict
    data: pd.DataFrame


# =========================================================
# CHI660E 测试类型识别
# =========================================================

TECHNIQUE_MARKERS = {
    "Cyclic Voltammetry": "CV",
    "Chronopotentiometry": "GCD",
    "A.C. Impedance": "EIS",
}


# =========================================================
# CHI 原始列名 → 软件内部统一列名
# =========================================================

COLUMN_MAP = {

    "CV": {
        "Potential/V": "potential_V",
        "Current/A": "current_A",
    },

    "GCD": {
        "Time/sec": "time_s",
        "Potential/V": "potential_V",
    },

    "EIS": {
        "Freq/Hz": "frequency_Hz",
        "Z'/ohm": "z_real_ohm",
        'Z"/ohm': "z_imag_ohm",
        "Z/ohm": "z_modulus_ohm",
        "Phase/deg": "phase_deg",
    },
}


# =========================================================
# 我们真正关心的 CHI 实验参数
# =========================================================

METADATA_MAP = {

    "CV": {
        "Init E (V)": "initial_potential_V",
        "High E (V)": "high_potential_V",
        "Low E (V)": "low_potential_V",
        "Scan Rate (V/s)": "scan_rate_V_s",
        "Segment": "segment_count",
        "Sample Interval (V)": "sample_interval_V",
        "Quiet Time (sec)": "quiet_time_s",
    },

    "GCD": {
        "Cathodic Current (A)": "cathodic_current_A",
        "Anodic Current (A)": "anodic_current_A",
        "Data Storage Interval (s)": "data_storage_interval_s",
        "High E Limit (V)": "high_potential_limit_V",
        "Low E Limit (V)": "low_potential_limit_V",
        "Cathodic Time (s)": "cathodic_time_s",
        "Anodic Time (s)": "anodic_time_s",
        "Segment": "segment_count",
    },

    "EIS": {
        "Init E (V)": "initial_potential_V",
        "High Frequency (Hz)": "high_frequency_Hz",
        "Low Frequency (Hz)": "low_frequency_Hz",
        "Amplitude (V)": "amplitude_V",
        "Quiet Time (sec)": "quiet_time_s",
    },
}


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


# =========================================================
# 文件编码
# =========================================================

def decode_text(raw_bytes):
    """
    尝试几种常见 TXT 编码。
    CHI 文件以后即使来自不同电脑，也尽量能够读取。
    """

    encodings = [
        "utf-8-sig",
        "gb18030",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("无法识别 TXT 文件编码。")


# =========================================================
# 从字符串中提取数字
# =========================================================

def extract_number(text):
    """
    Examples:

    '0.01'       -> 0.01
    '4e-4'       -> 0.0004
    '0.433V'     -> 0.433
    """

    match = NUMBER_PATTERN.search(text.strip())

    if match is None:
        return text.strip()

    return float(match.group())


# =========================================================
# 判断 CV / GCD / EIS
# =========================================================

def detect_technique(text):

    for marker, technique in TECHNIQUE_MARKERS.items():

        if marker in text:
            return technique

    raise ValueError(
        "无法识别测试类型。"
        "当前版本支持 CHI 的 CV、GCD 和 EIS TXT 文件。"
    )


# =========================================================
# 解析 metadata
# =========================================================

def parse_metadata(header_lines, technique):

    metadata = {}

    mapping = METADATA_MAP[technique]

    for line in header_lines:

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        key = key.strip()
        value = value.strip()

        if key not in mapping:
            continue

        internal_name = mapping[key]

        parsed_value = extract_number(value)

        if internal_name == "segment_count":
            parsed_value = int(parsed_value)

        metadata[internal_name] = parsed_value

    return metadata


# =========================================================
# 找到真正的数据表头
# =========================================================

def find_data_header(lines, technique):

    required_columns = list(
        COLUMN_MAP[technique].keys()
    )

    for index, line in enumerate(lines):

        if all(
            column in line
            for column in required_columns
        ):
            return index

    raise ValueError(
        f"没有找到 {technique} 数据表头。"
    )


# =========================================================
# 读取文件本身
# =========================================================

def load_source(source):
    """
    同时支持：

    1. 本地文件路径
    2. Streamlit UploadedFile
    3. bytes

    这样以后做 Streamlit 时不用重新写读取代码。
    """

    # 本地路径
    if isinstance(source, (str, Path)):

        path = Path(source)

        raw_bytes = path.read_bytes()

        filename = path.name

        return raw_bytes, filename

    # bytes
    if isinstance(source, bytes):

        return source, "uploaded.txt"

    # Streamlit UploadedFile
    if hasattr(source, "getvalue"):

        raw_bytes = source.getvalue()

        filename = getattr(
            source,
            "name",
            "uploaded.txt"
        )

        return raw_bytes, filename

    raise TypeError(
        "不支持的数据来源。"
    )


# =========================================================
# 主函数
# =========================================================

def read_chi_txt(source):
    """
    读取 CHI660E TXT。

    Return
    ------
    ElectrochemicalData
    """

    # -------------------------
    # 1. 获取文件
    # -------------------------

    raw_bytes, filename = load_source(source)

    # -------------------------
    # 2. 解码 TXT
    # -------------------------

    text = decode_text(raw_bytes)

    # -------------------------
    # 3. 判断 CV / GCD / EIS
    # -------------------------

    technique = detect_technique(text)

    # -------------------------
    # 4. 分行
    # -------------------------

    lines = text.splitlines()

    # -------------------------
    # 5. 找数据表头
    # -------------------------

    data_header_index = find_data_header(
        lines,
        technique
    )

    # 数据表头之前都是仪器信息
    header_lines = lines[:data_header_index]

    # -------------------------
    # 6. 解析实验参数
    # -------------------------

    metadata = parse_metadata(
        header_lines,
        technique
    )

    # -------------------------
    # 7. 读取数值数据
    # -------------------------

    numeric_text = "\n".join(
        lines[data_header_index:]
    )

    df = pd.read_csv(
        io.StringIO(numeric_text),
        sep=r"\s*,\s*",
        engine="python"
    )

    # 清理列名
    df.columns = [
        column.strip()
        for column in df.columns
    ]

    # -------------------------
    # 8. CHI 列名 → 内部标准列名
    # -------------------------

    df = df.rename(
        columns=COLUMN_MAP[technique]
    )

    expected_columns = list(
        COLUMN_MAP[technique].values()
    )

    df = df[expected_columns]

    # -------------------------
    # 9. 强制转换成数值
    # -------------------------

    for column in expected_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # 去掉无效数据
    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    # -------------------------
    # 10. 返回统一对象
    # -------------------------

    return ElectrochemicalData(
        technique=technique,
        filename=filename,
        metadata=metadata,
        data=df,
    )
