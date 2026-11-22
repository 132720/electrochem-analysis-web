"""Original uploaded GCD CHI parser, extracted without old prediction models.
Source: user-provided Stage-1 platform, same as v18 core parser.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
import re
import numpy as np
import pandas as pd
DEFAULT_CURRENT_UNIT="A/g"

@dataclass
class GCDTemperatureRecord:
    record_id: str
    filename: str
    temperature_C: float
    current_value: float | None
    current_unit: str
    metadata: dict[str, Any]
    raw_df: pd.DataFrame
    cycle_df: pd.DataFrame
    charge_df: pd.DataFrame
    discharge_df: pd.DataFrame
    total_period_s: float
    charge_duration_s: float
    discharge_duration_s: float
    potential_min_V: float
    potential_max_V: float

def infer_current_density_from_filename(
    filename: str,
) -> float | None:
    """
    The filename stem is interpreted as current density in A/g.

    Preferred examples:
        0.5.txt
        1.txt
        1.5.txt
        2.txt
        2.5.txt
    """
    stem = Path(filename).stem.strip()
    stem = (
        stem.replace("−", "-")
        .replace("–", "-")
        .replace("⁻", "-")
    )

    bare_number = re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        stem,
    )

    if bare_number:
        try:
            value = abs(float(stem))
        except ValueError:
            value = np.nan

        if np.isfinite(value) and value > 0:
            return float(value)

    # Backward compatibility only.
    patterns = [
        r"^\s*([-+]?\d+(?:\.\d+)?)\s*A\s*[/_\-]?\s*g(?:\^?-?1|-1)?\s*$",
        r"^\s*([-+]?\d+(?:\.\d+)?)\s*Ag-?1\s*$",
        r"^\s*([-+]?\d+(?:\.\d+)?)\s*A[_\-]g\s*$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            stem,
            flags=re.IGNORECASE,
        )

        if match:
            try:
                value = abs(float(match.group(1)))
            except ValueError:
                continue

            if np.isfinite(value) and value > 0:
                return float(value)

    return None

def _decode_bytes(raw: bytes) -> str:
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "latin1",
    ):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode(
        "latin1",
        errors="replace",
    )

def _clean_key(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )

def _to_float(value: str) -> float | None:
    value = (
        value.strip()
        .replace("−", "-")
        .replace("–", "-")
    )

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        value,
    )

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None

def _parse_metadata(
    lines: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if "=" in stripped:
            key, value = stripped.split("=", 1)

        elif ":" in stripped and not re.match(r"^[+-]?\d", stripped):
            key, value = stripped.split(":", 1)

        else:
            continue

        clean = _clean_key(key)
        number = _to_float(value)

        metadata[clean] = (
            value.strip()
            if number is None
            else number
        )

    return metadata

def _find_data_header(
    lines: list[str],
) -> int:
    for i, line in enumerate(lines):
        normalized = line.lower().replace(" ", "")

        if (
            "time/sec" in normalized
            and (
                "potential/v" in normalized
                or "voltage/v" in normalized
            )
        ):
            return i

    for i, line in enumerate(lines):
        normalized = line.lower()

        if (
            "time" in normalized
            and (
                "potential" in normalized
                or "voltage" in normalized
            )
            and (
                "," in line
                or "\t" in line
            )
        ):
            return i

    raise ValueError(
        "未找到 GCD 数据表头。需要 Time 与 Potential/Voltage 列。"
    )

def _read_numeric_table(
    lines: list[str],
    header_index: int,
) -> pd.DataFrame:
    text = "\n".join(
        lines[header_index:]
    )

    try:
        df = pd.read_csv(
            StringIO(text),
            sep=None,
            engine="python",
        )

    except Exception as exc:
        raise ValueError(
            f"无法读取 GCD 数值表：{exc}"
        ) from exc

    if df.shape[1] < 2:
        raise ValueError(
            "GCD 数据至少需要两列：Time 与 Potential。"
        )

    time_col = None
    potential_col = None

    for col in df.columns:
        name = str(col).lower().replace(" ", "")

        if time_col is None and "time" in name:
            time_col = col

        if (
            potential_col is None
            and (
                "potential" in name
                or "voltage" in name
            )
        ):
            potential_col = col

    if time_col is None:
        time_col = df.columns[0]

    if potential_col is None:
        potential_col = df.columns[1]

    out = pd.DataFrame(
        {
            "time_s": pd.to_numeric(
                df[time_col],
                errors="coerce",
            ),
            "potential_V": pd.to_numeric(
                df[potential_col],
                errors="coerce",
            ),
        }
    )

    out = (
        out.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .reset_index(drop=True)
    )

    if len(out) < 20:
        raise ValueError(
            "有效 GCD 数据点过少，无法执行温度预测。"
        )

    return out

def _fill_zero_direction(
    direction: np.ndarray,
) -> np.ndarray:
    series = pd.Series(
        direction,
        dtype=float,
    )

    series = (
        series.replace(
            0.0,
            np.nan,
        )
        .ffill()
        .bfill()
    )

    if series.isna().all():
        raise ValueError(
            "电位几乎没有变化，无法识别充电/放电段。"
        )

    return (
        series.fillna(1.0)
        .to_numpy(dtype=int)
    )

def _runs(
    direction: np.ndarray,
) -> list[tuple[int, int, int]]:
    if len(direction) == 0:
        return []

    result: list[
        tuple[int, int, int]
    ] = []

    start = 0
    current = int(direction[0])

    for i in range(
        1,
        len(direction),
    ):
        value = int(direction[i])

        if value != current:
            result.append(
                (
                    start,
                    i - 1,
                    current,
                )
            )

            start = i
            current = value

    result.append(
        (
            start,
            len(direction) - 1,
            current,
        )
    )

    return result

def _merge_short_runs(
    direction: np.ndarray,
    min_run_points: int = 8,
) -> np.ndarray:
    direction = direction.copy()

    for _ in range(20):
        runs = _runs(direction)
        changed = False

        for idx, (
            start,
            end,
            sign,
        ) in enumerate(runs):

            length = (
                end
                - start
                + 1
            )

            if length >= min_run_points:
                continue

            left_sign = (
                runs[idx - 1][2]
                if idx > 0
                else None
            )

            right_sign = (
                runs[idx + 1][2]
                if idx + 1 < len(runs)
                else None
            )

            replacement = None

            if (
                left_sign is not None
                and right_sign is not None
            ):
                if left_sign == right_sign:
                    replacement = left_sign

                else:
                    left_len = (
                        runs[idx - 1][1]
                        - runs[idx - 1][0]
                        + 1
                    )

                    right_len = (
                        runs[idx + 1][1]
                        - runs[idx + 1][0]
                        + 1
                    )

                    replacement = (
                        left_sign
                        if left_len >= right_len
                        else right_sign
                    )

            elif left_sign is not None:
                replacement = left_sign

            elif right_sign is not None:
                replacement = right_sign

            if (
                replacement is not None
                and replacement != sign
            ):
                direction[
                    start:end + 1
                ] = replacement

                changed = True

        if not changed:
            break

    return direction

def split_last_complete_cycle(
    df: pd.DataFrame,
    smoothing_window: int = 7,
    min_run_points: int = 8,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    potential = df[
        "potential_V"
    ].astype(float)

    window = max(
        3,
        int(smoothing_window),
    )

    if window % 2 == 0:
        window += 1

    smooth = potential.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).median()

    delta = (
        smooth.diff()
        .to_numpy()[1:]
    )

    finite_abs = np.abs(
        delta[
            np.isfinite(delta)
        ]
    )

    if len(finite_abs) == 0:
        raise ValueError(
            "无法计算 GCD 电位变化方向。"
        )

    median_step = float(
        np.median(finite_abs)
    )

    tolerance = max(
        median_step * 0.01,
        1e-12,
    )

    direction = np.zeros_like(
        delta,
        dtype=int,
    )

    direction[
        delta > tolerance
    ] = 1

    direction[
        delta < -tolerance
    ] = -1

    direction = _fill_zero_direction(
        direction
    )

    direction = _merge_short_runs(
        direction,
        min_run_points=
            min_run_points,
    )

    segments = []

    for (
        start_d,
        end_d,
        sign,
    ) in _runs(direction):

        start_point = int(start_d)
        end_point = int(end_d + 1)

        if (
            end_point
            - start_point
            + 1
            < min_run_points
        ):
            continue

        segments.append(
            {
                "sign":
                    int(sign),

                "start":
                    start_point,

                "end":
                    end_point,

                "df":
                    df.iloc[
                        start_point:
                        end_point + 1
                    ].copy(),
            }
        )

    if len(segments) < 2:
        raise ValueError(
            "未检测到完整的充电/放电段。"
        )

    pair = None

    for i in range(
        len(segments) - 1
    ):
        first = segments[i]
        second = segments[i + 1]

        if (
            first["sign"]
            * second["sign"]
            == -1
        ):
            pair = (
                first,
                second,
            )

    if pair is None:
        raise ValueError(
            "未找到相邻且方向相反的完整 GCD 充放电对。"
        )

    first, second = pair

    if first["sign"] == 1:
        charge = first["df"].copy()
        discharge = second["df"].copy()

    else:
        discharge = first["df"].copy()
        charge = second["df"].copy()

    cycle_start = min(
        first["start"],
        second["start"],
    )

    cycle_end = max(
        first["end"],
        second["end"],
    )

    cycle = df.iloc[
        cycle_start:
        cycle_end + 1
    ].copy()

    for part in (
        cycle,
        charge,
        discharge,
    ):
        part.reset_index(
            drop=True,
            inplace=True,
        )

    return (
        cycle,
        charge,
        discharge,
    )

def _duration(
    df: pd.DataFrame,
) -> float:
    if len(df) < 2:
        return 0.0

    return float(
        df["time_s"].iloc[-1]
        - df["time_s"].iloc[0]
    )

def parse_temperature_gcd_upload(
    uploaded_file,
    temperature_C: float,
) -> GCDTemperatureRecord:

    current_value = (
        infer_current_density_from_filename(
            uploaded_file.name
        )
    )

    raw = uploaded_file.getvalue()
    text = _decode_bytes(raw)
    lines = text.splitlines()

    header_index = _find_data_header(
        lines
    )

    metadata = _parse_metadata(
        lines[:header_index]
    )

    raw_df = _read_numeric_table(
        lines,
        header_index,
    )

    (
        cycle_df,
        charge_df,
        discharge_df,
    ) = split_last_complete_cycle(
        raw_df
    )

    total_period = _duration(
        cycle_df
    )

    charge_duration = _duration(
        charge_df
    )

    discharge_duration = _duration(
        discharge_df
    )

    if total_period <= 0:
        total_period = (
            charge_duration
            + discharge_duration
        )

    record_id = (
        f"{Path(uploaded_file.name).stem}_"
        f"{float(temperature_C):g}_"
        f"{abs(hash(raw))}"
    )

    return GCDTemperatureRecord(
        record_id=
            record_id,

        filename=
            uploaded_file.name,

        temperature_C=
            float(
                temperature_C
            ),

        current_value=(
            None
            if current_value is None
            else float(current_value)
        ),

        current_unit=
            DEFAULT_CURRENT_UNIT,

        metadata=
            metadata,

        raw_df=
            raw_df,

        cycle_df=
            cycle_df,

        charge_df=
            charge_df,

        discharge_df=
            discharge_df,

        total_period_s=
            float(
                total_period
            ),

        charge_duration_s=
            float(
                charge_duration
            ),

        discharge_duration_s=
            float(
                discharge_duration
            ),

        potential_min_V=
            float(
                cycle_df[
                    "potential_V"
                ].min()
            ),

        potential_max_V=
            float(
                cycle_df[
                    "potential_V"
                ].max()
            ),
    )
