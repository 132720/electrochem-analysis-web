"""Optional EIS functions via pyimpspec, isolated from the paper v3.3 kernels.

Numerical outputs use the original measured-frequency grid for comparable
evaluation, and preserve the exact analysis settings in exported metadata.
"""
from __future__ import annotations

import importlib.metadata
import numpy as np
import pandas as pd


def available():
    try:
        import pyimpspec
        return True
    except (ImportError, OSError):
        return False


def backend_version():
    try:
        return importlib.metadata.version("pyimpspec")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def dataset(record):
    import pyimpspec as p
    df = record.data
    f = df["frequency_Hz"].to_numpy(dtype=float)
    z = df["z_real_ohm"].to_numpy(dtype=float) + 1j * df["z_imag_ohm"].to_numpy(dtype=float)
    return p.DataSet(frequencies=f, impedances=z, label=record.filename)


def run_professional(record, operation="Z-HIT", cdc="R(RQ)",
                     method="auto", weight="auto", drt_lambda=-1.0):
    """Run one optional analysis; return summary, evaluated points, extra table.

    This tuple contract preserves v4 UI compatibility. No record is mutated.
    """
    import pyimpspec as p
    data = dataset(record)
    if operation == "Z-HIT":
        result = p.perform_zhit(data, smoothing="savgol", interpolation="pchip",
                                window="boxcar", num_procs=1)
        f = np.asarray(result.get_frequencies(), dtype=float)
        z = np.asarray(result.get_impedances(), dtype=complex)
        stats = result.to_statistics_dataframe()
        return stats, pd.DataFrame({
            "frequency_Hz": f, "z_real_ohm": z.real, "z_imag_ohm": z.imag,
        }), None

    if operation == "DRT (TR-NNLS)":
        # TR-NNLS uses a regularization lambda; the default -1 delegates
        # parameter selection to the library.
        if not np.isfinite(float(drt_lambda)):
            raise ValueError("DRT 正则化 λ 必须是有限数值")
        if float(drt_lambda) == 0.0 or float(drt_lambda) < -1.0:
            raise ValueError("DRT 正则化 λ 请输入 -1（自动）或正数")
        result = p.calculate_drt(data, method="tr-nnls",
                                 lambda_value=float(drt_lambda))
        tau, gamma = result.get_drt_data()
        frame = pd.DataFrame({
            "tau_s": np.asarray(tau, dtype=float),
            "gamma": np.asarray(gamma, dtype=float),
        })
        return pd.DataFrame({
            "method": ["TR-NNLS"], "lambda_setting": [drt_lambda],
            "pseudo_chisqr": [float(result.pseudo_chisqr)],
        }), frame, None

    if operation == "自定义等效电路":
        if not isinstance(cdc, str) or not cdc.strip() or len(cdc) > 100:
            raise ValueError("CDC 电路表达式不能为空且不能超过100个字符")
        circuit = p.parse_cdc(cdc)
        result = p.fit_circuit(circuit, data, method=method, weight=weight,
                               num_procs=1, timeout=90)
        f = np.asarray(result.get_frequencies(), dtype=float)
        z = np.asarray(result.get_impedances(), dtype=complex)
        fitted = pd.DataFrame({
            "frequency_Hz": f, "z_real_ohm": z.real, "z_imag_ohm": z.imag,
        })
        return result.to_parameters_dataframe(), fitted, result.to_statistics_dataframe()
    raise ValueError(f"未知专业分析任务：{operation}")
