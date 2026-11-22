from __future__ import annotations

import hashlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.analysis_export import build_batch_analysis_zip, quality_flags
from core.cv_parameters_v18 import kinetics_signature as cv_kinetics_signature
from core.research_validation import (
    validate_condition, apply_gcd_safety,
    cv_analysis_warnings, gcd_analysis_warnings,
)
from core.cv_analysis import build_cv_cycles, calculate_cv_parameters, get_cv_cycle
from core.data_loader import read_chi_txt
from core.eis_analysis import (
    CIRCUIT_CPE,
    CIRCUIT_CPE_W,
    CIRCUIT_RC,
    estimate_rct_initial,
    estimate_rs_intercept,
    fit_eis_equivalent_circuit,
)
from core.gcd_analysis import (
    build_gcd_cycles,
    calculate_gcd_parameters,
    get_default_discharge_current_A,
    get_discharge_segment,
    get_gcd_cycle,
)
from screens.madap_dashboard import set_style as madap_set_style, show_workspace as madap_show_workspace
from screens.import_screen_v14 import import_multiformat
from screens.kinetics_screen import render_supercap_kinetics
from screens.technique_extensions_v16 import render_extension, render_eis_comparison
from screens.temperature_eis_screen import render_temperature_eis_screen
from screens.temperature_gcd_screen import render_temperature_gcd_screen


st.set_page_config(page_title="电化学分析界面", page_icon="⚡", layout="wide")

TECHNIQUE_NAMES = {
    "CV": "循环伏安（CV）",
    "GCD": "恒流充放电（GCD）",
    "EIS": "电化学阻抗谱（EIS）",
}
COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
BASIS_OPTIONS = ["标准质量基准（k = 1）", "对称器件单电极等效（k = 4）"]
CIRCUIT_OPTIONS = [
    "仅计算特征值（不拟合）", CIRCUIT_CPE, CIRCUIT_RC, CIRCUIT_CPE_W,
]


def _mode_selector(prefix):
    return st.radio(
        "参数设置方式",
        ["所有文件参数一致", "不同文件单独设置", "混合模式（逐项决定）"],
        horizontal=True,
        key=f"{prefix}_parameter_mode",
    )


def _mode_scopes(mode, names, prefix, labels):
    if mode == "所有文件参数一致":
        return {name: "统一" for name in names}
    if mode == "不同文件单独设置":
        return {name: "逐文件" for name in names}
    columns = st.columns(min(4, len(names)))
    return {
        name: columns[index % len(columns)].selectbox(
            labels[name], ["统一", "逐文件"], key=f"{prefix}_scope_{name}"
        )
        for index, name in enumerate(names)
    }


def _basis_factor(basis):
    return 1.0 if basis.startswith("标准") else 4.0


def _cycle_options(cycles):
    return [int(item["cycle"]) for item in cycles]


def _render_cv_parameters(datasets):
    st.markdown("#### CV 参数")
    mode = _mode_selector("batch_cv")
    labels = {"mass": "活性质量", "basis": "计算基准", "scan": "扫描速率", "cycle": "分析循环"}
    scopes = _mode_scopes(mode, list(labels), "batch_cv", labels)
    common = {}

    is_mixed = mode == "混合模式（逐项决定）"
    show_common_parameters = mode == "所有文件参数一致" or (is_mixed and "统一" in scopes.values())
    show_file_parameters = mode == "不同文件单独设置" or (is_mixed and "逐文件" in scopes.values())

    # “所有文件参数一致”时只显示 CV 统一参数。
    if show_common_parameters:
        with st.expander("CV 统一参数", expanded=True):
            if scopes["mass"] == "统一":
                common["mass_mg"] = st.number_input(
                    "统一活性物质质量（mg，0 表示不计算质量比电容）",
                    min_value=0.0, value=0.0, step=0.01, key="batch_cv_mass_common",
                )
            if scopes["basis"] == "统一":
                common["basis"] = st.selectbox("统一计算基准", BASIS_OPTIONS, key="batch_cv_basis_common")
            if scopes["scan"] == "统一":
                common["scan_mode"] = st.selectbox(
                    "统一扫描速率设置", ["优先使用文件自动读取值", "所有文件使用同一手动值"],
                    key="batch_cv_scan_mode_common",
                )
                if common["scan_mode"].startswith("所有文件"):
                    detected = next(
                        (d.metadata.get("scan_rate_V_s") for d in datasets
                         if d.metadata.get("scan_rate_V_s") is not None),
                        0.01,
                    )
                    common["scan_rate_V_s"] = st.number_input(
                        "统一扫描速率（V/s）", min_value=1e-12, value=float(detected),
                        format="%.8g", key="batch_cv_scan_common",
                    )
            if scopes["cycle"] == "统一":
                common["cycle_policy"] = st.selectbox(
                    "统一循环规则", ["最后一个完整循环", "第一个完整循环"],
                    key="batch_cv_cycle_common",
                )

    configs = []
    for index, data in enumerate(datasets):
        cycles = build_cv_cycles(data.data)
        detected_scan = data.metadata.get("scan_rate_V_s")

        # 即使当前不显示单文件卡片，也完整生成后续计算所需配置。
        label = Path(data.filename).stem
        mass = float(common.get("mass_mg", 0.0))
        basis = common.get("basis", BASIS_OPTIONS[0])

        if scopes["scan"] == "统一":
            if common.get("scan_mode", "").startswith("所有文件"):
                scan_rate = float(common["scan_rate_V_s"])
            else:
                scan_rate = float(detected_scan if detected_scan is not None else 0.01)
        else:
            scan_rate = float(detected_scan if detected_scan is not None else 0.01)

        cycle_number = None
        if cycles and scopes["cycle"] == "统一":
            options = _cycle_options(cycles)
            cycle_number = options[-1] if common.get("cycle_policy", "").startswith("最后") else options[0]

        # “不同文件单独设置”时只显示 CV 1 / CV 2 / CV 3 ... 参数卡片。
        if show_file_parameters:
            with st.expander(f"CV {index + 1}｜{data.filename}", expanded=index < 2):
                label = st.text_input(
                    "图例名称", Path(data.filename).stem, key=f"batch_cv_label_{index}"
                )

                if scopes["mass"] == "统一":
                    mass = float(common.get("mass_mg", 0.0))
                else:
                    mass = st.number_input(
                        "活性物质质量（mg）", min_value=0.0, value=0.0, step=0.01,
                        key=f"batch_cv_mass_{index}",
                    )

                if scopes["basis"] == "统一":
                    basis = common.get("basis", BASIS_OPTIONS[0])
                else:
                    basis = st.selectbox(
                        "计算基准", BASIS_OPTIONS, key=f"batch_cv_basis_{index}"
                    )

                if scopes["scan"] == "统一":
                    if common.get("scan_mode", "").startswith("所有文件"):
                        scan_rate = float(common["scan_rate_V_s"])
                    else:
                        scan_rate = float(detected_scan if detected_scan is not None else 0.01)
                        st.caption(f"自动/回退扫描速率：{scan_rate:g} V/s")
                else:
                    scan_rate = st.number_input(
                        "扫描速率（V/s）", min_value=1e-12,
                        value=float(detected_scan if detected_scan is not None else 0.01),
                        format="%.8g", key=f"batch_cv_scan_{index}",
                    )

                if cycles:
                    options = _cycle_options(cycles)
                    if scopes["cycle"] == "统一":
                        cycle_number = (
                            options[-1]
                            if common.get("cycle_policy", "").startswith("最后")
                            else options[0]
                        )
                    else:
                        cycle_number = st.selectbox(
                            "分析循环", options, index=len(options) - 1,
                            key=f"batch_cv_cycle_{index}",
                        )
                    st.caption(
                        f"检测到 {len(cycles)} 个完整循环；当前分析第 {cycle_number} 循环。"
                    )
                else:
                    cycle_number = None
                    st.warning("未检测到完整循环，将使用全部原始数据。")

        scan_confirmed=(detected_scan is not None)
        if not scan_confirmed:
            scan_confirmed=st.checkbox(
                f"确认 {data.filename} 的扫描速率为 {scan_rate:g} V/s"
                "（文件未提供此值，需依据实验记录核对）",
                value=False,
                key=f"batch_cv_scan_confirm_v12_{index}",
            )
        configs.append({
            "label": label,
            "mass_mg": float(mass),
            "factor": _basis_factor(basis),
            "basis": basis,
            "scan_rate_V_s": float(scan_rate),
            "cycle_number": cycle_number,
            "scan_rate_confirmed": bool(scan_confirmed),
        })
    return configs

def _render_gcd_parameters(datasets):
    st.markdown("#### GCD 参数")
    measurement = st.selectbox(
        "测量体系（用于判断是否计算器件能量/功率）",
        ["三电极材料测试 / 未确认（默认）",
         "两电极完整器件（填写的是两电极总活性物质质量）"],
        key="batch_gcd_measurement_v12",
    )
    device_mode = measurement.startswith("两电极")
    if device_mode:
        st.warning("已选择两电极模式：请核对活性质量输入为两电极总活性质量、"
                   "放电电压为器件端电压，器件能量采用实测 V(t) 积分。")
    else:
        st.info("未确认两电极器件时仍可计算放电时间、比电容、比容量和 IR drop，"
                "但不输出器件能量密度和功率密度。")
    mode = _mode_selector("batch_gcd")
    labels = {"mass": "活性质量", "basis": "计算基准", "current": "放电电流", "cycle": "分析循环"}
    scopes = _mode_scopes(mode, list(labels), "batch_gcd", labels)
    common = {}

    # 三种模式采用真正的互斥显示：
    # 1) 所有文件参数一致 -> 只显示“GCD 统一参数”
    # 2) 不同文件单独设置 -> 只显示每个文件的 GCD 参数
    # 3) 混合模式 -> 仅显示当前实际需要的统一区 / 单文件区
    is_mixed = mode == "混合模式（逐项决定）"
    show_common_parameters = mode == "所有文件参数一致" or (is_mixed and "统一" in scopes.values())
    show_file_parameters = mode == "不同文件单独设置" or (is_mixed and "逐文件" in scopes.values())

    if show_common_parameters:
        with st.expander("GCD 统一参数", expanded=True):
            if scopes["mass"] == "统一":
                common["mass_mg"] = st.number_input(
                    "统一活性物质质量（mg，0 表示不计算质量归一化结果）",
                    min_value=0.0, value=0.0, step=0.01, key="batch_gcd_mass_common",
                )
            if scopes["basis"] == "统一":
                common["basis"] = st.selectbox("统一计算基准", BASIS_OPTIONS, key="batch_gcd_basis_common")
            if scopes["current"] == "统一":
                common["current_mode"] = st.selectbox(
                    "统一放电电流设置", ["优先使用文件自动读取值", "所有文件使用同一手动值"],
                    key="batch_gcd_current_mode_common",
                )
                if common["current_mode"].startswith("所有文件"):
                    detected = next(
                        (get_default_discharge_current_A(d.metadata) for d in datasets
                         if get_default_discharge_current_A(d.metadata) is not None), 0.0004,
                    )
                    common["current_A"] = st.number_input(
                        "统一放电电流（mA）", min_value=1e-12, value=float(detected) * 1000.0,
                        format="%.8g", key="batch_gcd_current_common",
                    ) / 1000.0
            if scopes["cycle"] == "统一":
                common["cycle_policy"] = st.selectbox(
                    "统一循环规则", ["最后一个完整循环", "第一个完整循环"], key="batch_gcd_cycle_common"
                )

    configs = []
    for index, data in enumerate(datasets):
        expected = data.metadata.get("segment_count")
        cycles = build_gcd_cycles(data.data, expected)
        detected_current = get_default_discharge_current_A(data.metadata)

        # 即使不显示单文件参数卡片，也为后续计算完整生成每个文件的配置。
        label = Path(data.filename).stem
        mass = float(common.get("mass_mg", 0.0))
        basis = common.get("basis", BASIS_OPTIONS[0])

        if scopes["current"] == "统一":
            if common.get("current_mode", "").startswith("所有文件"):
                current_A = float(common["current_A"])
            else:
                current_A = float(detected_current if detected_current is not None else 0.0004)
        else:
            current_A = float(detected_current if detected_current is not None else 0.0004)

        cycle_number = None
        if cycles and scopes["cycle"] == "统一":
            options = _cycle_options(cycles)
            cycle_number = options[-1] if common.get("cycle_policy", "").startswith("最后") else options[0]

        # 只有需要逐文件设置参数时，才显示 GCD 1 / GCD 2 / GCD 3 ... 参数卡片。
        if show_file_parameters:
            with st.expander(f"GCD {index + 1}｜{data.filename}", expanded=index < 2):
                label = st.text_input("图例名称", Path(data.filename).stem, key=f"batch_gcd_label_{index}")

                if scopes["mass"] == "统一":
                    mass = float(common.get("mass_mg", 0.0))
                else:
                    mass = st.number_input(
                        "活性物质质量（mg）", min_value=0.0, value=0.0, step=0.01,
                        key=f"batch_gcd_mass_{index}"
                    )

                if scopes["basis"] == "统一":
                    basis = common.get("basis", BASIS_OPTIONS[0])
                else:
                    basis = st.selectbox("计算基准", BASIS_OPTIONS, key=f"batch_gcd_basis_{index}")

                if scopes["current"] == "统一":
                    if common.get("current_mode", "").startswith("所有文件"):
                        current_A = float(common["current_A"])
                    else:
                        current_A = float(detected_current if detected_current is not None else 0.0004)
                        st.caption(f"自动/回退放电电流：{current_A * 1000:g} mA")
                else:
                    current_A = st.number_input(
                        "放电电流（mA）", min_value=1e-12,
                        value=float(detected_current if detected_current is not None else 0.0004) * 1000.0,
                        format="%.8g", key=f"batch_gcd_current_{index}",
                    ) / 1000.0

                if cycles:
                    options = _cycle_options(cycles)
                    if scopes["cycle"] == "统一":
                        cycle_number = options[-1] if common.get("cycle_policy", "").startswith("最后") else options[0]
                    else:
                        cycle_number = st.selectbox(
                            "分析循环", options, index=len(options) - 1, key=f"batch_gcd_cycle_{index}"
                        )
                    st.caption(f"检测到 {len(cycles)} 个完整循环；当前分析第 {cycle_number} 循环。")
                else:
                    cycle_number = None
                    st.warning("未检测到完整循环；该文件可识别，但无法计算放电参数。")

        current_confirmed=(detected_current is not None)
        if not current_confirmed:
            current_confirmed=st.checkbox(
                f"确认 {data.filename} 的放电电流为 {current_A*1000:g} mA"
                "（文件未提供此值，需依据实验记录核对）",
                value=False,
                key=f"batch_gcd_current_confirm_v12_{index}",
            )
        configs.append({
            "label": label, "mass_mg": float(mass), "factor": _basis_factor(basis),
            "basis": basis, "current_A": float(current_A), "cycle_number": cycle_number,
            "device_mode": device_mode,
            "current_confirmed": bool(current_confirmed),
        })
    return configs


def _render_eis_parameters(datasets):
    st.markdown("#### EIS 参数")
    mode = _mode_selector("batch_eis")
    labels = {"circuit": "等效电路", "weighting": "拟合权重", "inductive": "感抗点处理", "range": "频率范围"}
    scopes = _mode_scopes(mode, list(labels), "batch_eis", labels)
    common = {}

    is_mixed = mode == "混合模式（逐项决定）"
    show_common_parameters = mode == "所有文件参数一致" or (is_mixed and "统一" in scopes.values())
    show_file_parameters = mode == "不同文件单独设置" or (is_mixed and "逐文件" in scopes.values())

    # “所有文件参数一致”时只显示 EIS 统一参数。
    if show_common_parameters:
        with st.expander("EIS 统一参数", expanded=True):
            if scopes["circuit"] == "统一":
                common["circuit"] = st.selectbox(
                    "统一等效电路", CIRCUIT_OPTIONS, index=1,
                    key="batch_eis_circuit_common",
                )
            if scopes["weighting"] == "统一":
                common["weighting"] = st.selectbox(
                    "统一拟合权重", ["1/|Z|", "None"],
                    key="batch_eis_weighting_common",
                )
            if scopes["inductive"] == "统一":
                common["exclude_inductive"] = st.checkbox(
                    "统一排除感抗点（Z'' > 0）", value=True,
                    key="batch_eis_inductive_common",
                )
            if scopes["range"] == "统一":
                common["use_range"] = st.checkbox(
                    "统一限制拟合频率范围", value=False,
                    key="batch_eis_use_range_common",
                )
                if common["use_range"]:
                    c1, c2 = st.columns(2)
                    common["frequency_min_Hz"] = c1.number_input(
                        "统一最低频率（Hz）", min_value=1e-12, value=0.01,
                        format="%.8g", key="batch_eis_fmin_common",
                    )
                    common["frequency_max_Hz"] = c2.number_input(
                        "统一最高频率（Hz）", min_value=1e-12, value=100000.0,
                        format="%.8g", key="batch_eis_fmax_common",
                    )
                else:
                    common["frequency_min_Hz"] = None
                    common["frequency_max_Hz"] = None

    configs = []
    for index, data in enumerate(datasets):
        # 默认先生成完整配置；隐藏单文件参数区时仍可正常拟合/绘图。
        config = {
            "label": Path(data.filename).stem,
            "circuit": common.get("circuit", CIRCUIT_OPTIONS[1])
            if scopes["circuit"] == "统一" else CIRCUIT_OPTIONS[1],
            "weighting": common.get("weighting", "1/|Z|")
            if scopes["weighting"] == "统一" else "1/|Z|",
            "exclude_inductive": common.get("exclude_inductive", True)
            if scopes["inductive"] == "统一" else True,
        }

        if scopes["range"] == "统一":
            config["use_range"] = common.get("use_range", False)
            config["frequency_min_Hz"] = common.get("frequency_min_Hz")
            config["frequency_max_Hz"] = common.get("frequency_max_Hz")
        else:
            config["use_range"] = False
            config["frequency_min_Hz"] = None
            config["frequency_max_Hz"] = None

        # “不同文件单独设置”时只显示 EIS 1 / EIS 2 / EIS 3 ... 参数卡片。
        if show_file_parameters:
            with st.expander(f"EIS {index + 1}｜{data.filename}", expanded=index < 2):
                config["label"] = st.text_input(
                    "图例名称", Path(data.filename).stem,
                    key=f"batch_eis_label_{index}",
                )

                if scopes["circuit"] == "统一":
                    config["circuit"] = common.get("circuit", CIRCUIT_OPTIONS[1])
                else:
                    config["circuit"] = st.selectbox(
                        "等效电路", CIRCUIT_OPTIONS, index=1,
                        key=f"batch_eis_circuit_file_{index}",
                    )

                if scopes["weighting"] == "统一":
                    config["weighting"] = common.get("weighting", "1/|Z|")
                else:
                    config["weighting"] = st.selectbox(
                        "拟合权重", ["1/|Z|", "None"],
                        key=f"batch_eis_weighting_file_{index}",
                    )

                if scopes["inductive"] == "统一":
                    config["exclude_inductive"] = common.get("exclude_inductive", True)
                else:
                    config["exclude_inductive"] = st.checkbox(
                        "拟合时排除感抗点（Z'' > 0）", value=True,
                        key=f"batch_eis_inductive_file_{index}",
                    )

                if scopes["range"] == "统一":
                    config["use_range"] = common.get("use_range", False)
                    config["frequency_min_Hz"] = common.get("frequency_min_Hz")
                    config["frequency_max_Hz"] = common.get("frequency_max_Hz")
                else:
                    config["use_range"] = st.checkbox(
                        "限制拟合频率范围", value=False,
                        key=f"batch_eis_use_range_file_{index}",
                    )
                    if config["use_range"]:
                        c1, c2 = st.columns(2)
                        config["frequency_min_Hz"] = c1.number_input(
                            "最低频率（Hz）", min_value=1e-12, value=0.01,
                            format="%.8g", key=f"batch_eis_fmin_file_{index}",
                        )
                        config["frequency_max_Hz"] = c2.number_input(
                            "最高频率（Hz）", min_value=1e-12, value=100000.0,
                            format="%.8g", key=f"batch_eis_fmax_file_{index}",
                        )
                    else:
                        config["frequency_min_Hz"] = None
                        config["frequency_max_Hz"] = None

        configs.append(config)
    return configs

def _analyze_cv(data, config):
    if not config.get("scan_rate_confirmed",
                      data.metadata.get("scan_rate_V_s") is not None):
        raise ValueError("CV 扫描速率未在文件中识别到，请在参数设置中明确手动指定")
    validate_condition(config.get("scan_rate_V_s"), "CV 扫描速率")
    if config.get("mass_mg",0)>0 and config.get("cycle_number") is None:
        raise ValueError("未识别到完整 CV 循环，禁止按完整正反扫公式计算比电容")
    plot_df = data.data.copy() if config["cycle_number"] is None else get_cv_cycle(data.data, config["cycle_number"])
    result = calculate_cv_parameters(
        plot_df, scan_rate_V_s=config["scan_rate_V_s"],
        active_mass_mg=config["mass_mg"] if config["mass_mg"] > 0 else None,
        capacitance_factor=config["factor"],
    )
    row = {
        "文件名": data.filename, "类型": "CV", "图例": config["label"], "循环": config["cycle_number"],
        "扫描速率 (V/s)": result.get("scan_rate_V_s"),
        "电位窗口 (V)": result.get("voltage_window_V"),
        "比电容 (F/g)": result.get("specific_capacitance_F_g"),
        "峰间距 (V)": result.get("peak_separation_V"),
        "最大电流位置 (V; 峰需确认)": result.get("oxidation_peak_potential_V"),
        "最小电流位置 (V; 峰需确认)": result.get("reduction_peak_potential_V"),
    }
    notes=cv_analysis_warnings(data,config)
    result["analysis_warnings"]="; ".join(notes)
    return {"data": data, "config": config, "plot_df": plot_df,
            "result": result, "row": row, "analysis_warnings":notes}


def _analyze_gcd(data, config):
    if not config.get("current_confirmed",
                      get_default_discharge_current_A(data.metadata) is not None):
        raise ValueError("GCD 电流未在文件中识别到，请在参数设置中明确手动指定")
    validate_condition(config.get("current_A"),"GCD 放电电流")
    if config["cycle_number"] is None:
        raise ValueError("未检测到完整 GCD 循环。")
    expected = data.metadata.get("segment_count")
    cycle_df, cycle_info = get_gcd_cycle(data.data, config["cycle_number"], expected)
    discharge_df = get_discharge_segment(data.data, cycle_info)
    result = calculate_gcd_parameters(
        discharge_df, discharge_current_A=config["current_A"],
        active_mass_mg=config["mass_mg"] if config["mass_mg"] > 0 else None,
        capacitance_factor=config["factor"],
    )
    result=apply_gcd_safety(
        result,discharge_df,config["current_A"],
        config["mass_mg"] if config["mass_mg"]>0 else None,
        device_mode=config.get("device_mode",False),
    )
    normalized = cycle_df.copy()
    normalized["time_s"] = normalized["time_s"] - normalized["time_s"].iloc[0]
    row = {
        "文件名": data.filename, "类型": "GCD", "图例": config["label"], "循环": config["cycle_number"],
        "放电电流 (mA)": config["current_A"] * 1000.0, "放电时间 (s)": result.get("discharge_time_s"),
        "IR drop (V)": result.get("ir_drop_V"), "比电容 (F/g)": result.get("specific_capacitance_F_g"),
        "比容量 (C/g)": result.get("specific_capacity_C_g"),
        "比容量 (mAh/g)": result.get("specific_capacity_mAh_g"),
        "能量密度 (Wh/kg)": result.get("energy_density_Wh_kg"), "功率密度 (W/kg)": result.get("power_density_W_kg"),
        "器件能量计算依据":result.get("device_energy_method"),
    }
    notes=gcd_analysis_warnings(data,config,result)
    result["analysis_warnings"]="; ".join(notes)
    return {"data": data, "config": config, "plot_df": normalized,
            "result": result, "row": row, "analysis_warnings":notes}


def _analyze_eis(data, config):
    rs = estimate_rs_intercept(data.data)
    rct = estimate_rct_initial(data.data, rs)
    fit_result = None
    if config["circuit"] != CIRCUIT_OPTIONS[0]:
        fmin = config.get("frequency_min_Hz") if config.get("use_range") else None
        fmax = config.get("frequency_max_Hz") if config.get("use_range") else None
        if fmin is not None and fmax is not None and fmin >= fmax:
            raise ValueError("最低频率必须小于最高频率。")
        fit_result = fit_eis_equivalent_circuit(
            data.data, circuit=config["circuit"], weighting=config["weighting"],
            exclude_inductive=config["exclude_inductive"], frequency_min_Hz=fmin, frequency_max_Hz=fmax,
        )
    result = {"rs_intercept_estimate_ohm": rs, "rct_initial_estimate_ohm": rct}
    if fit_result:
        result.update({
            "circuit": fit_result["circuit"], "rmse_ohm": fit_result["rmse_ohm"],
            "relative_rmse_pct": fit_result["relative_rmse_pct"], **fit_result["parameters"],
        })
    row = {
        "文件名": data.filename, "类型": "EIS", "图例": config["label"], "等效电路": config["circuit"],
        "Rs 估计 (Ω)": rs, "Rct 估计 (Ω)": rct,
        "拟合 RMSE (Ω)": fit_result.get("rmse_ohm") if fit_result else None,
        "相对 RMSE (%)": fit_result.get("relative_rmse_pct") if fit_result else None,
    }
    if fit_result:
        for key, value in fit_result["parameters"].items():
            row[f"拟合 {key}"] = value
    return {
        "data": data, "config": config, "plot_df": data.data.copy(), "result": result,
        "fit_result": fit_result, "row": row,
    }


def _figure_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    return buffer.getvalue()


def _overlay_figure(technique, items):
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=120)
    for index, item in enumerate(items):
        df = item["plot_df"]
        color = COLORS[index % len(COLORS)]
        if technique == "CV":
            ax.plot(df["potential_V"], df["current_A"] * 1000.0, label=item["config"]["label"], color=color, linewidth=1.6)
        elif technique == "GCD":
            ax.plot(df["time_s"], df["potential_V"], label=item["config"]["label"], color=color, linewidth=1.6)
        else:
            ax.plot(
                df["z_real_ohm"], -df["z_imag_ohm"], "o-", label=item["config"]["label"],
                color=color, markersize=3.2, linewidth=1.2,
            )
            if item.get("fit_result"):
                fit_df = item["fit_result"]["fitted_data"]
                ax.plot(fit_df["z_real_fit_ohm"], -fit_df["z_imag_fit_ohm"], "--", color=color, linewidth=1.4)
    if technique == "CV":
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current (mA)")
    elif technique == "GCD":
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Potential (V)")
    else:
        ax.set_xlabel("Z' (Ω)")
        ax.set_ylabel("−Z'' (Ω)")
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#334c47")
        spine.set_linewidth(1.25)
    ax.tick_params(axis="both",which="both",direction="out",
                   top=True,right=True)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _show_type_results(technique, items):
    if not items:
        st.info(f"本批次没有成功分析的 {TECHNIQUE_NAMES[technique]} 文件。")
        return
    fig = _overlay_figure(technique, items)
    st.pyplot(fig, width="stretch")
    st.download_button(
        f"下载 {technique} 叠加图（PNG）", data=_figure_bytes(fig), file_name=f"{technique}_overlay.png",
        mime="image/png", key=f"download_{technique}_overlay",
    )
    plt.close(fig)
    summary = pd.DataFrame([item["row"] for item in items])
    st.markdown("##### 汇总表")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        f"下载 {technique} 汇总表（CSV）", data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{technique}_batch_results.csv", mime="text/csv", key=f"download_{technique}_summary",
    )
    st.markdown("##### 单文件结果")
    for item in items:
        with st.expander(item["data"].filename):
            result_rows = [
                {"参数": key, "数值": value}
                for key, value in item["result"].items()
                if not isinstance(value, (dict, list, tuple, pd.DataFrame))
            ]
            st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)


def _show_results(state):
    st.divider()
    st.subheader("分析结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功", sum(len(state["results"][key]) for key in ("CV", "GCD", "EIS")))
    c2.metric("CV", len(state["results"]["CV"]))
    c3.metric("GCD", len(state["results"]["GCD"]))
    c4.metric("EIS", len(state["results"]["EIS"]))
    with st.expander("所用计算公式、单位与适用条件",expanded=False):
        from core.research_validation import FORMULAS
        st.dataframe(pd.DataFrame(
            [{"计算项目":key,"公式 / 说明":formula}
             for key,formula in FORMULAS.items()
            ]),hide_index=True,width="stretch")
        st.warning("CV 最大/最小电流位置仅为峰值候选；"
                   "三电极或电极配置未确认时，不能报告器件能量密度。"
                   "两电极模式的质量必须为两极总活性质量。")
    st.markdown("### 论文计算结果与 Origin 数据导出")
    st.caption("打包各类参数汇总、原始与处理后的 CSV、EIS 拟合曲线、"
               "每文件分析配置和质量提示。图片为预览，不替代科研核验。")
    review_rows=[]
    for technique in ("CV","GCD","EIS"):
        for item in state["results"][technique]:
            flags=quality_flags(technique,item)
            scientific_notes=item.get("analysis_warnings",[])
            review_rows.append({"文件":item["data"].filename,
                                "类型":technique,
                                "规则提示":"; ".join(flags) if flags else "无自动提示",
                                "科研解释提醒":"; ".join(scientific_notes)})
    if review_rows:
        with st.expander("查看参数与数据质量检查",expanded=False):
            st.dataframe(pd.DataFrame(review_rows),hide_index=True,
                         width="stretch")
    if st.button("生成批量分析数据包",key="analysis_batch_build_v11"):
        try:
            st.session_state["analysis_batch_export_v11"] = build_batch_analysis_zip(state)
        except Exception as exc:
            st.error(f"导出失败：{type(exc).__name__}: {exc}")
    if st.session_state.get("analysis_batch_export_v11"):
        st.download_button(
            "下载 CV / GCD / EIS 计算结果与 Origin 数据 ZIP",
            data=st.session_state["analysis_batch_export_v11"],
            file_name=(str(st.session_state.get("madap_export_basename",
                            "electrochem_analysis_results")).strip()
                           .replace("/", "_").replace("\\\\", "_")[:90]
                           or "electrochem_analysis_results")+".zip",
            mime="application/zip",key="analysis_batch_download_v11",
        )
    if state["errors"]:
        with st.expander(f"分析失败/警告（{len(state['errors'])}）", expanded=True):
            for message in state["errors"]:
                st.error(message)
    render_supercap_kinetics(state)
    tabs = st.tabs(["CV 叠加与结果", "GCD 叠加与结果", "EIS 叠加与结果", "总汇总表"])
    with tabs[0]:
        _show_type_results("CV", state["results"]["CV"])
    with tabs[1]:
        _show_type_results("GCD", state["results"]["GCD"])
    with tabs[2]:
        _show_type_results("EIS", state["results"]["EIS"])
    with tabs[3]:
        rows = [item["row"] for technique in ("CV", "GCD", "EIS") for item in state["results"][technique]]
        if rows:
            total = pd.DataFrame(rows)
            st.dataframe(total, width="stretch", hide_index=True)
            st.download_button(
                "下载全部汇总表（CSV）", data=total.to_csv(index=False).encode("utf-8-sig"),
                file_name="electrochemical_batch_summary.csv", mime="text/csv", key="download_total_summary",
            )
        else:
            st.info("暂无成功结果。")


class _CachedUpload:
    """Keep uploaded bytes available when a different top-level page is opened."""
    def __init__(self,name,raw):
        self.name=name
        self._raw=raw
    def getvalue(self):
        return self._raw


def _v16_uploaded(technique):
    """Technique-specific retained uploads, independent of widget cleanup."""
    cache_key=f"v16_saved_uploads_{technique}"
    version_key=f"v16_uploader_version_{technique}"
    version=int(st.session_state.get(version_key,0))
    files=st.file_uploader(
        f"导入 {technique} 数据文件（TXT / CSV / TSV / XLSX / ZIP）",
        type=["txt","csv","tsv","xlsx","zip"],
        accept_multiple_files=True,
        key=f"v16_upload_{technique}_{version}")
    if files:
        # Streamlit's file_uploader widget may be destroyed when a different
        # top-level page opens. Store bytes under a separate durable key.
        captured=[(f.name,f.getvalue()) for f in files]
        if captured!=st.session_state.get(cache_key):
            st.session_state[cache_key]=captured
            st.session_state.pop(f"v16_result_{technique}",None)
            st.session_state.pop(f"v16_zip_{technique}",None)
    cached=st.session_state.get(cache_key,[])
    if cached:
        cols=st.columns([4,1])
        cols[0].caption("已保留文件："+"、".join(x[0] for x in cached[:4])+
            (" …" if len(cached)>4 else ""))
        if cols[1].button("清空",key=f"v16_clear_{technique}"):
            st.session_state.pop(cache_key,None)
            st.session_state.pop(f"v16_result_{technique}",None)
            st.session_state.pop(f"v16_zip_{technique}",None)
            st.session_state[version_key]=version+1
            st.rerun()
    return [_CachedUpload(n,r) for n,r in cached]


def _v16_state(technique,items):
    return {"results":{"CV":items if technique=="CV" else [],
                       "GCD":items if technique=="GCD" else [],
                       "EIS":items if technique=="EIS" else []},
            "errors":[]}


def _v16_module(technique):
    """One independent, stateful CV / GCD / EIS work area."""
    left,right=st.columns([.38,.62],gap="medium")
    samples=[]
    state=None
    signature=None
    with left:
        with st.container(border=True):
            st.markdown(f"### {technique} · 数据与参数")
            uploads=_v16_uploaded(technique)
            if uploads:
                datasets,errors,source_signature,count=import_multiformat(
                    uploads,context=f"v16_{technique.lower()}")
                samples=[d for d in datasets if d.technique==technique]
                other=[d for d in datasets if d.technique!=technique]
                if other:
                    st.warning("以下文件不是当前分析类型，未加入计算："+
                        "、".join(d.filename for d in other))
                for message in errors:
                    st.error(message)
                if samples:
                    table=pd.DataFrame([{
                        "文件":d.filename,"数据点":len(d.data),
                        "类型":d.technique} for d in samples])
                    st.dataframe(table,hide_index=True,
                                 width="stretch",
                                 height=min(230,max(90,54+35*len(table))))
                else:
                    st.info("当前没有已确认的 "+technique+" 数据。")
            else:
                st.caption("每个分析页独立保存上传文件。切换到其他工作区后再返回，数据仍会保留。")

            if samples:
                if technique=="CV":
                    missing=[d.filename for d in samples
                        if d.metadata.get("scan_rate_V_s") is None]
                    if missing:
                        st.warning("缺失扫描速率，请在下方手动填写并确认："+"、".join(missing))
                    configs=_render_cv_parameters(samples)
                elif technique=="GCD":
                    missing=[d.filename for d in samples
                        if get_default_discharge_current_A(d.metadata) is None]
                    if missing:
                        st.warning("缺失放电电流，请手动填写并确认："+"、".join(missing))
                    configs=_render_gcd_parameters(samples)
                else:
                    configs=_render_eis_parameters(samples)

                signature=hashlib.sha256(
                    (technique+source_signature+repr(configs)).encode("utf-8")
                ).hexdigest()
                stored=st.session_state.get(f"v16_result_{technique}")
                if stored and stored[0]!=signature:
                    st.warning("检测到文件或参数变化；请重新运行当前模块。")
                if st.button("▶ 运行 "+technique+" 分析",type="primary",
                             width="stretch",key=f"v16_run_{technique}"):
                    analyses={"CV":_analyze_cv,"GCD":_analyze_gcd,"EIS":_analyze_eis}
                    items=[]
                    failures=[]
                    with st.spinner("正在分析 "+technique+" 数据…"):
                        for data,config in zip(samples,configs):
                            try:
                                items.append(analyses[technique](data,config))
                            except Exception as exc:
                                failures.append(f"{data.filename}：{exc}")
                    state=_v16_state(technique,items)
                    state["errors"]=failures
                    st.session_state[f"v16_result_{technique}"]=(signature,state)
                    st.session_state.pop(f"v16_zip_{technique}",None)
                elif stored and stored[0]==signature:
                    state=stored[1]

                if state:
                    n=len(state["results"][technique])
                    if n:st.success(f"已分析 {n} 个文件")
                    for msg in state["errors"]:
                        st.error(msg)
            else:
                st.button("▶ 运行 "+technique+" 分析",
                    disabled=True,width="stretch",
                    key=f"v16_run_{technique}")

    with right:
        with st.container(border=True):
            st.markdown(f"### {technique} · 曲线与结果")
            verified_rate=None
            if technique=="GCD" and signature is not None:
                choices={f"{i+1}. {x['data'].filename}":i for i,x in
                         enumerate(state["results"]["GCD"])} if state else {}
                chosen=st.session_state.get("v16_gcd_rate_selected",[])
                agreed=bool(st.session_state.get("v16_gcd_rate_confirm",False))
                rate_signature=hashlib.sha256(
                    repr((signature,chosen,agreed)).encode()).hexdigest()
                stored_rate=st.session_state.get("v16_gcd_rate_result")
                if (agreed and chosen and all(k in choices for k in chosen)
                    and stored_rate and stored_rate[0]==rate_signature):
                    verified_rate=stored_rate[1]
            cv_kinetics_entry=None
            cv_kinetics_valid=False
            if technique=="CV" and state and signature:
                # An existing multi-scan kinetic fit may be displayed only
                # when the selected files, branch, confirmation and optional
                # peak windows exactly match the current input widgets.
                selected=st.session_state.get("v16_cv_kinetics_selected",[])
                verified=bool(st.session_state.get(
                    "v16_cv_kinetics_same_sample",False))
                branch=st.session_state.get("v16_cv_kinetics_branch","forward")
                manual=bool(st.session_state.get(
                    "v17_peak_manual_window",False))
                windows=None
                if manual and selected:
                    keys=("v17_anodic_min","v17_anodic_max",
                          "v17_cathodic_min","v17_cathodic_max")
                    if all(k in st.session_state for k in keys):
                        windows={
                            "anodic":(st.session_state[keys[0]],
                                      st.session_state[keys[1]]),
                            "cathodic":(st.session_state[keys[2]],
                                        st.session_state[keys[3]])}
                allowed_labels={
                    f"{i+1}. {x['data'].filename} · "
                    f"{x['config']['scan_rate_V_s']*1000:g} mV/s"
                    for i,x in enumerate(state["results"]["CV"])}
                entry=st.session_state.get("v16_cv_kinetics_result")
                calculated=cv_kinetics_signature(
                    signature,selected,branch,verified,manual,windows)
                if (entry and verified and len(selected)>=3 and
                        all(k in allowed_labels for k in selected) and
                        entry[0]==calculated):
                    cv_kinetics_entry=entry
                    cv_kinetics_valid=True
            madap_show_workspace(
                state=state,datasets=samples,technique=technique,
                rate_frame=verified_rate,
                cv_kinetics_entry=cv_kinetics_entry,
                cv_kinetics_valid=cv_kinetics_valid,
            )
            if state:
                items=state["results"][technique]
                if items:
                    st.markdown("**质量检查**")
                    review=pd.DataFrame([{
                        "文件":x["data"].filename,
                        "数据提示":"; ".join(quality_flags(technique,x)) or "无自动提示",
                        "科研解释":"; ".join(x.get("analysis_warnings",[])),
                    } for x in items])
                    with st.expander("查看每个文件的异常提示",expanded=False):
                        st.dataframe(review,hide_index=True,width="stretch")
                    if st.button("生成 "+technique+" 分析数据包",
                                 key=f"v16_prepare_zip_{technique}"):
                        try:
                            st.session_state[f"v16_zip_{technique}"]=(
                                signature,build_batch_analysis_zip(state))
                        except Exception as exc:
                            st.error(f"导出失败：{exc}")
                    content=st.session_state.get(f"v16_zip_{technique}")
                    if content and content[0]==signature:
                        st.download_button(
                            "下载 "+technique+" 结果与 Origin CSV",
                            data=content[1],
                            file_name=f"{technique}_analysis_origin.zip",
                            mime="application/zip",
                            key=f"v16_download_{technique}",
                            width="stretch",
                        )
                if state["errors"]:
                    with st.expander("未完成分析的文件"):
                        for msg in state["errors"]:st.error(msg)
    if state and state["results"][technique]:
        if technique in ("CV","GCD"):
            with st.expander(f"{technique} 进阶科研分析",expanded=False):
                render_extension(technique,state["results"][technique],signature)
        elif technique=="EIS":
            with st.expander("EIS 多样品拟合参数比较",expanded=False):
                render_eis_comparison(state["results"][technique],signature)


def render_electrochemical_analysis():
    madap_set_style()
    st.title("电化学分析平台")
    # Streamlit tabs all mount together, so each uploader and control remains
    # instantiated during CV<->GCD<->EIS tab changes.
    cv_tab,gcd_tab,eis_tab=st.tabs(
        ["CV · 循环伏安","GCD · 恒流充放电","EIS · 阻抗谱"])
    with cv_tab:
        _v16_module("CV")
    with gcd_tab:
        _v16_module("GCD")
    with eis_tab:
        _v16_module("EIS")


def render_home():
    madap_set_style()
    st.title("超级电容器电化学分析平台")
    st.markdown("CV、GCD、EIS 分区分析，独立保留数据、参数与结果。")
    st.markdown("进入 **电化学分析**，选择对应标签页上传实验文件并运行。"
                "GCD 温度预测作为独立功能保留。")


# Keys belonging to Streamlit buttons, downloads and uploads must never be
# restored from session_state. This predicate intentionally has no module-level
# dependencies so both app entrypoint and regression-test isolation can use it.
def _v17_is_nonrestorable_widget_key(key):
    if not isinstance(key,str):
        return True
    lower=key.lower()
    disallowed_parts=(
        "download","upload","button","run","submit","clear",
        "prepare_zip","export","generate",
        "validation_csv","prediction_csv",
        "dataset_zip",
    )
    if any(token in lower for token in disallowed_parts):
        return True
    if lower.startswith((
        "analysis_batch_","electrochem_batch_",
        "electrochem_result_","electrochem_uploaded_",
        "v16_result_","v16_zip_","v16_saved_",
        "v16_control_shadow","v16_gcd_temperature_file_",
        "madap_annotations_",
    )):
        return True
    return not lower.startswith((
        "batch_","v14_","v16_","v17_",
        "gcd_","six_mode_","stage33_",
        "temp_","madap_",
    ))


def _v16_restore_controls():
    """Restore only allowed scalar/list inputs after a page switch.

    Streamlit 1.x forbids assigning session-state values to button,
    download-button and file-uploader widgets. In particular, the old
    v17 shadow incorrectly restored madap_notes_download_CV and crashed.
    """
    saved=st.session_state.get("v16_control_shadow",{})
    clean={k:v for k,v in saved.items()
           if not _v17_is_nonrestorable_widget_key(k)}
    st.session_state["v16_control_shadow"]=clean
    # DO NOT delete active Streamlit button keys here.
    # A clicked st.button()/st.download_button() has True in session_state
    # *before* this script registers the widget; deleting it would swallow
    # the click. Only sanitize our own shadow cache, never the live widget
    # state. This fixes "PCA-Ridge 预测点击后没有反应".
    for key,value in clean.items():
        if key not in st.session_state:
            st.session_state[key]=value


def _v16_store_controls():
    """Persist only user-editable scalar/list input controls."""
    snapshot={k:v for k,v in
              st.session_state.get("v16_control_shadow",{}).items()
              if not _v17_is_nonrestorable_widget_key(k)}
    for key,value in list(st.session_state.items()):
        if _v17_is_nonrestorable_widget_key(key):
            continue
        if isinstance(value,(str,int,float,bool,type(None))):
            snapshot[key]=value
        elif isinstance(value,list) and len(value)<=100 and all(
                isinstance(x,(str,int,float,bool)) for x in value):
            snapshot[key]=list(value)
    st.session_state["v16_control_shadow"]=snapshot


PAGE_ANALYSIS="电化学分析"
PAGE_TEMP="GCD 温度预测"

_v16_restore_controls()

st.markdown(
    '<div style="font-size:14px;font-weight:700;color:#32665a;'
    'letter-spacing:.12em;padding:4px 0 8px 0">'
    'ELECTROCHEMICAL ANALYSIS  /  RESEARCH WORKSPACE</div>',
    unsafe_allow_html=True,
)
page=st.radio("功能导航",[PAGE_ANALYSIS,PAGE_TEMP],
    horizontal=True,label_visibility="collapsed",key="v16_page")

if page==PAGE_ANALYSIS:
    render_electrochemical_analysis()
else:
    # Temperature module owns its data source and reset button.
    # The old v16 uploader reset would not clear paper-stage3.3 PCA-GPR.
    render_temperature_gcd_screen()

_v16_store_controls()
