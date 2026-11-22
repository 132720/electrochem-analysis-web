"""Research integrity checks and explicit electrochemical formula provenance.

This module adds scientific guardrails without altering legacy CV/GCD kernels.
Do not infer an electrode configuration or replace missing conditions with defaults.
"""
from __future__ import annotations
import numpy as np

FORMULAS = {
 "CV_path_integral": "A = sum(0.5*(|I_i|+|I_{i+1}|)*|V_{i+1}-V_i|) [A V]",
 "CV_specific_capacitance": "Csp = k*A/(2*m_g*scan_rate_V_s*voltage_window_V) [F/g]; complete forward+reverse cycle",
 "GCD_specific_capacitance": "Csp = k*I_A*delta_t_s/(m_g*delta_V_V) [F/g]",
 "GCD_capacity": "Qsp = I_A*delta_t_s/m_g [C/g] = (I_A*delta_t_s/m_g)/3.6 [mAh/g]",
 "GCD_device_energy": "E = I_A*integral(V_device(t) dt)/(3.6*m_total_g) [Wh/kg]; measured discharge curve",
 "GCD_device_power": "P = 3600*E/delta_t_s [W/kg]",
 "EIS_fit": "weighted nonlinear least squares of real and imaginary Z(f); circuit-dependent parameters",
}

def validate_condition(value,label):
    if value is None:
        raise ValueError(f"{label}缺失：必须从原始数据读取，或由用户明确手动设置。")
    try:value=float(value)
    except (TypeError,ValueError):
        raise ValueError(f"{label}必须是有效数值") from None
    if not np.isfinite(value) or value<=0:
        raise ValueError(f"{label}必须是大于零的有限数值")
    return value

def apply_gcd_safety(result,discharge_df,current_A,mass_mg,device_mode=False):
    """Keep basic GCD results, suppress device claims absent explicit device mode.

    Device energy integrates measured terminal voltage, not the triangular
    approximation. Its reference is 0 V and requires a positive discharge curve.
    """
    result=dict(result)
    t=discharge_df.time_s.to_numpy(float)
    v=discharge_df.potential_V.to_numpy(float)
    start=int(result["analysis_start_index"])
    t=t[start:]
    v=v[start:]
    if len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0):
        raise ValueError("GCD 放电时间轴无效或不严格递增")
    if not np.all(np.isfinite(v)):
        raise ValueError("GCD 放电电压含非有限值")
    result["specific_capacity_C_g"]=None
    result["specific_capacity_mAh_g"]=None
    result["energy_density_Wh_kg"]=None
    result["power_density_W_kg"]=None
    result["device_energy_method"]="not computed (electrode configuration unverified)"
    if mass_mg is not None and float(mass_mg)>0:
        mass_g=float(mass_mg)/1000.0
        capacity=float(current_A)*float(result["discharge_time_s"])/mass_g
        result["specific_capacity_C_g"]=capacity
        result["specific_capacity_mAh_g"]=capacity/3.6
        if device_mode:
            if np.min(v)<-1e-9 or np.max(v)<=0:
                raise ValueError("两电极器件电压必须非负且含正电压；不计算器件能量/功率")
            area=float(np.trapezoid(v,t))
            energy=float(current_A)*area/(3.6*mass_g)
            result["energy_density_Wh_kg"]=energy
            result["power_density_W_kg"]=3600*energy/float(result["discharge_time_s"])
            result["device_energy_method"]="measured integral I*Vdt / total active mass"
    return result

def cv_analysis_warnings(data, config):
    notes=[]
    if config.get("cycle_number") is None:
        notes.append("未找到完整 CV 循环，不能可靠报告采用双向积分的比电容")
    if config.get("mass_mg",0)<=0:
        notes.append("未填写有效活性质量；不报告质量比电容")
    notes.append("氧化/还原峰为当前曲线全局最大/最小电流位置，须人工确认属于真实氧化还原峰")
    return notes

def gcd_analysis_warnings(data,config,result):
    notes=[]
    if not config.get("device_mode",False):
        notes.append("未确认两电极器件及总活性质量，已抑制器件能量密度/功率密度")
    if config.get("factor",1)!=1:
        notes.append("使用 k=4 换算时须确认对称两电极单电极等效约定与质量基准")
    if result.get("linearity_r2") is not None and result["linearity_r2"]<.9:
        notes.append("GCD 放电曲线明显非线性：IΔt/(mΔV) 是表观平均电容指标")
    return notes
