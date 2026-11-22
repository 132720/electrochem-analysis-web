"""Supercapacitor EIS paper-oriented analysis and export.

All reported Rs/Rct/CPE/W values come from fitted impedance functions.
Fits use measured frequencies only; raw data is never modified.
Publication exports (PDF/SVG/600dpi PNG) use the very same plotted data.
"""
from __future__ import annotations

import io
import json
import warnings
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

MODEL_SIMPLE = "Rs + CPE"
MODEL_ARC = "Rs + (Rct || CPE)"
MODEL_WARBURG = "Rs + (Rct || CPE) + W"
MODELS = (MODEL_SIMPLE, MODEL_ARC, MODEL_WARBURG)

COLORS = ["#17668c", "#dd8950", "#6c8b69", "#9a6799",
          "#b39b4a", "#4e8792", "#bd6578", "#777777"]

@dataclass
class PaperFit:
    label: str
    model: str
    parameters: dict
    success: bool
    rmse_ohm: float
    relative_rmse_pct: float
    n_points: int
    warnings: list[str]
    curve: pd.DataFrame

def impedance(f,model,params):
    f=np.asarray(f,dtype=float)
    w=2*np.pi*f
    jw=1j*w
    rs=params["Rs_ohm"]
    q=params["Q_S_s_n"]
    n=params["n"]
    cpe=1/(q*jw**n)
    if model==MODEL_SIMPLE:
        return rs+cpe
    zpar=1/(1/params["Rct_ohm"]+q*jw**n)
    if model==MODEL_ARC:
        return rs+zpar
    if model==MODEL_WARBURG:
        return rs+zpar+params["sigma_W"]/np.sqrt(jw)
    raise ValueError("未知电路")

def _params_from_x(model,x):
    d=dict(Rs_ohm=float(x[0]),Q_S_s_n=float(x[1]),n=float(x[2]))
    if model!=MODEL_SIMPLE:
        d["Rct_ohm"]=float(x[3])
    if model==MODEL_WARBURG:
        d["sigma_W"]=float(x[4])
    return d

def fit_one(record,model=MODEL_ARC,exclude_inductive=True):
    if model not in MODELS:raise ValueError("不支持的电路模型")
    df=record.data[["frequency_Hz","z_real_ohm","z_imag_ohm"]].copy()
    df=df.replace([np.inf,-np.inf],np.nan).dropna()
    df=df[df.frequency_Hz>0]
    if exclude_inductive:
        df=df[df.z_imag_ohm<=0].copy()
    df=df.sort_values("frequency_Hz",ascending=False)
    if len(df)<9:
        raise ValueError("可用 EIS 数据点不足 9 个；请检查是否过度排除了感抗点")
    f=df.frequency_Hz.to_numpy(float)
    z=df.z_real_ohm.to_numpy(float)+1j*df.z_imag_ohm.to_numpy(float)
    rs=max(float(np.percentile(df.z_real_ohm,5)),1e-7)
    span=max(float(np.max(df.z_real_ohm)-np.min(df.z_real_ohm)),rs*.5,1e-4)
    peak=max(float(np.max(-df.z_imag_ohm)),1e-4)
    f_characteristic=float(f[np.argmax(-df.z_imag_ohm)])
    q_guess=np.clip(1/(2*np.pi*f_characteristic*max(span,1e-5)),1e-9,1e2)
    p0=[rs,q_guess,.85]
    lower=[1e-10,1e-12,.2]
    upper=[max(float(np.max(np.abs(z)))*20,1e5),1e4,1.0]
    if model!=MODEL_SIMPLE:
        p0.append(max(span,1e-5));lower.append(1e-10);upper.append(max(float(np.max(abs(z)))*30,1e6))
    if model==MODEL_WARBURG:
        p0.append(max(peak*np.sqrt(2*np.pi*float(np.min(f)))*0.1,1e-7))
        lower.append(0);upper.append(max(float(np.max(abs(z)))*np.sqrt(2*np.pi*float(np.max(f)))*30,1e6))
    p0=np.clip(np.asarray(p0),np.asarray(lower)*1.01,
               np.asarray(upper)*.95)
    scale=np.maximum(np.abs(z),max(float(np.median(np.abs(z)))*1e-6,1e-12))
    def residual(x):
        zz=impedance(f,model,_params_from_x(model,x))
        v=(zz-z)/scale
        return np.r_[v.real,v.imag]
    # 3 physically valid starts reduce local-minimum sensitivity.
    starts=[p0]
    for mul in (.4,2.5):
        trial=p0.copy()
        trial[1]=np.clip(trial[1]*mul,lower[1]*1.01,upper[1]*.99)
        if model!=MODEL_SIMPLE:
            trial[3]=np.clip(trial[3]*mul,lower[3]*1.01,upper[3]*.99)
        starts.append(trial)
    solutions=[]
    for x0 in starts:
        result=least_squares(residual,x0,bounds=(lower,upper),method="trf",
                             x_scale="jac",max_nfev=9000)
        solutions.append(result)
    best=min(solutions,key=lambda r:float(np.sum(r.fun**2)))
    pars=_params_from_x(model,best.x)
    fitted=impedance(f,model,pars)
    resid=fitted-z
    pointwise=np.abs(resid)/np.maximum(abs(z),1e-12)
    messages=[]
    if not best.success:messages.append("优化未声明收敛")
    ratio_lower=np.abs(best.x-np.asarray(lower))/np.maximum(abs(best.x),1e-10)
    ratio_upper=np.abs(best.x-np.asarray(upper))/np.maximum(abs(best.x),1e-10)
    if (ratio_lower<1e-4).any() or (ratio_upper<1e-4).any():
        messages.append("有拟合参数接近边界；物理解释需谨慎")
    if model==MODEL_WARBURG:
        messages.append("W 为半无限扩散模型，仅在可辨识且有物理依据时使用")
        if pars["sigma_W"] < 1e-8:
            messages.append("Warburg 系数接近零，不能解释为检测到了扩散过程")
    if model!=MODEL_SIMPLE and pars["Rct_ohm"] > max(float(np.max(abs(z)))*10,100):
        messages.append("Rct 极大且未出现可辨识半圆，Rct 可能不可辨识；不要报告为确定的电荷转移电阻")
    if np.sqrt(np.mean(pointwise**2))*100>10:
        messages.append("相对复阻抗拟合误差超过10%；建议检查模型")
    if len(df)!=len(record.data):
        messages.append(f"排除了 {len(record.data)-len(df)} 个感抗/无效频点")
    output=pd.DataFrame({
       "frequency_Hz":f,
       "Zreal_exp_ohm":z.real,"Zimag_exp_ohm":z.imag,
       "Zreal_fit_ohm":fitted.real,"Zimag_fit_ohm":fitted.imag,
       "relative_residual_pct":pointwise*100,
    })
    return PaperFit(record.filename,model,pars,bool(best.success),
          float(np.sqrt(np.mean(abs(resid)**2))),
          float(np.sqrt(np.mean(pointwise**2))*100),len(f),messages,output)

def fit_many(records,model=MODEL_ARC,exclude_inductive=True):
    fits=[];errors=[]
    for rec in records:
        try:fits.append(fit_one(rec,model,exclude_inductive))
        except Exception as exc:errors.append({"sample":rec.filename,"error":str(exc)})
    return fits,pd.DataFrame(errors)

def parameter_table(fits):
    rows=[]
    for fit in fits:
        row={"Sample":fit.label,"Circuit":fit.model,"Rs (Ω)":fit.parameters.get("Rs_ohm"),
             "Rct / Rinterface (Ω)":fit.parameters.get("Rct_ohm",np.nan),
             "Q (S·s^n)":fit.parameters.get("Q_S_s_n"),
             "n":fit.parameters.get("n"),
             "Warburg σ (Ω·s^-1/2)":fit.parameters.get("sigma_W",np.nan),
             "Complex RMSE (Ω)":fit.rmse_ohm,
             "Relative RMSE (%)":fit.relative_rmse_pct,
             "Points fitted":fit.n_points,
             "Converged":fit.success,
             "Warnings":"; ".join(fit.warnings)}
        rows.append(row)
    return pd.DataFrame(rows)

def complex_capacitance(record):
    """C*=1/(jωZ); full-cell apparent complex capacitance, not specific capacitance."""
    df=record.data
    f=df.frequency_Hz.to_numpy(float)
    z=df.z_real_ohm.to_numpy(float)+1j*df.z_imag_ohm.to_numpy(float)
    c=1/(1j*2*np.pi*f*z)
    return pd.DataFrame({"frequency_Hz":f,"C_real_F":c.real,
                          "C_loss_F":-c.imag})

def relaxation_summary(record):
    frame=complex_capacitance(record)
    x=frame["C_loss_F"].to_numpy()
    if len(x)<5 or not np.isfinite(x).all():
        return {"Sample":record.filename,"f0 (Hz)":np.nan,"tau0 (s)":np.nan,
                "Status":"invalid"}
    at=int(np.argmax(x))
    if at==0 or at==len(x)-1:
        return {"Sample":record.filename,"f0 (Hz)":np.nan,"tau0 (s)":np.nan,
                "Status":"peak at frequency boundary - not reportable"}
    f=float(frame["frequency_Hz"].iloc[at])
    return {"Sample":record.filename,"f0 (Hz)":f,"tau0 (s)":1/f,
            "Status":"peak sampled inside frequency range"}

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
       "font.family":"DejaVu Sans","font.size":9,
       "axes.labelsize":9,"axes.linewidth":.8,
       "xtick.labelsize":8,"ytick.labelsize":8,
       "legend.fontsize":8,
       "pdf.fonttype":42,"ps.fonttype":42,
       "svg.fonttype":"none","savefig.facecolor":"white",
    })
    return plt

def _palette(labels):
    return {label:COLORS[i%len(COLORS)] for i,label in enumerate(labels)}

def draw_figure(records,fits,kind="Nyquist",width=3.5,height=2.8,
                inset=True,labels=None):
    plt=_mpl()
    fig,ax=plt.subplots(figsize=(width,height),dpi=140)
    names=[r.filename for r in records]
    pal=_palette(names)
    fitmap={r.label:r for r in fits}
    for rec in records:
        display=(labels or {}).get(rec.filename,rec.filename)
        df=rec.data; f=df.frequency_Hz.to_numpy(float)
        zr=df.z_real_ohm.to_numpy(float);zi=df.z_imag_ohm.to_numpy(float)
        color=pal[rec.filename]
        if kind=="Nyquist":
            ax.scatter(zr,-zi,color=color,s=9,marker="o",label=display,zorder=3,
                       edgecolors="none")
            fit=fitmap.get(rec.filename)
            if fit is not None:
                tab=fit.curve
                ax.plot(tab["Zreal_fit_ohm"],-tab["Zimag_fit_ohm"],
                        color=color,lw=1.1,alpha=.9)
        elif kind=="Bode modulus":
            order=np.argsort(f)
            ax.semilogx(f[order],np.hypot(zr,zi)[order],color=color,
                        lw=1.35,label=display)
        elif kind=="Bode phase":
            order=np.argsort(f)
            ax.semilogx(f[order],np.degrees(np.arctan2(zi,zr))[order],
                        color=color,lw=1.35,label=display)
        elif kind=="C' and C''":
            cdf=complex_capacitance(rec)
            order=np.argsort(cdf.frequency_Hz)
            ax.semilogx(cdf.frequency_Hz.iloc[order],
                        cdf.C_real_F.iloc[order],lw=1.3,color=color,
                        label=display+" C'")
            ax.semilogx(cdf.frequency_Hz.iloc[order],
                        cdf.C_loss_F.iloc[order],lw=1.2,color=color,
                        ls="--",label=display+" C''")
        elif kind=="Residuals":
            fit=fitmap.get(rec.filename)
            if fit is None:continue
            tab=fit.curve
            ax.semilogx(tab.frequency_Hz,tab.relative_residual_pct,
                        color=color,lw=1.15,label=display)
        else:
            plt.close(fig)
            raise ValueError(kind)
    if kind=="Nyquist":
        ax.set_xlabel(r"$Z'$ ($\Omega$)")
        ax.set_ylabel(r"$-Z''$ ($\Omega$)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=min(0,float(ax.get_ylim()[0])))
        ax.set_aspect("equal",adjustable="box")
        if inset and records:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            # Separate high-frequency panel outside the main plot:
            # publication data must never be hidden underneath an inset.
            axin=fig.add_axes([.73,.57,.24,.32])
            for rec in records:
                df=rec.data
                # close-up based on the high-frequency third of each curve
                subset=df.nlargest(max(5,len(df)//3),"frequency_Hz")
                axin.scatter(subset.z_real_ohm,-subset.z_imag_ohm,s=4,
                             color=pal[rec.filename],edgecolors="none")
                fit=fitmap.get(rec.filename)
                if fit is not None:
                    fitted=fit.curve.nlargest(max(5,len(fit.curve)//3),"frequency_Hz")
                    axin.plot(fitted.Zreal_fit_ohm,-fitted.Zimag_fit_ohm,
                              color=pal[rec.filename],lw=.7)
            axin.tick_params(labelsize=6,length=2)
            axin.set_title("High-frequency",fontsize=6)
    elif kind=="Bode modulus":
        ax.set_xlabel("Frequency (Hz)");ax.set_ylabel(r"$|Z|$ ($\Omega$)")
    elif kind=="Bode phase":
        ax.set_xlabel("Frequency (Hz)");ax.set_ylabel("Phase (deg)")
    elif kind=="C' and C''":
        ax.set_xlabel("Frequency (Hz)");ax.set_ylabel("Complex capacitance (F)")
    elif kind=="Residuals":
        ax.set_xlabel("Frequency (Hz)");ax.set_ylabel("Relative residual (%)")
    ax.tick_params(direction="in",length=3)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.grid(False)
    if len(records):
        if kind=="Nyquist" and inset:
            handles, legends=ax.get_legend_handles_labels()
            fig.legend(handles,legends,loc="upper left",bbox_to_anchor=(.72,.49),
                       frameon=False,fontsize=6.5,ncol=1)
        else:
            ax.legend(loc="best",frameon=False,fontsize=7,
                      ncol=1 if len(records)<5 else 2)
    if kind=="Nyquist" and inset:
        fig.subplots_adjust(left=.12,right=.69,bottom=.16,top=.93)
    else:
        fig.tight_layout(pad=.7)
    return fig

def draw_parameters(fits,width=4.5,height=2.8):
    plt=_mpl()
    table=parameter_table(fits)
    fig,axes=plt.subplots(1,2,figsize=(width,height),dpi=140)
    labels=list(table["Sample"])
    xs=np.arange(len(labels))
    for ax,key in zip(axes,["Rs (Ω)","Rct / Rinterface (Ω)"]):
        y=table[key].to_numpy(float)
        present=np.isfinite(y)
        ax.bar(xs[present],y[present],color=COLORS[:len(y)] if len(y)<len(COLORS) else
               [COLORS[i%len(COLORS)] for i in range(len(y))],width=.64)
        ax.set_xticks(xs,labels,rotation=35,ha="right")
        ax.tick_params(axis="x",labelsize=6)
        ax.set_ylabel(key);ax.grid(False)
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=.8)
    return fig

def draw_circuit(model,width=4.5,height=1.8):
    """A schematic of the actual impedance expression, not a fitted mechanism claim."""
    plt=_mpl()
    fig,ax=plt.subplots(figsize=(width,height),dpi=140)
    ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis("off")
    def line(x1,y1,x2,y2):
        ax.plot([x1,x2],[y1,y2],color="#23435a",lw=1.35)
    def resistor(x1,x2,y,label):
        line(x1,y,x1+.2,y);line(x2-.2,y,x2,y)
        ax.add_patch(plt.Rectangle((x1+.2,y-.26),x2-x1-.4,.52,
                                   facecolor="#f4f6f7",edgecolor="#23435a",lw=1))
        ax.text((x1+x2)/2,y,label,ha="center",va="center",fontsize=9)
    line(.3,2,1,2);resistor(1,3,2,"Rs")
    if model==MODEL_SIMPLE:
        line(3,2,4,2);resistor(4,6.5,2,"CPE");line(6.5,2,9.6,2)
    else:
        line(3,2,4.1,2);line(4.1,1,4.1,3);line(4.1,3,4.4,3)
        line(4.1,1,4.4,1)
        resistor(4.4,6.6,3,"Rct")
        resistor(4.4,6.6,1,"CPE")
        line(6.6,3,7,3);line(6.6,1,7,1)
        line(7,1,7,3);line(7,2,7.55,2)
        if model==MODEL_WARBURG:
            resistor(7.55,9.1,2,"W");line(9.1,2,9.6,2)
        else:line(7.55,2,9.6,2)
    ax.text(5,-.15,model,ha="center",fontsize=10)
    fig.tight_layout(pad=.25)
    return fig

def _write_figure(z,stem,fig,formats=("png","pdf","svg")):
    for ext in formats:
        buf=io.BytesIO()
        fig.savefig(buf,format=ext,dpi=600,bbox_inches="tight",pad_inches=.07)
        z.writestr("figures/"+stem+"."+ext,buf.getvalue())
    _mpl().close(fig)

def make_paper_bundle(records,fits,model,figure_types=None,
                      width=3.6,height=3.0,inset=True,labels=None):
    if not records:raise ValueError("没有可导出的谱线")
    figure_types=figure_types or ["Nyquist","Bode modulus","Bode phase",
                                  "Residuals","Rs/Rct bars","Circuit"]
    table=parameter_table(fits)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("data/fit_parameters.csv",table.to_csv(index=False).encode("utf-8-sig"))
        frames=[]
        for rec in records:
            d=rec.data.copy();d.insert(0,"Sample",rec.filename);frames.append(d)
        z.writestr("data/original_EIS.csv",
                   pd.concat(frames,ignore_index=True).to_csv(index=False).encode("utf-8-sig"))
        if fits:
            curves=[]
            for fit in fits:
                d=fit.curve.copy();d.insert(0,"Sample",fit.label);curves.append(d)
            z.writestr("data/fitted_EIS.csv",
                       pd.concat(curves,ignore_index=True).to_csv(index=False).encode("utf-8-sig"))
        for kind in figure_types:
            if kind=="Rs/Rct bars":
                if fits:_write_figure(z,"06_Rs_Rct",draw_parameters(fits,max(4,width),height))
            elif kind=="Circuit":
                _write_figure(z,"07_equivalent_circuit",draw_circuit(model))
            else:
                fig=draw_figure(records,fits,kind,width,height,inset,labels=labels)
                safe={"Nyquist":"01_Nyquist","Bode modulus":"02_Bode_modulus",
                      "Bode phase":"03_Bode_phase",
                      "Residuals":"04_fit_residuals",
                      "C' and C''":"05_complex_capacitance"}[kind]
                _write_figure(z,safe,fig)
        manifest={
          "model":model,"figure_size_inches":[width,height],
          "raster_dpi":600,"vector_formats":["PDF","SVG"],
          "sample_labels":labels or {},
          "parameters_provenance":"SciPy weighted complex least-squares fits",
          "note_Rct":"Rct nomenclature needs model/experiment-specific mechanistic justification",
          "note_capacitance":"C*=1/(j omega Z), not gravimetric/specific capacitance",
          "note_relaxation":"tau0=1/f0 only if C'' peak lies inside measured frequency interval",
          "note_inset":"high-frequency third of measured frequencies; inset uses native data",
          "source":"original imported frequencies and complex impedances",
        }
        z.writestr("README_metadata.json",json.dumps(manifest,indent=2,
                            ensure_ascii=False).encode("utf-8"))
    return buf.getvalue()
