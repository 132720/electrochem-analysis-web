# -*- coding: utf-8 -*-
"""EIS v3.3 shared kernel. linkk_fit is verbatim from the paper's
EIS_GCD/.../final_evidence_audit.py. KK-like numeric diagnostics are
not a certification of physical Kramers-Kronig validity.
"""
import numpy as np
from scipy.optimize import least_squares

def linkk_fit(freq, z, cutoff=.85, max_m=35, add_cap=True):
    freq=np.asarray(freq,float);z=np.asarray(z,complex)
    if len(freq)<30 or np.any(freq<=0) or not np.isfinite(z).all() or not np.isfinite(freq).all():
        raise ValueError('Need >=30 finite points and positive frequencies')
    omega=2*np.pi*freq
    scale=np.maximum(abs(z),1e-12)
    selected=None
    history=[]
    for m in range(3,max_m+1):
        tau=np.geomspace(1/omega.max(),1/omega.min(),m)
        rc=1/(1+1j*omega[:,None]*tau)
        columns=[np.ones(len(z)),*[rc[:,k] for k in range(m)],1j*omega]
        if add_cap:columns.append(-1j/omega)
        a=np.column_stack(columns)
        aw=np.vstack([a.real/scale[:,None],a.imag/scale[:,None]])
        yy=np.r_[z.real/scale,z.imag/scale]
        coef=np.linalg.lstsq(aw,yy,rcond=1e-12)[0]
        r=coef[1:m+1]
        pos=np.sum(np.abs(r[r>=0]));neg=np.sum(np.abs(r[r<0]))
        mu=1-neg/pos if pos>0 else np.nan
        pred=a@coef
        residual=(z-pred)/scale
        rms=float(np.sqrt(np.mean(abs(residual)**2)))
        history.append((m,mu,rms))
        selected=(m,mu,pred,residual,coef,history)
        if cutoff is not None and np.isfinite(mu) and mu<cutoff:break
    return selected


def analyze_spectrum(record):
    """Apply paper v3.3 independent linear-KK model to an app EIS record."""
    frame = record.data.sort_values("frequency_Hz")
    f=frame["frequency_Hz"].to_numpy(dtype=float)
    z=frame["z_real_ohm"].to_numpy(dtype=float)+1j*frame["z_imag_ohm"].to_numpy(dtype=float)
    m,mu,pred,resid,coef,history=linkk_fit(f,z,max_m=35,add_cap=True)
    return dict(temperature_C=float(record.temperature_C), filename=record.filename,
                replicate=int(record.replicate_index), n_points=len(f),
                linKK_M=int(m), mu=float(mu), relative_complex_RMS=float(np.sqrt(np.mean(abs(resid)**2))),
                positive_imag=bool(np.any(z.imag>0)), physical_KK_status="NOT_ESTABLISHED")


def cpe_blocked_tl_fit(record):
    """CPE / blocked finite-length transmission-line comparison:
    parameterization, bounds, weighting and optimizer settings copied from
    paper-stage3.3/新增分析_v3.3/additional_checks.py.
    """
    df=record.data
    f=df["frequency_Hz"].to_numpy(dtype=float)
    zr=df["z_real_ohm"].to_numpy(dtype=float)
    zi=df["z_imag_ohm"].to_numpy(dtype=float)
    selection=np.isfinite(f)&np.isfinite(zr)&np.isfinite(zi)&(f>=.1)&(f<=1e4)&(zi<0)
    f,zr,zi=f[selection],zr[selection],zi[selection]
    if len(f)<8:
        return []
    order=np.argsort(f);f,zr,zi=f[order],zr[order],zi[order]
    w=2*np.pi*f
    observed=zr+1j*zi
    wt=np.maximum(abs(observed),1e-12)
    fits=[]
    for model in ['CPE','Blocked-TL']:
        def impedance(p):
            rs=np.exp(p[0])
            if model=='CPE':return rs+1/(np.exp(p[1])*(1j*w)**p[2])
            rp,cp=np.exp(p[1:]);return rs+np.sqrt(rp/(1j*w*cp))/np.tanh(np.sqrt(1j*w*rp*cp))
        def residual(p):
            r=(impedance(p)-observed)/wt;return np.r_[r.real,r.imag]
        rs=max(.01,zr.min())
        if model=='CPE': x=[np.log(rs),np.log(.004),.8];lo=[np.log(.001),np.log(1e-7),.3];hi=[np.log(10),0,1.]
        else:x=[np.log(rs),np.log(30),np.log(.004)];lo=[np.log(.001),np.log(.001),np.log(1e-7)];hi=[np.log(10),np.log(1e5),np.log(10)]
        opt=least_squares(residual,np.clip(x,np.array(lo)+1e-8,np.array(hi)-1e-8),bounds=(lo,hi),max_nfev=350)
        fits.append(dict(temperature_C=float(record.temperature_C),filename=record.filename,
           replicate=int(record.replicate_index),model=model,
           magnitude_RMSE=float(np.sqrt(np.mean(opt.fun**2)*2)),
           Rs=float(np.exp(opt.x[0])),Q_or_Rp=float(np.exp(opt.x[1])),
           alpha_or_Cp=float(opt.x[2] if model=='CPE' else np.exp(opt.x[2])),
           success=bool(opt.success),n_points_used=len(f)))
    return fits


def frequency_features(record):
    """Paper v3.3 measured-frequency feature extraction, no model fit."""
    frame=record.data
    f=frame["frequency_Hz"].to_numpy(float)
    zr=frame["z_real_ohm"].to_numpy(float)
    zi=frame["z_imag_ohm"].to_numpy(float)
    rows=[]
    for target,name in ((1e4,"10kHz"),(1.0,"1Hz"),(.1,"0p1Hz")):
        k=int(np.argmin(np.abs(np.log(f/target))))
        rows.append(dict(temperature_C=float(record.temperature_C),filename=record.filename,
                         replicate=int(record.replicate_index), target=name,
                         actual_frequency_Hz=float(f[k]),Zreal_ohm=float(zr[k]),
                         Zimag_ohm=float(zi[k]),phase_deg=float(np.degrees(np.angle(zr[k]+1j*zi[k])))))
    return rows
