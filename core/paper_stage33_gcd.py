# -*- coding: utf-8 -*-
"""GCD v3.3 shared calculation kernel.
Verbatim computational functions taken from:
paper-stage3.3/GCD/01_周期预测与模型评价/01_完整周期预测与误差分解.py
Only imports and source-file execution entry points are omitted.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

NODES = np.array([0, .04, .15, .35, .60, .82, .94, 1.])
GRID = np.linspace(0., 1., 401)

def features(records):
    return np.array([[float(r.temperature_C), np.log(float(r.current_value))] for r in records])


def branch_shape(frame, node=NODES):
    t = frame['time_s'].to_numpy(float)
    v = frame['potential_V'].to_numpy(float)
    t = t-t[0]
    keep = np.r_[True,np.diff(t)>1e-10]
    return np.interp(node, t[keep]/t[keep][-1], v[keep])


def mono(y, rising):
    return np.maximum.accumulate(y) if rising else np.minimum.accumulate(y)


def train_piecewise(records):
    x=features(records)
    y=[]
    for r in records:
        c=branch_shape(r.charge_df)
        d=branch_shape(r.discharge_df)
        y.append(np.r_[c,d,np.log(max(r.charge_duration_s,1.e-5)),np.log(max(r.discharge_duration_s,1.e-5))])
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
    model.fit(x,np.vstack(y))
    return model


def piecewise_predict(model,temp,current):
    y=model.predict(np.array([[temp,np.log(current)]]))[0]
    c=mono(y[:len(NODES)],True)
    d=mono(y[len(NODES):2*len(NODES)],False)
    cp=np.exp(np.clip(y[-2], -15,20)); dp=np.exp(np.clip(y[-1], -15,20))
    # The two separate branch endpoints intentionally need not be continuous;
    # a switch-time potential step is physically possible.
    grid=np.linspace(0,1,401)
    cshape=PchipInterpolator(NODES,c)(grid)
    dshape=PchipInterpolator(NODES,d)(grid)
    split=cp/(cp+dp)
    full=np.where(GRID<=split,np.interp(np.clip(GRID/split,0,1),grid,cshape),
                  np.interp(np.clip((GRID-split)/(1-split),0,1),grid,dshape))
    return full,float(cp+dp),float(split)


def normalized_metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    e=p-y
    mse=float(np.mean(e*e))
    var=float(np.sum((y-y.mean())**2))
    r2=float(1-np.sum(e*e)/var) if var>1e-15 else np.nan
    return np.sqrt(mse),float(np.mean(np.abs(e))),r2


def interpolated_voltage(shape,total,split,t,actual_total=None,actual_split=None):
    """time-warp normalized curve with independently controllable branch durations."""
    total=max(float(total),1e-9)
    split=float(np.clip(split,1e-5,1-1e-5))
    if actual_total is None: actual_total=total
    if actual_split is None: actual_split=split
    actual_split=float(np.clip(actual_split,1e-5,1-1e-5))
    u=np.where(t<=actual_total*actual_split,
               (np.minimum(t,actual_total*actual_split)/max(actual_total*actual_split,1e-9))*split,
               split+np.maximum(t-actual_total*actual_split,0)/max(actual_total*(1-actual_split),1e-9)*(1-split))
    u=np.clip(u,0,1)
    return np.interp(u,GRID,shape)


def reconstruct_for_diagnostics(shape, total, predicted_split, r):
    times=r.cycle_df['time_s'].to_numpy(float)
    times=times-times[0]
    true_total=float(r.total_period_s)
    true_split=float(r.charge_duration_s/true_total)
    # All outputs on identical measured physical-time samples, constant end continuation.
    base=interpolated_voltage(shape,total,predicted_split,times)
    total_oracle=interpolated_voltage(shape,total,predicted_split,times,actual_total=true_total)
    branches_oracle=interpolated_voltage(shape,total,predicted_split,times,actual_total=true_total,actual_split=true_split)
    return times,base,total_oracle,branches_oracle

