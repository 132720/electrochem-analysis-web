"""Numerical regression for literature-aligned CV peak b and EIS comparisons."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.literature_methods_v17 import cv_peak_b_fits,eis_compare_fits

ROOT=Path(__file__).resolve().parents[1]


class Data:
    def __init__(self,name):self.filename=name

def cv_set():
    result=[]
    for rate in [.005,.01,.02,.05,.1]:
        v=np.r_[np.linspace(0,1,120),np.linspace(1,0,120)]
        current=np.r_[np.full(120,.002*rate**.65),
                      np.full(120,-.003*rate**.58)]
        result.append({"data":Data(str(rate)),
             "plot_df":pd.DataFrame({"potential_V":v,"current_A":current}),
             "config":{"scan_rate_V_s":rate,"scan_rate_confirmed":True,
                       "cycle_number":1}})
    return result

def fit(name="test",model="Rs+(Rct||CPE)",weight="1/|Z|",rct=2):
    return {"data":Data(name),"config":{"use_range":False},
            "fit_result":{"success":True,"circuit":model,
                          "weighting":weight,
                          "exclude_inductive_requested":True,
                          "parameters":{"Rs":1,"Rct":rct,"Q":.001,"n":.9},
                          "relative_rmse_pct":1.1,"rmse_ohm":.1,
                          "number_of_points":25,
                          "frequency_min_Hz":.1,"frequency_max_Hz":1e5}}

class LiteratureV17Tests(unittest.TestCase):
    def test_anodic_and_cathodic_peak_b_ground_truth(self):
        summary,points=cv_peak_b_fits(cv_set())
        self.assertEqual(len(points),10)
        a=summary[summary.Branch=="anodic"].iloc[0]
        c=summary[summary.Branch=="cathodic"].iloc[0]
        self.assertAlmostEqual(a["b"],.65,places=8)
        self.assertAlmostEqual(c["b"],.58,places=8)
        self.assertAlmostEqual(a.R2,1,places=8)
        self.assertAlmostEqual(c.R2,1,places=8)

    def test_invalid_rates_and_peak_windows_rejected(self):
        inputs=cv_set()
        inputs[0]["config"]["scan_rate_confirmed"]=False
        with self.assertRaises(ValueError):cv_peak_b_fits(inputs)
        inputs[0]["config"]["scan_rate_confirmed"]=True
        with self.assertRaises(ValueError):
            cv_peak_b_fits(inputs,{"anodic":(2,3)})
        with self.assertRaises(ValueError):cv_peak_b_fits(inputs[:2])

    def test_eis_compare_success_and_mismatch(self):
        a,b=fit("a",rct=3),fit("b",rct=2)
        df=eis_compare_fits([a,b])
        self.assertEqual(df.iloc[0].Sample,"b")
        self.assertAlmostEqual(df.iloc[1]["Rct fit (ohm)"],3)
        b["fit_result"]["circuit"]="different model"
        with self.assertRaisesRegex(ValueError,"电路"):
            eis_compare_fits([a,b])
        b["fit_result"]["circuit"]=a["fit_result"]["circuit"]
        b["fit_result"]["weighting"]="None"
        with self.assertRaisesRegex(ValueError,"设置"):
            eis_compare_fits([a,b])
        b["fit_result"]["weighting"]="1/|Z|"
        b["fit_result"]["success"]=False
        with self.assertRaisesRegex(ValueError,"未收敛"):
            eis_compare_fits([a,b])

    def test_gcd_rate_requires_verified_rate_dataframe(self):
        spec=importlib.util.spec_from_file_location(
             "v17_gcd_figure",ROOT/"screens"/"madap_dashboard.py")
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"streamlit":types.ModuleType("streamlit")}):
            spec.loader.exec_module(module)
        fake=types.SimpleNamespace(
            filename="GCD.csv",technique="GCD",
            data=pd.DataFrame({"time_s":[0,1,2,3],
                               "potential_V":[.6,.4,.2,0]}))
        item={"data":fake,"plot_df":fake.data,"config":{"label":"GCD"},
              "result":{"current_density_A_g":1.,
                        "specific_capacitance_F_g":100.}}
        baseline=module.diagnostic_figure("GCD",[item])
        rate=pd.DataFrame({"Current density (A/g)":[1,2],
                           "Rate retention vs lowest current (%)":[100,70]})
        approved=module.diagnostic_figure("GCD",[item],rate_frame=rate)
        self.assertEqual(len(approved.data),len(baseline.data)+1)

    def test_grid_free_and_full_frame_plot(self):
        spec=importlib.util.spec_from_file_location(
             "v17_figure",ROOT/"screens"/"madap_dashboard.py")
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"streamlit":types.ModuleType("streamlit")}):
            spec.loader.exec_module(module)
        fake=types.SimpleNamespace(
            filename="sample",technique="EIS",
            data=pd.DataFrame({
                "frequency_Hz":np.geomspace(.1,1e5,24),
                "z_real_ohm":np.linspace(5,1,24),
                "z_imag_ohm":np.linspace(-4,-.1,24)
            }))
        figure=module.diagnostic_figure("EIS",raw=[fake])
        self.assertGreaterEqual(figure.layout.margin.l,85)
        self.assertGreaterEqual(figure.layout.margin.r,85)
        for name in figure.layout:
            if str(name).startswith(("xaxis","yaxis")):
                axis=figure.layout[name]
                self.assertTrue(axis.showline)
                self.assertTrue(axis.mirror)
                self.assertFalse(axis.showgrid)
                self.assertEqual(axis.zeroline,False)
        # The right subplot has a distinct domain, leaving a 24% gutter.
        left=figure.layout.xaxis.domain
        right=figure.layout.xaxis2.domain
        self.assertGreater(right[0]-left[1],.20)

if __name__=="__main__":
    unittest.main()
