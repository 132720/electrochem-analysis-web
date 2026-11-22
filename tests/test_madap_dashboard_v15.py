"""Visual workspace smoke tests; no Streamlit runtime necessary."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load_dashboard():
    path=ROOT/"screens"/"madap_dashboard.py"
    fake_streamlit=types.ModuleType("streamlit")
    with patch.dict(sys.modules,{"streamlit":fake_streamlit}):
        spec=importlib.util.spec_from_file_location("v15_dashboard_module",path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class DummyData:
    def __init__(self,name,kind,df):
        self.filename=name
        self.technique=kind
        self.data=df
        self.metadata={}


class DashboardTests(unittest.TestCase):
    def test_empty_eis(self):
        mod=load_dashboard()
        fig=mod.diagnostic_figure("EIS",[],[])
        self.assertEqual(len(fig.layout.annotations),4)

    def test_eis_four_panels_with_actual_residual(self):
        mod=load_dashboard()
        f=np.geomspace(1e5,.1,30)
        zr=1+10/(1+f/300)
        zi=-10/(1+300/f)
        raw=pd.DataFrame({"frequency_Hz":f,"z_real_ohm":zr,
                          "z_imag_ohm":zi})
        simulated=pd.DataFrame({"frequency_Hz":f,
            "z_real_fit_ohm":zr+0.03,"z_imag_fit_ohm":zi})
        d=DummyData("sample","EIS",raw)
        item={"data":d,"plot_df":raw,"config":{"label":"sample"},
             "fit_result":{"fit_data":raw,"fitted_data":simulated},"result":{}}
        fig=mod.diagnostic_figure("EIS",[item])
        self.assertGreaterEqual(len(fig.data),6)
        residual=fig.data[-1]
        self.assertTrue(np.isfinite(np.array(residual.y,dtype=float)).all())
        self.assertTrue((np.array(residual.y,dtype=float)>0).all())
        self.assertEqual(fig.layout.xaxis2.type,"log")
        self.assertEqual(fig.layout.xaxis4.type,"log")

    def test_cv_and_gcd_results(self):
        mod=load_dashboard()
        v=np.r_[np.linspace(0,.8,30),np.linspace(.8,0,30)]
        i=np.r_[np.full(30,.0008),np.full(30,-.0008)]
        df=pd.DataFrame({"potential_V":v,"current_A":i})
        item={"data":DummyData("CV.txt","CV",df),"plot_df":df,
              "config":{"label":"CV","scan_rate_V_s":.01},
              "result":{"oxidation_peak_potential_V":.4,
                        "specific_capacitance_F_g":80}}
        fig=mod.diagnostic_figure("CV",[item])
        self.assertGreaterEqual(len(fig.data),3)
        gc=pd.DataFrame({"time_s":[0,1,2,3,4,5],
                         "potential_V":[0,.2,.4,.5,.25,0]})
        item2={"data":DummyData("GCD.txt","GCD",gc),"plot_df":gc,
               "config":{"label":"GCD","cycle_number":None},
               "result":{"current_density_A_g":1,
                         "specific_capacitance_F_g":150,
                         "ir_drop_V":.02}}
        fig2=mod.diagnostic_figure("GCD",[item2])
        # With no valid discharge segment or verified rate-series, show only
        # measured GCD and capacitance; never fabricate IR drop/retention.
        self.assertGreaterEqual(len(fig2.data),2)


if __name__=="__main__":
    unittest.main()
