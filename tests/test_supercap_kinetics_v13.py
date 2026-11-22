"""Regression tests for real data prerequisites, Dunn and GCD statistics."""
import io
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_loader import read_chi_txt
from core.gcd_analysis import (
    build_gcd_cycles, get_gcd_cycle, get_discharge_segment,
    calculate_gcd_parameters,
)
from core.research_validation import apply_gcd_safety
from core.supercap_kinetics import (
    analyse_cv_kinetics,gcd_rate_summary,gcd_cycling_summary,make_kinetics_zip,
)

ROOT=Path(__file__).resolve().parents[1]

class Upload:
    def __init__(self,name,raw):self.name=name;self.raw=raw
    def getvalue(self):return self.raw

class ExampleData:
    def __init__(self,name,df):
        self.filename=name;self.data=df;self.metadata={}

def synthetic_cv_samples(rates=(.005,.01,.02,.05,.1),
                         k1=.08,k2=.0009):
    items=[]
    for rate in rates:
        voltage=np.r_[np.linspace(0,1,121),np.linspace(1,0,121)]
        intensity=k1*rate+k2*np.sqrt(rate)
        current=np.r_[np.full(121,intensity),np.full(121,-intensity)]
        df=pd.DataFrame({"potential_V":voltage,"current_A":current})
        items.append({"data":ExampleData(f"rate_{rate}",df),
             "config":{"label":str(rate),"scan_rate_V_s":rate,
                       "scan_rate_confirmed":True,"cycle_number":1},
             "plot_df":df})
    return items

class KineticsV13Tests(unittest.TestCase):
    def test_known_dunn_coefficients_and_fraction(self):
        items=synthetic_cv_samples()
        response=analyse_cv_kinetics(items,"forward",121)
        points=response["b_vs_potential"]
        self.assertAlmostEqual(float(points["Dunn k1"].median()),.08,places=7)
        self.assertAlmostEqual(float(points["Dunn k2"].median()),.0009,places=7)
        self.assertTrue(response["dunn_by_scan_rate"]
                        ["Capacitive contribution (%)"].notna().all())
        self.assertTrue((points["b"]>.5).all() and (points["b"]<1).all())
        self.assertLess(float(points["Dunn current fit RMSE (A)"].max()),1e-10)

    def test_kinetics_rejects_unverified_and_insufficient_rates(self):
        items=synthetic_cv_samples()
        with self.assertRaises(ValueError):
            analyse_cv_kinetics(items[:2])
        items[0]["config"]["scan_rate_confirmed"]=False
        with self.assertRaisesRegex(ValueError,"未经确认"):
            analyse_cv_kinetics(items)
        items[0]["config"]["scan_rate_confirmed"]=True
        items[0]["config"]["scan_rate_V_s"]=items[1]["config"]["scan_rate_V_s"]
        with self.assertRaises(ValueError):
            analyse_cv_kinetics(items[:3])

    def test_no_claim_if_signed_dunn_components_inconsistent(self):
        items=synthetic_cv_samples(k1=-.02,k2=.002)
        result=analyse_cv_kinetics(items,"forward")
        df=result["dunn_by_scan_rate"]
        self.assertTrue(df["Capacitive contribution (%)"].isna().all())
        self.assertTrue(df["Quality status"].str.contains("INCONSISTENT").all())

    @classmethod
    def _gcd_example(cls):
        data=read_chi_txt(Upload("gcd_test.txt",
               (ROOT/"data"/"gcd_test.txt").read_bytes()))
        cy=build_gcd_cycles(data.data,data.metadata.get("segment_count"))
        c,info=get_gcd_cycle(data.data,int(cy[-1]["cycle"]),
                            data.metadata.get("segment_count"))
        discharge=get_discharge_segment(data.data,info)
        result=calculate_gcd_parameters(discharge,.0004,1.,1.)
        result=apply_gcd_safety(result,discharge,.0004,1.,False)
        cfg={"label":"specimen","current_A":.0004,"current_confirmed":True,
             "mass_mg":1.,"factor":1.,"device_mode":False}
        return {"data":data,"plot_df":c,"result":result,"config":cfg,"row":{}}

    def test_gcd_cycling_reference(self):
        item=self._gcd_example()
        frame,failed,ref=gcd_cycling_summary(item,reference_cycle=1)
        self.assertEqual(ref,1)
        self.assertEqual(len(frame),2)
        self.assertEqual(len(failed),0)
        self.assertAlmostEqual(float(frame["Retention vs reference (%)"].iloc[0]),100)
        with self.assertRaisesRegex(ValueError,"参考循环"):
            gcd_cycling_summary(item,reference_cycle=999)

    def test_rate_summary_reference_and_consistency(self):
        a=self._gcd_example()
        b=self._gcd_example()
        a["data"].filename="low_current.txt"
        b["data"].filename="high_current.txt"
        a["config"]["current_A"]=.0004
        b["config"]["current_A"]=.0008
        a["result"]["current_density_A_g"]=.4
        b["result"]["current_density_A_g"]=.8
        a["result"]["specific_capacitance_F_g"]=100.
        b["result"]["specific_capacitance_F_g"]=70.
        frame=gcd_rate_summary([b,a])
        self.assertAlmostEqual(frame["Rate retention vs lowest current (%)"].iloc[0],100)
        self.assertAlmostEqual(frame["Rate retention vs lowest current (%)"].iloc[1],70)
        b["config"]["mass_mg"]=2.
        with self.assertRaisesRegex(ValueError,"质量"):
            gcd_rate_summary([a,b])

    def test_origin_zip(self):
        items=synthetic_cv_samples()
        result=analyse_cv_kinetics(items,"forward")
        content=make_kinetics_zip({"CV/b_values.csv":result["b_vs_potential"]},
                                  result["summary"])
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            self.assertIsNone(z.testzip())
            self.assertIn("CV/b_values.csv",z.namelist())
            frame=pd.read_csv(io.BytesIO(z.read("CV/b_values.csv")))
            self.assertEqual(len(frame),120)

if __name__=="__main__":
    unittest.main()
