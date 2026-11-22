"""CV Analysis parameters: order, actual kinetic values and stale-result safety."""
import unittest

import numpy as np
import pandas as pd

from core.cv_parameters_v18 import (
    BASE_ORDER, KINETIC_ORDER,
    build_cv_parameters_table, kinetics_signature,
)
from core.supercap_kinetics import analyse_cv_kinetics
from core.literature_methods_v17 import cv_peak_b_fits


class CVFile:
    def __init__(self,name):
        self.filename=name


def make_items():
    rows=[]
    for rate in [.005,.01,.02,.05]:
        v=np.r_[np.linspace(0,1,121),np.linspace(1,0,121)]
        # Faradaic-like positive/negative scan-dependent peaks plus a
        # known rate dependence suitable for peak-current log regression.
        i=np.r_[np.full(121,.001*rate**.72),
                np.full(121,-.002*rate**.63)]
        data=pd.DataFrame({"potential_V":v,"current_A":i})
        name=f"sample_{rate:.3f}.txt"
        rows.append({
            "data":CVFile(name),"plot_df":data,
            "config":{"label":name,"scan_rate_V_s":rate,
                      "scan_rate_confirmed":True,"cycle_number":1},
            "row":{
                "文件名":name,"类型":"CV","图例":name,"循环":1,
                "扫描速率 (V/s)":rate,
                "电位窗口 (V)":1.0,
                "最大电流位置 (V; 峰需确认)":0.5,
                "最小电流位置 (V; 峰需确认)":0.5,
                "峰间距 (V)":0,
                "比电容 (F/g)":10+rate,
            }})
    return rows


class ParameterTableV18Tests(unittest.TestCase):
    def test_order_and_uncomputed_for_single_cv(self):
        frame=build_cv_parameters_table(make_items()[:1])
        actual=list(frame.columns)
        self.assertLess(actual.index("电位窗口 (V)"),actual.index("比电容 (F/g)"))
        self.assertEqual(actual.index("比电容 (F/g)")+1,
                         actual.index("峰间距 (V)"))
        self.assertTrue(np.isnan(frame.iloc[0]["氧化峰 b（同组拟合）"]))
        self.assertTrue(np.isnan(frame.iloc[0]["Dunn 电容控制贡献 (%)"]))
        self.assertIn("未计算",frame.iloc[0]["动力学拟合状态"])

    def test_verified_group_populates_only_selected_scans(self):
        items=make_items()
        subset=items[:3]
        analysis=analyse_cv_kinetics(subset,"forward")
        peak_sum,peak_points=cv_peak_b_fits(subset)
        entry=("match",analysis,peak_sum,peak_points,None)
        table=build_cv_parameters_table(items,entry,True)
        self.assertAlmostEqual(float(table.iloc[0]["氧化峰 b（同组拟合）"]),.72,places=6)
        self.assertAlmostEqual(float(table.iloc[1]["还原峰 b（同组拟合）"]),.63,places=6)
        self.assertTrue(np.isfinite(table.iloc[0]["Dunn 电流拟合 RMSE (A)"]))
        self.assertTrue(np.isnan(table.iloc[3]["氧化峰 b（同组拟合）"]))
        self.assertIn("未计算",table.iloc[3]["动力学拟合状态"])
        contribution=table.iloc[0]["Dunn 电容控制贡献 (%)"]
        if np.isfinite(contribution):
            self.assertGreaterEqual(contribution,0)
            self.assertLessEqual(contribution,100)

    def test_invalid_kinetics_entry_does_not_populate(self):
        items=make_items()[:3]
        analysis=analyse_cv_kinetics(items,"forward")
        sums,points=cv_peak_b_fits(items)
        entry=("previous",analysis,sums,points,None)
        frame=build_cv_parameters_table(items,entry,False)
        self.assertTrue(frame["氧化峰 b（同组拟合）"].isna().all())

    def test_signature_changes_with_selection_or_peak_windows(self):
        original=kinetics_signature("source",["1","2","3"],"forward",True,
                                    False,None)
        changed=kinetics_signature("source",["1","2","4"],"forward",True,
                                    False,None)
        altered=kinetics_signature("source",["1","2","3"],"forward",True,
                                   True,{"anodic":(.1,.4),"cathodic":(.1,.4)})
        self.assertNotEqual(original,changed)
        self.assertNotEqual(original,altered)


if __name__=="__main__":
    unittest.main()
