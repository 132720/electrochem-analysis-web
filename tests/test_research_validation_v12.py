"""Ground-truth numerical tests and error/experiment guardrails for v12."""
import io
import json
import unittest
import zipfile

import numpy as np
import pandas as pd

from core.cv_analysis import calculate_cv_integral,calculate_cv_specific_capacitance
from core.gcd_analysis import calculate_gcd_parameters
from core.research_validation import (
    validate_condition, apply_gcd_safety, FORMULAS,
)
from core.analysis_export import build_batch_analysis_zip

class ValidationV12Tests(unittest.TestCase):
    def test_cv_rectangular_cycle_integral_and_capacitance(self):
        # Rectangular current ±2mA across 1V in forward and reverse sweep:
        # total path integral=0.004 A V; m=2mg, rate=.1V/s -> 10 F/g.
        v=np.r_[np.linspace(0,1,101),np.linspace(1,0,101)]
        i=np.r_[np.full(101,.002),np.full(101,-.002)]
        frame=pd.DataFrame({"potential_V":v,"current_A":i})
        self.assertAlmostEqual(calculate_cv_integral(frame),.004,places=9)
        self.assertAlmostEqual(calculate_cv_specific_capacitance(
            frame,.1,2.,1.),10.,places=8)

    def test_gcd_linear_discharge_known_answer(self):
        # I=2mA, mass=2mg, 1V over 10s -> 10 F/g
        # capacity=10 C/g and measured E=1.3889 Wh/kg
        t=np.linspace(0,10,101)
        v=1-t/10
        discharge=pd.DataFrame({"time_s":t,"potential_V":v})
        base=calculate_gcd_parameters(discharge,.002,2.,1.)
        self.assertAlmostEqual(base["specific_capacitance_F_g"],10,places=5)
        no_device=apply_gcd_safety(base,discharge,.002,2.,False)
        self.assertIsNone(no_device["energy_density_Wh_kg"])
        self.assertAlmostEqual(no_device["specific_capacity_C_g"],10)
        self.assertAlmostEqual(no_device["specific_capacity_mAh_g"],10/3.6)
        device=apply_gcd_safety(base,discharge,.002,2.,True)
        self.assertAlmostEqual(device["energy_density_Wh_kg"],1.38888888889,places=6)
        self.assertAlmostEqual(device["power_density_W_kg"],500,places=6)

    def test_invalid_device_voltage_rejected(self):
        t=np.linspace(0,10,101)
        discharge=pd.DataFrame({"time_s":t,"potential_V":np.linspace(.4,-.5,101)})
        base=calculate_gcd_parameters(discharge,.001,1.,1.)
        with self.assertRaisesRegex(ValueError,"两电极"):
            apply_gcd_safety(base,discharge,.001,1.,True)

    def test_missing_conditions_and_nan_rejected(self):
        for bad in [None,0,-1,float("nan"),float("inf"),"unknown"]:
            with self.subTest(value=bad),self.assertRaises(ValueError):
                validate_condition(bad,"测试值")

    def test_formula_audit_export(self):
        class D:pass
        d=D()
        d.filename="sample.txt"
        d.data=pd.DataFrame({"potential_V":[0.,1.],"current_A":[1.,2.]})
        d.metadata={}
        state={"results":{"CV":[{"data":d,"config":{"label":"test",
              "mass_mg":1,"scan_rate_V_s":.1,"cycle_number":1,
              "scan_rate_confirmed":True},"plot_df":d.data,
              "result":{"value":5.},"row":{"Sample":"sample.txt"},
              "analysis_warnings":["Check peak assignment"]}],
             "GCD":[],"EIS":[]},"errors":[]}
        output=build_batch_analysis_zip(state)
        with zipfile.ZipFile(io.BytesIO(output)) as z:
            self.assertIsNone(z.testzip())
            manifest=[x for x in z.namelist() if x.endswith("_analysis.json")][0]
            info=json.loads(z.read(manifest))
            self.assertIn("Check peak assignment",info["scientific_warnings"])
            self.assertIn("CV_specific_capacitance",info["formula_references"])
            self.assertIn("00_formulas.json",z.namelist())

if __name__=="__main__":
    unittest.main()
