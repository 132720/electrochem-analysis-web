"""Self-contained regression tests for the EIS v5 numerical comparison."""
import io
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from core.eis_temperature_analysis import parse_temperature_eis_upload
from core.eis_temperature_final import fit_final_dataset
from core.eis_benchmark import matched_metrics, compare_with_paper, make_comparison_bundle
from core.eis_flexible_import import list_candidates

ROOT=Path(__file__).resolve().parents[1]

class Upload:
    def __init__(self,name,payload):
        self.name=name
        self.payload=payload
    def getvalue(self):
        return self.payload

class EISBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(ROOT/"data"/"eis_test.txt").read_bytes()
        cls.record=parse_temperature_eis_upload(
            Upload("EIS_25C.txt",cls.raw),25,1)

    def test_identity_is_zero(self):
        metric,_=matched_metrics(
            self.record,self.record.data,model_name="identity")
        self.assertEqual(metric["frequency_matched_count"],84)
        self.assertLess(metric["complex_RMSE_ohm"],1e-12)

    def test_missing_frequencies_rejected(self):
        fitted=self.record.data.head(5)
        with self.assertRaises(ValueError):
            matched_metrics(self.record,fitted,model_name="too-few")

    def test_paper_fit_vs_identical_curve_agree(self):
        result=fit_final_dataset([self.record])[0]
        points=result.fitted_data[
            ["frequency_Hz","z_real_fit_ohm","z_imag_fit_ohm"]].copy()
        points.columns=["frequency_Hz","z_real_ohm","z_imag_ohm"]
        metrics,detail,_=compare_with_paper(
            self.record,points,"自定义等效电路","R(RQ)")
        self.assertEqual(len(metrics),2)
        self.assertAlmostEqual(
            metrics["relative_RMSE_pct"].iloc[0],
            metrics["relative_RMSE_pct"].iloc[1],places=10)
        self.assertTrue(
            (~metrics["direct_parameter_equivalence_verified"]).all())
        output=make_comparison_bundle(
            metrics,detail,self.record,"fit","R(RQ)","auto","auto","test")
        with zipfile.ZipFile(io.BytesIO(output)) as z:
            self.assertIsNone(z.testzip())
            self.assertIn("04_reproducibility_manifest.json",z.namelist())

    def test_eis_only_zip_ignores_cv_gcd(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,"w") as z:
            z.writestr("25/EIS/eis.txt",self.raw)
            z.writestr("25/GCD/data.txt",b"Chronopotentiometry")
            z.writestr("25/CV/data.txt",b"Cyclic Voltammetry")
        rows,errors=list_candidates([Upload("mixed.zip",b.getvalue())])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0].temperature,25.)

if __name__=="__main__":
    unittest.main()
