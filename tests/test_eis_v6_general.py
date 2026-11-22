"""Regression checks for temperature-optional general EIS analysis."""
import hashlib
import unittest
from pathlib import Path
import numpy as np

from core.eis_flexible_import import list_candidates, parse_candidate
from core.eis_temperature_final import (
    run_final_kk_screen, screen_candidate_models, fit_final_dataset,
    final_fit_table, direct_high_frequency_intercept,
    model_free_relaxation_descriptors,
)

ROOT=Path(__file__).resolve().parents[1]

class Upload:
    def __init__(self,name,body):
        self.name=name
        self.body=body
    def getvalue(self):
        return self.body

class GeneralEISTests(unittest.TestCase):
    def test_temperature_not_required_for_spectrum(self):
        body=(ROOT/"data"/"eis_test.txt").read_bytes()
        candidates, errors=list_candidates([Upload("test_unknown.txt",body)])
        self.assertEqual(len(candidates),1)
        self.assertIsNone(candidates[0].temperature)
        record=parse_candidate(candidates[0],10000,1)[0]
        record.metadata["actual_temperature_C"]=None
        record.metadata["sample_index"]=1
        record.record_id=hashlib.sha256(b"sample_1").hexdigest()
        self.assertEqual(len(record.data),84)
        kk=run_final_kk_screen([record])
        candidates_fit=screen_candidate_models([record])
        fits=fit_final_dataset([record])
        fit_table=final_fit_table(fits)
        self.assertEqual(len(kk),1)
        self.assertEqual(len(fits),1)
        self.assertEqual(len(fit_table),1)
        self.assertGreater(len(candidates_fit.candidate_summary),0)
        self.assertEqual(len(direct_high_frequency_intercept([record])),1)
        self.assertEqual(len(model_free_relaxation_descriptors([record])),1)

    def test_temperature_path_still_supported(self):
        body=(ROOT/"data"/"eis_test.txt").read_bytes()
        candidates,errors=list_candidates([Upload("EIS_35C.txt",body)])
        self.assertEqual(candidates[0].temperature,35.)

if __name__ == "__main__":
    unittest.main()
