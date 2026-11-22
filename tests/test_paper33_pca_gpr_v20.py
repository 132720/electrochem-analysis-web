"""Exact PCA-GPR paper-stage3.3 source reproduction, no other predictor."""
import unittest
from pathlib import Path
import numpy as np

from core.pca_gpr_paper33 import (
 training_manifest,load_training_records,find_heldout_record,
 fit_and_predict_paper33,paper33_heldout_comparison,
 physical_prediction_as_frame,predict_pca_gpr_full_curve,
)
ROOT=Path(__file__).resolve().parents[1]

class GPRPaper33(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=(ROOT/"data/gcd_paper_stage33_training.zip").read_bytes()
  cls.m=training_manifest(cls.raw)

 def test_manifests_and_no_mixed_concentration(self):
  self.assertEqual(len(self.m),69)
  self.assertEqual(set(self.m.concentration_mol_L),{1,2,4})
  records=load_training_records(self.raw,self.m,2,60)
  self.assertEqual(len(records),15)
  self.assertTrue(all(x.temperature_C<60 for x in records))
  self.assertTrue(all(x.metadata["concentration_mol_L"]==2 for x in records))

 def test_paper_1mol_70C_GPR_exact_reproduction(self):
  records=load_training_records(self.raw,self.m,1,70)
  model,pred=fit_and_predict_paper33(records,70,1.,uncertainty_samples=20)
  self.assertAlmostEqual(pred.predicted_total_period_s,183.79990561909526,places=4)
  self.assertEqual(len(pred.physical_curve),401)
  self.assertTrue(np.isfinite(pred.physical_curve.predicted_potential_V).all())
  obs=find_heldout_record(self.raw,self.m,1,70,1.)
  self.assertAlmostEqual(obs.total_period_s,174.5,places=2)
  _,metrics=paper33_heldout_comparison(pred,obs)
  self.assertAlmostEqual(abs(metrics["Period relative error (%)"]),5.3294588075,places=3)
  self.assertEqual(len(physical_prediction_as_frame(pred)),401)
  self.assertGreater(model.explained_variance_ratio_sum,.8)

 def test_other_representative_periods(self):
  for c,t,expected in [(2,60,346.0489375908853),(4,50,386.6253095755462)]:
   with self.subTest(c=c):
    records=load_training_records(self.raw,self.m,c,t)
    _,pred=fit_and_predict_paper33(records,t,1.0,uncertainty_samples=0)
    self.assertAlmostEqual(pred.predicted_total_period_s,expected,places=3)

 def test_other_model_modules_deleted_and_screen_only_one(self):
  core=ROOT/"core"
  for name in ("pca_ridge_only.py","gcd_pca_gpr_final.py",
               "gcd_temperature_neural_v2.py",
               "gcd_temperature_v3_models.py","gcd_temperature_v4_models.py"):
   self.assertFalse((core/name).exists(),name)
  screen=(ROOT/"screens/temperature_gcd_screen.py").read_text(encoding="utf8")
  self.assertIn("PCA-GPR 训练并预测",screen)
  self.assertNotIn("PCA-Ridge",screen)
  self.assertNotIn("Piecewise-Ridge",screen)

if __name__=="__main__": unittest.main()
