"""Paper graphics and EIS circuit numerical regression."""
import io
import unittest
import zipfile
import numpy as np
import pandas as pd

from core.eis_temperature_analysis import TemperatureEISRecord
from core.eis_paper_figures import (
    MODEL_ARC, MODEL_SIMPLE, MODEL_WARBURG, MODELS, impedance,
    fit_one, parameter_table, complex_capacitance, relaxation_summary,
    make_paper_bundle,
)

def synthetic():
    f=np.geomspace(1e5,.01,86)
    truth={"Rs_ohm":.9,"Rct_ohm":3.4,
           "Q_S_s_n":.0025,"n":.86}
    z=impedance(f,MODEL_ARC,truth)
    frame=pd.DataFrame({"frequency_Hz":f,"z_real_ohm":z.real,
                        "z_imag_ohm":z.imag})
    record=TemperatureEISRecord("test","sample_one.csv",0.,1,{},frame)
    return record,truth

class PaperTests(unittest.TestCase):
    def test_arc_parameter_recovery(self):
        rec,truth=synthetic()
        fit=fit_one(rec,MODEL_ARC,exclude_inductive=False)
        self.assertTrue(fit.success)
        self.assertAlmostEqual(fit.parameters["Rs_ohm"],truth["Rs_ohm"],places=2)
        self.assertAlmostEqual(fit.parameters["Rct_ohm"],truth["Rct_ohm"],places=2)
        self.assertLess(fit.relative_rmse_pct,.05)
        table=parameter_table([fit])
        self.assertAlmostEqual(table["Rct / Rinterface (Ω)"].iloc[0],3.4,places=2)

    def test_all_model_types(self):
        rec,truth=synthetic()
        for model in MODELS:
            with self.subTest(model=model):
                fit=fit_one(rec,model,False)
                self.assertTrue(np.isfinite(fit.rmse_ohm))
                self.assertGreater(fit.n_points,10)

    def test_capacitance_complex_definition(self):
        rec,_=synthetic()
        cap=complex_capacitance(rec)
        f=rec.data.frequency_Hz.to_numpy()
        z=rec.data.z_real_ohm.to_numpy()+1j*rec.data.z_imag_ohm.to_numpy()
        expected=1/(1j*2*np.pi*f*z)
        np.testing.assert_allclose(cap["C_real_F"],expected.real)
        np.testing.assert_allclose(cap["C_loss_F"],-expected.imag)

    def test_publication_bundle_vector_and_raster(self):
        rec,_=synthetic()
        fit=fit_one(rec,MODEL_ARC,False)
        pack=make_paper_bundle([rec],[fit],MODEL_ARC,
             figure_types=["Nyquist","Bode modulus","Bode phase",
                           "Residuals","Rs/Rct bars","Circuit","C' and C''"])
        with zipfile.ZipFile(io.BytesIO(pack)) as z:
            self.assertIsNone(z.testzip())
            names=z.namelist()
            for name in ("figures/01_Nyquist.svg","figures/01_Nyquist.pdf",
                         "figures/01_Nyquist.png",
                         "data/fit_parameters.csv","README_metadata.json"):
                self.assertIn(name,names)
            self.assertTrue(z.read("figures/01_Nyquist.pdf").startswith(b"%PDF"))

if __name__=="__main__":
    unittest.main()
