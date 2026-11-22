"""Controlled format, unit, Nyquist sign and ZIP import tests."""
import io
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.universal_import_v14 import (
    InputBlob, VirtualUpload, auto_map_columns, identify_technique,
    make_dataset, preview_blob, unpack_uploads,
)

ROOT=Path(__file__).resolve().parents[1]

class ImportV14Tests(unittest.TestCase):
    def _from_csv(self,name,df):
        blob=InputBlob(name,df.to_csv(index=False).encode("utf-8-sig"))
        return blob,preview_blob(blob)

    def test_cv_millivolts_milliamps(self):
        v=np.array([0,100,200,300,200,100,0],float)
        i=np.array([1,1,1,1,-1,-1,-1],float)
        blob,pre=self._from_csv("CV.csv",
            pd.DataFrame({"Potential (mV)":v,"Current (mA)":i}))
        self.assertEqual(pre["technique"],"CV")
        d=make_dataset(blob,pre,"CV",pre["suggested"])
        np.testing.assert_allclose(d.data.potential_V,v*.001)
        np.testing.assert_allclose(d.data.current_A,i*.001)
        self.assertEqual(d.metadata["column_mapping"]["current_A"]["factor_to_SI"],.001)

    def test_gcd_excel_time_ms(self):
        df=pd.DataFrame({"Time/ms":[0,1000,2000,3000,4000],
                         "Potential/V":[.5,.4,.3,.2,.1]})
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w:
            df.to_excel(w,index=False,sheet_name="Cycle data")
        blob=InputBlob("GCD.xlsx",buf.getvalue())
        pre=preview_blob(blob)
        self.assertEqual(pre["technique"],"GCD")
        d=make_dataset(blob,pre,"GCD",pre["suggested"])
        np.testing.assert_allclose(d.data.time_s,[0,1,2,3,4])
        self.assertEqual(d.metadata["source_sheet"],"Cycle data")

    def test_eis_negative_imag_nyquist_sign(self):
        f=np.array([1e5,1e3,1e1,1.,.1])
        source=pd.DataFrame({
            "Frequency (Hz)":f,
            "Z' (Ω)":[1.,2.,4.,6.,10.],
            "-Z'' (Ω)":[.2,.3,1,3,7]
        })
        blob,pre=self._from_csv("Nyquist.csv",source)
        self.assertEqual(pre["technique"],"EIS")
        d=make_dataset(blob,pre,"EIS",pre["suggested"])
        np.testing.assert_allclose(d.data.z_imag_ohm,[-.2,-.3,-1,-3,-7])
        self.assertEqual(d.metadata["column_mapping"]["z_imag_ohm"]["sign_multiplier"],-1)

    def test_malformed_row_is_not_silently_dropped(self):
        df=pd.DataFrame({"Potential/V":[0,.1,.2,.3],"Current/A":[.1,"bad",.3,.4]})
        blob,pre=self._from_csv("bad.csv",df)
        with self.assertRaisesRegex(ValueError,"未擅自丢弃"):
            make_dataset(blob,pre,"CV",pre["suggested"])

    def test_ambiguous_headers_require_mapping(self):
        df=pd.DataFrame({"x":[0,.1,.2,.3],"y":[1,2,3,4]})
        blob,pre=self._from_csv("unknown.csv",df)
        self.assertIsNone(pre["technique"])
        with self.assertRaisesRegex(ValueError,"缺少"):
            make_dataset(blob,pre,"CV",pre["suggested"])
        d=make_dataset(blob,pre,"CV",{"potential_V":("x",1,1),
                                    "current_A":("y",.001,1)})
        self.assertAlmostEqual(d.data.current_A.iloc[0],.001)

    def test_zip_mixed_valid_bad_and_traversal(self):
        cv=pd.DataFrame({"Potential/V":[0,.1,.2,.3],"Current/A":[.01,.02,.03,.04]})
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr("good/test.csv",cv.to_csv(index=False))
            z.writestr("../malicious.csv",cv.to_csv(index=False))
            z.writestr("readme.md","note")
            z.writestr("wrong.csv","hello world\n")
        blobs,errors=unpack_uploads([VirtualUpload("batch.zip",buf.getvalue())])
        self.assertEqual(len(blobs),2)
        self.assertTrue(any("危险" in e for e in errors))
        valid=[b for b in blobs if "test.csv" in b.name][0]
        d=make_dataset(valid,preview_blob(valid),"CV",
                       preview_blob(valid)["suggested"])
        self.assertEqual(len(d.data),4)
        with self.assertRaises(Exception):
            preview_blob([b for b in blobs if "wrong" in b.name][0])

    def test_original_chi_txt_backward_compatible(self):
        blob=InputBlob("cv_test.txt",(ROOT/"data"/"cv_test.txt").read_bytes())
        pre=preview_blob(blob)
        self.assertIsNotNone(pre["chi"])
        self.assertEqual(pre["chi"].technique,"CV")
        self.assertEqual(pre["chi"].metadata["scan_rate_V_s"],.01)

    def test_invalid_eis_frequency_and_gcd_time(self):
        for technique,df in [
            ("EIS",pd.DataFrame({"frequency_Hz":[0,-1,2,3],
              "z_real_ohm":[1,2,3,4],"z_imag_ohm":[-1,-2,-3,-4]})),
            ("GCD",pd.DataFrame({"time_s":[0,2,1,3],
               "potential_V":[.5,.4,.3,.2]})),
        ]:
            blob,pre=self._from_csv(f"{technique}.csv",df)
            with self.assertRaises(ValueError):
                make_dataset(blob,pre,technique,
                             {c:(c,1,1) for c in df.columns})

if __name__=="__main__":
    unittest.main()
