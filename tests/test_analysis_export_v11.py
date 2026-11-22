"""Test Origin-friendly EIS numeric data export without a graphical interface."""
import io
import json
import unittest
import zipfile
from pathlib import Path
import pandas as pd

from core.analysis_export import build_eis_analysis_zip, build_batch_analysis_zip
from core.eis_paper_figures import MODEL_ARC, fit_one
from core.eis_temperature_analysis import parse_temperature_eis_upload

ROOT=Path(__file__).resolve().parents[1]
class Upload:
    def __init__(self,name,content):
        self.name=name
        self.content=content
    def getvalue(self):
        return self.content

class AnalysisExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=parse_temperature_eis_upload(
            Upload("EIS_sample.txt",(ROOT/"data"/"eis_test.txt").read_bytes()),25,1)
        cls.fit=fit_one(cls.record,MODEL_ARC)

    def test_eis_origin_files(self):
        content=build_eis_analysis_zip([self.record],[self.fit],pd.DataFrame(),MODEL_ARC,True)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            self.assertIsNone(z.testzip())
            self.assertIn("01_EIS_parameter_summary.csv",z.namelist())
            self.assertIn("analysis_conditions.json",z.namelist())
            names=[name for name in z.namelist() if name.endswith("_fit.csv")]
            self.assertEqual(len(names),1)
            table=pd.read_csv(io.BytesIO(z.read(names[0])))
            self.assertGreater(len(table),8)

    def test_batch_export_and_quality_flags(self):
        raw=self.record.data.copy()
        class Data: pass
        d=Data()
        d.data=raw
        d.filename="example.txt"
        d.metadata={}
        state={"results":{
            "CV":[],"GCD":[],
            "EIS":[{"data":d,"config":{"label":"example","circuit":"test"},
                    "plot_df":raw,"result":{"rs":1.2,"not_finite":float("nan")},
                    "fit_result":None,"row":{"Sample":"example"}}]
            },"errors":["bad_file.txt: unable to read"]}
        content=build_batch_analysis_zip(state)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            self.assertIsNone(z.testzip())
            self.assertIn("00_failed_files.csv",z.namelist())
            audit=[x for x in z.namelist() if x.endswith("_analysis.json")][0]
            metadata=json.loads(z.read(audit))
            self.assertIsNone(metadata["result"]["not_finite"])
            self.assertIn("NO_CIRCUIT_FIT",metadata["quality_flags"])

if __name__=="__main__":
    unittest.main()
