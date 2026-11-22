"""v16 separated CV/GCD/EIS UI and fully framed chart regression."""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class State(dict):
    pass


class FakeFile:
    def __init__(self,name,raw):
        self.name=name;self.raw=raw
    def getvalue(self):
        return self.raw


class FakeStreamlit:
    def __init__(self):
        self.session_state=State()
        self.new_files=[]
        self.rerun_count=0
    def file_uploader(self,*args,**kwargs):
        return self.new_files
    def columns(self,*args,**kwargs):
        return [types.SimpleNamespace(caption=lambda *a,**kw:None,
             button=lambda *a,**kw:False) for _ in range(2)]
    def rerun(self):
        self.rerun_count+=1


def extract_function(name):
    source=(ROOT/"app.py").read_text(encoding="utf-8")
    syntax=ast.parse(source)
    dependency="_v17_is_nonrestorable_widget_key"
    targets=[n for n in syntax.body if isinstance(n,ast.FunctionDef)
             and n.name in ({name,dependency} if name in (
                 "_v16_store_controls","_v16_restore_controls") else {name})]
    exec(compile(ast.Module(body=targets,type_ignores=[]),"<v16 functions>","exec"),
         environment)
    return environment[name]


environment={}
class WorkspaceTests(unittest.TestCase):
    def test_separate_saved_uploads_cv_gcd_eis(self):
        fake=FakeStreamlit()
        environment.update({"st":fake,"_CachedUpload":None})
        class Cached:
            def __init__(self,name,raw):self.name=name;self.raw=raw
            def getvalue(self):return self.raw
        environment["_CachedUpload"]=Cached
        load=extract_function("_v16_uploaded")
        fake.new_files=[FakeFile("CV.txt",b"CV raw")]
        self.assertEqual(load("CV")[0].getvalue(),b"CV raw")
        fake.new_files=[FakeFile("GCD.txt",b"GCD raw")]
        self.assertEqual(load("GCD")[0].name,"GCD.txt")
        fake.new_files=[FakeFile("EIS.txt",b"EIS raw")]
        self.assertEqual(load("EIS")[0].name,"EIS.txt")
        # Simulate upload widget being unloaded when another page is selected.
        fake.new_files=[]
        self.assertEqual(load("CV")[0].getvalue(),b"CV raw")
        self.assertEqual(load("GCD")[0].getvalue(),b"GCD raw")
        self.assertEqual(load("EIS")[0].getvalue(),b"EIS raw")
        self.assertEqual(len(fake.session_state),3)

    def test_scalar_widget_keys_restored_after_other_page(self):
        fake=FakeStreamlit()
        environment["st"]=fake
        store=extract_function("_v16_store_controls")
        restore=extract_function("_v16_restore_controls")
        fake.session_state.update({
            "batch_cv_mass_common":1.23,"batch_gcd_mass_common":2.34,
            "batch_eis_circuit_common":"Rs + CPE",
            "v16_uploaded_files_GCD":"not a widget", # non-upload test key
            "temp_prediction_rate":1.5,
        })
        store()
        for key in list(fake.session_state):
            if key!="v16_control_shadow":
                fake.session_state.pop(key)
        restore()
        self.assertEqual(fake.session_state["batch_cv_mass_common"],1.23)
        self.assertEqual(fake.session_state["batch_gcd_mass_common"],2.34)
        self.assertEqual(fake.session_state["batch_eis_circuit_common"],"Rs + CPE")
        self.assertEqual(fake.session_state["temp_prediction_rate"],1.5)

    def test_eis_subplots_are_full_solid_frames(self):
        path=ROOT/"screens"/"madap_dashboard.py"
        spec=importlib.util.spec_from_file_location("test_v16_dashboard",path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"streamlit":types.ModuleType("streamlit")}):
            spec.loader.exec_module(module)
        f=np.geomspace(1e5,.1,40)
        sample=types.SimpleNamespace(
            filename="EIS.csv",technique="EIS",
            data=pd.DataFrame({"frequency_Hz":f,
                               "z_real_ohm":1+1/(1+f/10),
                               "z_imag_ohm":-1/(1+f/10)}))
        fig=module.diagnostic_figure("EIS",raw=[sample])
        for axis in ["xaxis","xaxis2","xaxis3","xaxis4",
                     "yaxis","yaxis2","yaxis3","yaxis4"]:
            obj=getattr(fig.layout,axis)
            self.assertTrue(obj.showline,axis)
            self.assertTrue(obj.mirror,axis)
            self.assertGreater(obj.linewidth,1.0)
            self.assertNotEqual(obj.linecolor,"rgba(0,0,0,0)")

    def test_navigation_and_removed_copy(self):
        source=(ROOT/"app.py").read_text(encoding="utf-8")
        self.assertIn('st.tabs(',source)
        for k in ['_v16_module("CV")','_v16_module("GCD")','_v16_module("EIS")']:
            self.assertIn(k,source)
        for removed in ("MADAP 风格工作台", "MADAP 界面布局参考",
                        "MADAP 风格的超级电容器 CV / GCD / EIS 数据分析工作台",
                        'PAGE_EIS_TEMP'):
            self.assertNotIn(removed,source)

if __name__=="__main__":
    unittest.main()
