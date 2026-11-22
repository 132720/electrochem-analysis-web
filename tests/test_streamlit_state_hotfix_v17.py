"""Prevent StreamlitValueAssignmentNotAllowedError after switching tabs.

The test reproduces a historical session containing button values in
v16_control_shadow, notably madap_notes_download_CV.
"""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=(ROOT/"app.py").read_text(encoding="utf-8")
TREE=ast.parse(SRC)

def functions():
    names={"_v17_is_nonrestorable_widget_key",
           "_v16_restore_controls","_v16_store_controls"}
    nodes=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/"app.py"),"exec"),ns)
    return ns

class FakeStreamlit:
    def __init__(self):self.session_state={}

class SessionTests(unittest.TestCase):
    def test_legacy_button_shadow_removed_cv_gcd_eis(self):
        ns=functions()
        st=FakeStreamlit()
        st.session_state["v16_control_shadow"]={
            "madap_notes_download_CV":False,
            "madap_notes_download_GCD":True,
            "madap_notes_download_EIS":False,
            "v16_run_CV":False,
            "v16_download_EIS":False,
            "v16_prepare_zip_GCD":False,
            "v16_cv_kinetics_download":False,
            "v17_eis_compare_run":False,
            "batch_cv_mass_common":1.5,
            "batch_gcd_mass_common":2.5,
            "batch_eis_circuit_common":"CPE",
        }
        # Simulate a previously restored button value already in session.
        st.session_state["madap_notes_download_CV"]=False
        st.session_state["batch_cv_mass_common"]=1.5
        ns["st"]=st
        ns["_v16_restore_controls"]()
        for key in [
            "madap_notes_download_CV","madap_notes_download_GCD",
            "madap_notes_download_EIS","v16_run_CV",
            "v16_download_EIS","v16_prepare_zip_GCD",
            "v16_cv_kinetics_download","v17_eis_compare_run",
        ]:
            # Shadow values must be purged, but Streamlit owns live widget
            # values and can already have marked a clicked button True.
            # Deleting a live value before rendering swallows the click.
            self.assertNotIn(key,st.session_state["v16_control_shadow"])
        self.assertIn("madap_notes_download_CV",st.session_state)
        self.assertEqual(st.session_state["batch_cv_mass_common"],1.5)
        self.assertEqual(st.session_state["batch_gcd_mass_common"],2.5)
        self.assertEqual(st.session_state["batch_eis_circuit_common"],"CPE")

    def test_store_excludes_click_widgets_but_keeps_inputs(self):
        ns=functions()
        st=FakeStreamlit()
        st.session_state.update({
            "v16_control_shadow":{"madap_notes_download_CV":False},
            "madap_notes_download_CV":False,
            "v16_run_GCD":True,
            "v17_eis_compare_download":False,
            "v17_peak_manual_window":True,
            "v17_anodic_min":0.21,
            "batch_cv_scan_common":.01,
            "v16_cv_kinetics_selected":["test1","test2"],
            "v16_saved_uploads_CV":[("CV.txt",b"raw")],
        })
        ns["st"]=st
        ns["_v16_store_controls"]()
        shadow=st.session_state["v16_control_shadow"]
        for forbidden in ["madap_notes_download_CV","v16_run_GCD",
                          "v17_eis_compare_download","v16_saved_uploads_CV"]:
            self.assertNotIn(forbidden,shadow)
        self.assertEqual(shadow["v17_anodic_min"],.21)
        self.assertEqual(shadow["v16_cv_kinetics_selected"],["test1","test2"])

    def test_does_not_reset_user_edits_or_export_cache(self):
        ns=functions()
        st=FakeStreamlit()
        st.session_state.update({
            "v16_control_shadow":{},
            "madap_annotations_CV":{"edited_rows":{"0":{"论文图例":"Updated"}}},
            "analysis_batch_export_v11":b"some zip content",
            "v16_saved_uploads_CV":[("CV.txt",b"raw")],
            "v16_result_CV":("digest",{"results":{}}),
        })
        ns["st"]=st
        ns["_v16_restore_controls"]()
        self.assertIn("madap_annotations_CV",st.session_state)
        self.assertIn("analysis_batch_export_v11",st.session_state)
        self.assertIn("v16_saved_uploads_CV",st.session_state)
        self.assertIn("v16_result_CV",st.session_state)

if __name__=="__main__":
    unittest.main()
