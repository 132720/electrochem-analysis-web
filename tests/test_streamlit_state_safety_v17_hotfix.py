"""Regression for StreamlitValueAssignmentNotAllowedError (button keys)."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENV={}


def load(name):
    src=(ROOT/"app.py").read_text(encoding="utf-8")
    parsed=ast.parse(src)
    assignment_names={"_PERSISTENCE_DISALLOWED_PARTS",
                      "_PERSISTENCE_DISALLOWED_PREFIXES"}
    declarations=[
        n for n in parsed.body if (
            isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id in assignment_names
                for t in n.targets))
    ]
    target=next(n for n in parsed.body if isinstance(n,ast.FunctionDef)
                and n.name==name)
    exec(compile(ast.Module(body=declarations+[target],type_ignores=[]),
                 "<app.py state safety>","exec"),ENV)
    return ENV[name]


class FakeStreamlit:
    def __init__(self):
        self.session_state={}


class StateSafetyTests(unittest.TestCase):
    def test_exclude_all_known_action_widget_keys(self):
        unsafe=load("_v17_is_nonrestorable_widget_key")
        action_keys=[
            "madap_notes_download_CV","madap_notes_download_GCD",
            "madap_notes_download_EIS","v16_cv_kinetics_download",
            "v16_gcd_rate_download","v17_eis_compare_download",
            "v16_run_CV","v16_prepare_zip_EIS",
            "v16_clear_GCD","v16_upload_CV_0",
            "gcd_ghosh_parameterization_run",
            "gcd_ghosh_parameterization_summary_download",
            "six_mode_prediction_csv",
            "six_mode_train_button","six_mode_validation_csv",
            "gcd_complete_temperature_dataset_zip",
            "analysis_batch_download_v11",
        ]
        for key in action_keys:
            with self.subTest(key=key):
                self.assertTrue(unsafe(key))
        for key in [
            "batch_cv_mass_common","batch_gcd_mass_common",
            "batch_eis_circuit_common","temp_prediction_rate",
            "v17_peak_manual_window","v16_cv_kinetics_selected",
            "v14_kind_testid","gcd_web_e_temperature",
        ]:
            with self.subTest(key=key):
                self.assertFalse(unsafe(key))

    def test_corrupt_old_shadow_cleaned_before_restore(self):
        unsafe=load("_v17_is_nonrestorable_widget_key")
        ENV["_v17_is_nonrestorable_widget_key"]=unsafe
        st=FakeStreamlit()
        st.session_state["v16_control_shadow"]={
            "madap_notes_download_CV":False,
            "v16_run_CV":False,
            "v16_upload_CV_0":None,
            "batch_cv_mass_common":1.5,
            "batch_gcd_mass_common":2.5,
            "temp_prediction_rate":4.0,
        }
        ENV["st"]=st
        restore=load("_v16_restore_controls")
        restore()
        self.assertNotIn("madap_notes_download_CV",st.session_state)
        self.assertNotIn("v16_run_CV",st.session_state)
        self.assertNotIn("v16_upload_CV_0",st.session_state)
        self.assertEqual(st.session_state["batch_cv_mass_common"],1.5)
        self.assertEqual(st.session_state["temp_prediction_rate"],4.0)
        self.assertNotIn("madap_notes_download_CV",
                         st.session_state["v16_control_shadow"])

    def test_store_does_not_capture_command_states(self):
        unsafe=load("_v17_is_nonrestorable_widget_key")
        ENV["_v17_is_nonrestorable_widget_key"]=unsafe
        st=FakeStreamlit()
        st.session_state.update({
            "v16_control_shadow":{"madap_notes_download_CV":False},
            "madap_notes_download_CV":False,
            "v17_eis_compare_run":False,
            "batch_cv_mass_common":1.5,
            "v17_peak_manual_window":True,
            "v16_cv_kinetics_selected":["a","b","c"],
        })
        ENV["st"]=st
        store=load("_v16_store_controls")
        store()
        shadow=st.session_state["v16_control_shadow"]
        self.assertNotIn("madap_notes_download_CV",shadow)
        self.assertNotIn("v17_eis_compare_run",shadow)
        self.assertEqual(shadow["batch_cv_mass_common"],1.5)
        self.assertTrue(shadow["v17_peak_manual_window"])
        self.assertEqual(shadow["v16_cv_kinetics_selected"],["a","b","c"])


if __name__=="__main__":
    unittest.main()
