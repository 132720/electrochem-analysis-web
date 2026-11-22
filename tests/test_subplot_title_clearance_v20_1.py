"""All CV/GCD/EIS four-panel titles clear plot top frames."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.util

ROOT=Path(__file__).resolve().parents[1]


class PlotTitleClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name="madap_subplot_spacing_test"
        spec=importlib.util.spec_from_file_location(
            name,ROOT/"screens"/"madap_dashboard.py")
        mod=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"streamlit":types.ModuleType("streamlit")}):
            spec.loader.exec_module(mod)
        cls.dashboard=mod

    def test_all_four_titles_do_not_overlap_axes(self):
        for technique in ("CV","GCD","EIS"):
            with self.subTest(technique=technique):
                figure=self.dashboard.diagnostic_figure(technique)
                subtitles=list(figure.layout.annotations[:4])
                self.assertEqual(len(subtitles),4)
                self.assertGreaterEqual(figure.layout.margin.t,110)
                for note in subtitles:
                    self.assertGreaterEqual(note.yshift,14)
                    self.assertEqual(note.yanchor,"bottom")
                    self.assertEqual(note.yref,"paper")
                self.assertEqual(
                    [round(n.y,3) for n in subtitles],
                    [1.0,1.0,.385,.385])

    def test_spacing_is_adjustable_without_changing_axes(self):
        figure=self.dashboard.diagnostic_figure("CV")
        coordinates=[a.y for a in figure.layout.annotations]
        self.dashboard._space_subplot_titles(figure,gap_px=24)
        self.assertTrue(all(a.yshift==24 for a in figure.layout.annotations[:4]))
        self.assertEqual(coordinates,[a.y for a in figure.layout.annotations])
        with self.assertRaises(ValueError):
            self.dashboard._space_subplot_titles(figure,gap_px=-1)

if __name__=="__main__":
    unittest.main()
