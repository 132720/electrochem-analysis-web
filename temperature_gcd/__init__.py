"""
Ghosh et al. (2025) style GCD temperature-prediction prototype.
"""

from .io import GCDTemperatureRecord, parse_chi_gcd_upload, infer_temperature_from_filename
from .model import (
    build_feature_table,
    compare_time_period_models,
    tune_decision_tree,
    train_parameter_random_forests,
    predict_paper_gcd,
)

__all__ = [
    "GCDTemperatureRecord",
    "parse_chi_gcd_upload",
    "infer_temperature_from_filename",
    "build_feature_table",
    "compare_time_period_models",
    "tune_decision_tree",
    "train_parameter_random_forests",
    "predict_paper_gcd",
]
