"""Verify that the Python environment running Streamlit has pyimpspec."""
from __future__ import annotations
import sys
import importlib.metadata

def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    try:
        import pyimpspec as library
        import numpy as np
        import pandas as pd
        print("pyimpspec:", importlib.metadata.version("pyimpspec"))
        print("numpy:",np.__version__)
        print("pandas:",pd.__version__)
        assert callable(getattr(library, "perform_zhit", None))
        assert callable(getattr(library, "calculate_drt", None))
        assert callable(getattr(library, "fit_circuit", None))
        assert callable(getattr(library, "parse_cdc", None))
        print("PASS: Z-HIT, DRT and equivalent circuit APIs are importable.")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print("Install: python -m pip install -r requirements_eis_professional_optional.txt")
        return 1

if __name__=="__main__":
    raise SystemExit(main())
