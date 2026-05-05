"""
Compatibility shim — re-exports TwoLineResult and TolanskyPipeline from the
hyphenated-filename tolansky-2line.py module so that normal Python imports work.
"""
import importlib.util
import pathlib
import sys

_mod_path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "processing" / "tolansky-2line.py"
)
_spec = importlib.util.spec_from_file_location("tolansky_2line", _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["tolansky_2line"] = _mod   # required for @dataclass resolution
_spec.loader.exec_module(_mod)

TwoLineResult    = _mod.TwoLineResult
TolanskyAnalyser = _mod.TolanskyAnalyser
TolanskyResult   = _mod.TolanskyResult

__all__ = ["TwoLineResult", "TolanskyAnalyser", "TolanskyResult"]
