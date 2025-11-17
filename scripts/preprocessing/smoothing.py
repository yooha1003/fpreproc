"""Wrapper module to expose spatial smoothing for the pipeline."""

from importlib import util
from pathlib import Path

_SCRIPT_NAME = "05_smoothing.py"
_MODULE_NAME = "preprocessing.smoothing._source"

def _load_class(class_name: str):
    script_path = Path(__file__).with_name(_SCRIPT_NAME)
    spec = util.spec_from_file_location(_MODULE_NAME, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


SpatialSmoothing = _load_class("SpatialSmoothing")
__all__ = ["SpatialSmoothing"]
