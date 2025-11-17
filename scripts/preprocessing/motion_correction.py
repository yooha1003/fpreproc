"""Wrapper module to expose the motion correction class for the pipeline."""

from importlib import util
from pathlib import Path

_SCRIPT_NAME = "01_motion_correction.py"
_MODULE_NAME = "preprocessing.motion_correction._source"

def _load_class(class_name: str):
    script_path = Path(__file__).with_name(_SCRIPT_NAME)
    spec = util.spec_from_file_location(_MODULE_NAME, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


MotionCorrection = _load_class("MotionCorrection")
__all__ = ["MotionCorrection"]
