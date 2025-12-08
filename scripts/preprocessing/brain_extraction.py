"""Wrapper module to expose brain extraction for the pipeline."""

from importlib import util
from pathlib import Path

_SCRIPT_NAME = "00_brain_extraction.py"
_MODULE_NAME = "preprocessing.brain_extraction._source"


def _load_class(class_name: str):
    script_path = Path(__file__).with_name(_SCRIPT_NAME)
    spec = util.spec_from_file_location(_MODULE_NAME, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


BrainExtraction = _load_class("BrainExtraction")
__all__ = ["BrainExtraction"]
