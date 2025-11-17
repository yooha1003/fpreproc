"""Wrapper module to expose spatial normalization for the pipeline."""

from importlib import util
from pathlib import Path

_SCRIPT_NAME = "04_normalization.py"
_MODULE_NAME = "preprocessing.normalization._source"

def _load_class(class_name: str):
    script_path = Path(__file__).with_name(_SCRIPT_NAME)
    spec = util.spec_from_file_location(_MODULE_NAME, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


SpatialNormalization = _load_class("SpatialNormalization")
__all__ = ["SpatialNormalization"]
