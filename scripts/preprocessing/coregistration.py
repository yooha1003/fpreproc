"""Wrapper module to expose the coregistration class for the pipeline."""

from importlib import util
from pathlib import Path

_SCRIPT_NAME = "03_coregistration.py"
_MODULE_NAME = "preprocessing.coregistration._source"

def _load_class(class_name: str):
    script_path = Path(__file__).with_name(_SCRIPT_NAME)
    spec = util.spec_from_file_location(_MODULE_NAME, script_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


Coregistration = _load_class("Coregistration")
__all__ = ["Coregistration"]
