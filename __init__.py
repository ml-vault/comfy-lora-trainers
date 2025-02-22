"""Top-level package for my_custom_nodepack."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """test_node_py"""
__email__ = "laptise@live.jp"
__version__ = "0.0.1"

from .src.comfy_kohya_trainers.nodes import NODE_CLASS_MAPPINGS
from .src.comfy_kohya_trainers.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
