"""
Kohya's SD-Scripts config fields
https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-ja.md
"""

from ..const import DATASET_CONFIG_TYPE, DreamBoothDatasetConfigDict, DREAMBOOTH_SUBSET_CONFIG_TYPE

class DreamBoothDatasetConfigNode:
    RETURN_TYPES = (DATASET_CONFIG_TYPE,)
    RETURN_NAMES = ("dataset",)
    DESCRIPTION = "Load dataset with image and caption file"
    FUNCTION = "load_subsets"

    CATEGORY = "Example"


    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dream_booth_subset_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                }),
                "width": ("INT", {
                    "default": 512,
                }),
                "height": ("INT", {
                    "default": 512,
                }),
                "dream-booth-subset-config-1": (DREAMBOOTH_SUBSET_CONFIG_TYPE, {"default": {}}),
            },
        }

    def load_subsets(self, *_, **kwargs):
        to_exclude_keys = ["width", "height"]
        keys_without_count = [key for key in kwargs.keys() if "count" not in key and key not in to_exclude_keys]
        values_without_count = [kwargs[key] for key in keys_without_count]
        width = kwargs["width"]
        height = kwargs["height"]
        return (DreamBoothDatasetConfigDict(
            subsets=values_without_count,
            resolution=[width, height],
        ),)

