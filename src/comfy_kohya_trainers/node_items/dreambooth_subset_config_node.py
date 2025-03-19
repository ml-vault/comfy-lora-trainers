"""
Kohya's SD-Scripts config fields
https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-ja.md
"""
from ..const import DREAMBOOTH_SUBSET_CONFIG_TYPE, DreamBoothSubsetDict

class DreamBoothSubsetConfigNode:
    RETURN_TYPES = (DREAMBOOTH_SUBSET_CONFIG_TYPE,)
    RETURN_NAMES = ("dream_booth_subset_config",)
    DESCRIPTION = "Load dataset with image and caption file"
    FUNCTION = "config_subset"

    CATEGORY = "Example"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "multiline": False,
                }),
                "class_tokens": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "is_reg": ("BOOLEAN", {
                    "default": False,
                }),
                "cache_info": ("BOOLEAN", {
                    "default": False,
                }),
                "keep_tokens": ("INT", {
                    "default": 0,
                }),
                "shuffle_caption": ("BOOLEAN", {
                    "default": False,
                }),
                "color_aug": ("BOOLEAN", {
                    "default": False,
                }),
                "flip_aug": ("BOOLEAN", {
                    "default": False,
                }),
                "num_repeats": ("INT", {
                    "default": 1,
                }),
                "random_crop": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }

    def config_subset(
        self,
        image_dir,
        caption_extension=".txt",
        class_tokens=None,
        is_reg=False,
        cache_info=False,
        keep_tokens=None,
        color_aug=False,
        flip_aug=False,
        num_repeats=1,
        random_crop=False,
        shuffle_caption=False,
        caption_prefix="",
        caption_suffix="",
        caption_separator=",",
    ):
        return (
            DreamBoothSubsetDict(
                image_dir=image_dir,
                caption_extension=caption_extension,
                class_tokens=class_tokens,
                is_reg=is_reg,
                cache_info=cache_info,
                keep_tokens=keep_tokens,
                color_aug=color_aug,
                flip_aug=flip_aug,
                num_repeats=num_repeats,
                random_crop=random_crop,
                shuffle_caption=shuffle_caption,
                caption_prefix=caption_prefix,
                caption_suffix=caption_suffix,
                caption_separator=caption_separator,
            ),
        )
