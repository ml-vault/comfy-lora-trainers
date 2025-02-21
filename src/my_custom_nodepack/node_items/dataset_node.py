from ..const import DATASET_TYPE, DatasetDict
import os

class ContainsAnyDict(dict):
    def __contains__(self, key):
        return True


class DatasetNode:
    RETURN_TYPES = (DATASET_TYPE,)
    RETURN_NAMES = ("dataset",)
    DESCRIPTION = "Load dataset with image and caption file"
    FUNCTION = "load_dataset"

    CATEGORY = "Example"


    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir": ("STRING", {
                    "default": os.getcwd(),
                    "tooltip": "The directory of the datasets"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "The width of the images"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "The height of the images"
                }),
                "caption_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "The extension of the caption file"
                }),
                "repeat_times": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "The number of times to repeat the dataset"
                }),
                "shuffle_captions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Shuffle the captions"
                }),
            },
        }

    def load_dataset(self,
                     dir: str,
                     width: int,
                     height: int,
                     caption_extension: str,
                     repeat_times: int,
                     shuffle_captions: bool
                     ):
        return (DatasetDict(
            dir=dir,
            width=width,
            height=height,
            caption_extension=caption_extension,
            repeat_times=repeat_times,
            shuffle_captions=shuffle_captions,
        ),)
