import toml
import os
from datetime import datetime

from ..const import DATASET_LOADER_TYPE, DATASET_CONFIG_TYPE, DatasetLoaderDict


class ContainsAnyDict(dict):
    def __contains__(self, key):
        print(f"checking key: {key}")
        print(self)
        return True

    def __setitem__(self, key, value):
        print(f"setting key: {key} to value: {value}")
        super().__setitem__(key, value)

class DatasetLoaderNode:
    RETURN_TYPES = (DATASET_LOADER_TYPE,)
    RETURN_NAMES = ("dataset",)
    DESCRIPTION = "Load a dataset from a huggingface repo"
    FUNCTION = "load_datasets"
    CATEGORY = "Example"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "toml_config_output_dir": ("STRING", {"default": "output"}),
                "dataset_count": ("INT", {"default": 1, "min": 1, "max": 100 }),
                "dataset-config-1": (DATASET_CONFIG_TYPE, {"default": {}}),
            },
        }

    def load_datasets(self, *_, **kwargs):
        keys_without_count = [key for key in kwargs.keys() if "-" in key]
        values_without_count = [kwargs[key] for key in keys_without_count]
        print(values_without_count)
        toml_str = toml.dumps({"datasets": values_without_count})
        toml_config_output_dir = kwargs["toml_config_output_dir"]
        os.makedirs(toml_config_output_dir, exist_ok=True)
        toml_file_name = f"{toml_config_output_dir}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.toml"
        with open(toml_file_name, "w") as f:
            f.write(toml_str)
        return (DatasetLoaderDict(
            datasets=values_without_count,
            dataset_config=toml_file_name,
        ),)
