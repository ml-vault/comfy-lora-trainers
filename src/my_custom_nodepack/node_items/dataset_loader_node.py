
from ..const import DATASET_LOADER_TYPE, DatasetLoaderDict

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
    FUNCTION = "load_dataset"

    CATEGORY = "Example"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {}

    def load_dataset(self, *_, **kwargs):
        return (DatasetLoaderDict(datasets=list(kwargs.values())),)
