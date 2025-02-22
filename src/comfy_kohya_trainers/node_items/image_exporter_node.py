from mlvault.util import load_dataset_for_dpack
from mlvault.config import set_auth_config
from mlvault.datapack.main import export_datataset_by_filters
import os
from ..const import DATAPACK_REPO_INFO_TYPE, DatapackRepoInfo

class ImageExporterNode:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("exported_path",)
    DESCRIPTION = "Export an image to a huggingface repo"
    FUNCTION = "export_image"

    OUTPUT_NODE = True
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset": (DATAPACK_REPO_INFO_TYPE, {
                    "tooltip": "The repo id of the dataset to use"
                }),
                "export_path": ("STRING", {
                    "default": os.getcwd(),
                    "tooltip": "The path to export the image to"
                }),
            },
            "optional": {
                "filters": ("STRING", {
                    "default": "filter",
                    "tooltip": "Filters to tag with split by comma"
                }),
                "exclude_filters": ("STRING", {
                    "default": "exclude_filter",
                    "tooltip": "Filters to exclude split by comma"
                }),
            }
        }

    def export_image(self, dataset: DatapackRepoInfo, export_path: str, filters: str | None, exclude_filters: str | None):
        set_auth_config(r_token=dataset["auth_token"], w_token=dataset["auth_token"])
        dataset = load_dataset_for_dpack(dataset["dataset_repo_id"])
        filters = [f.strip() for f in filters.split(",")] if filters else None
        exclude_filters = [f.strip() for f in exclude_filters.split(",")] if exclude_filters else None
        export_datataset_by_filters(dataset, export_path, filters, exclude_filters)
        return (export_path, )

