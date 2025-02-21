from mlvault.util import load_dataset_for_dpack
from mlvault.config import set_auth_config
from ..const import DATAPACK_REPO_INFO_TYPE, DatapackRepoInfo

class DataPackLoaderNode:
    RETURN_TYPES = (DATAPACK_REPO_INFO_TYPE,)
    RETURN_NAMES = ("dataset",)
    DESCRIPTION = "Load a dataset from a huggingface repo"
    FUNCTION = "load_datapack"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Example"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "dataset_repo_id": ("STRING", {
                    "default": "dataset_repo_id",
                    "tooltip": "The repo id of the dataset to use"
                }),
                "huggingface_token": ("STRING", {
                    "default": "huggingface_token",
                    "tooltip": "The token of the huggingface account to use"
                }),
            },
        }


    def load_datapack(self, dataset_repo_id, huggingface_token):
        set_auth_config(r_token=huggingface_token, w_token=huggingface_token)
        dset = load_dataset_for_dpack(dataset_repo_id)
        print(f"loaded dataset: {dataset_repo_id}")
        return (DatapackRepoInfo(
            dataset_repo_id=dataset_repo_id,
            auth_token=huggingface_token,
        ), )
