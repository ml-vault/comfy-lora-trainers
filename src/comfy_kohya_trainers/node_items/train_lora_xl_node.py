import argparse
from library.train_util import read_config_from_file
from ..train_lora_xl import SdxlNetworkTrainer
from ..const import SAMPLER_CONFIG_TYPE, TRAIN_CONFIG_TYPE, OUTPUT_CONFIG_TYPE, DATASET_LOADER_TYPE, DatasetLoaderDict, TrainConfigDict
from ..args import ClassfiedArgs
from ..args import setup_parser_sdxl
from accelerate import Accelerator

PRECISIONS = ["fp16", "bf16"]

class TrainLoraXlNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset": (DATASET_LOADER_TYPE, {
                    "tooltip": "Dataset"
                }),
                "train_config": (TRAIN_CONFIG_TYPE, {
                    "tooltip": "Train config"
                }),
                "sampler_config": (SAMPLER_CONFIG_TYPE, {
                    "tooltip": "Sampler config"
                }),
                "output_config": (OUTPUT_CONFIG_TYPE, {
                    "tooltip": "Output config"
                }),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("dataset", "sample_image")
    DESCRIPTION = "Train LoRA XL Node"
    FUNCTION = "train_lora_xl"
    OUTPUT_NODE = True

    CATEGORY = "Example"

    def train_lora_xl(self,
                      dataset: DatasetLoaderDict,
                      train_config: TrainConfigDict,
                      sampler_config,
                      output_config,
                     ):
        args = ClassfiedArgs(
            **train_config,
            network_module="networks.lora",
            dataset_config=dataset['dataset_config']
        )
        parser = setup_parser_sdxl()
        args = read_config_from_file(args, parser)
        # Accelerator(mixed_precision=train_config["mixed_precision"], cpu=False)
        print(f"args: {args}")
        # print(f"dataset: {dataset}")
        # print(f"train_config: {train_config}")
        # print(f"sampler_config: {sampler_config}")
        # print(f"output_config: {output_config}")
        trainer = SdxlNetworkTrainer()
        trainer.train(args)
        # set_auth_config(r_token=huggingface_token, w_token=huggingface_token)
        # dset = load_dataset_for_dpack(dataset_repo_id)
        # print(f"loaded dataset: {dataset_repo_id}")
        # return (dset_info, )
        return ("a", )
