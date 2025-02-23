import argparse
import os
import pathlib
import subprocess
from venv import logger
from library.train_util import read_config_from_file
import toml
from ..train_lora_xl import SdxlNetworkTrainer
from ..const import SAMPLER_CONFIG_TYPE, TRAIN_CONFIG_TYPE, OUTPUT_CONFIG_TYPE, DATASET_LOADER_TYPE, DatasetLoaderDict, TrainConfigDict
from ..args import ClassfiedArgs
from ..args import setup_parser_sdxl
from accelerate import Accelerator
import os

PRECISIONS = ["fp16", "bf16"]

def run_cli(command: str):
    import subprocess
    import sys

    print(f"run cli: {command}")

    # bufsize=1, text=True（universal_newlines=True）を指定して行単位で処理
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 標準エラーも標準出力にマージ
        text=True,
        bufsize=1
    )

    # 標準出力をリアルタイムで読み取る
    for line in process.stdout:
        print(line, end="")

    process.stdout.close()
    return_code = process.wait()

    return return_code

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
            **sampler_config,
            **output_config,
            network_module="networks.lora",
            dataset_config=dataset['dataset_config']
        )

        file_dir = os.path.dirname(os.path.abspath(__file__))
        node_root = file_dir.rsplit("/", 3)[0]
        kohya_repo_dir = os.path.join(node_root, "oss", "kohya-main")
        # check if kohya-main exists
        if not os.path.exists(kohya_repo_dir):
            #clone kohya-main
            print(f"kohya-main not found at {kohya_repo_dir}, cloning...")
            run_cli(f"git clone https://github.com/kohya-ss/sd-scripts.git {kohya_repo_dir}")
        parser = setup_parser_sdxl()
        args = read_config_from_file(args, parser)

        config_path = f"{os.getcwd()}/argstest.toml"

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info(f"Saved config file / 設定ファイルを保存しました: {config_path}")

        # Accelerator(mixed_precision=train_config["mixed_precision"], cpu=False)
        print(f"args: {args}")
        cmd = f"accelerate launch --mixed_precision {train_config['mixed_precision']} {kohya_repo_dir}/sdxl_train_network.py --config_file {config_path} "
        run_cli(cmd)

        return ("a", )
