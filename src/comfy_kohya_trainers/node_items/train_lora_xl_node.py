import argparse
import os
import pathlib
import subprocess
from venv import logger
from library.train_util import read_config_from_file
import toml
from ..train_lora_xl import SdxlNetworkTrainer
from ..const import SAMPLER_CONFIG_TYPE, TRAIN_CONFIG_TYPE, OUTPUT_CONFIG_TYPE, DATASET_LOADER_TYPE, DatasetLoaderDict, TrainConfigDict, SamplerConfigDict
from ..args import ClassfiedArgs
from ..args import setup_parser_sdxl
from accelerate import Accelerator
import os

PRECISIONS = ["fp16", "bf16"]
file_dir = os.path.dirname(os.path.abspath(__file__))
node_root = file_dir.rsplit("/", 3)[0]
kohya_repo_dir = os.path.join(node_root, "oss", "kohya-main")
compfy_root = node_root.rsplit("/", 2)[0]

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
        bufsize=1,
        env={
            **os.environ,
            "NCCL_DEBUG": "INFO",
            # "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_BLOCKING_WAIT": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
        }
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
                "model_name": ("STRING", {
                    "tooltip": "Model name"
                }),
                "output_dir": ("STRING", {
                    "tooltip": "Output directory"
                }),
                "multi_gpu": ("BOOLEAN", {
                    "tooltip": "Multi GPU",
                    "default": False
                }),
                "num_processes": ("INT", {
                    "min": 0,
                    "tooltip": "Number of processes, if 0, follows config"
                }),
                "persistent_data_loader_workers": ("BOOLEAN", {
                    "tooltip": "Persistent data loader workers",
                    "default": False
                }),
                "cache_latents_to_disk": ("BOOLEAN", {
                    "tooltip": "Cache latents to disk",
                    "default": False
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
                      model_name: str,
                      output_dir: str,
                      dataset: DatasetLoaderDict,
                      train_config: TrainConfigDict,
                      sampler_config: SamplerConfigDict,
                      output_config,
                      multi_gpu: bool,
                      num_processes: int,
                      persistent_data_loader_workers: bool,
                      cache_latents_to_disk: bool,
                     ):

        # Write sample prompts to file

        output_dir = os.path.join(output_dir, model_name) if output_dir.startswith("/") else os.path.join(compfy_root, output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_name = model_name

        # Sample config
        sample_prompts = sampler_config["sample_prompts"]
        print(f"sample_prompts: {sample_prompts}")
        sample_configs = {key: value for key, value in sampler_config.items() if key not in ["sample_prompts"]}
        sample_prompts_path = os.path.join(output_dir, "sample_prompts.txt")
        with open(sample_prompts_path, "w", encoding="utf-8") as f:
            f.write(sample_prompts)

        args = ClassfiedArgs(
            **train_config,
            **sample_configs,
            sample_prompts=sample_prompts_path,
            **output_config,
            huggingface_path_in_repo=model_name,
            output_dir=output_dir,
            output_name=output_name,
            network_module="networks.lora",
            dataset_config=dataset['dataset_config'],
            persistent_data_loader_workers=persistent_data_loader_workers,
            cache_latents_to_disk=cache_latents_to_disk,
        )

        config_path = f"{output_dir}/configs.toml"

        # check if kohya-main exists
        if not os.path.exists(kohya_repo_dir):
            #clone kohya-main
            print(f"kohya-main not found at {kohya_repo_dir}, cloning...")
            run_cli(f"git clone https://github.com/kohya-ss/sd-scripts.git {kohya_repo_dir}")
        parser = setup_parser_sdxl()
        args = read_config_from_file(args, parser)


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
        multi_gpu_flag = "--multi_gpu" if multi_gpu else ""
        num_processes_flag = f"--num_processes {num_processes}" if num_processes > 0 else ""
        cmd = f"accelerate launch --mixed_precision {train_config['mixed_precision']} {multi_gpu_flag} {num_processes_flag} {kohya_repo_dir}/sdxl_train_network.py --config_file {config_path} "
        run_cli(cmd)

        return ("a", )
