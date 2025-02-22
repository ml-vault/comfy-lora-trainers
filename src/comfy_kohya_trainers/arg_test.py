import sys
import os
import argparse
from .train_network import setup_parser as train_network_setup_parser
from library import train_util
from .train_lora_xl import SdxlNetworkTrainer
from library import sdxl_train_util, train_util

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network_setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser


if __name__ == "__main__":
    print("arg_test.py")
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    print(args)
