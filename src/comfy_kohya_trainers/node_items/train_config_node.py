import os
from ..const import TRAIN_CONFIG_TYPE, TrainConfigDict, OPTIMIZER_CONFIG_TYPE

PRECISIONS = ["fp16", "bf16", "no"]

class TrainConfigNode:

    RETURN_TYPES = (TRAIN_CONFIG_TYPE,)
    RETURN_NAMES = ("train_config",)
    DESCRIPTION = "Train Config Node"
    FUNCTION = "parse"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "checkpoint": ("STRING", {
                    "default": os.path.join(os.getcwd(), "models"),
                    "tooltip": "Checkpoint"
                }),
                "lora": ("STRING", {
                    "default": os.path.join(os.getcwd(), "models", "loras"),
                    "tooltip": "Lora"
                }),
                "xformers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use xformers"
                }),
                "mixed_precision": (PRECISIONS, {
                    "default": "fp16",
                    "tooltip": "Mixed precision"
                }),
                "save_precision": (PRECISIONS, {
                    "default": "fp16",
                    "tooltip": "Save precision"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0005,
                    "step": 1e-7,
                    "tooltip": "Learning rate"
                }),
                "train_batch_size": ("INT", {
                    "default": 1,
                    "tooltip": "Train batch size"
                }),
                "prior_loss_weight": ("FLOAT", {
                    "default": 1.0,
                    "tooltip": "Prior loss weight"
                }),
                "cache_latents": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Cache latents"
                }),
            },
            "optional": {
                "optimizer": (OPTIMIZER_CONFIG_TYPE, {
                    "default": {"optimizer_type": "AdamW"},
                    "tooltip": "Optimizer"
                }),
                "max_train_epochs": ("INT", {
                    "default": 1,
                    "tooltip": "Max train epochs"
                }),
                "max_train_steps": ("INT", {
                    "default": 1,
                    "tooltip": "Max train steps"
                }),
            }
        }

    def parse(self,
              checkpoint: str,
              lora: str,
              xformers: bool,
              mixed_precision: str,
              save_precision: str,
              learning_rate: float,
              train_batch_size: int,
              prior_loss_weight: float,
              max_train_epochs: int,
              max_train_steps: int,
              cache_latents: bool,
              optimizer,
            ):
        return (TrainConfigDict(
            pretrained_model_name_or_path=checkpoint,
            network_weights=lora or None,
            xformers=xformers if xformers else None,
            mixed_precision=mixed_precision,
            save_precision=save_precision,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            prior_loss_weight=prior_loss_weight,
            max_train_epochs=max_train_epochs,
            max_train_steps=max_train_steps,
            cache_latents=cache_latents,
            **optimizer,
        ),)





