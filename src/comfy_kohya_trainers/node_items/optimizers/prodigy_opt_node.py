from ...const import OPTIMIZER_CONFIG_TYPE
from .opt_args import get_optimizer_args


class ProdigyOptimizerNode:
    RETURN_TYPES = (OPTIMIZER_CONFIG_TYPE,)
    RETURN_NAMES = ("optimizer",)
    DESCRIPTION = "Prodigy optimizer"
    FUNCTION = "get_optimizer"
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eps": ("FLOAT", {
                    "tooltip": "Term added to the denominator outside of the root operation to improve numerical stability.",
                    "default": 1e-8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.0000001,
                    "precision": 8,
                }),
                "weight_decay": ("FLOAT", {
                    "tooltip": "Weight decay, i.e. a L2 penalty",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "precision": 9,
                }),
                "decouple": ("BOOLEAN", {
                    "tooltip": "Use AdamW style decoupled weight decay",
                    "default": True,
                }),
                "use_bias_correction": ("BOOLEAN", {
                    "tooltip": "Turn on Adam's bias correction. Off by default.",
                    "default": False,
                }),
                "safeguard_warmup": ("BOOLEAN", {
                    "tooltip": "Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.",
                    "default": False,
                }),
            },
        }

    def get_optimizer(self, eps, weight_decay, decouple, use_bias_correction, safeguard_warmup):
        args = get_optimizer_args({
            "eps": eps,
            "weight_decay": weight_decay,
            "decouple": decouple,
            "use_bias_correction": use_bias_correction,
            "safeguard_warmup": safeguard_warmup,
        })

        return ({
            "optimizer_type": "Prodigy",
            **args,
        },)
