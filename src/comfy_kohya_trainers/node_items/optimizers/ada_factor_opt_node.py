from ...const import OPTIMIZER_CONFIG_TYPE

class AdaFactorOptimizerNode:
    RETURN_TYPES = (OPTIMIZER_CONFIG_TYPE,)
    RETURN_NAMES = ("optimizer",)
    DESCRIPTION = "AdaFactor optimizer"
    FUNCTION = "get_optimizer"
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lr": ("FLOAT", {
                    "tooltip": "The external learning rate.",
                    "default": 0.0001,
                }),
                "eps": ("FLOAT", {
                    "tooltip": "Regularization constants for square gradient and parameter scale respectively",
                    "default": 1e-30,
                }),
                "clip_threshold": ("FLOAT", {
                    "tooltip": "Threshold of root mean square of final gradient update",
                    "default": 1.0,
                }),
                "decay_rate": ("FLOAT", {
                    "tooltip": "Coefficient used to compute running averages of square",
                    "default": -0.8,
                }),
                "beta1": ("FLOAT", {
                    "tooltip": "Coefficient used for computing running averages of gradient",
                    "default": 0.9,
                }),
                "weight_decay": ("FLOAT", {
                    "tooltip": "Weight decay (L2 penalty)",
                    "default": 0.0,
                }),
                "scale_parameter": ("BOOLEAN", {
                    "tooltip": "If True, learning rate is scaled by root mean square",
                    "default": True,
                }),
                "relative_step": ("BOOLEAN", {
                    "tooltip": "If True, time-dependent learning rate is computed instead of external learning rate",
                    "default": True,
                }),
                "warmup_init": ("BOOLEAN", {
                    "tooltip": "Time-dependent learning rate computation depends on whether warm-up initialization is being used",
                    "default": False,
                }),
            },
        }

    def get_optimizer(self, lr, eps, clip_threshold, decay_rate, beta1, weight_decay, scale_parameter, relative_step, warmup_init):
        return ({
            "optimizer_type": "AdaFactor",
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        },)
