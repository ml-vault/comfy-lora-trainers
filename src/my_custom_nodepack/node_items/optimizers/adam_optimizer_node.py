from ...const import OPTIMIZER_CONFIG_TYPE


class AdamOptimizerNode:
    RETURN_TYPES = (OPTIMIZER_CONFIG_TYPE,)
    RETURN_NAMES = ("optimizer",)
    DESCRIPTION = "Adam optimizer"
    FUNCTION = "get_optimizer"
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    def get_optimizer(self):
        return ({
            "optimizer_type": "AdamW",
        },)
