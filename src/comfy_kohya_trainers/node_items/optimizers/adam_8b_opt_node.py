from ...const import OPTIMIZER_CONFIG_TYPE

class AdamW8BitOptimizerNode:
    RETURN_TYPES = (OPTIMIZER_CONFIG_TYPE,)
    RETURN_NAMES = ("optimizer",)
    DESCRIPTION = "AdamW optimizer with 8-bit precision"
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
            "optimizer_type": "AdamW8Bit",
        },)
