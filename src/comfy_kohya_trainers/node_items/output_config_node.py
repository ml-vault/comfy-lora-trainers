from ..const import OUTPUT_CONFIG_TYPE

class OutputConfigNode:
    RETURN_TYPES = (OUTPUT_CONFIG_TYPE,)
    RETURN_NAMES = ("output_config",)
    DESCRIPTION = "Output Config Node"
    FUNCTION = "parse"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_every_n_epochs": ("INT", {"default": 1, "min": 1, "max": 1000000, "step": 1, "display": "slider"}),
                "save_every_n_steps": ("INT", {"default": 1, "min": 1, "max": 1000000, "step": 1, "display": "slider"}),
                "save_model_as": ("STRING", {"default": "safetensors"}),
                "output_dir": ("STRING", {"default": "output"}),
                "output_name": ("STRING", {"default": "output"}),
            }
        }


    def parse(self, save_every_n_epochs: int, save_every_n_steps: int, save_model_as: str, output_dir: str, output_name: str):
        return ({"save_every_n_epochs": save_every_n_epochs, "save_every_n_steps": save_every_n_steps, "save_model_as": save_model_as, "output_dir": output_dir, "output_name": output_name},)
