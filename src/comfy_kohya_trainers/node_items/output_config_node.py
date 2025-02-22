from ..const import OUTPUT_CONFIG_TYPE, OutputConfigDict

class OutputConfigNode:
    RETURN_TYPES = (OUTPUT_CONFIG_TYPE,)
    RETURN_NAMES = ("output_config",)
    DESCRIPTION = "Output Config Node"
    FUNCTION = "parse"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_model_as": ("STRING", {"default": "safetensors"}),
                "output_dir": ("STRING", {"default": "output/train"}),
                "output_name": ("STRING", {"default": "output/train"}),
            },
            "optional": {
                "save_every_n_epochs": ("INT", {"default": 1, "step": 1}),
                "save_every_n_steps": ("INT", {"default": 0, "step": 1}),
                "save_last_n_epochs": ("INT", {"default": 0, "step": 1}),
                "save_last_n_steps": ("INT", {"default": 0, "step": 1}),
                "huggingface_repo_id": ("STRING", {"default": None}),
                "huggingface_repo_type": ("STRING", {"default": None}),
                "huggingface_path_in_repo": ("STRING", {"default": None}),
                "huggingface_token": ("STRING", {"default": None}),
                "huggingface_repo_visibility": (["public", "private"], {"default": "private"}),
                "save_state_to_huggingface": ("BOOLEAN", {"default": False}),
                "resume_from_huggingface": ("BOOLEAN", {"default": False}),
                "async_upload": ("BOOLEAN", {"default": False}),
            },
        }

    def parse(self, *_, **kwargs):
        parsed = OutputConfigDict( **kwargs,
                                  save_every_n_epochs=kwargs["save_every_n_epochs"] or None,
                                  save_every_n_steps=kwargs["save_every_n_steps"] or None,
                                  save_last_n_epochs=kwargs["save_last_n_epochs"] or None,
                                  save_last_n_steps=kwargs["save_last_n_steps"] or None,
                                  huggingface_repo_id=kwargs["huggingface_repo_id"] or None,
                                  huggingface_repo_type=kwargs["huggingface_repo_type"] or None,
                                  huggingface_path_in_repo=kwargs["huggingface_path_in_repo"] or None,
                                  huggingface_token=kwargs["huggingface_token"] or None,
                                  huggingface_repo_visibility=kwargs["huggingface_repo_visibility"] or None,
                                  save_state_to_huggingface=kwargs["save_state_to_huggingface"] or None,
                                  resume_from_huggingface=kwargs["resume_from_huggingface"] or None,
                                  async_upload=kwargs["async_upload"] or None,
                                  )
        return (parsed,)
