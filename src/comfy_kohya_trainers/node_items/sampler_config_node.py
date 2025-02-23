from ..const import SAMPLER_CONFIG_TYPE

SAMPLERS = ['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'heun', 'dpm_2', 'dpm_2_a', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a',]

class SamplerConfigNode:

    RETURN_TYPES = (SAMPLER_CONFIG_TYPE,)
    RETURN_NAMES = ("sampler_config",)
    DESCRIPTION = "Sampler Config Node"
    FUNCTION = "parse"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": (SAMPLERS, {
                    "tooltip": "Sampler"
                }),
                "prompts": ("STRING", {
                    "tooltip": "Prompts",
                    "multiline": True,
                    "label": "Prompts"
                }),
            },
            "optional": {
                "sample_every_n_epochs": ("INT", {
                    "default": 1,
                    "tooltip": "Sample every n epochs"
                }),
                "sample_every_n_steps": ("INT", {
                    "default": 1,
                    "tooltip": "Sample every n steps"
                }),
            }
        }

    def parse(self, sampler: str, prompts: str, sample_every_n_epochs: int, sample_every_n_steps: int):
        return ({"sampler": sampler, "prompts": prompts, "sample_every_n_epochs": sample_every_n_epochs, "sample_every_n_steps": sample_every_n_steps},)
