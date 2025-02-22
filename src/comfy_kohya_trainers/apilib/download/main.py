from  huggingface_hub import hf_hub_download
from ..util.env import HF_USER, R_TOKEN
from huggingface_hub import snapshot_download

sd_lora_dir = "/workspace/sd/stable-diffusion-webui/models/Lora"
sd_model_dir='/workspace/sd/stable-diffusion-webui/models/Stable-diffusion'

def download_from_hf(repo_id:str, filename:str, local_dir:str):
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, force_download=True, local_dir_use_symlinks=False)

def ask(msg:str):
    print(msg)
    return input()

def download_lora_from_hf(hf_repo_name=""):
    repo_name=hf_repo_name or ask("hf lora repo name?")
    repo_id=f"{HF_USER}/{repo_name}"
    download_dir = f"{sd_lora_dir}/{repo_name}"
    snapshot_download(repo_id=repo_id, local_dir=download_dir, token=R_TOKEN, cache_dir="/workspace/hub-cache")
    
def download_ckpt_from_civit_ai(version:str, ckpt_model_id=""):
    model_id= ckpt_model_id or ask("ckpt model id?")
    down_dir=f"{sd_model_dir}/{version}"
    print(f"!wget -P {down_dir} https://civitai.com/api/download/models/{model_id} --content-disposition")

def download_lora_from_civit_ai(lora_model_id=""):
    model_id=lora_model_id or ask("lora model id?")
    print(f"!wget -P {sd_lora_dir} https://civitai.com/api/download/models/{model_id} --content-disposition")
