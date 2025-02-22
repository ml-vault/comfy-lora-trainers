sd_scripts_path = "/workspace/difflex"
import os
from huggingface_hub import hf_hub_download
from typing import Literal, Optional
from abc import ABCMeta, abstractmethod
from apilib.util.env.main import DOWNLOAD_DIR
from mlvault.datapack import DataPack
from mlvault.config import get_r_token
from ..util import run_cli
from ..util.env import SDXL


alpha = "abcdefghijklmnopqrstuvwxyz"

def done():
    print("done!")

class OutputConfig:
    base_path:str
    model_name:str
    save_model_as:str

    def __init__(self, base_path:str, model_name:str, save_every_n_epochs:int|None = None, save_every_n_steps:int|None = None, save_model_as: Literal["safetensors", "ckpt"] = "safetensors"):
        self.base_path = base_path
        self.model_name = model_name
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.save_model_as = save_model_as
        print("output config done!")
        pass

    @property
    def out_dir(self):
        return f"{self.base_path}/output"

    def getArgs(self):
        dynamic = ""
        if self.save_every_n_epochs:
            dynamic = f"--save_every_n_epochs {self.save_every_n_epochs} "
        if self.save_every_n_steps:
            dynamic = f"--save_every_n_steps {self.save_every_n_steps} "
        os.makedirs(self.out_dir, exist_ok=True)
        return f"--output_dir {self.out_dir} \
        --output_name {self.model_name} \
        --save_model_as {self.save_model_as} {dynamic}"

class OptimizerConfig(metaclass=ABCMeta):
    optimizer_type:str
    def __init__(self, optimizer_type:Literal["AdaFactor","AdamW8bit" ]) -> None:
        self.optimizer_type = optimizer_type

    @abstractmethod
    def getArgs(self) -> str:
        pass

class AdamW8bitConfig(OptimizerConfig):
    def __init__(self):
        super().__init__("AdamW8bit")

    def getArgs(self) -> str:
        return f"--optimizer_type {self.optimizer_type}"


class AdaFactorConfig(OptimizerConfig):
    def __init__(self):
        super().__init__("AdaFactor")

    def getArgs(self) -> str:
        return f'--optimizer_type {self.optimizer_type} \
        --optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=False'

class TrainConfig:
    pretrained_model_name_or_path:str
    max_train_epochs:int |None
    max_train_steps:int| None
    train_batch_size:int
    prior_loss_weight:float
    learning_rate:float
    mixed_precision:str
    max_data_loader_n_workers:int
    config_file_path:str
    def __init__(self, config_file_path:str, pretrained_model_name_or_path:str,  train_batch_size:int, learning_rate:float, 
                   mixed_precision: Literal["no", "fp16", "bf16"] = "bf16",
                   max_data_loader_n_workers:int =3000,
                   max_train_epochs:int|None = None,
                   max_train_steps:int|None = None
                   ) -> None:
        self.config_file_path = config_file_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path 
        self.max_train_epochs = max_train_epochs
        self.max_train_steps = max_train_steps
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.max_data_loader_n_workers = max_data_loader_n_workers
        print("train config done!")
        pass

    def getArgs(self) -> str:
        return f"--dataset_config {self.config_file_path} \
        --pretrained_model_name_or_path {self.pretrained_model_name_or_path} \
        --max_train_epochs {self.max_train_epochs} \
        --train_batch_size {self.train_batch_size} \
        --learning_rate {self.learning_rate} \
        --mixed_precision {self.mixed_precision} \
        --max_data_loader_n_workers {self.max_data_loader_n_workers} \
        --max_train_epochs {self.max_train_epochs}"

class TrainNetworkConfig(TrainConfig):
    def __init__(self, config_file_path: str, pretrained_model_name_or_path: str, max_train_epochs: int|None, max_train_steps:int|None, train_batch_size: int, learning_rate: float, network_dim: int, network_alpha: int, mixed_precision: Literal['no', 'fp16', 'bf16'] = "bf16", network_module: Literal['networks.lora', 'lycoris.kohya'] = "networks.lora", continue_from: str | None = None, max_data_loader_n_workers: int = 3000, prior_loss_weight=1) -> None:
        super().__init__(config_file_path, pretrained_model_name_or_path, train_batch_size, learning_rate, mixed_precision, max_data_loader_n_workers, max_train_epochs=max_train_epochs, max_train_steps=max_train_steps)
        self.network_alpha = network_alpha
        self.network_dim = network_dim
        self.network_module = network_module
        self.prior_loss_weight = prior_loss_weight
        self.continue_from = continue_from
    def getArgs(self) -> str:
        dynamic = ""
        continue_from = ""
        if self.continue_from:
            continue_from = f"--network_weights {resolve_model_name(self.continue_from)}"
        return f"{continue_from} --dataset_config {self.config_file_path} \
        --pretrained_model_name_or_path {self.pretrained_model_name_or_path} \
        --train_batch_size {self.train_batch_size} \
        --network_dim {self.network_dim} --network_alpha {self.network_alpha} \
        --prior_loss_weight {self.prior_loss_weight} \
        --learning_rate {self.learning_rate} \
        --mixed_precision {self.mixed_precision} \
        --network_module {self.network_module} \
        --max_data_loader_n_workers {self.max_data_loader_n_workers} \
        --max_train_epochs {self.max_train_epochs} {dynamic}"

class SampleConfig:
    sampler: str
    prompt_path:str
    def __init__(self,
               sampler: Literal['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'heun', 'dpm_2', 'dpm_2_a', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'],
               prompt_path:str,
               sample_every_n_epochs:int|None = None,
               sample_every_n_steps:int|None = None,
               ) -> None:
        self.sampler = sampler
        self.sample_every_n_epochs = sample_every_n_epochs
        self.sample_every_n_steps = sample_every_n_steps
        self.prompt_path = prompt_path
        print("sample config done!")
        pass
    def getArgs(self) -> str:
        dynamic = ""
        if self.sample_every_n_epochs:
            dynamic = f"--sample_every_n_epochs {self.sample_every_n_epochs} "
        if self.sample_every_n_steps:
            dynamic = f"--sample_every_n_steps {self.sample_every_n_steps} "
        return f"{dynamic} --sample_prompts {self.prompt_path} --sample_sampler {self.sampler}"

def gen_train_lora_args(train_config:TrainConfig, output_config:OutputConfig, optimizer_config:OptimizerConfig, sample_config:Optional[SampleConfig] = None):
   
    basic_args = "--cache_latents --gradient_checkpointing --xformers"
    train_args = train_config.getArgs()
    output_args = output_config.getArgs()
    opt_args = optimizer_config.getArgs()
    sample_args = sample_config.getArgs() if sample_config else ""
    args = f"{basic_args} {train_args} {output_args} {opt_args} {sample_args}"
    return args
    
def resolve_model_name(model_name:str):
    if ":" in model_name:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        print("downloading model from hf")
        repo_id, model_path_in_repo = model_name.split(":")
        downloaded = hf_hub_download(repo_id=repo_id,  filename=model_path_in_repo, force_download=True, local_dir=DOWNLOAD_DIR, token=get_r_token(), local_dir_use_symlinks=False)
        print(f"downloaded model to {downloaded}")
        return downloaded
    else:
        return model_name

def train_xl_lora_from_datapack(datapack: DataPack, job_input:dict):
    try:
        print("Train lora for sdxl")
        toml_config = datapack.toml_path
        base_dir = os.path.dirname(toml_config)
        output_config = OutputConfig(
            base_path=base_dir,
            model_name=job_input['output']['model_name'],
            save_every_n_epochs=job_input['output'].get('save_every_n_epochs', None),
            save_every_n_steps=job_input['output'].get('save_every_n_steps', None)
        )
        train_config = TrainNetworkConfig(
            config_file_path=toml_config,
            pretrained_model_name_or_path=resolve_model_name(job_input['train'].get('pretrained_model_name_or_path', SDXL)),
            max_train_epochs=job_input['train'].get('max_train_epochs', None),
            max_train_steps=job_input['train'].get('max_train_steps', None),
            train_batch_size=job_input['train'].get('train_batch_size', 1),
            learning_rate=job_input['train'].get('learning_rate', 1e-4),
            network_dim=job_input['train'].get('network_dim', 1),
            network_alpha=job_input['train'].get('network_alpha', 1),
            mixed_precision=job_input['train'].get('mixed_precision', "bf16"),
            continue_from=job_input['train'].get('continue_from', None)
        )
        sample_config = SampleConfig(sampler= job_input['sample']['sampler'],
                                    sample_every_n_steps=job_input['sample'].get('sample_every_n_steps', None),
                                    sample_every_n_epochs=job_input['sample'].get('sample_every_n_epochs', None),
                                    prompt_path= f"{base_dir}/sample.txt"
                                    )
        args = gen_train_lora_args(output_config=output_config, train_config=train_config, sample_config=sample_config, optimizer_config=AdamW8bitConfig())
        cmd = f"accelerate launch --mixed_precision {train_config.mixed_precision} {sd_scripts_path}/sdxl_train_network.py --resolution 1024 {args}"
        run_cli(cmd)
        return
    except:
        print("train failed!")
        raise

def train_xl_model(datapack: DataPack, job_input:dict):
    try:
        print("train model for sdxl")
        toml_config = datapack.toml_path
        base_dir = os.path.dirname(toml_config)
        output_config = OutputConfig(
            base_path=base_dir,
            model_name=job_input['output']['model_name'],
            save_every_n_epochs=job_input['output'].get('save_every_n_epochs', None),
            save_every_n_steps=job_input['output'].get('save_every_n_steps', None)
        )
        train_config = TrainConfig(
            config_file_path=toml_config,
            pretrained_model_name_or_path=resolve_model_name(job_input['train'].get('pretrained_model_name_or_path', SDXL)),
            max_train_epochs=job_input['train'].get('max_train_epochs', None),
            max_train_steps=job_input['train'].get('max_train_steps', None),
            train_batch_size=job_input['train'].get('train_batch_size', 1),
            learning_rate=job_input['train'].get('learning_rate', 1e-4),
            mixed_precision=job_input['train'].get('mixed_precision', "bf16"),
        )
        sample_config = SampleConfig(sampler=job_input['sample']['sampler'],
                                    sample_every_n_steps=job_input['sample'].get('sample_every_n_steps', None),
                                    sample_every_n_epochs=job_input['sample'].get('sample_every_n_epochs', None),
                                    prompt_path= f"{base_dir}/sample.txt"
                                    )
        args = gen_train_lora_args(output_config=output_config, train_config=train_config, sample_config=sample_config, optimizer_config=AdamW8bitConfig())
        cmd = f"accelerate launch --mixed_precision {train_config.mixed_precision} {sd_scripts_path}/sdxl_train.py --resolution 1024 {args} "
        run_cli(cmd)
    except:
        print("train failed!")
        raise
