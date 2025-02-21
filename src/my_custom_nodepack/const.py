from typing import TypedDict
NODE_PREFIX = "KHXL_CF_"

# Node types
DATAPACK_REPO_INFO_TYPE = NODE_PREFIX + "DATASET_INFO"

class DatapackRepoInfo(TypedDict):
    dataset_repo_id: str
    auth_token: str

SAMPLER_CONFIG_TYPE = NODE_PREFIX + "SAMPLER_CONFIG"
TRAIN_CONFIG_TYPE = NODE_PREFIX + "TRAIN_CONFIG"
class TrainConfigDict(TypedDict):
    pretrained_model_name_or_path: str
    network_weights: str | None
    xformers: bool
    mixed_precision: str
    save_precision: str
    learning_rate: float
    train_batch_size: int
    prior_loss_weight: float
    max_train_epochs: int
    max_train_steps: int

OUTPUT_CONFIG_TYPE = NODE_PREFIX + "OUTPUT_CONFIG"

DATASET_TYPE = NODE_PREFIX + "DATASET"
class DatasetDict(TypedDict):
    dir: str
    width: int
    height: int
    caption_extension: str
    repeat_times: int
    shuffle_captions: bool

DATASET_LOADER_TYPE = NODE_PREFIX + "DATASET_LOADER"

class DatasetLoaderDict(TypedDict):
    datasets: list[DatasetDict]
