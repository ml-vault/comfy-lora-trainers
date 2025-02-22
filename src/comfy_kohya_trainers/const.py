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

DREAMBOOTH_SUBSET_TYPE = NODE_PREFIX + "DREAMBOOTH_SUBSET"

class DreamBoothSubsetDict(TypedDict):
    image_dir: str
    caption_extension: str = ".txt"
    class_tokens: str | None = None
    is_reg: bool = False
    cache_info: bool = False
    keep_tokens: int | None = None
    color_aug: bool = False
    flip_aug: bool = False
    num_repeats: int = 1
    random_crop: bool = False
    shuffle_caption: bool = False
    caption_prefix: str = ""
    caption_suffix: str = ""
    caption_separator: str = ","
    # secondary_separator
    # enable_wildcard


DATASET_CONFIG_TYPE = NODE_PREFIX + "DATASET_CONFIG"
class DreamBoothDatasetConfigDict(TypedDict):
    subsets: list[DreamBoothSubsetDict] = []
    resolution: list[int] = []

DATASET_LOADER_TYPE = NODE_PREFIX + "DATASET_LOADER"
DREAMBOOTH_SUBSET_CONFIG_TYPE = NODE_PREFIX + "DREAMBOOTH_SUBSET_CONFIG"

class DatasetLoaderDict(TypedDict):
    datasets: list[DreamBoothDatasetConfigDict] = []
    dataset_config: str

OPTIMIZER_CONFIG_TYPE = NODE_PREFIX + "OPTIMIZER_CONFIG"
