import os
import re
import toml
from time import time
import warnings

warnings.filterwarnings("ignore")

# These carry information from past executions
if "model_url" in globals():
  old_model_url = model_url
else:
  old_model_url = None
if "dependencies_installed" not in globals():
  dependencies_installed = False
if "model_file" not in globals():
  model_file = None

# These may be set by other cells, some are legacy
if "custom_dataset" not in globals():
  custom_dataset = None
if "override_dataset_config_file" not in globals():
  override_dataset_config_file = None
if "continue_from_lora" not in globals():
  continue_from_lora = ""
if "override_config_file" not in globals():
  override_config_file = None

#try:
#  LOWRAM = int(next(line.split()[1] for line in open('/proc/meminfo') if "MemTotal" in line)) / (1024**2) < 15
#except:
#  LOWRAM = True

LOWRAM = True
LOAD_TRUNCATED_IMAGES = True
BETTER_EPOCH_NAMES = True
FIX_DIFFUSERS = True


#@title ## 🚩 Start Here

#@markdown ### ▶️ Setup
#@markdown Your project name will be the same as the folder containing your images. Spaces aren't allowed.
project_name = "" #@param {type:"string"}
project_name = project_name.strip()
#@markdown The folder structure doesn't matter and is purely for comfort. Make sure to always pick the same one. I like organizing by project.
folder_structure = "Organize by project (MyDrive/Loras/project_name/dataset)" #@param ["Organize by category (MyDrive/lora_training/datasets/project_name)", "Organize by project (MyDrive/Loras/project_name/dataset)"]
#@markdown Decide the model that will be downloaded and used for training. You can also choose your own by pasting its download link, or a file in your Google Drive starting with `/content/drive/MyDrive`.
training_model = "Pony Diffusion V6 XL" #@param ["Pony Diffusion V6 XL", "Illustrious XL 0.1", "NoobAI V-Pred 1.0", "NoobAI Eps 1.1", "Animagine XL V3", "Stable Diffusion XL 1.0 base"]
optional_custom_training_model = "" #@param {type:"string"}
custom_model_is_diffusers = False #@param {type:"boolean"}
custom_model_is_vpred = False #@param {type:"boolean"}
#@markdown Use wandb if you want to visualize the progress of your training over time.
wandb_key = "" #@param {type:"string"}

load_diffusers = custom_model_is_diffusers and len(optional_custom_training_model) > 0
vpred = custom_model_is_vpred and len(optional_custom_training_model) > 0

if optional_custom_training_model:
  model_url = optional_custom_training_model
elif "Pony" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/hollowstrawberry/67AB2F"
  else:
    model_url = "https://civitai.com/api/download/models/290640"
    model_file = "/content/ponyDiffusionV6XL.safetensors"
elif "Animagine" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/cagliostrolab/animagine-xl-3.0"
  else:
    model_url = "https://civitai.com/api/download/models/293564"
    model_file = "/content/animagineXLV3.safetensors"
elif "Illustrious" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0"
  else:
    model_url = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors"
elif "NoobAI Eps" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-1.1"
  else:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors"
elif "NoobAI V-Pred" in training_model:
  vpred = True
  if load_diffusers:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0"
  else:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/resolve/main/NoobAI-XL-Vpred-v1.0.safetensors"
else:
  if load_diffusers:
    model_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
  else:
    model_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

if load_diffusers:
  vae_file = "stabilityai/sdxl-vae"
else:
  vae_url = "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
  vae_file = "/content/sdxl_vae.safetensors"

model_url = model_url.strip()

#@markdown ### ▶️ Processing
resolution = 1024 #param {type:"slider", min:768, max:1536, step:128}
caption_extension = ".txt" #@param [".txt", ".caption"]
#@markdown Shuffling anime tags in place improves learning and prompting. An activation tag goes at the start of every text file and will not be shuffled.
shuffle_tags = True #@param {type:"boolean"}
shuffle_caption = shuffle_tags
activation_tags = "1" #@param [0,1,2,3]
keep_tokens = int(activation_tags)

#@markdown ### ▶️ Steps
#@markdown Your images will repeat this number of times during training. I recommend that your images multiplied by their repeats is around 100, or 1 repeat with more than 100 images.
num_repeats = 2 #@param {type:"number"}
#@markdown Choose how long you want to train for. A good starting point is around 10 epochs or around 2000 steps.
#@markdown One epoch is a number of steps equal to: your number of images multiplied by their repeats, divided by batch size.
preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
how_many = 10 #@param {type:"number"}
max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None
#@markdown Saving more epochs will let you compare your Lora's progress better.
save_every_n_epochs = 1 #@param {type:"number"}
keep_only_last_n_epochs = 10 #@param {type:"number"}
if not save_every_n_epochs:
  save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
  keep_only_last_n_epochs = max_train_epochs

#@markdown ### ▶️ Learning
#@markdown The learning rate is the most important for your results. If your Lora produces black images, lower the unet and text encoder to 1e-4 and 1e-5 respectively, or even lower.
#@markdown If you're training a style you can choose to set the text encoder to 0.
unet_lr = 3e-4 #@param {type:"number"}
text_encoder_lr = 6e-5 #@param {type:"number"}
#@markdown The scheduler is the algorithm that guides the learning rate. If you're not sure, pick `constant` and ignore the number. I personally recommend `cosine_with_restarts` with 3 restarts.
lr_scheduler = "cosine_with_restarts" #@param ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial", "rex"]
lr_scheduler_number = 3 #@param {type:"number"}
#@markdown Steps spent "warming up" the learning rate during training for efficiency. I recommend leaving it at 5%.
lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.2, step:0.01}
lr_warmup_steps = 0
#@markdown These settings may produce better results. min_snr_gamma adjusts loss over time. ip_noise_gamma adjusts random noise.
min_snr_gamma_enabled = True #@param {type:"boolean"}
min_snr_gamma = 8.0 #@param {type:"slider", min:4, max:16.0, step:0.5}
ip_noise_gamma_enabled = False #@param {type:"boolean"}
ip_noise_gamma = 0.05 #@param {type:"slider", min:0.05, max:0.1, step:0.01}
#@markdown Multinoise may help with color balance (darker darks, lighter lights).
multinoise = True #@param {type:"boolean"}

#@markdown ### ▶️ Structure
#@markdown LoRA is the classic type and good for a variety of purposes. LoCon is good with artstyles as it has more layers to learn more aspects of the dataset.
lora_type = "LoRA" #@param ["LoRA", "LoCon"]

#@markdown Below are some recommended XL values for the following settings:

#@markdown | type | network_dim | network_alpha | conv_dim | conv_alpha |
#@markdown | :---: | :---: | :---: | :---: | :---: |
#@markdown | Regular LoRA | 8 | 4 |   |   |
#@markdown | Style LoCon | 16 | 8 | 16 | 8 |

#@markdown More dim means larger Lora, it can hold more information but more isn't always better.
network_dim = 8 #@param {type:"slider", min:1, max:32, step:1}
network_alpha = 4 #@param {type:"slider", min:1, max:32, step:1}
#@markdown The following two values only apply to the additional layers of LoCon.
conv_dim = 16 #@param {type:"slider", min:1, max:32, step:1}
conv_alpha = 8 #@param {type:"slider", min:1, max:32, step:1}

network_module = "networks.lora"
network_args = None
if lora_type.lower() == "locon":
  network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]

#@markdown ### ▶️ Training
#@markdown Adjust these parameters depending on your colab configuration.
#@markdown
#@markdown Higher batch size is often faster but uses more memory.
train_batch_size = 4 #@param {type:"slider", min:1, max:16, step:1}
#@markdown I have found no substantial difference between sdpa and xformers.
cross_attention = "sdpa" #@param ["sdpa", "xformers"]
#@markdown Use `full fp16` for lowest memory usage. The `bf16` options may not work correctly on the free GPU tier.
#@markdown The Lora will be trained with the selected precision, but will always be saved in fp16 format for compatibility reasons.
precision = "full fp16" #@param ["float", "full fp16", "full bf16", "mixed fp16", "mixed bf16"]
#@markdown Caching latents to drive will add a 250KB file next to each image but will use considerably less memory.
cache_latents = True #@param {type:"boolean"}
cache_latents_to_drive = True #@param {type:"boolean"}
#@markdown The following option will turn off shuffle_tags and disable text encoder training.
cache_text_encoder_outputs  = False  # @param {type:"boolean"}

mixed_precision = "no"
if "fp16" in precision:
  mixed_precision = "fp16"
elif "bf16" in precision:
  mixed_precision = "bf16"
full_precision = "full" in precision

#@markdown ### ▶️ Advanced
#@markdown The optimizer is the algorithm used for training. AdamW8bit is the default and works great, while Prodigy manages learning rate automatically and may have several advantages such as training faster due to needing less steps as well as working better for small datasets.
optimizer = "AdamW8bit" #@param ["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor", "Came"]
#@markdown Recommended args for AdamW8bit: `weight_decay=0.1 betas=[0.9,0.99]`
#@markdown Recommended args for Prodigy: `decouple=True weight_decay=0.01 betas=[0.9,0.999] d_coef=2 use_bias_correction=True safeguard_warmup=True`
#@markdown Recommended args for AdaFactor: `scale_parameter=False relative_step=False warmup_init=False`
#@markdown Recommended args for Came: `weight_decay=0.04`
#@markdown If Dadapt or Prodigy are selected and the recommended box is checked, the following values will override any previous settings:
#@markdown `unet_lr=0.75`, `text_encoder_lr=0.75`, `network_alpha=network_dim`, `full_precision=False`
recommended_values = True #@param {type:"boolean"}
#@markdown Alternatively set your own optimizer arguments separated by spaces (not commas). Recommended box must be disabled.
optimizer_args = "" #@param {type:"string"}
optimizer_args = [a.strip() for a in optimizer_args.split(' ') if a]

if recommended_values:
  if any(opt in optimizer.lower() for opt in ["dadapt", "prodigy"]):
    unet_lr = 0.75
    text_encoder_lr = 0.75
    network_alpha = network_dim
    full_precision = False
  if optimizer == "Prodigy":
    optimizer_args = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
  elif optimizer == "AdamW8bit":
    optimizer_args = ["weight_decay=0.1", "betas=[0.9,0.99]"]
  elif optimizer == "AdaFactor":
    optimizer_args = ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
  elif optimizer == "Came":
    optimizer_args = ["weight_decay=0.04"]

if optimizer == "Came":
  optimizer = "LoraEasyCustomOptimizer.came.CAME"

lr_scheduler_type = None
lr_scheduler_args = None
lr_scheduler_num_cycles = lr_scheduler_number
lr_scheduler_power = lr_scheduler_number

if "rex" in lr_scheduler:
  lr_scheduler = "cosine"
  lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
  lr_scheduler_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]

# Misc
seed = 42
gradient_accumulation_steps = 1
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 4096


#@markdown ### ▶️ Ready
#@markdown You can now run this cell to cook your Lora. Good luck!


# 👩‍💻 Cool code goes here

root_dir = "/content"
trainer_dir = os.path.join(root_dir, "trainer")
kohya_dir = os.path.join(trainer_dir, "sd_scripts")

venv_python = os.path.join(kohya_dir, "venv/bin/python")
venv_pip = os.path.join(kohya_dir, "venv/bin/pip")
train_network = os.path.join(kohya_dir, "sdxl_train_network.py")

if "/Loras" in folder_structure:
  main_dir      = os.path.join(root_dir, "drive/MyDrive/Loras")
  log_folder    = os.path.join(main_dir, "_logs")
  config_folder = os.path.join(main_dir, project_name)
  images_folder = os.path.join(main_dir, project_name, "dataset")
  output_folder = os.path.join(main_dir, project_name, "output")
else:
  main_dir      = os.path.join(root_dir, "drive/MyDrive/lora_training")
  images_folder = os.path.join(main_dir, "datasets", project_name)
  output_folder = os.path.join(main_dir, "output", project_name)
  config_folder = os.path.join(main_dir, "config", project_name)
  log_folder    = os.path.join(main_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")


def install_trainer():
#   !apt -y update -qq
#   !apt install -y python3.10-venv aria2 -qq
#   !git clone https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend {trainer_dir}
#   !chmod 755 /content/trainer/colab_install.sh
#   os.chdir(trainer_dir)
#   !./colab_install.sh

#   # fix logging
#   !{venv_pip} uninstall -y rich

#   # patch kohya
#   os.chdir(kohya_dir)
#   if LOAD_TRUNCATED_IMAGES:
#     !sed -i 's/from PIL import Image/from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py # fix truncated jpegs error
#   if BETTER_EPOCH_NAMES:
#     !sed -i 's/{:06d}/{:02d}/g' library/train_util.py # make epoch names shorter
#     !sed -i 's/"." + args.save_model_as)/"-{:02d}.".format(num_train_epochs) + args.save_model_as)/g' train_network.py # name of the last epoch will match the rest
#   if FIX_DIFFUSERS:
#     deprecation_utils = os.path.join(kohya_dir, "venv/lib/python3.10/site-packages/diffusers/utils/deprecation_utils.py")
#     !sed -i 's/if version.parse/if False:#/g' {deprecation_utils}
#   if wandb_key:
#     !sed -i 's/accelerator.log(logs, step=epoch + 1)//g' train_network.py  # fix warning
#     !sed -i 's/accelerator.log(logs, step=epoch + 1)//g' sdxl_train.py  # fix warning

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"
  os.environ["PYTHONWARNINGS"] = "ignore"
  os.chdir(root_dir)


def validate_dataset():
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, model_url
  supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

  if model_url.startswith("/content/drive/") and not os.path.exists(model_url):
    print("💥 Error: The custom training model you specified was not found in your Google Drive.")
    return

  print("\n💿 Checking dataset...")
  if not project_name.strip() or any(c in project_name for c in " .()\"'\/"):
    print("💥 Error: Please choose a valid project name.")
    return

  # Find the folders and files
  if custom_dataset:
    try:
      datconf = toml.loads(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except:
      print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
      return
    reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)]
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
    folders = datasets_dict.keys()
    files = [f for folder in folders for f in os.listdir(folder)]
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else:
    reg = []
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  # Validation
  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
      return
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
      return
  test_files = []
  for f in files:
    if not f.lower().endswith((caption_extension, ".npz")) and not f.lower().endswith(supported_types):
      print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
      return
    for ff in test_files:
      if f.endswith(supported_types) and ff.endswith(supported_types) \
          and os.path.splitext(f)[0] == os.path.splitext(ff)[0]:
        print(f"💥 Error: The files {f} and {ff} cannot have the same name. Aborting.")
        return
    test_files.append(f)

  if caption_extension and not [txt for txt in files if txt.lower().endswith(caption_extension)]:
    caption_extension = ""
  if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
    print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
    return

  # Show estimations to the user

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio)

  for folder, (img, rep) in images_repeats.items():
    print("📁"+folder.replace("/content/drive/", "") + (" (Regularization)" if folder in reg else ""))
    print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
  print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
  if max_train_epochs:
    print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
  else:
    print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

  if total_steps > 10000:
    print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
    return

  return True


def create_config():
  global dataset_config_file, config_file, model_file

  if override_config_file:
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "network_arguments": {
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr if not cache_text_encoder_outputs else 0,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_module": network_module,
        "network_args": network_args,
        "network_train_unet_only": text_encoder_lr == 0 or cache_text_encoder_outputs,
        "network_weights": continue_from_lora or None
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_args": lr_scheduler_args,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler not in ("cosine", "constant") else None,
        "optimizer_type": optimizer,
        "optimizer_args": optimizer_args or None,
        "loss_type": "l2",
        "max_grad_norm": 1.0,
      },
      "training_arguments": {
        "lowram": LOWRAM,
        "pretrained_model_name_or_path": model_file,
        "vae": vae_file,
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "train_batch_size": train_batch_size,
        "seed": seed,
        "max_token_length": 225,
        "xformers": cross_attention == "xformers",
        "sdpa": cross_attention == "sdpa",
        "min_snr_gamma": min_snr_gamma if min_snr_gamma_enabled else None,
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma_enabled else None,
        "no_half_vae": True,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_data_loader_n_workers": 1,
        "persistent_data_loader_workers": True,
        "mixed_precision": mixed_precision,
        "full_fp16": mixed_precision == "fp16" and full_precision,
        "full_bf16": mixed_precision == "bf16" and full_precision,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_drive,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "min_timestep": 0,
        "max_timestep": 1000,
        "prior_loss_weight": 1.0,
        "multires_noise_iterations": 6 if multinoise else None,
        "multires_noise_discount": 0.3 if multinoise else None,
        "v_parameterization": vpred or None,
        "scale_v_pred_loss_like_noise_pred": vpred or None,
        "zero_terminal_snr": vpred or None,
      },
      "saving_arguments": {
        "save_precision": "fp16",
        "save_model_as": "safetensors",
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "output_name": project_name,
        "output_dir": output_folder,
        "log_prefix": project_name,
        "logging_dir": log_folder,
        "wandb_api_key": wandb_key or None,
        "log_with": "wandb" if wandb_key else None,
      }
    }

    for key in config_dict:
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file:
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:
    dataset_config_dict = {
      "general": {
        "resolution": resolution,
        "shuffle_caption": shuffle_caption and not cache_text_encoder_outputs,
        "keep_tokens": keep_tokens,
        "flip_aug": False,
        "caption_extension": caption_extension,
        "enable_bucket": True,
        "bucket_no_upscale": False,
        "bucket_reso_steps": bucket_reso_steps,
        "min_bucket_reso": min_bucket_reso,
        "max_bucket_reso": max_bucket_reso,
      },
      "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict:
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Dataset config saved to {dataset_config_file}")


def download_model():
  global old_model_url, model_url, model_file, vae_url, vae_file
  real_model_url = model_url  # There was a reason for having a separate variable but I forgot what it was.

  if real_model_url.startswith("/content/drive/"):
    # Local model, already checked to exist
    model_file = real_model_url
    print(f"📁 Using local model file: {model_file}")
    # Validation
    if model_file.lower().endswith(".safetensors"):
      from safetensors.torch import load_file as load_safetensors
      try:
        test = load_safetensors(model_file)
        del test
      except:
        return False
    elif model_file.lower().endswith(".ckpt"):
      from torch import load as load_ckpt
      try:
        test = load_ckpt(model_file)
        del test
      except:
        return False
    return True

  else:
    # Downloadable model
    if load_diffusers:
      if 'huggingface.co' in real_model_url:
          match = re.search(r'huggingface\.co/([^/]+)/([^/]+)', real_model_url)
          if match:
              username = match.group(1)
              model_name = match.group(2)
              model_file = f"{username}/{model_name}"
              from huggingface_hub import HfFileSystem
              fs = HfFileSystem()
              existing_folders = set(fs.ls(model_file, detail=False))
              necessary_folders = [ "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae" ]
              if all(f"{model_file}/{folder}" in existing_folders for folder in necessary_folders):
                print("🍃 Diffusers model identified.")  # Will be handled by kohya
                return True
      raise ValueError("💥 Failed to load Diffusers model. If this model is not Diffusers, have you tried turning it off at the top of the colab?")

    # Define local filename
    if not model_file or old_model_url and old_model_url != model_url:
      if real_model_url.lower().endswith((".ckpt", ".safetensors")):
        model_file = f"/content{real_model_url[real_model_url.rfind('/'):]}"
      else:
        model_file = "/content/downloaded_model.safetensors"
        if os.path.exists(model_file):
          !rm "{model_file}"

    # HuggingFace
    if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", real_model_url):
      real_model_url = real_model_url.replace("blob", "resolve")
    # Civitai
    elif m := re.search(r"(?:https?://)?(?:www\.)?civitai\.com/models/([0-9]+)(/[A-Za-z0-9-_]+)?", real_model_url):
      if m.group(2):
        model_file = f"/content{m.group(2)}.safetensors"
      if m := re.search(r"modelVersionId=([0-9]+)", real_model_url):
        real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"
      else:
        raise ValueError("💥 optional_custom_training_model contains a civitai link, but the link doesn't include a modelVersionId. You can also right click the download button to copy the direct download link.")

    # Download checkpoint
    !aria2c "{real_model_url}" --console-log-level=warn -c -s 16 -x 16 -k 10M -d / -o "{model_file}"

    # Download VAE
    if not os.path.exists(vae_file):
      !aria2c "{vae_url}" --console-log-level=warn -c -s 16 -x 16 -k 10M -d / -o "{vae_file}"

    # Validation

    if model_file.lower().endswith(".safetensors"):
      from safetensors.torch import load_file as load_safetensors
      try:
        test = load_safetensors(model_file)
        del test
      except:
        new_model_file = os.path.splitext(model_file)[0]+".ckpt"
        !mv "{model_file}" "{new_model_file}"
        model_file = new_model_file
        print(f"Renamed model to {os.path.splitext(model_file)[0]}.ckpt")

    if model_file.lower().endswith(".ckpt"):
      from torch import load as load_ckpt
      try:
        test = load_ckpt(model_file)
        del test
      except:
        return False

  return True


def calculate_rex_steps():
  # https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend/blob/c34084b0435e6e19bb7a01ac1ecbadd185ee8c1e/utils/validation.py#L268
  global max_train_steps
  print("\n🤔 Calculating Rex steps")
  if max_train_steps:
    calculated_max_steps = max_train_steps
  else:
    from library.train_util import BucketManager
    from PIL import Image
    from pathlib import Path
    import math

    with open(dataset_config_file, "r") as f:
      subsets = toml.load(f)["datasets"][0]["subsets"]

    supported_types = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    res = (resolution, resolution)
    bucketManager = BucketManager(False, res, min_bucket_reso, max_bucket_reso, bucket_reso_steps)
    bucketManager.make_buckets()
    for subset in subsets:
        for image in Path(subset["image_dir"]).iterdir():
            if image.suffix not in supported_types:
                continue
            with Image.open(image) as img:
                bucket_reso, _, _ = bucketManager.select_bucket(img.width, img.height)
                for _ in range(subset["num_repeats"]):
                    bucketManager.add_image(bucket_reso, image)
    steps_before_acc = sum(math.ceil(len(bucket) / train_batch_size) for bucket in bucketManager.buckets)
    calculated_max_steps = math.ceil(steps_before_acc / gradient_accumulation_steps) * max_train_epochs
    del bucketManager

  cycle_steps = calculated_max_steps // (lr_scheduler_num_cycles or 1)
  print(f"  cycle steps: {cycle_steps}")
  lr_scheduler_args.append(f"first_cycle_max_steps={cycle_steps}")

  warmup_steps = round(calculated_max_steps * lr_warmup_ratio) // (lr_scheduler_num_cycles or 1)
  if warmup_steps > 0:
    print(f"  warmup steps: {warmup_steps}")
    lr_scheduler_args.append(f"warmup_steps={warmup_steps}")


def main():
  global dependencies_installed

  for dir in (main_dir, trainer_dir, log_folder, images_folder, output_folder, config_folder):
    os.makedirs(dir, exist_ok=True)

  if not validate_dataset():
    return

  if not dependencies_installed:
    print("\n🏭 Installing trainer...\n")
    t0 = time()
    install_trainer()
    t1 = time()
    dependencies_installed = True
    print(f"\n✅ Installation finished in {int(t1-t0)} seconds.")
  else:
    print("\n✅ Dependencies already installed.")

  if old_model_url != model_url or not model_file or not os.path.exists(model_file):
    print("\n🔄 Getting model...")
    if not download_model():
      print("\n💥 Error: The model you specified is invalid or corrupted."
            "\nIf you're using an URL, please check that the model is accessible without being logged in."
            "\nYou can try civitai or huggingface URLs, or a path in your Google Drive starting with /content/drive/MyDrive")
      return
    print()
  else:
    print("\n🔄 Model already downloaded.\n")

  if lr_scheduler_type:
    create_config()
    os.chdir(kohya_dir)
    calculate_rex_steps()
    os.chdir(root_dir)

  create_config()

  print("\n⭐ Starting trainer...\n")

  os.chdir(kohya_dir)
  !{venv_python} {train_network} --config_file={config_file} --dataset_config={dataset_config_file}
  os.chdir(root_dir)

  if not get_ipython().__dict__['user_ns']['_exit_code']:
    display(Markdown("### ✅ Done! [Go download your Lora from Google Drive](https://drive.google.com/drive/my-drive)\n"
                     "### There will be several files, you should try the latest version (the file with the largest number next to it)"))

main()
