from library.utils import setup_logging, add_logging_arguments
from library import train_util
import argparse
from library import deepspeed_utils, model_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
from library import sdxl_model_util, sdxl_train_util, train_util


class ClassfiedArgs:
    def __init__(
        self,
        console_log_level=None,
        console_log_file=None,
        console_log_simple=False,
        v2=False,
        v_parameterization=False,
        pretrained_model_name_or_path=None,
        tokenizer_cache_dir=None,
        train_data_dir=None,
        cache_info=False,
        shuffle_caption=False,
        caption_separator=',',
        caption_extension='.caption',
        caption_extention=None,
        keep_tokens=0,
        keep_tokens_separator='',
        secondary_separator=None,
        enable_wildcard=False,
        caption_prefix=None,
        caption_suffix=None,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        debug_dataset=False,
        resolution=None,
        cache_latents=False,
        vae_batch_size=1,
        cache_latents_to_disk=False,
        enable_bucket=False,
        min_bucket_reso=256,
        max_bucket_reso=1024,
        bucket_reso_steps=64,
        bucket_no_upscale=False,
        token_warmup_min=1,
        token_warmup_step=0,
        alpha_mask=False,
        dataset_class=None,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        reg_data_dir=None,
        in_json=None,
        dataset_repeats=1,
        output_dir=None,
        output_name=None,
        huggingface_repo_id=None,
        huggingface_repo_type=None,
        huggingface_path_in_repo=None,
        huggingface_token=None,
        huggingface_repo_visibility=None,
        save_state_to_huggingface=False,
        resume_from_huggingface=False,
        async_upload=False,
        save_precision='fp16',
        save_every_n_epochs=None,
        save_every_n_steps=None,
        save_n_epoch_ratio=None,
        save_last_n_epochs=None,
        save_last_n_epochs_state=None,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
        save_state=False,
        save_state_on_train_end=False,
        resume=None,
        train_batch_size=1,
        max_token_length=None,
        mem_eff_attn=False,
        torch_compile=False,
        dynamo_backend='inductor',
        xformers=True,
        sdpa=False,
        vae=None,
        max_train_steps=1600,
        max_train_epochs=None,
        max_data_loader_n_workers=8,
        persistent_data_loader_workers=False,
        seed=None,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
        full_fp16=False,
        full_bf16=False,
        fp8_base=False,
        ddp_timeout=None,
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=False,
        clip_skip=None,
        logging_dir=None,
        log_with=None,
        log_prefix=None,
        log_tracker_name=None,
        wandb_run_name=None,
        log_tracker_config=None,
        wandb_api_key=None,
        log_config=False,
        noise_offset=None,
        noise_offset_random_strength=False,
        multires_noise_iterations=None,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        multires_noise_discount=0.3,
        adaptive_noise_scale=None,
        zero_terminal_snr=False,
        min_timestep=None,
        max_timestep=None,
        loss_type='l2',
        huber_schedule='snr',
        huber_c=0.1,
        lowram=False,
        highvram=False,
        sample_every_n_steps=None,
        sample_at_first=False,
        sample_every_n_epochs=None,
        sample_prompts=None,
        sample_sampler='ddim',
        config_file=None,
        output_config=False,
        metadata_title=None,
        metadata_author=None,
        metadata_description=None,
        metadata_license=None,
        metadata_tags=None,
        prior_loss_weight=1.0,
        conditioning_data_dir=None,
        masked_loss=False,
        deepspeed=False,
        zero_stage=2,
        offload_optimizer_device=None,
        offload_optimizer_nvme_path=None,
        offload_param_device=None,
        offload_param_nvme_path=None,
        zero3_init_flag=False,
        zero3_save_16bit_model=False,
        fp16_master_weights_and_gradients=False,
        optimizer_type='',
        use_8bit_adam=False,
        use_lion_optimizer=False,
        learning_rate=2e-06,
        max_grad_norm=1.0,
        optimizer_args=None,
        lr_scheduler_type='',
        lr_scheduler_args=None,
        lr_scheduler='constant',
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1,
        fused_backward_pass=False,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        dataset_config=None,
        min_snr_gamma=None,
        scale_v_pred_loss_like_noise_pred=False,
        v_pred_like_loss=None,
        debiased_estimation_loss=False,
        weighted_captions=False,
        no_metadata=False,
        save_model_as='safetensors',
        unet_lr=None,
        text_encoder_lr=None,
        network_weights=None,
        network_module=None,
        network_dim=None,
        network_alpha=1,
        network_dropout=None,
        network_args=None,
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
        training_comment=None,
        dim_from_weights=False,
        scale_weight_norms=None,
        base_weights=None,
        base_weights_multiplier=None,
        no_half_vae=False,
        skip_until_initial_step=False,
        initial_epoch=None,
        initial_step=None,
        cache_text_encoder_outputs=False,
        cache_text_encoder_outputs_to_disk=False,
        disable_mmap_load_safetensors=False
    ):
        self.console_log_level = console_log_level
        self.console_log_file = console_log_file
        self.console_log_simple = console_log_simple
        self.v2 = v2
        self.v_parameterization = v_parameterization
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_cache_dir = tokenizer_cache_dir
        self.train_data_dir = train_data_dir
        self.cache_info = cache_info
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.caption_extension = caption_extension
        self.caption_extention = caption_extention
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.debug_dataset = debug_dataset
        self.resolution = resolution
        self.cache_latents = cache_latents
        self.vae_batch_size = vae_batch_size
        self.cache_latents_to_disk = cache_latents_to_disk
        self.enable_bucket = enable_bucket
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.bucket_reso_steps = bucket_reso_steps
        self.bucket_no_upscale = bucket_no_upscale
        self.token_warmup_min = token_warmup_min
        self.token_warmup_step = token_warmup_step
        self.alpha_mask = alpha_mask
        self.dataset_class = dataset_class
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.reg_data_dir = reg_data_dir
        self.in_json = in_json
        self.dataset_repeats = dataset_repeats
        self.output_dir = output_dir
        self.output_name = output_name
        self.huggingface_repo_id = huggingface_repo_id
        self.huggingface_repo_type = huggingface_repo_type
        self.huggingface_path_in_repo = huggingface_path_in_repo
        self.huggingface_token = huggingface_token
        self.huggingface_repo_visibility = huggingface_repo_visibility
        self.save_state_to_huggingface = save_state_to_huggingface
        self.resume_from_huggingface = resume_from_huggingface
        self.async_upload = async_upload
        self.save_precision = save_precision
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps = save_every_n_steps
        self.save_n_epoch_ratio = save_n_epoch_ratio
        self.save_last_n_epochs = save_last_n_epochs
        self.save_last_n_epochs_state = save_last_n_epochs_state
        self.save_last_n_steps = save_last_n_steps
        self.save_last_n_steps_state = save_last_n_steps_state
        self.save_state = save_state
        self.save_state_on_train_end = save_state_on_train_end
        self.resume = resume
        self.train_batch_size = train_batch_size
        self.max_token_length = max_token_length
        self.mem_eff_attn = mem_eff_attn
        self.torch_compile = torch_compile
        self.dynamo_backend = dynamo_backend
        self.xformers = xformers
        self.sdpa = sdpa
        self.vae = vae
        self.max_train_steps = max_train_steps
        self.max_train_epochs = max_train_epochs
        self.max_data_loader_n_workers = max_data_loader_n_workers
        self.persistent_data_loader_workers = persistent_data_loader_workers
        self.seed = seed
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.full_fp16 = full_fp16
        self.full_bf16 = full_bf16
        self.fp8_base = fp8_base
        self.ddp_timeout = ddp_timeout
        self.ddp_gradient_as_bucket_view = ddp_gradient_as_bucket_view
        self.ddp_static_graph = ddp_static_graph
        self.clip_skip = clip_skip
        self.logging_dir = logging_dir
        self.log_with = log_with
        self.log_prefix = log_prefix
        self.log_tracker_name = log_tracker_name
        self.wandb_run_name = wandb_run_name
        self.log_tracker_config = log_tracker_config
        self.wandb_api_key = wandb_api_key
        self.log_config = log_config
        self.noise_offset = noise_offset
        self.noise_offset_random_strength = noise_offset_random_strength
        self.multires_noise_iterations = multires_noise_iterations
        self.ip_noise_gamma = ip_noise_gamma
        self.ip_noise_gamma_random_strength = ip_noise_gamma_random_strength
        self.multires_noise_discount = multires_noise_discount
        self.adaptive_noise_scale = adaptive_noise_scale
        self.zero_terminal_snr = zero_terminal_snr
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.loss_type = loss_type
        self.huber_schedule = huber_schedule
        self.huber_c = huber_c
        self.lowram = lowram
        self.highvram = highvram
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_at_first = sample_at_first
        self.sample_every_n_epochs = sample_every_n_epochs
        self.sample_prompts = sample_prompts
        self.sample_sampler = sample_sampler
        self.config_file = config_file
        self.output_config = output_config
        self.metadata_title = metadata_title
        self.metadata_author = metadata_author
        self.metadata_description = metadata_description
        self.metadata_license = metadata_license
        self.metadata_tags = metadata_tags
        self.prior_loss_weight = prior_loss_weight
        self.conditioning_data_dir = conditioning_data_dir
        self.masked_loss = masked_loss
        self.deepspeed = deepspeed
        self.zero_stage = zero_stage
        self.offload_optimizer_device = offload_optimizer_device
        self.offload_optimizer_nvme_path = offload_optimizer_nvme_path
        self.offload_param_device = offload_param_device
        self.offload_param_nvme_path = offload_param_nvme_path
        self.zero3_init_flag = zero3_init_flag
        self.zero3_save_16bit_model = zero3_save_16bit_model
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients
        self.optimizer_type = optimizer_type
        self.use_8bit_adam = use_8bit_adam
        self.use_lion_optimizer = use_lion_optimizer
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.optimizer_args = optimizer_args
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_args = lr_scheduler_args
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_scheduler_num_cycles = lr_scheduler_num_cycles
        self.lr_scheduler_power = lr_scheduler_power
        self.fused_backward_pass = fused_backward_pass
        self.lr_scheduler_timescale = lr_scheduler_timescale
        self.lr_scheduler_min_lr_ratio = lr_scheduler_min_lr_ratio
        self.dataset_config = dataset_config
        self.min_snr_gamma = min_snr_gamma
        self.scale_v_pred_loss_like_noise_pred = scale_v_pred_loss_like_noise_pred
        self.v_pred_like_loss = v_pred_like_loss
        self.debiased_estimation_loss = debiased_estimation_loss
        self.weighted_captions = weighted_captions
        self.no_metadata = no_metadata
        self.save_model_as = save_model_as
        self.unet_lr = unet_lr
        self.text_encoder_lr = text_encoder_lr
        self.network_weights = network_weights
        self.network_module = network_module
        self.network_dim = network_dim
        self.network_alpha = network_alpha
        self.network_dropout = network_dropout
        self.network_args = network_args
        self.network_train_unet_only = network_train_unet_only
        self.network_train_text_encoder_only = network_train_text_encoder_only
        self.training_comment = training_comment
        self.dim_from_weights = dim_from_weights
        self.scale_weight_norms = scale_weight_norms
        self.base_weights = base_weights
        self.base_weights_multiplier = base_weights_multiplier
        self.no_half_vae = no_half_vae
        self.skip_until_initial_step = skip_until_initial_step
        self.initial_epoch = initial_epoch
        self.initial_step = initial_step
        self.cache_text_encoder_outputs = cache_text_encoder_outputs
        self.cache_text_encoder_outputs_to_disk = cache_text_encoder_outputs_to_disk
        self.disable_mmap_load_safetensors = disable_mmap_load_safetensors


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser

def setup_parser_sdxl() -> argparse.ArgumentParser:
    parser = setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser
