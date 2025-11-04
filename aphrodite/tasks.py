from typing import Literal, get_args

GenerationTask = Literal["generate", "transcription"]
GENERATION_TASKS = get_args(GenerationTask)

PoolingTask = Literal["embed", "classify", "score", "token_embed", "token_classify", "plugin"]
POOLING_TASKS = get_args(PoolingTask)

VAETask = Literal["vae_encode", "vae_decode"]
VAE_TASKS = get_args(VAETask)

UNetTask = Literal["denoise_step"]
UNET_TASKS = get_args(UNetTask)

SupportedTask = Literal[GenerationTask, PoolingTask, VAETask, UNetTask]
