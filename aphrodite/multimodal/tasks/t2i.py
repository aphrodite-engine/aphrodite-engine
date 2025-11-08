import io
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import regex as re
import torch

# To use shared tensor serialization with torch.multiprocessing,
# call 'init_reductions' after importing to register custom reducer functions.
import torch.multiprocessing  # noqa: F401
from transformers.modeling_outputs import BaseModelOutputWithPast

from aphrodite.distributed import (
    broadcast_tensor_dict,
    get_tensor_model_parallel_rank,
)
from aphrodite.logger import init_logger
from aphrodite.modeling.models.hunyuan_image3_utils import (
    create_hunyuan_image_attention_meta,
)
from aphrodite.modeling.models.text_to_image.hunyuan import HunyuanImage3ForCausalMM

logger = init_logger(__name__)


class HunyuanImage3TokenConfig:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.cot_prefix_tokens = {
            "think": "<think>",
            "recaption": "<recaption>",
        }
        self.extra_auto_stops = [self.tokenizer.special_token_map[f"<img_ratio_{i}>"] for i in range(33)]
        self.eos_token_id = tokenizer.eos_token_id
        self.end_recaption_token_id = tokenizer.end_recaption_token_id
        self.end_think_token_id = tokenizer.tokenizer.convert_tokens_to_ids("</think>")
        self.end_answer_token_id = tokenizer.end_answer_token_id
        self.boi_token_id = tokenizer.boi_token_id

        self.stop_token_ids_map = dict(
            auto=[self.eos_token_id] + self.extra_auto_stops,
            image=[self.eos_token_id],
            recaption=[
                self.end_recaption_token_id,
                self.end_answer_token_id,
                self.eos_token_id,
            ],
            think=[
                self.end_think_token_id,
                self.end_answer_token_id,
                self.eos_token_id,
            ],
            img_ratio=[self.boi_token_id],
        )

    def get_stop_tokens(self, mode: str) -> list[int]:
        assert mode in self.stop_token_ids_map, f"Invalid mode: {mode}"
        return self.stop_token_ids_map[mode]

    def get_cot_prefix_tokens(self, mode: str) -> str:
        assert mode in self.cot_prefix_tokens, f"Invalid mode: {mode}"
        return self.cot_prefix_tokens[mode]


@dataclass
class HunyuanImage3ModelConfig:
    multi_moda_path: str = ""
    image_base_size: int = 1024
    gen_image_sequence_template: str = "pretrain"
    token_cfg: HunyuanImage3TokenConfig | None = None
    reso_group: Any | None = None


class HunyuanImage3SizeParser:
    """
    Parses a string in the form "{height}x{width}" and provides validation
    and 16-pixel upward-alignment utilities.
    """

    def __init__(self, size_str: str):
        """
        :param size_str: A string like "1080*1920"
        :raises ValueError: If the string is not in the expected format
        """
        try:
            h_str, w_str = size_str.strip().split("x")
            self.original_height = int(h_str)
            self.original_width = int(w_str)
        except Exception as e:
            raise ValueError(f'Invalid size string "{size_str}": {e}') from e

        # Values after 16-pixel upward alignment
        self.aligned_height = self._ceil16(self.original_height)
        self.aligned_width = self._ceil16(self.original_width)

    def is_valid(self) -> bool:
        """
        Returns True only if both original width and height
        are within the inclusive range [512, 2048].
        """
        return 512 <= self.original_width <= 2048 and 512 <= self.original_height <= 2048

    def get_aligned_size(self):
        """
        Returns a tuple:
        (aligned_height, aligned_width)
        """
        return (self.aligned_height, self.aligned_width)

    def get_origin_size(self):
        """
        Returns a tuple:
        (original_height, original_width)
        """
        return (self.original_height, self.original_width)

    @staticmethod
    def _ceil16(x: int) -> int:
        """Rounds an integer up to the nearest multiple of 16."""
        return (x + 15) // 16 * 16

    def __eq__(self, other) -> bool:
        if not isinstance(other, HunyuanImage3SizeParser):
            return False
        return (self.aligned_height, self.aligned_width) == (
            other.aligned_height,
            other.aligned_width,
        )

    def __hash__(self) -> int:
        return hash((self.aligned_height, self.aligned_width))

    # ---------------- Debugging aid ----------------
    def __repr__(self):
        return (
            f"SizeParser(original={self.original_height}*{self.original_width}, "
            f"aligned={self.aligned_height}*{self.aligned_width})"
        )


@contextmanager
def suppress_transformers_unused_weight_log():
    """
    Suppress only the 'Some weights of the model checkpoint at ... were not used ...'
    INFO log from transformers, auto-restore afterwards.
    """
    logger = logging.getLogger("transformers.modeling_utils")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(old_level)


class ModelProvider:
    @staticmethod
    def ext_forward(self, *args, **kwargs):
        assert len(args) == 0, "args should be empty"
        hidden_states = kwargs.pop("inputs_embeds")
        attention_mask = kwargs.pop("attention_mask")
        position_ids = kwargs.pop("position_ids")
        custom_pos_emb = kwargs.pop("custom_pos_emb")
        first_step = kwargs.pop("first_step")
        ext_forward_agent = kwargs.pop("ext_forward_agent")
        assert ext_forward_agent is not None, "ext_forward_agent should not be None"
        hidden_states = ext_forward_agent(hidden_states, position_ids, attention_mask, custom_pos_emb, first_step)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @staticmethod
    def _create_selective_device_map():
        device_map = {}

        # call aphrodite transformer_blocks, filter hf weights loading.
        device_map["model.layers"] = "cpu"

        # others keep on cuda:0
        device_map["vae"] = "cuda:0"
        device_map["vision_model"] = "cuda:0"
        device_map["vision_aligner"] = "cuda:0"
        device_map["timestep_emb"] = "cuda:0"
        device_map["patch_embed"] = "cuda:0"
        device_map["time_embed"] = "cuda:0"
        device_map["time_embed_2"] = "cuda:0"
        device_map["final_layer.model"] = "cuda:0"
        device_map["model.wte"] = "cuda:0"
        device_map["model.ln_f"] = "cuda:0"
        device_map["lm_head"] = "cuda:0"

        return device_map

    @staticmethod
    def load_multi_moda(path, infer_mode="aphrodite") -> tuple[str, Any]:
        """
        infer_mode: aphrodite or hf.
        """
        if infer_mode == "aphrodite":
            device_map = ModelProvider._create_selective_device_map()
        else:
            device_map = "auto"
        kwargs = dict(
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map=device_map,
            moe_impl="eager",
            low_cpu_mem_usage=True,
        )

        with suppress_transformers_unused_weight_log():
            model = HunyuanImage3ForCausalMM.from_pretrained(path, **kwargs)
        model.load_tokenizer(path)
        print("layer device_map:")
        print(model.hf_device_map)

        return model, model.tokenizer


@contextmanager
def ext_forward_context(model, ext_forward_agent):
    origin_forward = model.forward
    model.forward = partial(ModelProvider.ext_forward, self=model, ext_forward_agent=ext_forward_agent)
    yield
    model.forward = origin_forward


@dataclass
class HunyuanImage3RequestMeta:
    prompt: list[int]
    seed: int
    raw_conversation: list[dict]
    task_extra_kwargs: dict[str, Any]
    start_time: float
    sys_prompt: str = ""
    user_prompt: str = ""
    image_size: str = ""
    bot_task: str = ""
    cot: str = ""
    tokens: list[int] = field(default_factory=list)


class RequestRepository:
    def __init__(self):
        self._store: dict[str, HunyuanImage3RequestMeta] = {}

    # ---------- CRUD ----------
    def add(self, rid: str, **kw) -> None:
        self._store[rid] = HunyuanImage3RequestMeta(start_time=time.time(), **kw)

    def get(self, rid: str) -> HunyuanImage3RequestMeta:
        return self._store[rid]

    def pop(self, rid: str) -> HunyuanImage3RequestMeta:
        return self._store.pop(rid)

    def exists(self, rid: str) -> bool:
        return rid in self._store


class PromptStrategy(ABC):
    @abstractmethod
    def make_prompt(
        self,
        raw_conv: list[dict],
        output_tokens: list[int],
        req_meta: "HunyuanImage3RequestMeta",
    ) -> tuple[str, str]:
        """return (system_prompt, user_prompt, cot_txt, image_size)"""

    def allowed_token_ids(self) -> list[int]:
        return []

    def stop_token_ids(self) -> list[int]:
        return []


class CotStrategy(PromptStrategy):
    def __init__(self, tokenizer, cfg: "HunyuanImage3ModelConfig", mode="recaption"):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.token_cfg = cfg.token_cfg
        self.mode = mode
        if mode == "recaption":
            self.stop_id = self.token_cfg.end_recaption_token_id
        elif mode == "think":
            self.stop_id = self.token_cfg.end_think_token_id
        self.start_txt = self.token_cfg.get_cot_prefix_tokens(mode)

    def make_prompt(self, raw_conv, output_tokens, req_meta):
        user_ctx = next(m["content"] for m in raw_conv if m["role"] == "user")
        sys_ctx = next(m["content"] for m in raw_conv if m["role"] == "system")
        if output_tokens[-1] != self.stop_id:
            output_tokens = output_tokens[:-1] + [self.stop_id]
        cot = self.start_txt + self.tokenizer.decode(output_tokens, skip_special_tokens=False)

        return sys_ctx, user_ctx, cot, req_meta.image_size

    def stop_token_ids(self):
        return self.token_cfg.get_stop_tokens(self.mode)


class AutoShapeStrategy(PromptStrategy):
    def __init__(self, tokenizer, cfg: "HunyuanImage3ModelConfig", reso_group):
        self.tokenizer = tokenizer
        self.token_cfg = cfg.token_cfg
        self.reso_group = reso_group
        self._allowed_token_ids = [self.tokenizer.special_token_map[f"<img_ratio_{i}>"] for i in range(33)]

    def make_prompt(self, raw_conv, output_tokens, req_meta):
        sys_ctx = next(m["content"] for m in raw_conv if m["role"] == "system")
        user_ctx = next(m["content"] for m in raw_conv if m["role"] == "user")
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
        ratio_idx = int(re.search(r"<img_ratio_(\d+)>", text).group(1))
        w, h = self.reso_group.data[ratio_idx].w, self.reso_group.data[ratio_idx].h
        image_size = f"{h}x{w}"
        return sys_ctx, user_ctx, "", image_size

    def allowed_token_ids(self):
        return self._allowed_token_ids

    def stop_token_ids(self):
        return self.token_cfg.get_stop_tokens("auto")


class PlainStrategy(PromptStrategy):
    def __init__(self, cfg: "HunyuanImage3ModelConfig"):
        self.token_cfg = cfg.token_cfg

    def make_prompt(self, raw_conv, output_tokens, req_meta):
        sys_cotx = next(m["content"] for m in raw_conv if m["role"] == "system")
        user_ctx = next(m["content"] for m in raw_conv if m["role"] == "user")
        return sys_cotx, user_ctx, "", req_meta.image_size

    def stop_token_ids(self):
        return self.token_cfg.get_stop_tokens("image")


class RatioStrategy(PromptStrategy):
    def __init__(self, cfg: "HunyuanImage3ModelConfig", reso_group):
        self.token_cfg = cfg.token_cfg
        self.reso_group = reso_group

    def make_prompt(self, raw_conv, output_tokens, req_meta):
        sys_ctx = next(m["content"] for m in raw_conv if m["role"] == "system")
        user_ctx = next(m["content"] for m in raw_conv if m["role"] == "user")
        # Attempt to convert image_size to integer, default to 16 if conversion fails
        try:
            ratio_idx = int(req_meta.image_size)
        except (ValueError, TypeError) as e:
            # Handle cases where image_size is not convertible to integer
            # Example: image_size is None, string, or invalid format
            ratio_idx = 16
            print(f"Failed to convert image_size to integer, using default value 16. Error: {e}")
        except AttributeError:
            # Handle case where image_size attribute does not exist
            ratio_idx = 16
            print("image_size attribute not found in req_meta, using default value 16")

        w, h = self.reso_group.data[ratio_idx].w, self.reso_group.data[ratio_idx].h
        image_size = f"{h}x{w}"
        return sys_ctx, user_ctx, "", image_size

    def stop_token_ids(self):
        return self.token_cfg.get_stop_tokens("img_ratio")


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


# @dataclass
# class CustomForwardAgentContext:
#     shared_tensor: torch.Tensor


class ImageGenerationEngine:
    def __init__(
        self,
        multi_moda_model,
        cfg: "HunyuanImage3ModelConfig",
        save_dir: str,
        custom_forward_agent: Callable | None = None,
        custom_forward_agent_context: Callable | None = None,
    ):
        self.model = multi_moda_model
        self.cfg = cfg
        self._custom_forward_agent = custom_forward_agent
        self._custom_forward_agent_context = custom_forward_agent_context
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def generate_image(self, gen_image_output, meta: "HunyuanImage3RequestMeta") -> list:
        sys_prompt = meta.sys_prompt
        sys_prompt_type = meta.task_extra_kwargs.pop("use_system_prompt", None)
        user_prompt = meta.user_prompt
        cot = meta.cot
        seed = meta.seed

        size_parser = HunyuanImage3SizeParser(meta.image_size)
        aligned_height, aligned_width = size_parser.get_aligned_size()
        origin_image_size = size_parser.get_origin_size()
        image_size = f"{aligned_height}x{aligned_width}"
        image_info = self.model.image_processor.build_image_info(image_size)
        image_size = f"{image_info.h}x{image_info.w}"
        num_image_tokens = (
            image_info.image_token_length
            + (1 if image_info.add_timestep_token else 0)
            + (1 if image_info.add_guidance_token else 0)
        )

        extra_kwargs = meta.task_extra_kwargs
        save_path = os.path.join(self.save_dir, f"{gen_image_output.request_id}.png")

        try:
            image = self._generate_core(
                sys_prompt,
                sys_prompt_type,
                user_prompt,
                cot,
                seed,
                image_size,
                num_image_tokens,
                extra_kwargs,
            )
        except Exception as e:
            print(f"Image generation failed: {e}")
            image = None

        if image is None:
            return gen_image_output
        self.save_batch_image([image], [save_path], origin_image_sizes=[origin_image_size])
        gen_image_output.image = save_path
        return gen_image_output

    @torch.inference_mode()
    def _generate_core(
        self,
        sys_prompt: str,
        sys_prompt_type: str,
        user_prompt: str,
        cot: str,
        seed: int,
        image_size: str,
        num_image_tokens: int,
        extra_kwargs: dict,
    ) -> str:
        start = time.time()

        ext_forward_agent = partial(self._custom_forward_agent, num_image_tokens)

        with ext_forward_context(self.model.model, ext_forward_agent):
            image = self.model.generate_image(
                user_prompt,
                seed=seed,
                image_size=image_size,
                use_system_prompt=sys_prompt_type,
                system_prompt=sys_prompt,
                bot_task="image",
                output_type="np",
                cot_text=cot,
                verbose=1,
                **extra_kwargs,
            )

        logger.info("[ImageEngine] finished in %.2fs", time.time() - start)
        return image

    @staticmethod
    def save_batch_image(images, save_names, origin_image_sizes=None):
        """
        images can be:
        * torch.Tensor, shape (B, C, H, W)
        * numpy.ndarray, values in interval [0, 1], shape (H, W, C) or (B, H, W, C)
        * a list of PIL.Image.
        """
        if len(images) != len(save_names):
            raise ValueError(
                f"Length of images ({len(images)}) should be equal to length of save_names ({len(save_names)})."
            )

        if isinstance(images, torch.Tensor):
            # Tensor -> numpy.ndarray
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        if isinstance(images, np.ndarray):
            # numpy.ndarray -> PIL.Image
            if images.ndim == 3:
                images = images[None, ...]
            images = (images * 255).round().astype("uint8")
            if images.shape[-1] == 1:
                # special case for grayscale (single channel) images
                images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in images]
            else:
                images = [PIL.Image.fromarray(image) for image in images]

        if origin_image_sizes is not None:
            resized_images = []
            for img, (h, w) in zip(images, origin_image_sizes):
                # PIL resize order is (width, height)
                resized_images.append(img.resize((w, h), PIL.Image.BICUBIC))
            images = resized_images

        for i, (image, save_name) in enumerate(zip(images, save_names)):
            image.save(safe_file(save_name))


class HunyuanImage3Orchestrator:
    def __init__(
        self,
        aphrodite_config,
        custom_forward_agent: Callable | None = None,
        custom_forward_agent_context: Callable | None = None,
    ):
        """
        Initialize the Hunyuan-Image3 Orchestrator with the given configuration.

        Args:
            aphrodite_config: Configuration object for the Aphrodite engine.
        """
        model_path = aphrodite_config.model_config.model
        self.model, self.tokenizer = ModelProvider.load_multi_moda(model_path)
        gen_image_sequence_template = self.model.generation_config.sequence_template
        image_base_size = self.model.config.image_base_size
        print(f"[HunyuanImage3]:image_base_size: {image_base_size}, gen_image_template: {gen_image_sequence_template}")
        self.token_cfg = HunyuanImage3TokenConfig(self.tokenizer)
        self.reso_group = self.model.image_processor.reso_group
        self.cfg = HunyuanImage3ModelConfig(
            multi_moda_path=model_path,
            image_base_size=image_base_size,
            gen_image_sequence_template=gen_image_sequence_template,
            token_cfg=self.token_cfg,
            reso_group=self.reso_group,
        )
        self.repo = RequestRepository()

        self.engine = ImageGenerationEngine(
            multi_moda_model=self.model,
            cfg=self.cfg,
            save_dir=os.getenv("MULTI_MODA_SAVE_PATH", "/tmp/hunyuanimage3/png/"),
            custom_forward_agent=custom_forward_agent,
            custom_forward_agent_context=custom_forward_agent_context,
        )

        self._strategies = {
            "image": PlainStrategy(self.cfg),
            "auto": AutoShapeStrategy(self.tokenizer, self.cfg, self.reso_group),
            "img_ratio": RatioStrategy(self.cfg, self.reso_group),
            "recaption": CotStrategy(self.tokenizer, self.cfg, mode="recaption"),
            "think": CotStrategy(self.tokenizer, self.cfg, mode="think"),
        }

    def on_new_requests(self, scheduled_new_reqs):
        for req in scheduled_new_reqs:
            if req.task_type not in ["t2i"]:
                continue

            image_size = f"{self.cfg.image_base_size}x{self.cfg.image_base_size}"
            image_size = req.task_extra_kwargs.pop("image_size", image_size)
            bot_task = req.task_extra_kwargs.pop("bot_task", "image")
            if bot_task not in self._strategies:
                # invalid 'bot_task', fallback to 'image'
                bot_task = "image"

            rid = req.req_id if hasattr(req, "req_id") else req.request_id
            self.repo.add(
                rid=rid,
                prompt=req.prompt_token_ids,
                seed=req.sampling_params.seed or 0,
                raw_conversation=req.raw_conversation,
                bot_task=bot_task,
                image_size=image_size,
                task_extra_kwargs=req.task_extra_kwargs,
            )

    def on_step_outputs(self, outputs) -> None:
        for out in outputs:
            if not self.repo.exists(out.request_id):
                continue
            if not out.finished:
                continue
            meta = self.repo.get(out.request_id)
            meta.tokens = out.new_token_ids
            strategy = self._strategies[meta.bot_task]
            sys_prompt, user_prompt, cot, size = strategy.make_prompt(meta.raw_conversation, meta.tokens, meta)
            meta.sys_prompt = sys_prompt
            meta.user_prompt = user_prompt
            meta.cot = cot
            meta.image_size = size
            # one-by-one generate image
            self.engine.generate_image(out, meta)

    def allowed_token_ids(self, task_type, task_extra_kwargs):
        if task_type not in ["t2i"]:
            return []
        bot_task = task_extra_kwargs["bot_task"]
        if bot_task not in self._strategies:
            # invalid 'bot_task', fallback to 'image'
            bot_task = "image"
        return self._strategies[bot_task].allowed_token_ids()

    def stop_token_ids(self, task_type, task_extra_kwargs):
        if task_type not in ["t2i"]:
            return []
        bot_task = task_extra_kwargs["bot_task"]
        if bot_task not in self._strategies:
            # invalid 'bot_task', fallback to 'image'
            bot_task = "image"
        return self._strategies[bot_task].stop_token_ids()

    def skip_txt_infer(self, task_type, task_extra_kwargs):
        if task_type not in ["t2i"]:
            return False
        bot_task = task_extra_kwargs["bot_task"]
        if bot_task not in self._strategies:
            # invalid 'bot_task', fallback to 'image'
            bot_task = "image"
        return bot_task in ["image", "img_ratio"]


class HunyuanImage3AphroditeEngineCoreAgent:
    @staticmethod
    def _preprocess_wrapper(
        self_obj,
        *args,
        **kwargs,
    ) -> tuple[
        tuple[
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor],
            bool,
        ],
        dict[str, Any],
    ]:
        """
        Returns
        -------
        ( (num_image_tokens, flat_len, pos_ids_cpu, attn_mask_cpu,
        (cos_cpu, sin_cpu), first_step),
        kwargs )
        """
        # Argument resolution: prefer keyword arguments, fall back to positional ones.
        try:
            num_image_tokens = kwargs.pop("num_image_tokens", None) or args[0]
            hidden_states = kwargs.pop("hidden_states", None) or args[1]
            positions = kwargs.pop("positions", None) or args[2]
            attention_mask = kwargs.pop("attention_mask", None) or args[3]
            custom_pos_emb = kwargs.pop("custom_pos_emb", None) or args[4]
            first_step = kwargs.pop("first_step", None) or args[5]
        except IndexError as e:
            raise ValueError("Missing required argument") from e
        hidden_states = hidden_states.view(-1, hidden_states.size(-1)).contiguous()
        positions = positions.contiguous()
        attention_mask = attention_mask.contiguous()
        cos, sin = custom_pos_emb
        custom_pos_emb = (cos.contiguous(), sin.contiguous())

        # Call synchronize to ensure pre-task completion before dispatching to sub-workers.
        torch.cuda.synchronize()
        src_rank = hidden_states.device.index
        meta_tuple = (num_image_tokens, first_step, src_rank)
        tensor_tuple = (
            hidden_states,
            attention_mask,
            custom_pos_emb,
        )
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor_tuple)
        serialized_obj = buf.getvalue()
        kwargs["serialized_obj"] = serialized_obj
        # print(f"core: serialized_obj: {len(serialized_obj)}")
        return meta_tuple, kwargs

    @staticmethod
    def _postprocess_wrapper(
        self_obj,
        output,
        *args,
        **kwargs,
    ) -> list[torch.Tensor]:
        hidden_states = args[1] if len(args) > 1 else kwargs["hidden_states"]
        B, S, H = hidden_states.size()
        hidden_states = output.view(B, S, H)
        return hidden_states


def register_hunyuan_image3_engine_core_hooks(engine, aphrodite_config):
    """把 hunyuan-image3 的 hooks 注册到 EngineCore"""
    ctx = None
    custom_foward_agent = partial(
        engine.custom_execute_model,
        preprocess_fn=partial(HunyuanImage3AphroditeEngineCoreAgent._preprocess_wrapper, engine),
        postprocess_fn=partial(HunyuanImage3AphroditeEngineCoreAgent._postprocess_wrapper, engine),
        task_type="t2i",
    )
    orch = HunyuanImage3Orchestrator(
        aphrodite_config=aphrodite_config,
        custom_forward_agent=custom_foward_agent,
        custom_forward_agent_context=ctx,
    )

    engine.register_hook("on_new_requests", orch.on_new_requests)
    engine.register_hook("on_step_outputs", orch.on_step_outputs)
    engine.task_allowed_token_ids["t2i"] = orch.allowed_token_ids
    engine.task_stop_token_ids["t2i"] = orch.stop_token_ids
    engine.task_skip_inference["t2i"] = orch.skip_txt_infer


class HunyuanImage3ModelRunnerTaskCallable:
    @staticmethod
    def _preprocess_wrapper(self_obj, *args, **kwargs):
        try:
            num_image_tokens = kwargs.pop("num_image_tokens", None) or args[0]
            first_step = kwargs.pop("first_step", None) or args[1]
            src_rank = kwargs.pop("src_rank", None) or args[2]
            serialized_obj = kwargs.pop("serialized_obj", None) or args[3]
        except IndexError as e:
            raise ValueError("Missing required argument") from e

        curr_rank = get_tensor_model_parallel_rank()
        if curr_rank == src_rank:
            tensor_tuple = pickle.loads(serialized_obj)
            hidden_states, attention_mask, custom_pos_emb = tensor_tuple
            tensor_dict = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "cos": custom_pos_emb[0],
                "sin": custom_pos_emb[1],
            }
            broadcast_tensor_dict(tensor_dict, src_rank)
        else:
            tensor_dict = broadcast_tensor_dict(src=src_rank)

        hidden_states = tensor_dict["hidden_states"]
        attention_mask = tensor_dict["attention_mask"]
        custom_pos_emb = (tensor_dict["cos"], tensor_dict["sin"])

        # torch.distributed.barrier()
        attn_meta = create_hunyuan_image_attention_meta(
            attention_mask, num_image_tokens=num_image_tokens, first_step=first_step
        )

        # rm 'num_image_tokens'
        exec_args = (hidden_states, attention_mask, custom_pos_emb)
        return attn_meta, exec_args, kwargs

    @staticmethod
    def _postprocess_wrapper(self_obj, output, *args, **kwargs):
        return output


def register_hunyuan_image3_modelrunner_task_callable(model_runner):
    model_runner.register_task_callables(
        task_type="t2i",
        preprocess_fn=partial(HunyuanImage3ModelRunnerTaskCallable._preprocess_wrapper, model_runner),
        postprocess_fn=partial(HunyuanImage3ModelRunnerTaskCallable._postprocess_wrapper, model_runner),
        custom_forward_method="forward_block",
    )
