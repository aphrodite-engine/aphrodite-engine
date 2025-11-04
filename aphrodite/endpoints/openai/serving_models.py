import gc
import time
from argparse import Namespace
from asyncio import Lock
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from http import HTTPStatus

from aphrodite import envs
from aphrodite.endpoints.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    LoadLoRAAdapterRequest,
    ModelCard,
    ModelList,
    ModelPermission,
    UnloadLoRAAdapterRequest,
)
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.protocol import EngineClient
from aphrodite.logger import init_logger
from aphrodite.lora.request import LoRARequest
from aphrodite.lora.resolver import LoRAResolver, LoRAResolverRegistry
from aphrodite.modeling.model_loader.weight_utils import get_model_config_yaml
from aphrodite.usage.usage_lib import UsageContext
from aphrodite.utils.counter import AtomicCounter

logger = init_logger(__name__)


@dataclass
class BaseModelPath:
    name: str
    model_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: str | None = None


class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    - /v1/load_lora_adapter
    - /v1/unload_lora_adapter
    """

    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None,
    ):
        super().__init__()

        self.engine_client = engine_client
        self.base_model_paths = base_model_paths

        self.static_lora_modules = lora_modules
        self.lora_requests: dict[str, LoRARequest] = {}
        self.lora_id_counter = AtomicCounter(0)

        self.lora_resolvers: list[LoRAResolver] = []
        for lora_resolver_name in LoRAResolverRegistry.get_supported_resolvers():
            self.lora_resolvers.append(LoRAResolverRegistry.get_resolver(lora_resolver_name))
        self.lora_resolver_lock: dict[str, Lock] = defaultdict(Lock)

        self.processor = self.engine_client.processor
        self.io_processor = self.engine_client.io_processor
        self.model_config = self.engine_client.model_config
        self.max_model_len = self.model_config.max_model_len

    async def init_static_loras(self):
        """Loads all static LoRA modules.
        Raises if any fail to load"""
        if self.static_lora_modules is None:
            return
        for lora in self.static_lora_modules:
            load_request = LoadLoRAAdapterRequest(lora_path=lora.path, lora_name=lora.name)
            load_result = await self.load_lora_adapter(request=load_request, base_model_name=lora.base_model_name)
            if isinstance(load_result, ErrorResponse):
                raise ValueError(load_result.error.message)

    def is_base_model(self, model_name) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        """Returns the appropriate model name depending on the availability
        and support of the LoRA or base model.
        Parameters:
        - lora: LoRARequest that contain a base_model_name.
        Returns:
        - str: The name of the base model or the first available model path.
        """
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        """Show available models. This includes the base model and all
        adapters"""
        model_cards = [
            ModelCard(
                id=base_model.name,
                max_model_len=self.max_model_len,
                root=base_model.model_path,
                permission=[ModelPermission()],
            )
            for base_model in self.base_model_paths
        ]
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.local_path,
                parent=lora.base_model_name if lora.base_model_name else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    async def load_lora_adapter(
        self, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_load_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            lora_path = request.lora_path
            unique_id = self.lora_id_counter.inc(1)
            lora_request = LoRARequest(lora_name=lora_name, lora_int_id=unique_id, lora_path=lora_path)
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also pre-load it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(e):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND

                return create_error_response(message=str(e), err_type=error_type, status_code=status_code)

            self.lora_requests[lora_name] = lora_request
            logger.info("Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path)
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def unload_lora_adapter(self, request: UnloadLoRAAdapterRequest) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_unload_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            # Safe to delete now since we hold the lock
            del self.lora_requests[lora_name]
            logger.info("Removed LoRA adapter: name '%s'", lora_name)
            return f"Success: LoRA adapter '{lora_name}' removed successfully."

    async def _check_load_lora_adapter_request(self, request: LoadLoRAAdapterRequest) -> ErrorResponse | None:
        # Check if both 'lora_name' and 'lora_path' are provided
        if not request.lora_name or not request.lora_path:
            return create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Check if the lora adapter with the given name already exists
        if request.lora_name in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' has already been loaded.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        return None

    async def _check_unload_lora_adapter_request(self, request: UnloadLoRAAdapterRequest) -> ErrorResponse | None:
        # Check if 'lora_name' is not provided return an error
        if not request.lora_name:
            return create_error_response(
                message="'lora_name' needs to be provided to unload a LoRA adapter.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Check if the lora adapter with the given name exists
        if request.lora_name not in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' cannot be found.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        return None

    async def resolve_lora(self, lora_name: str) -> LoRARequest | ErrorResponse:
        """Attempt to resolve a LoRA adapter using available resolvers.

        Args:
            lora_name: Name/identifier of the LoRA adapter

        Returns:
            LoRARequest if found and loaded successfully.
            ErrorResponse (404) if no resolver finds the adapter.
            ErrorResponse (400) if adapter(s) are found but none load.
        """
        async with self.lora_resolver_lock[lora_name]:
            # First check if this LoRA is already loaded
            if lora_name in self.lora_requests:
                return self.lora_requests[lora_name]

            base_model_name = self.model_config.model
            unique_id = self.lora_id_counter.inc(1)
            found_adapter = False

            # Try to resolve using available resolvers
            for resolver in self.lora_resolvers:
                lora_request = await resolver.resolve_lora(base_model_name, lora_name)

                if lora_request is not None:
                    found_adapter = True
                    lora_request.lora_int_id = unique_id

                    try:
                        await self.engine_client.add_lora(lora_request)
                        self.lora_requests[lora_name] = lora_request
                        logger.info(
                            "Resolved and loaded LoRA adapter '%s' using %s",
                            lora_name,
                            resolver.__class__.__name__,
                        )
                        return lora_request
                    except BaseException as e:
                        logger.warning(
                            "Failed to load LoRA '%s' resolved by %s: %s. Trying next resolver.",
                            lora_name,
                            resolver.__class__.__name__,
                            e,
                        )
                        continue

            if found_adapter:
                # An adapter was found, but all attempts to load it failed.
                return create_error_response(
                    message=(f"LoRA adapter '{lora_name}' was found but could not be loaded."),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            else:
                # No adapter was found
                return create_error_response(
                    message=f"LoRA adapter {lora_name} does not exist",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )

    async def unload_model(self, old_engine_client: EngineClient) -> dict[str, str | float]:
        """Unload the current model and free GPU memory.

        Returns a dict with status information including timing metrics.
        """
        start_time = time.time()

        logger.info("Model unload requested - waiting for in-flight requests to drain...")

        try:
            await old_engine_client.wait_for_requests_to_drain(drain_timeout=300)
            drain_time = time.time() - start_time
            logger.info("All requests drained in %.2fs. Shutting down engine.", drain_time)
        except TimeoutError:
            drain_time = time.time() - start_time
            logger.warning(
                "Timeout waiting for requests to drain after %.2fs. Proceeding with shutdown anyway...",
                drain_time,
            )

        # Mark engine as already dead to prevent monitor from triggering API server shutdown
        if hasattr(old_engine_client, "engine_core") and hasattr(old_engine_client.engine_core, "resources"):
            old_engine_client.engine_core.resources.engine_dead = True

        shutdown_start = time.time()
        old_engine_client.shutdown()

        import torch.cuda

        gc.collect()
        torch.cuda.empty_cache()

        shutdown_time = time.time() - shutdown_start
        total_time = time.time() - start_time
        logger.info(
            "Engine shutdown complete in %.2fs. Total unload time: %.2fs. GPU memory freed.",
            shutdown_time,
            total_time,
        )

        return {
            "status": "success",
            "message": f"Model unloaded successfully in {total_time:.2f}s. All GPU memory has been freed.",
            "drain_time_s": round(drain_time, 2),
            "shutdown_time_s": round(shutdown_time, 2),
            "total_time_s": round(total_time, 2),
        }

    async def load_model(
        self,
        original_args: Namespace,
        model: str | None = None,
        config_data: dict | None = None,
    ) -> tuple[EngineClient, Namespace, dict[str, str | float | dict]]:
        """Load a model with optional config overrides.

        Args:
            original_args: The original server startup arguments
            model: Optional model name/path to load (different from original)
            config_data: Optional config dict to override settings

        Returns:
            Tuple of (new_engine_client, updated_args, response_data)
        """
        start_time = time.time()

        # Start with the ORIGINAL server startup args
        args = deepcopy(original_args)
        config_applied = {}

        # If a different model is specified, update it
        if model is not None:
            old_model = args.model
            args.model = model
            logger.info("Switching model from %s to %s", old_model, model)
            config_applied["model"] = {"old": old_model, "new": model, "source": "request"}

            if envs.APHRODITE_ENABLE_MULTI_MODEL:
                # Clear potentially conflicting config parameters when loading a different model
                # These will be re-set by the model's config if needed
                if hasattr(args, "cudagraph_capture_sizes"):
                    args.cudagraph_capture_sizes = None
                if (
                    hasattr(args, "compilation_config")
                    and args.compilation_config
                    and hasattr(args.compilation_config, "cudagraph_capture_sizes")
                ):
                    args.compilation_config.cudagraph_capture_sizes = None

        # Only auto-load aphrodite_config.yaml from model directory if NO explicit config was provided
        if config_data is None:
            model_config_yaml = get_model_config_yaml(args.model, getattr(args, "download_dir", None))

            if model_config_yaml:
                logger.info("Found aphrodite_config in model directory with %d settings", len(model_config_yaml))
                for key, value in model_config_yaml.items():
                    attr_name = key.replace("-", "_")
                    # Don't override the model path if it was explicitly provided in request
                    if attr_name == "model" and model is not None:
                        continue
                    if hasattr(args, attr_name):
                        old_value = getattr(args, attr_name)
                        setattr(args, attr_name, value)
                        config_applied[key] = {"old": old_value, "new": value, "source": "model_dir"}
                        logger.info("Config from model dir: %s = %s (was: %s)", key, value, old_value)
                    else:
                        logger.warning("Unknown config key in model directory: %s - ignoring", key)

        # If config was provided, apply it (this overrides everything)
        if config_data is not None:
            logger.info("Config data provided - applying...")

            if not isinstance(config_data, dict):
                raise ValueError("Config must be a dictionary/object")

            # Apply config values to args
            for key, value in config_data.items():
                attr_name = key.replace("-", "_")

                if hasattr(args, attr_name):
                    old_value = getattr(args, attr_name)
                    setattr(args, attr_name, value)
                    config_applied[key] = {"old": old_value, "new": value, "source": "uploaded"}
                    logger.info("Config override (uploaded): %s = %s (was: %s)", key, value, old_value)
                else:
                    logger.warning("Unknown config key: %s - ignoring", key)

            logger.info("Applied %d config overrides", len(config_applied))

        logger.info("Model load requested - initializing engine...")

        from aphrodite.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs.from_cli_args(args)
        aphrodite_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)

        new_client = AsyncLLM.from_aphrodite_config(
            aphrodite_config=aphrodite_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
        )

        total_time = time.time() - start_time
        logger.info("Model load complete in %.2fs!", total_time)

        response_data = {
            "status": "success",
            "message": f"Model loaded successfully in {total_time:.2f}s.",
            "load_time_s": round(total_time, 2),
            "model": args.model,
        }

        if config_applied:
            response_data["config_applied"] = {
                key: {"value": value["new"], "source": value["source"]} for key, value in config_applied.items()
            }

        return new_client, args, response_data


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    return ErrorResponse(error=ErrorInfo(message=message, type=err_type, code=status_code.value))
