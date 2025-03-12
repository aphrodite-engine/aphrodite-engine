# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import json
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from openai.types.chat import ChatCompletionContentPartParam
from pydantic import (AliasChoices, BaseModel, ConfigDict, Field,
                      model_validator)
from transformers import PreTrainedTokenizer
from typing_extensions import Annotated, Required, TypedDict

from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import (GuidedDecodingParams,
                                              RequestOutputKind,
                                              SamplingParams)
from aphrodite.common.sequence import Logprob
from aphrodite.common.utils import random_uuid
from aphrodite.endpoints.chat_utils import ChatCompletionMessageParam


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""
    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""
    name: str
    """An optional name for the participant.
    Provides the model information to differentiate between participants of the
    same role.
    """
    tool_call_id: Optional[str]
    tool_calls: Optional[List[dict]]


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pygmalionai"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: Optional[UsageInfo] = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None,
                                ge=torch.iinfo(torch.long).min,
                                le=torch.iinfo(torch.long).max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"], Literal["auto"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    # NOTE this will be ignored by Aphrodite - the model determines the behavior
    parallel_tool_calls: Optional[bool] = False
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = False
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    top_a: Optional[float] = 0.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    no_repeat_ngram_size: Optional[int] = 0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    temperature_last: Optional[bool] = False
    prompt_logprobs: Optional[int] = None
    xtc_threshold: Optional[float] = 0.1
    xtc_probability: Optional[float] = 0.0
    dry_multiplier: Optional[float] = 0
    dry_base: Optional[float] = 1.75
    dry_allowed_length: Optional[int] = 2
    dry_sequence_breakers: Optional[List[str]] = Field(
        default=["\n", ":", "\"", "*"])
    dry_range: Optional[int] = Field(
        default=0,
        validation_alias=AliasChoices("dry_range",
                                      "dry_penalty_last_n"))
    dry_max_ngram: Optional[int] = 12
    dry_max_occurrences: Optional[int] = 8
    dry_early_exit_match_len: Optional[int] = 8
    dynatemp_min: Optional[float] = 0.0
    dynatemp_max: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    nsigma: Optional[float] = 0.0
    skew: Optional[float] = 0.0
    custom_token_bans: Optional[List[int]] = None
    token_ban_ranges: Optional[List[Tuple[List[int], int, int]]] = None
    sampler_priority: Optional[Union[List[int], List[str]]] = Field(
        default=[],
        validation_alias=AliasChoices("sampler_priority",
                                      "sampler_order"))
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: Optional[bool] = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: Optional[bool] = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=
        ("If this is set, the chat will be formatted so that the final "
         "message in the chat is open-ended, without any EOS tokens. The "
         "model will continue this message rather than starting a new one. "
         "This allows you to \"prefill\" part of the model's response for it. "
         "Cannot be used at the same time as `add_generation_prompt`."),
    )
    add_special_tokens: Optional[bool] = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to False (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    include_stop_str_in_output: Optional[bool] = Field(
        default=False,
        description=(
            "Whether to include the stop string in the output. "
            "This is only applied when the stop or stop_token_ids is set."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))

    # doc: end-chat-completion-extra-params

    def to_sampling_params(
            self, tokenizer: PreTrainedTokenizer,
            default_max_tokens: int) -> SamplingParams:
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens

        guided_json_object = None
        if (self.response_format is not None
                and self.response_format.type == "json_object"):
            guided_json_object = True

        guided_decoding = None
        if any([self.guided_json, self.guided_regex, self.guided_choice, 
                self.guided_grammar, guided_json_object]):
            guided_decoding = GuidedDecodingParams.from_optional(
                json=self.guided_json,
                regex=self.guided_regex,
                choice=self.guided_choice,
                grammar=self.guided_grammar,
                json_object=guided_json_object,
                backend=self.guided_decoding_backend,
                whitespace_pattern=self.guided_whitespace_pattern)

        dry_sequence_breaker_ids = []
        if self.dry_sequence_breakers:
            for s in self.dry_sequence_breakers:
                token_id = tokenizer.encode(f'a{s}')[-1]
                dry_sequence_breaker_ids.append(token_id)

        return SamplingParams(
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            max_tokens=max_tokens,
            min_tokens=self.min_tokens,
            logprobs=self.top_logprobs if self.logprobs else None,
            prompt_logprobs=self.prompt_logprobs if self.prompt_logprobs else
            (self.top_logprobs if self.echo else None),
            best_of=self.best_of,
            top_k=self.top_k,
            top_a=self.top_a,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            smoothing_factor=self.smoothing_factor,
            smoothing_curve=self.smoothing_curve,
            ignore_eos=self.ignore_eos,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            temperature_last=self.temperature_last,
            xtc_threshold=self.xtc_threshold,
            xtc_probability=self.xtc_probability,
            dry_multiplier=self.dry_multiplier,
            dry_base=self.dry_base,
            dry_allowed_length=self.dry_allowed_length,
            dry_sequence_breaker_ids=dry_sequence_breaker_ids,
            dry_range=self.dry_range,
            dry_max_ngram=self.dry_max_ngram,
            dry_max_occurrences=self.dry_max_occurrences,
            dry_early_exit_match_len=self.dry_early_exit_match_len,
            dynatemp_min=self.dynatemp_min,
            dynatemp_max=self.dynatemp_max,
            dynatemp_exponent=self.dynatemp_exponent,
            nsigma=self.nsigma,
            skew=self.skew,
            custom_token_bans=self.custom_token_bans,
            token_ban_ranges=self.token_ban_ranges,
            sampler_priority=self.sampler_priority,
            output_kind=RequestOutputKind.DELTA if self.stream \
                else RequestOutputKind.FINAL_ONLY,
            guided_decoding=guided_decoding,
            logit_bias=self.logit_bias)

    def _get_guided_json_from_tool(
            self) -> Optional[Union[str, dict, BaseModel]]:
        # user has chosen to not use any tool
        if self.tool_choice == "none" or self.tools is None:
            return None
        # user has chosen to use a named tool
        if type(self.tool_choice) is ChatCompletionNamedToolChoiceParam:
            tool_name = self.tool_choice.function.name
            tools = {tool.function.name: tool.function for tool in self.tools}
            if tool_name not in tools:
                raise ValueError(
                    f"Tool '{tool_name}' has not been passed in `tools`.")
            tool = tools[tool_name]
            return tool.parameters
        return None

    @model_validator(mode='before')
    @classmethod
    def validate_stream_options(cls, values):
        if (values.get('stream_options') is not None
                and not values.get('stream')):
            raise ValueError(
                "stream_options can only be set if stream is true")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        if isinstance(data, ValueError):
            raise data
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        # you can only use one kind of guided decoding
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        # you can only either use guided decoding or tools, not both
        if guide_count > 1 and data.get("tool_choice",
                                        "none") not in ("none", "auto"):
            raise ValueError(
                "You can only either use guided decoding or tools, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):
        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"
        # if "tool_choice" is specified -- validation
        if "tool_choice" in data:
            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
            # make sure that tool choice is either a named tool
            # OR that it's set to "auto"
            if data["tool_choice"] != "auto" and not isinstance(
                    data["tool_choice"], dict):
                raise ValueError(
                    "`tool_choice` must either be a named tool or \"auto\". "
                    "`tool_choice=\"none\" is not supported.")
            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                specified_function = data["tool_choice"]["function"]
                if not specified_function:
                    raise ValueError(
                        "Incorrectly formatted `tool_choice`. Should be like "
                        "`{\"type\": \"function\","
                        " \"function\": {\"name\": \"my_function\"}}`")
                specified_function_name = specified_function["name"]
                if not specified_function_name:
                    raise ValueError(
                        "Incorrectly formatted `tool_choice`. Should be like "
                        "`{\"type\": \"function\", "
                        "\"function\": {\"name\": \"my_function\"}}`")
                for tool in data["tools"]:
                    if tool["function"]["name"] == specified_function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if "top_logprobs" in data and data["top_logprobs"] is not None:
            if "logprobs" not in data or data["logprobs"] is False:
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )
            elif data["top_logprobs"] < 0:
                raise ValueError(
                    "`top_logprobs` must be a value a positive value.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None,
                                ge=torch.iinfo(torch.long).min,
                                le=torch.iinfo(torch.long).max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: Optional[bool] = False
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    top_a: Optional[float] = 0.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    no_repeat_ngram_size: Optional[int] = 0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[List[int]] = None
    include_stop_str_in_output: Optional[bool] = False
    add_special_tokens: Optional[bool] = False
    temperature_last: Optional[bool] = False
    prompt_logprobs: Optional[int] = None
    xtc_threshold: Optional[float] = 0.1
    xtc_probability: Optional[float] = 0.0
    dry_multiplier: Optional[float] = 0
    dry_base: Optional[float] = 1.75
    dry_allowed_length: Optional[int] = 2
    dry_sequence_breakers: Optional[List[str]] = Field(
        default=["\n", ":", "\"", "*"])
    dry_range: Optional[int] = Field(
        default=0,
        validation_alias=AliasChoices("dry_range",
                                      "dry_penalty_last_n"))
    dry_max_ngram: Optional[int] = 12
    dry_max_occurrences: Optional[int] = 8
    dry_early_exit_match_len: Optional[int] = 8
    dynatemp_min: Optional[float] = 0.0
    dynatemp_max: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    nsigma: Optional[float] = 0.0
    skew: Optional[float] = 0.0
    custom_token_bans: Optional[List[int]] = None
    token_ban_ranges: Optional[List[Tuple[List[int], int, int]]] = None
    sampler_priority: Optional[Union[List[int], List[str]]] = Field(
        default=[],
        validation_alias=AliasChoices("sampler_priority",
                                      "sampler_order"))
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'json_object'} or {'type': 'text' } is "
         "supported."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))

    # doc: end-completion-extra-params

    def to_sampling_params(
            self, tokenizer: PreTrainedTokenizer,
            default_max_tokens: int) -> SamplingParams:
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens

        echo_without_generation = self.echo and self.max_tokens == 0

        guided_json_object = None
        if (self.response_format is not None
                and self.response_format.type == "json_object"):
            guided_json_object = True

        guided_decoding = None
        if any([self.guided_json, self.guided_regex, self.guided_choice, 
                self.guided_grammar, guided_json_object]):
            guided_decoding = GuidedDecodingParams.from_optional(
                json=self.guided_json,
                regex=self.guided_regex,
                choice=self.guided_choice,
                grammar=self.guided_grammar,
                json_object=guided_json_object,
                backend=self.guided_decoding_backend,
                whitespace_pattern=self.guided_whitespace_pattern)

        dry_sequence_breaker_ids = []
        if self.dry_sequence_breakers:
            for s in self.dry_sequence_breakers:
                s = bytes(s, "utf-8").decode("unicode_escape")
                token_id = tokenizer.encode(f'a{s}')[-1]
                dry_sequence_breaker_ids.append(token_id)

        return SamplingParams(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            top_a=self.top_a,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            smoothing_factor=self.smoothing_factor,
            smoothing_curve=self.smoothing_curve,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs
            if self.prompt_logprobs else self.logprobs if self.echo else None,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=(self.spaces_between_special_tokens),
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            temperature_last=self.temperature_last,
            xtc_threshold=self.xtc_threshold,
            xtc_probability=self.xtc_probability,
            dry_multiplier=self.dry_multiplier,
            dry_base=self.dry_base,
            dry_allowed_length=self.dry_allowed_length,
            dry_sequence_breaker_ids=dry_sequence_breaker_ids,
            dry_range=self.dry_range,
            dry_max_ngram=self.dry_max_ngram,
            dry_max_occurrences=self.dry_max_occurrences,
            dry_early_exit_match_len=self.dry_early_exit_match_len,
            dynatemp_min=self.dynatemp_min,
            dynatemp_max=self.dynatemp_max,
            dynatemp_exponent=self.dynatemp_exponent,
            nsigma=self.nsigma,
            skew=self.skew,
            custom_token_bans=self.custom_token_bans,
            token_ban_ranges=self.token_ban_ranges,
            sampler_priority=self.sampler_priority,
            output_kind=RequestOutputKind.DELTA if self.stream \
                else RequestOutputKind.FINAL_ONLY,
            guided_decoding=guided_decoding,
            logit_bias=self.logit_bias)

    def _get_guided_json_from_tool(
            self) -> Optional[Union[str, dict, BaseModel]]:
        # user has chosen to not use any tool
        if self.tool_choice == "none" or self.tools is None:
            return None
        # user has chosen to use a named tool
        if type(self.tool_choice) is ChatCompletionNamedToolChoiceParam:
            tool_name = self.tool_choice.function.name
            tools = {tool.function.name: tool.function for tool in self.tools}
            if tool_name not in tools:
                raise ValueError(
                    f"Tool '{tool_name}' has not been passed in `tools`.")
            tool = tools[tool_name]
            return tool.parameters
        return None

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if "logprobs" in data and data[
                "logprobs"] is not None and not data["logprobs"] >= 0:
            raise ValueError("if passed, `logprobs` must be a positive value.")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when stream is True.")
        return data

    @model_validator(mode='before')
    @classmethod
    def parse_dry_sequence_breakers(cls, data):
        if 'dry_sequence_breakers' in data:
            breakers = data['dry_sequence_breakers']
            if isinstance(breakers, str):
                try:
                    # Try to parse as JSON string
                    data['dry_sequence_breakers'] = json.loads(breakers)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON for dry_sequence_breakers:"
                                     f" {e}") from e

            # Validate that we now have a list of strings
            is_list = isinstance(data['dry_sequence_breakers'], list)
            all_strings = all(
                isinstance(x, str)
                for x in data['dry_sequence_breakers']
            )
            if not is_list or not all_strings:
                raise ValueError(
                    "dry_sequence_breakers must be a list of strings or a "
                    "JSON string representing a list of strings"
                )

        return data


class EmbeddingRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: str
    input: Union[List[int], List[List[int]], str, List[str]]
    encoding_format: Optional[str] = Field('float', pattern='^(float|base64)$')
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None

    # doc: begin-embedding-pooling-params
    additional_data: Optional[Any] = None

    # doc: end-embedding-pooling-params

    # doc: begin-embedding-extra-params
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."))
    # doc: end-embedding-extra-params

    def to_pooling_params(self):
        return PoolingParams(additional_data=self.additional_data)


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str,
                                     float]]] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[List[float], str]


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: List[EmbeddingResponseData]
    usage: UsageInfo


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    index: int
    function: Optional[DeltaFunctionCall] = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool
    # extracted tool calls
    tool_calls: List[ToolCall]
    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: List[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    # per OpenAI spec this is the default
    finish_reason: Optional[str] = "stop"
    # not part of the OpenAI spec but included in Aphrodite for legacy reasons
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.

    NOTE: Currently only the `/v1/chat/completions` endpoint is supported.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request. Currently
    # /v1/chat/completions is supported.
    url: str

    # The parameters of the request.
    body: Union[ChatCompletionRequest, EmbeddingRequest]


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Optional[Union[ChatCompletionResponse, EmbeddingResponse]] = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[BatchResponseData]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]


class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str
    prompt: str
    add_special_tokens: bool = Field(default=True)


class TokenizeChatRequest(OpenAIBaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    add_generation_prompt: bool = Field(default=True)
    continue_final_message: bool = Field(default=False)
    add_special_tokens: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get(
                "add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and "
                             "`add_generation_prompt` to True.")
        return data


TokenizeRequest = Union[TokenizeCompletionRequest, TokenizeChatRequest]


class TokenizeResponse(OpenAIBaseModel):
    tokens: List[int]
    count: int
    max_model_len: int


class DetokenizeRequest(OpenAIBaseModel):
    model: Optional[str]
    tokens: List[int]


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str


# ========== KoboldAI ========== #


class KAIGenerationInputSchema(BaseModel):
    genkey: Optional[str] = None
    prompt: str
    n: Optional[int] = 1
    max_context_length: int
    max_length: int
    rep_pen: Optional[float] = 1.0
    top_k: Optional[int] = 0
    top_a: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    tfs: Optional[float] = 1.0
    eps_cutoff: Optional[float] = 0.0
    eta_cutoff: Optional[float] = 0.0
    typical: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    dynatemp_range: Optional[float] = 0.0
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
    xtc_threshold: Optional[float] = 0.1
    xtc_probability: Optional[float] = 0.0
    use_default_badwordsids: Optional[bool] = None
    quiet: Optional[bool] = None
    # pylint: disable=unexpected-keyword-arg
    sampler_seed: Optional[int] = None
    stop_sequence: Optional[List[str]] = None
    include_stop_str_in_output: Optional[bool] = False

    @model_validator(mode='before')
    def check_context(cls, values):  # pylint: disable=no-self-argument
        assert values.get("max_length") <= values.get(
            "max_context_length"
        ), "max_length must not be larger than max_context_length"
        return values
