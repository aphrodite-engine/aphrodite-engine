import json
from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

import jinja2
from fastapi import Request

from aphrodite.config import ModelConfig
from aphrodite.endpoints.logger import RequestLogger
from aphrodite.endpoints.openai.protocol import (AnthropicContentBlockDelta,
                                                 AnthropicContentBlockStart,
                                                 AnthropicContentBlockStop,
                                                 AnthropicError,
                                                 AnthropicImageBlock,
                                                 AnthropicMessage,
                                                 AnthropicMessageDelta,
                                                 AnthropicMessagesRequest,
                                                 AnthropicMessagesResponse,
                                                 AnthropicMessageStart,
                                                 AnthropicMessageStop,
                                                 AnthropicTextBlock,
                                                 AnthropicThinkingBlock,
                                                 AnthropicToolResultBlock,
                                                 AnthropicToolUseBlock,
                                                 AnthropicUsage,
                                                 ChatCompletionMessageParam,
                                                 ChatCompletionRequest,
                                                 ErrorResponse)
from aphrodite.endpoints.openai.serving_engine import OpenAIServing
from aphrodite.endpoints.openai.serving_models import OpenAIServingModels
from aphrodite.engine.protocol import EngineClient
from aphrodite.logger import init_logger
from aphrodite.lora.request import LoRARequest
from aphrodite.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingMessages(OpenAIServing):
    """Anthropic Messages API serving class that wraps the chat completion
    functionality."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack
        )

        self.response_role = response_role
        self.chat_template = chat_template
        self._primary_model_name = (
            self.models.base_model_paths[0].name if
            self.models.base_model_paths else None
        )
        self.enable_auto_tools = enable_auto_tools
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser

        from aphrodite.endpoints.openai.serving_chat import OpenAIServingChat

        self.chat_serving = OpenAIServingChat(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            response_role=response_role,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format="string",  # Default to string format
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            reasoning_parser=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
        )

        self.default_sampling_params = \
            self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default messages sampling params from %s: %s",
                        source, self.default_sampling_params)

    async def create_message(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], AnthropicMessagesResponse,
               ErrorResponse]:
        """
        Main entry point for Anthropic Messages API.

        Converts Anthropic format to OpenAI format, delegates to chat serving,
        then converts the response back to Anthropic format.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            # Convert Anthropic request to OpenAI ChatCompletion format
            chat_request = self._convert_to_chat_request(request)

            # Delegate to the chat serving implementation
            chat_response = await self.chat_serving.create_chat_completion(
                chat_request, raw_request
            )

            # Handle streaming vs non-streaming responses
            if request.stream:
                return self._convert_streaming_response(
                    chat_response, request, raw_request
                )
            else:
                return self._convert_response(
                    chat_response, request, raw_request)

        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in Messages API processing")
            return self.create_error_response(str(e))

    async def _check_model(self, request) -> Optional[ErrorResponse]:
        """
        Override model checking to allow Claude model name aliases.
        """
        # Check if it's a Claude alias - if so, treat it as valid
        if self._is_claude_model_alias(request.model):
            return None

        # For non-Claude models, use the base implementation
        return await super()._check_model(request)

    def _is_claude_model_alias(self, model_name: Optional[str]) -> bool:
        """
        Check if the model name is a Claude alias that should be routed
        to the primary model.
        """
        if not model_name:
            return False
        return model_name.startswith("claude-")

    def _is_model_supported(self, model_name: Optional[str]) -> bool:
        """
        Override model support checking to allow Claude aliases.
        """
        if not model_name:
            return True

        # Claude aliases are always supported (routed to primary model)
        if self._is_claude_model_alias(model_name):
            return True

        # For non-Claude models, use the base implementation
        return super()._is_model_supported(model_name)

    def _get_model_name(
        self,
        model_name: Optional[str] = None,
        lora_request: Optional[LoRARequest] = None
    ) -> str:
        """
        Override model name resolution to handle Claude aliases.
        """
        if lora_request:
            return lora_request.lora_name

        if not model_name:
            return self.models.base_model_paths[0].name

        if self._is_claude_model_alias(model_name):
            return self._primary_model_name or \
                self.models.base_model_paths[0].name

        return model_name

    def _convert_to_chat_request(
        self, request: AnthropicMessagesRequest) -> ChatCompletionRequest:
        """
        Convert Anthropic Messages request to OpenAI ChatCompletion request."""

        messages = []

        if request.system:
            if isinstance(request.system, str):
                system_content = request.system
            else:
                # Convert list of text blocks to string
                system_content = "\n".join(
                    block.text for block in request.system)
            messages.append({"role": "system", "content": system_content})

        # Convert user/assistant messages
        for msg in request.messages:
            openai_msg = self._convert_message(msg)
            messages.append(openai_msg)

        # Convert tools if present
        tools = None
        if request.tools:
            tools = []
            for tool in request.tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema
                    }
                }
                tools.append(openai_tool)

        # Convert tool_choice
        tool_choice = "none"
        if request.tool_choice:
            if request.tool_choice.type == "auto":
                tool_choice = "auto"
            elif request.tool_choice.type == "any":
                tool_choice = "auto"  # Map "any" to "auto" for OpenAI
            elif request.tool_choice.type == "tool":
                tool_choice = {
                    "type": "function",
                    "function": {"name": request.tool_choice.name}
                }
            # "none" maps to "none"
        elif request.tools:
            # If tools are provided but no choice specified, default to auto
            tool_choice = "auto"

        # Convert stop sequences
        stop = request.stop_sequences or []

        actual_model = self._get_model_name(request.model)

        chat_request_dict = {
            "model": actual_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 1.0,
            "top_k": request.top_k or -1,
            "frequency_penalty": request.frequency_penalty or 0.0,
            "presence_penalty": request.presence_penalty or 0.0,
            "stop": stop,
            "stream": request.stream or False,
            "seed": request.seed,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        # Handle thinking/reasoning configuration
        if request.thinking and request.thinking.type == "enabled":
            # Map thinking to reasoning parser if available
            # This is implementation-specific and may need adjustment
            pass

        return ChatCompletionRequest(**chat_request_dict)

    def _convert_message(self, msg: AnthropicMessage) -> ChatCompletionMessageParam:
        """Convert a single Anthropic message to OpenAI format."""

        if isinstance(msg.content, str):
            return {"role": msg.role, "content": msg.content}

        # Handle complex content blocks
        if msg.role == "user":
            content_parts = []
            for block in msg.content:
                if isinstance(block, AnthropicTextBlock):
                    content_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, AnthropicImageBlock):
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source.media_type};base64,{block.source.data}"
                        }
                    })
                elif isinstance(block, AnthropicToolResultBlock):
                    # Convert tool result to text for now
                    content_text = block.content if isinstance(block.content, str) else "Tool result"
                    content_parts.append({"type": "text", "text": f"Tool result: {content_text}"})
                # Skip other block types for user messages

            return {"role": "user", "content": content_parts if content_parts else ""}

        elif msg.role == "assistant":
            content_text = ""
            tool_calls = []

            for block in msg.content:
                if isinstance(block, AnthropicTextBlock):
                    content_text += block.text
                elif isinstance(block, AnthropicThinkingBlock):
                    # Include thinking content in the text
                    content_text += f"<thinking>{block.thinking}</thinking>"
                elif isinstance(block, AnthropicToolUseBlock):
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })

            result = {"role": "assistant", "content": content_text or None}
            if tool_calls:
                result["tool_calls"] = tool_calls

            return result

        # Fallback
        return {"role": msg.role, "content": str(msg.content)}

    def _convert_response(
        self,
        chat_response: Any,
        request: AnthropicMessagesRequest,
        raw_request: Optional[Request] = None
    ) -> Union[AnthropicMessagesResponse, ErrorResponse]:
        """Convert OpenAI ChatCompletion response to Anthropic Messages format."""

        if isinstance(chat_response, ErrorResponse):
            return chat_response

        # Generate message ID
        message_id = f"msg_{random_uuid()}"

        # Convert content blocks
        content_blocks = []

        # Get the first choice (Anthropic doesn't support multiple choices)
        if hasattr(chat_response, 'choices') and chat_response.choices:
            choice = chat_response.choices[0]
            message = choice.message

            # Handle reasoning content if present
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                content_blocks.append(AnthropicThinkingBlock(
                    thinking=message.reasoning_content
                ))

            # Handle regular content
            if message.content:
                content_blocks.append(AnthropicTextBlock(text=message.content))

            # Handle tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function'):
                        try:
                            input_data = json.loads(tool_call.function.arguments)
                        except (json.JSONDecodeError, AttributeError):
                            input_data = {}

                        content_blocks.append(AnthropicToolUseBlock(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=input_data
                        ))

            # Determine stop reason
            stop_reason = "end_turn"  # default
            if choice.finish_reason == "length":
                stop_reason = "max_tokens"
            elif choice.finish_reason == "stop":
                stop_reason = "end_turn"
            elif choice.finish_reason == "tool_calls":
                stop_reason = "tool_use"

        # Convert usage
        usage = AnthropicUsage(
            input_tokens=chat_response.usage.prompt_tokens,
            output_tokens=chat_response.usage.completion_tokens
        )

        return AnthropicMessagesResponse(
            id=message_id,
            content=content_blocks or [AnthropicTextBlock(text="")],
            model=request.model,
            stop_reason=stop_reason,
            usage=usage
        )

    async def _convert_streaming_response(
        self,
        chat_stream: AsyncGenerator[str, None],
        request: AnthropicMessagesRequest,
        raw_request: Optional[Request] = None
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI streaming response to Anthropic streaming format."""

        message_id = f"msg_{random_uuid()}"
        content_index = 0
        current_text = ""
        total_input_tokens = 0
        total_output_tokens = 0
        first_chunk = True

        try:
            async for chunk_str in chat_stream:
                if not chunk_str.strip():
                    continue

                # Parse SSE format
                if chunk_str.startswith("data: "):
                    data_str = chunk_str[6:].strip()

                    if data_str == "[DONE]":
                        # Send final message stop event
                        stop_event = AnthropicMessageStop()
                        yield f"data: {stop_event.model_dump_json()}\n\n"
                        return

                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # First chunk - send message_start event
                    if first_chunk:
                        first_chunk = False

                        # Create initial message structure
                        initial_message = AnthropicMessagesResponse(
                            id=message_id,
                            content=[],
                            model=request.model,
                            stop_reason=None,
                            usage=AnthropicUsage(input_tokens=0, output_tokens=0)
                        )

                        start_event = AnthropicMessageStart(message=initial_message)
                        yield f"data: {start_event.model_dump_json()}\n\n"

                    # Process the chunk
                    if "choices" in chunk_data and chunk_data["choices"]:
                        choice = chunk_data["choices"][0]
                        delta = choice.get("delta", {})

                        # Handle content delta
                        if "content" in delta and delta["content"]:
                            # If this is the first content, send content_block_start
                            if not current_text:
                                content_start = AnthropicContentBlockStart(
                                    index=content_index,
                                    content_block=AnthropicTextBlock(text="")
                                )
                                yield f"data: {content_start.model_dump_json()}\n\n"

                            # Send content delta
                            content_delta = AnthropicContentBlockDelta(
                                index=content_index,
                                delta={"type": "text_delta", "text": delta["content"]}
                            )
                            yield f"data: {content_delta.model_dump_json()}\n\n"

                            current_text += delta["content"]

                        # Handle tool calls (simplified)
                        if "tool_calls" in delta and delta["tool_calls"]:
                            # This would need more sophisticated handling
                            pass

                        # Handle finish reason
                    if choice.get("finish_reason") and current_text:
                        # Send content_block_stop
                        content_stop = AnthropicContentBlockStop(index=content_index)
                        yield f"data: {content_stop.model_dump_json()}\n\n"

                    # Handle usage information
                    if "usage" in chunk_data:
                        usage = chunk_data["usage"]
                        total_input_tokens = usage.get("prompt_tokens", 0)
                        total_output_tokens = usage.get("completion_tokens", 0)

                        # Send message delta with usage
                        message_delta = AnthropicMessageDelta(
                            delta={},
                            usage=AnthropicUsage(
                                input_tokens=total_input_tokens,
                                output_tokens=total_output_tokens
                            )
                        )
                        yield f"data: {message_delta.model_dump_json()}\n\n"

        except Exception as e:
            error_event = AnthropicError(
                error={
                    "type": "api_error",
                    "message": str(e)
                }
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

        # Always send message_stop at the end
        stop_event = AnthropicMessageStop()
        yield f"data: {stop_event.model_dump_json()}\n\n"