import json
import uuid
from collections.abc import Sequence

import regex as re

from aphrodite.endpoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from aphrodite.endpoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from aphrodite.logger import init_logger
from aphrodite.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("qwen3_next")
class Qwen3NextToolParser(ToolParser):
    """
    Tool parser for Qwen3 Next models that use JSON-in-XML format for tool
    calls.

    Expected format:
    <tool_call>
    {"name": "function_name", "arguments":
     {"arg1": "value1", "arg2": "value2"}}
    </tool_call>
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns for JSON-in-XML format
        self.tool_call_complete_regex = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        self.tool_call_regex = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>|<tool_call>\s*(\{.*?)$", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 Next JSON-XML Tool parser could not locate tool call start/end tokens in the tokenizer!"
            )

        logger.info("Successfully imported tool parser %s !", self.__class__.__name__)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_json_content = ""
        self.json_started = False
        self.json_closed = False
        self.streaming_request = None
        self.accumulated_json = ""

    def _parse_json_function_call(self, json_str: str) -> ToolCall | None:
        """Parse a JSON string into a ToolCall object."""
        try:
            # Clean up the JSON string
            json_str = json_str.strip()

            # Parse the JSON
            tool_data = json.loads(json_str)

            if not isinstance(tool_data, dict):
                logger.warning("Tool call JSON is not a dictionary: {}", json_str)
                return None

            if "name" not in tool_data:
                logger.warning("Tool call JSON missing 'name' field: {}", json_str)
                return None

            function_name = tool_data["name"]
            arguments = tool_data.get("arguments", {})

            # Ensure arguments is properly formatted as JSON string
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments, ensure_ascii=False)
            elif isinstance(arguments, str):
                # Validate it's valid JSON
                try:
                    json.loads(arguments)
                    arguments_str = arguments
                except json.JSONDecodeError:
                    # If not valid JSON, wrap in quotes
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
            else:
                arguments_str = json.dumps(arguments, ensure_ascii=False)

            return ToolCall(
                type="function",
                function=FunctionCall(name=function_name, arguments=arguments_str),
            )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool call JSON '{}': {}", json_str, e)
            return None
        except Exception as e:
            logger.warning("Unexpected error parsing tool call '{}': {}", json_str, e)
            return None

    def _get_json_function_calls(self, model_output: str) -> list[str]:
        """Extract JSON strings from tool_call XML tags."""
        # Find all complete tool calls first
        complete_matches = self.tool_call_complete_regex.findall(model_output)

        if complete_matches:
            return complete_matches

        # Fall back to partial matches for incomplete responses
        partial_matches = self.tool_call_regex.findall(model_output)
        json_calls = []

        for match in partial_matches:
            # match is a tuple (complete_json, partial_json)
            if match[0]:  # complete JSON
                json_calls.append(match[0])
            elif match[1]:  # partial JSON
                json_calls.append(match[1])

        return json_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output."""
        # Quick check to avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        try:
            json_function_calls = self._get_json_function_calls(model_output)
            if len(json_function_calls) == 0:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

            tool_calls = []
            for json_str in json_function_calls:
                parsed_tool = self._parse_json_function_call(json_str)
                if parsed_tool:
                    tool_calls.append(parsed_tool)

            # Populate prev_tool_call_arr for serving layer to set
            # finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            content = model_output[:content_index].rstrip() if content_index >= 0 else ""

            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output."""
        # Store request for reference
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(self.tool_call_complete_regex.findall(current_text))

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(self.tool_call_start_token) - current_text.count(
                        self.tool_call_end_token
                    )
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            if self.tool_call_start_token_id in delta_token_ids or self.tool_call_start_token in delta_text:
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[: delta_text.index(self.tool_call_start_token)]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # We're in a tool call - extract current tool call portion
        # Find all tool call start positions
        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            # No more tool calls to process yet
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)

        if tool_end_idx == -1:
            # Tool call not complete yet
            tool_text = current_text[tool_start_idx:]
        else:
            # Complete tool call
            tool_text = current_text[tool_start_idx : tool_end_idx + len(self.tool_call_end_token)]

        # Extract JSON content from within the tool_call tags
        json_start = tool_text.find(">") + 1 if ">" in tool_text else len(self.tool_call_start_token)
        json_end = tool_text.find(self.tool_call_end_token)

        if json_end == -1:
            # Still streaming the JSON content
            current_json = tool_text[json_start:].strip()
        else:
            # Complete JSON content
            current_json = tool_text[json_start:json_end].strip()

        # Check if we have a complete JSON object
        if current_json and current_json != self.accumulated_json:
            # Send header if not sent yet
            if not self.header_sent:
                try:
                    # Try to parse JSON to get function name
                    if current_json.startswith("{") and "name" in current_json:
                        # Try to extract name even from partial JSON
                        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', current_json)
                        if name_match:
                            self.current_function_name = name_match.group(1)
                            self.current_tool_id = self._generate_tool_call_id()
                            self.header_sent = True

                            # Add to prev_tool_call_arr immediately
                            self.prev_tool_call_arr.append(
                                {
                                    "name": self.current_function_name,
                                    "arguments": "{}",  # Placeholder
                                }
                            )

                            # Send header with function info
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        id=self.current_tool_id,
                                        function=DeltaFunctionCall(name=self.current_function_name, arguments=""),
                                        type="function",
                                    )
                                ]
                            )
                except Exception:
                    pass  # Continue processing

            # Check if JSON is complete
            if tool_end_idx != -1 and current_json.endswith("}"):
                # Complete tool call - parse and update prev_tool_call_arr
                try:
                    parsed_tool = self._parse_json_function_call(current_json)
                    if parsed_tool:
                        # Update existing entry in prev_tool_call_arr
                        for i, tool in enumerate(self.prev_tool_call_arr):
                            if tool.get("name") == parsed_tool.function.name:
                                self.prev_tool_call_arr[i]["arguments"] = parsed_tool.function.arguments
                                break

                        # Send the complete arguments
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    function=DeltaFunctionCall(arguments=parsed_tool.function.arguments),
                                )
                            ]
                        )
                except Exception:
                    pass  # Ignore parsing errors during streaming

                # Move to next tool
                self.current_tool_index += 1
                self.header_sent = False
                self.accumulated_json = ""
                return None

            # Stream partial arguments if we have more JSON content
            if len(current_json) > len(self.accumulated_json) and self.header_sent:
                try:
                    # Extract arguments portion that's new
                    new_content = current_json[len(self.accumulated_json) :]
                    self.accumulated_json = current_json

                    if new_content:
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_index,
                                    function=DeltaFunctionCall(arguments=new_content),
                                )
                            ]
                        )
                except Exception:
                    pass

        return None
