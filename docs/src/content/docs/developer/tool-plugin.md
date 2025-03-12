---
title: Aphrodite Tool Parser Plugin System
---

Tool parsers in Aphrodite help convert model outputs into OpenAI-compatible function/tool calls. This guide goes through creating and using custom tool parsers.

## Creating a Tool Parser Plugin
### Basic Structure

Create a new Python file (e.g. `my_tool_parser.py`) with the following structure:

```py
from typing import Dict, Sequence, Union

from aphrodite.endpoints.openai.protocol import (
    ChatCompletionRequest, 
    DeltaMessage, 
    ExtractedToolCallInformation,
    ToolCall, 
    FunctionCall
)
from aphrodite.endpoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, 
    ToolParserManager
)

@ToolParserManager.register_module(["my_parser"])  # register the unique tool name
class MyToolParser(ToolParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # add any custom initialization here

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Optional: Modify the request before processing"""
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Handle non-streaming tool call extraction"""
        # implementation here

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """Handle streaming tool call extraction"""
        # implementation here
```

### Required Methods

#### `extracted_tool_calls`
This method handles complete (non-streaming) responses.

- Input: Complete model output string.
- Output: `ExtractedToolCallInformation` obj containing:
    - `tools_called`: bool indicating if tool calls were detected
    - `tool_calls`: List of parsed tool calls
    - `content`: Any non-tool-call content
Example:
```py
def extract_tool_calls(
    self,
    model_output: str,
    request: ChatCompletionRequest,
) -> ExtractedToolCallInformation:
    # Example: Parse format "FUNCTION_NAME(arguments)"
    if "(" not in model_output:
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output
        )

    try:
        name, args = model_output.split("(", 1)
        args = args.rsplit(")", 1)[0]
        
        tool_calls = [
            ToolCall(
                function=FunctionCall(
                    name=name.strip(),
                    arguments=args.strip()
                )
            )
        ]

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=None
        )
    except Exception:
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output
        )
```

#### `extract_tool_calls_streaming`
This method handles streaming responses token by token:
- called for each new token
- must track state between calls
- return `None` if not ready to send delta
- return `DeltaMessage` when ready to send updates

### Optional Methods

#### `adjust_request`
Modify the request before processing if needed.

```py
def adjust_request(
    self, 
    request: ChatCompletionRequest
) -> ChatCompletionRequest:
    # Example: Disable special token skipping
    request.skip_special_tokens = False
    return request
```

## Using the Tool Parser

### Starting the Server
Launch a local Aphrodite instance:

```sh
aphrodite run <your_model> \
    --tool-parser-plugin /path/to/my_tool_parser.py \
    --tool-call-parser my_parser \
    --enable-auto-tool-choice \
    --chat-template /path/to/chat_template.jinja
```

### Making API calls
Use the OpenAI API format:

```py
import openai

response = openai.ChatCompletion.create(
    model="your_model",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)
```


## Example Implementations

For a more complete example set, see [tool_parsers](https://github.com/aphrodite-engine/aphrodite-engine/tree/main/aphrodite/endpoints/openai/tool_parsers).