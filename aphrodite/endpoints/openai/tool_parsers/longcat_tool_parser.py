import regex as re

from aphrodite.endpoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParserManager)
from aphrodite.endpoints.openai.tool_parsers.hermes_tool_parser import (
    Hermes2ProToolParser)
from aphrodite.transformers_utils.tokenizer import AnyTokenizer


@ToolParserManager.register_module("longcat")
class LongcatFlashToolParser(Hermes2ProToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<longcat_tool_call>"
        self.tool_call_end_token: str = "</longcat_tool_call>"

        self.tool_call_regex = re.compile(
            r"<longcat_tool_call>(.*?)</longcat_tool_call>|<longcat_tool_call>(.*)",
            re.DOTALL,
        )

        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(
            self.tool_call_end_token, add_special_tokens=False
        )

        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_start_token_ids
        ]

        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_end_token_ids
        ]
