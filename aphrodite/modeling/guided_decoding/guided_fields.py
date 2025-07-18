from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel


# These classes are deprecated, see SamplingParams
class LLMGuidedOptions(TypedDict, total=False):
    guided_json: Union[Dict, BaseModel, str]
    guided_regex: str
    guided_choice: List[str]
    guided_grammar: str
    guided_decoding_backend: str
    guided_whitespace_pattern: str
    guided_json_object: bool


@dataclass
class GuidedDecodingRequest:
    """One of the fields will be used to retrieve the logit processor."""
    guided_json: Optional[Union[Dict, BaseModel, str]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None
    guided_json_object: Optional[bool] = None
    structural_tag: Optional[str] = None

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        # Helper function to check if a value is effectively None
        def is_effectively_none(value):
            if value is None:
                return True
            if isinstance(value, str) and value.strip() == "":
                return True
            if isinstance(value, dict) and len(value) == 0:
                return True
            return False
        
        guide_count = sum(not is_effectively_none(x)
                          for x in (self.guided_json, self.guided_regex,
                                    self.guided_choice, self.guided_grammar,
                                    self.guided_json_object,
                                    self.structural_tag))
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")
