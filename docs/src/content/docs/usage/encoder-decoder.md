---
title: Encoder-Decoder Model Support in Aphrodite
---

Aphrodite now supports encoder-decoder language models (only available if built from source for now), such as [BART](https://huggingface.co/facebook/bart-large-cnn), in addition to decoder-only models. This document will guide you through using encoder-decoder models with Aphrodite.

## Introduction
Encoder-decoder models, like BART, consist of two main components: an encoder that processes the input sequence, and a decoder that generates the output sequence. Aphrodite's support for these models allows you to leverage their capabilities for tasks such as summarization, translation, and more.

## Setting up an Encoder-Decoder Model

To use an encoder-decoder model with Aphrodite, you need to initialize an `LLM` instance with the appropriate model name. Here's an example using BART model:

```py
from aphrodite import LLM

llm = LLM(model="facebook/bart-large-cnn", dtype="float")
```

Keep in mind that it's recommended to use float (FP32) data type for BART models.

The `LLM` class automatically detects whether the model is encoder-decoder or a decoder-only model and sets up the appropriate internal configurations.

## Input Types
We support various input types for encoder-decoder models. The main types are defined in the `aphrodite.inputs.data` module:

```py
    if prompt is None:
        return 'None'

    required_keys_dict = {
        'TextPrompt': {'prompt'}, # [!code highlight]
        'TokensPrompt': {'prompt_token_ids'}, # [code! highlight]
        'ExplicitEncoderDecoder': {'encoder_prompt', 'decoder_prompt'}, # [code! highlight]
    }

    if isinstance(prompt, dict):
        for (ptype, required_keys) in required_keys_dict.items():
            # Ignore type checking in the conditional below because type
            # checker does not understand that is_dict(prompt) narrows
            # down the possible types
            if _has_required_keys(
                    prompt,  # type: ignore
                    required_keys):
                return ptype

        raise ValueError(f"Invalid prompt {prompt}, valid types are "
                         "required_keys_dict={required_keys_dict}")

    if isinstance(prompt, str):
        return "str"
```

For encoder-decoder models, you can use these input types in different combinations:

### Single Input (Implicit Encoder Input)
You can provide a single input, which will be treated as the encoder input. The decoder input will be assumed to be empty (None).

```py
single_text_prompt_raw = "Hello, my name is"
single_text_prompt = TextPrompt(prompt="The president of the United States is")
single_tokens_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode("The capital of France is"))
```

### Explicit Encoder and Decoder Inputs
For more control, you can explicitly specify both encoder and decoder inputs using the `ExplicitEncoderDecoderPrompt` class:

```py
class ExplicitEncoderDecoderPrompt(TypedDict):
    """Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a 
    decoder prompt.
    The encoder and decoder prompts, respectively,
    may formatted according to any of the
    SingletonPromptInputs schemas, and are not
    required to have the same schema.
    Only the encoder prompt may have multi-modal data.
    Note that an ExplicitEncoderDecoderPrompt may not
    be used as an input to a decoder-only model,
    and that the `encoder_prompt` and `decoder_prompt`
    fields of this data structure may not themselves
    must be SingletonPromptInputs instances.
    """

    encoder_prompt: SingletonPromptInputs

    decoder_prompt: SingletonPromptInputs
```

Example usage:

```py
enc_dec_prompt = ExplicitEncoderDecoderPrompt(
    encoder_prompt="Summarize this text:",
    decoder_prompt="Summary:"
)
```

## Generating Text
To generate text with an encoder-decoder model, use the `generate` method of the `LLM` instance. You can pass a single prompt, or a list of prompts, along with sampling parameters:

```py
from aphrodite import SamplingParams

sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    min_tokens=0,
    max_tokens=20,
)

outputs = llm.generate(prompts, sampling_params)
```

The `generate` method returns a list of `RequestOutput` objects containing the generated text and other information.

## Advanced Usage
### Mixing Input Types
You can mix different input types in a single generation request:

```py
prompts = [
    single_text_prompt_raw,
    single_text_prompt,
    single_tokens_prompt,
    enc_dec_prompt1,
    enc_dec_prompt2,
    enc_dec_prompt3
]

outputs = llm.generate(prompts, sampling_params)
```

### Batching Encoder and Decoder Prompts
For efficient processing of multiple encoder-decoder pairs, use the `zip_enc_dec_prompt_lists` helper function:

```py
from aphrodite.common.utils import zip_enc_dec_prompt_lists

zipped_prompt_list = zip_enc_dec_prompt_lists(
    ['An encoder prompt', 'Another encoder prompt'],
    ['A decoder prompt', 'Another decoder prompt']
)
```

### Accessing Generated Text
After generation, you can access the generated text and other information from the `RequestOutput` objects:

```py
for output in outputs:
    prompt = output.prompt
    encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    print(f"Encoder prompt: {encoder_prompt!r}, "
          f"Decoder prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")
```

## API Reference
### LLM Class
The `LLM` class in the `aphrodite.endpoints.llm` module is the main interface for working with both decoder-only and encoder-decoder models.

Key methods:

- `__init__(self, model: str, ...)`: Initialize an LLM instance with the specified model.
- `generate(self, prompts: Union[PromptInputs, Sequence[PromptInputs]], ...)`: Generate text based on the given prompts and sampling parameters.

### Input Types
- `TextPrompt`: Represents a text prompt.
- `TokensPrompt`: Represents a tokenized prompt.
- `ExplicitEncoderDecoderPrompt`: Represents an explicit encoder-decoder prompt pair.

### RequestOutput
The `RequestOutput` class in the `aphrodite.common.outputs` module contains the results of a generation request.

Key attributes:

- `prompt`: The input (decoder prompt for encoder-decoder models).
- `encoder_prompt`: The encoder prompt for encoder-decoder models.
- `outputs`: A list of `CompletionOutput` objects containing the generate text and other information.

For detailed info on these classes and their methods, please refer to the source code.