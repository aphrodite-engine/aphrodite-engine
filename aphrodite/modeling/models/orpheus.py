# Adapted from
# https://github.com/canopyai/Orpheus-TTS/tree/main/orpheus_tts_pypi/orpheus_tts

import asyncio
import os
import queue
import threading

import numpy as np
import torch
from snac import SNAC
from transformers import AutoTokenizer

from aphrodite.v1.engine.async_llm import AsyncLLM
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.common.sampling_params import SamplingParams

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

snac_device = os.environ.get(
    "SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(snac_device)


def convert_to_audio(multiframe, count):

    if len(multiframe) < 7:
        return

    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]

    for j in range(num_frames):
        i = 7*j
        if codes_0.shape[0] == 0:
            codes_0 = torch.tensor(
                [frame[i]], device=snac_device, dtype=torch.int32)
        else:
            codes_0 = torch.cat(
                [codes_0, torch.tensor([frame[i]], device=snac_device,
                                       dtype=torch.int32)])

        if codes_1.shape[0] == 0:
            codes_1 = torch.tensor(
                [frame[i+1]], device=snac_device, dtype=torch.int32)
            codes_1 = torch.cat(
                [codes_1, torch.tensor([frame[i+4]], device=snac_device,
                                       dtype=torch.int32)])
        else:
            codes_1 = torch.cat(
                [codes_1, torch.tensor([frame[i+1]], device=snac_device,
                                       dtype=torch.int32)])
            codes_1 = torch.cat(
                [codes_1, torch.tensor([frame[i+4]], device=snac_device,
                                       dtype=torch.int32)])

        if codes_2.shape[0] == 0:
            codes_2 = torch.tensor(
                [frame[i+2]], device=snac_device, dtype=torch.int32)
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+3]], device=snac_device,
                                       dtype=torch.int32)])
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+5]], device=snac_device,
                                       dtype=torch.int32)])
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+6]], device=snac_device,
                                       dtype=torch.int32)])
        else:
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+2]], device=snac_device,
                                       dtype=torch.int32)])
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+3]], device=snac_device,
                                       dtype=torch.int32)])
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+5]], device=snac_device,
                                       dtype=torch.int32)])
            codes_2 = torch.cat(
                [codes_2, torch.tensor([frame[i+6]], device=snac_device,
                                       dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    # check that all tokens are between 0 and 4096 otherwise return *
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or \
       torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or \
       torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return

    with torch.inference_mode():
        audio_hat = model.decode(codes)

    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes


def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()

    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")

    if last_token_start == -1:
        print("No token found in the string")
        return None

    # Extract the last token
    last_token = token_string[last_token_start:]

    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None


async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples


# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator
        # that processes tokens.
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()


class OrpheusModel:
    def __init__(
        self,
        model_name,
        dtype=torch.bfloat16,
        tokenizer='canopylabs/orpheus-3b-0.1-pretrained',
        **engine_kwargs,
    ):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs
        self.engine = self._setup_engine()
        self.available_voices = [
            "zoe", "zac", "jess", "leo", "mia", "julia", "leah",
        ]

        # Use provided tokenizer path or default to model_name
        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        try:
            # Check if tokenizer_path is a local directory
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(
                    tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")

    def _map_model_params(self, model_name):
        model_map = {
            # "nano-150m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            # "micro-400m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            # "small-1b":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if (model_name in unsupported_models):
            raise ValueError(f"Model {model_name} is not supported. "
                             "Only medium-3b is supported, small, micro "
                             "and nano models will be released very soon")
        elif model_name in model_map:
            return model_name[model_name]["repo_id"]
        else:
            return model_name

    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            **self.engine_kwargs
        )
        return AsyncLLM.from_engine_args(engine_args)

    def validate_voice(self, voice):
        if voice:
            if voice not in self.available_voices:
                raise ValueError(f"Voice {voice} is not available for model "
                                 "{self.model_name}")

    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return (f"<custom_token_3>{prompt}[{voice}]<custom_token_4>"
                        "<custom_token_5>")
            else:
                return (f"<custom_token_3>{prompt}<custom_token_4>"
                        "<custom_token_5>")
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(
                    adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor(
                    [[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat(
                    [start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor(
                    [[128259]], dtype=torch.int64)
                end_tokens = torch.tensor(
                    [[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat(
                    [start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

    def generate_tokens_sync(
        self,
        prompt,
        voice=None,
        request_id="req-001",
        temperature=0.6,
        top_p=0.8,
        max_tokens=1200,
        stop_token_ids=[49158],
        repetition_penalty=1.3,
    ):
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,  # Adjust max_tokens as needed.
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            async for result in self.engine.generate(
                prompt=prompt_string,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                # Place each token text into the queue.
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion.

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()

    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
