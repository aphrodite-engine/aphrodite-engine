from aphrodite import LLM, SamplingParams
from aphrodite.plugins.sampling_registry import SamplingPluginRegistry
from aphrodite.plugins.example_chaos_sampler import ChaosSamplerPlugin

# Register the plugin
SamplingPluginRegistry.register("chaos", ChaosSamplerPlugin)

params = SamplingParams(
    custom_sampler={"chaos_enabled": 1}
)

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompts = [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The quick brown fox jumps over the lazy dog.",
    "The meaning of life is",
]

outputs = llm.generate(prompts, params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
