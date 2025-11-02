from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings(
        "hmellor/tiny-random-LlamaForCausalLM", [], ["--cpu-offload-gb", "1"]
    )
