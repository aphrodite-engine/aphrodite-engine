import pytest

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
max_model_len = 128

input_str = """Immerse yourself in the enchanting chronicle of calculus, a 
mathematical domain that has radically transformed our comprehension of 
change and motion. Despite its roots in ancient civilizations, the 
formal birth of calculus predominantly occurred in the 17th century, 
primarily under the influential guidance of Sir Isaac Newton and Gottfried 
Wilhelm Leibniz. The earliest traces of calculus concepts are found in 
ancient Greek mathematics,most notably in the works of Eudoxus and 
Archimedes, around 300 BCE. They utilized the 'method of exhaustion'—a 
technique for computing areas and volumes through the use of finite sums. 
This methodology laid crucial foundational work for integral calculus. 
In the 17th century, both Newton and Leibniz independently pioneered 
calculus, each contributing unique perspectives that would shape this new 
field."""


def test_smaller_truncation_size(aphrodite_runner,
                                 model_name=MODEL_NAME,
                                 input_str=input_str):

    truncate_prompt_tokens = 10

    with aphrodite_runner(model_name, task="embed",
                     max_model_len=max_model_len) as aphrodite_model:
        aphrodite_output = aphrodite_model.model.encode(
            input_str, truncate_prompt_tokens=truncate_prompt_tokens)

    prompt_tokens = aphrodite_output[0].prompt_token_ids

    assert len(prompt_tokens) == truncate_prompt_tokens


def test_max_truncation_size(aphrodite_runner,
                             model_name=MODEL_NAME,
                             input_str=input_str):
    truncate_prompt_tokens = -1

    with aphrodite_runner(model_name, task="embed",
                     max_model_len=max_model_len) as aphrodite_model:
        aphrodite_output = aphrodite_model.model.encode(
            input_str, truncate_prompt_tokens=truncate_prompt_tokens)

    prompt_tokens = aphrodite_output[0].prompt_token_ids

    assert len(prompt_tokens) == max_model_len


def test_bigger_truncation_size(aphrodite_runner,
                                model_name=MODEL_NAME,
                                input_str=input_str):

    truncate_prompt_tokens = max_model_len + 1

    with pytest.raises(ValueError), aphrodite_runner(
            model_name, task="embed",
            max_model_len=max_model_len) as aphrodite_model:

        llm_output = aphrodite_model.model.encode(
            input_str, truncate_prompt_tokens=truncate_prompt_tokens)

        assert llm_output == f"""truncate_prompt_tokens value 
                ({truncate_prompt_tokens}) is greater than 
                max_model_len ({max_model_len}). Please, select 
                a smaller truncation size."""
