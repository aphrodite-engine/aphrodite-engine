import pytest

from aphrodite.endpoints.chat_utils import (apply_chat_template,
                                            load_chat_template)
from aphrodite.endpoints.openai.protocol import ChatCompletionRequest
from aphrodite.transformers_utils.tokenizer import get_tokenizer

from ..utils import APHRODITE_PATH

chatml_jinja_path = APHRODITE_PATH / "examples/chat_templates/chatml.jinja"
assert chatml_jinja_path.exists()

# Define models, templates, and their corresponding expected outputs
MODEL_TEMPLATE_GENERATON_OUTPUT = [
    ("facebook/opt-125m", chatml_jinja_path, True, """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of<|im_end|>
<|im_start|>assistant
"""),
    ("facebook/opt-125m", chatml_jinja_path, False, """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of""")
]

TEST_MESSAGES = [
    {
        'role': 'user',
        'content': 'Hello'
    },
    {
        'role': 'assistant',
        'content': 'Hi there!'
    },
    {
        'role': 'user',
        'content': 'What is the capital of'
    },
]


def test_load_chat_template():
    # Testing chatml template
    template_content = load_chat_template(chat_template=chatml_jinja_path)

    # Test assertions
    assert template_content is not None
    # Hard coded value for template_chatml.jinja
    assert template_content == """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""  # noqa: E501


def test_no_load_chat_template_filelike():
    # Testing chatml template
    template = "../../examples/does_not_exist"

    with pytest.raises(ValueError, match="looks like a file path"):
        load_chat_template(chat_template=template)


def test_no_load_chat_template_literallike():
    # Testing chatml template
    template = "{{ messages }}"

    template_content = load_chat_template(chat_template=template)

    assert template_content == template


@pytest.mark.parametrize(
    "model,template,add_generation_prompt,expected_output",
    MODEL_TEMPLATE_GENERATON_OUTPUT)
def test_get_gen_prompt(model, template, add_generation_prompt,
                        expected_output):
    # Initialize the tokenizer
    tokenizer = get_tokenizer(tokenizer_name=model)
    template_content = load_chat_template(chat_template=template)

    # Create a mock request object using keyword arguments
    mock_request = ChatCompletionRequest(
        model=model,
        messages=TEST_MESSAGES,
        add_generation_prompt=add_generation_prompt)

    # Call the function and get the result
    result = apply_chat_template(
        tokenizer,
        conversation=mock_request.messages,
        chat_template=mock_request.chat_template or template_content,
        add_generation_prompt=mock_request.add_generation_prompt,
    )

    # Test assertion
    assert result == expected_output, (
        f"The generated prompt does not match the expected output for "
        f"model {model} and template {template}")
