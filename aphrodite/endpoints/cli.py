# The CLI entrypoint to Aphrodite.
import argparse
import os
import signal
import subprocess
import sys
from typing import Optional

import uvloop
import yaml
from loguru import logger
from openai import OpenAI

from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.openai.api_server import run_server
from aphrodite.endpoints.openai.args import make_arg_parser


def registrer_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def serve(args: argparse.Namespace) -> None:
    # EngineArgs expects the model name to be passed as --model.
    args.model = args.model_tag

    uvloop.run(run_server(args))


def interactive_cli(args: argparse.Namespace) -> None:
    registrer_signal_handlers()

    base_url = args.url
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models.data[0].id

    print(f"Using model: {model_name}")

    if args.command == "complete":
        complete(model_name, openai_client)
    elif args.command == "chat":
        chat(args.system_prompt, model_name, openai_client)


def complete(model_name: str, client: OpenAI) -> None:
    print("Please enter prompt to complete:")
    while True:
        input_prompt = input("> ")

        completion = client.completions.create(model=model_name,
                                               prompt=input_prompt)
        output = completion.choices[0].text
        print(output)


def chat(system_prompt: Optional[str], model_name: str,
         client: OpenAI) -> None:
    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        input_message = input("> ")
        message = {"role": "user", "content": input_message}
        conversation.append(message)

        chat_completion = client.chat.completions.create(model=model_name,
                                                         messages=conversation)

        response_message = chat_completion.choices[0].message
        output = response_message.content

        conversation.append(response_message)
        print(output)


STR_BOOLS = ['enforce_eager', 'enable_chunked_prefill']
ADAPTERS = ['lora_modules', 'prompt_adapters']


# TODO: refactor this to directly call run_server with the config file
def serve_yaml(args: argparse.Namespace) -> None:

    def append_cmd_args(cmd, key, value):
        if value:  # Skip appending if value is empty
            if key in ADAPTERS and isinstance(value, list):
                adapters = [f"{k}={v}" for k, v in value[0].items() if v]
                if adapters:
                    cmd.append(f"--{key}")
                    cmd.extend(adapters)
            else:
                cmd.append(f"--{key}")
                if isinstance(value, bool):
                    if key in STR_BOOLS:
                        cmd.append(str(value).lower())
                    elif value:
                        cmd.append(f"--{key}")
                else:
                    cmd.append(str(value))

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    cmd = ["python3", "-m", "aphrodite.endpoints.openai.api_server"]
    for key, value in config.items():
        if isinstance(value, list):
            for item in value:
                for sub_key, sub_value in item.items():
                    append_cmd_args(cmd, sub_key, sub_value)
        else:
            append_cmd_args(cmd, key, value)

    process = subprocess.Popen(cmd)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()


def _add_query_options(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:2242/v1",
        help="url of the running OpenAI-Compatible RESTful API server")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=("The model name used in prompt completion, default to "
              "the first model in list models API call."))
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for OpenAI services. If provided, this api key "
            "will overwrite the api key obtained through environment variables."
        ))
    return parser


def env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using Aphrodite as a library. `spawn` will
    # cause unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "APHRODITE_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting APHRODITE_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["APHRODITE_WORKER_MULTIPROC_METHOD"] = "spawn"


def main():
    env_setup()

    parser = FlexibleArgumentParser(description="Aphrodite CLI")
    subparsers = parser.add_subparsers(required=True)

    serve_parser = subparsers.add_parser(
        "run",
        help="Start the Aphrodite OpenAI Compatible API server",
        usage="aphrodite run <model_tag> [options]")
    serve_parser.add_argument("model_tag",
                              type=str,
                              help="The model tag to serve")
    serve_parser = make_arg_parser(serve_parser)
    serve_parser.set_defaults(dispatch_function=serve)

    complete_parser = subparsers.add_parser(
        "complete",
        help=("Generate text completions based on the given prompt "
              "via the running API server"),
        usage="aphrodite complete [options]")
    _add_query_options(complete_parser)
    complete_parser.set_defaults(dispatch_function=interactive_cli,
                                 command="complete")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Generate chat completions via the running API server",
        usage="aphrodite chat [options]")
    _add_query_options(chat_parser)
    chat_parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=("The system prompt to be added to the chat template, "
              "used for models that support system prompts."))
    chat_parser.set_defaults(dispatch_function=interactive_cli, command="chat")

    yaml_parser = subparsers.add_parser(
        "yaml",
        help="Start the Aphrodite OpenAI Compatible API server with a YAML "
        "config file",
        usage="aphrodite yaml <config.yaml>")
    yaml_parser.add_argument("config_file",
                             type=str,
                             help="The YAML configuration file to use")
    yaml_parser.set_defaults(dispatch_function=serve_yaml)

    args = parser.parse_args()
    # One of the sub commands should be executed.
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
