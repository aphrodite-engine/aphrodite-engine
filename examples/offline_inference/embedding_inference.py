import os
from argparse import Namespace

from aphrodite import LLM, EngineArgs
from aphrodite.utils import FlexibleArgumentParser

# needs V0 engine
os.environ["APHRODITE_USE_V1"] = "0"


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="openai/clip-vit-base-patch32",
        runner="pooling",
        enforce_eager=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass runner="pooling" for embedding models
    llm = LLM(**vars(args))

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = llm.embed(prompts)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} "
              f"(size={len(embeds)})")
        print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
