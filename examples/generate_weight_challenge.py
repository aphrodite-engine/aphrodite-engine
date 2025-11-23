#!/usr/bin/env python3
"""Generate weight challenge JSON for a given model.

This script generates valid challenge requests for the /weights/execute API
by loading model weights from the Hugging Face cache and inspecting their shapes.

Usage:
    python generate_weight_challenge.py <model_id> [--output <file>] [--num-challenges <n>]

Example:
    python generate_weight_challenge.py Qwen/Qwen3-0.6B --output challenges.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def find_model_path(model_id: str) -> Path:
    """Find the model in Hugging Face cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_slug = f"models--{model_id.replace('/', '--')}"
    model_dir = cache_dir / model_slug

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model {model_id} not found in HF cache. "
            f"Please download it first using: huggingface-cli download {model_id}"
        )

    snapshots = list((model_dir / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found for {model_id}")

    snapshot_path = snapshots[0]

    weight_files = list(snapshot_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = list(snapshot_path.glob("pytorch_model*.bin"))

    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {snapshot_path}")

    return weight_files[0]


def load_weight_info(weight_file: Path) -> dict[str, tuple[int, ...]]:
    """Load weight shapes from a safetensors file."""
    weights_info = {}

    if weight_file.suffix == ".safetensors":
        with safe_open(str(weight_file), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                tensor = f.get_tensor(key)
                weights_info[key] = tuple(tensor.shape)
    else:
        # Load pytorch .bin file
        state_dict = torch.load(weight_file, map_location="cpu")
        for key, tensor in state_dict.items():
            weights_info[key] = tuple(tensor.shape)

    return weights_info


def create_challenge(
    layer_name: str,
    weight_shape: tuple[int, ...],
    model_id: str,
    challenge_id: str,
    num_nonzero: int = 5,
) -> dict:
    """Create a challenge JSON for a specific layer."""
    # For 2D weights, assume [out_features, in_features]
    if len(weight_shape) == 2:
        out_dim, in_dim = weight_shape
    else:
        # For other shapes, treat as 1D
        in_dim = weight_shape[-1]
        out_dim = weight_shape[0] if len(weight_shape) > 0 else in_dim

    # sparse input vector
    indices = torch.linspace(0, in_dim - 1, min(num_nonzero, in_dim), dtype=torch.long).tolist()
    values = [0.5, -0.75, 0.125, 0.875, -0.25, 0.625, -0.375, 0.1875, -0.5625]

    input_data = [0.0] * in_dim
    for i, idx in enumerate(indices):
        input_data[idx] = values[i % len(values)]

    challenge = {
        "model": model_id,
        "worker_id": "worker-0",
        "challenge_id": challenge_id,
        "layer": layer_name,  # full layer name with .weight
        "input": {
            "data": input_data,
            "dtype": "float32",
            "shape": [in_dim],
        },
        "metadata": {
            "weight_shape": list(weight_shape),
            "input_size": in_dim,
            "output_size": out_dim,
            "nonzero_indices": indices,
        },
    }

    return challenge


def generate_challenges(
    model_id: str,
    num_challenges: int = 5,
    layer_pattern: str | None = None,
) -> dict:
    """Generate challenges for a model."""
    print(f"Finding model: {model_id}", file=sys.stderr)
    weight_file = find_model_path(model_id)
    print(f"Loading weights from: {weight_file}", file=sys.stderr)

    weights_info = load_weight_info(weight_file)
    print(f"Found {len(weights_info)} weight tensors", file=sys.stderr)

    # only .weight parameters (not .bias)
    weight_layers = {k: v for k, v in weights_info.items() if k.endswith(".weight")}
    print(f"Filtered to {len(weight_layers)} .weight tensors", file=sys.stderr)

    # filter by pattern if provided
    if layer_pattern:
        weight_layers = {k: v for k, v in weight_layers.items() if layer_pattern in k}
        print(f"Filtered to {len(weight_layers)} layers matching '{layer_pattern}'", file=sys.stderr)

    # select diverse layers
    selected_layers = list(weight_layers.items())[:num_challenges]

    # generate challenges
    challenges = []
    for layer_name, weight_shape in selected_layers:
        print(f"\nLayer: {layer_name}", file=sys.stderr)
        print(f"  Shape: {weight_shape}", file=sys.stderr)

        challenge_id = f"{model_id.replace('/', '_')}_{layer_name.replace('.', '_')}_v1"
        challenge = create_challenge(layer_name, weight_shape, model_id, challenge_id)
        challenges.append(challenge)

    bundle = {
        "model_key": model_id,
        "version": "auto-generated",
        "weight_file": str(weight_file),
        "challenges": challenges,
    }

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Generate weight challenge JSON for a model")
    parser.add_argument(
        "model_id",
        type=str,
        help="Hugging Face model ID (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument(
        "--num-challenges",
        "-n",
        type=int,
        default=5,
        help="Number of challenges to generate (default: 5)",
    )
    parser.add_argument(
        "--layer-pattern",
        "-p",
        type=str,
        help="Filter layers by name pattern (e.g., 'layers.0' or 'mlp')",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Output as bundle (default: output single challenge or first challenge only)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        help="Output directory for multiple challenge files (one per challenge)",
    )

    args = parser.parse_args()

    try:
        bundle = generate_challenges(
            args.model_id,
            num_challenges=args.num_challenges,
            layer_pattern=args.layer_pattern,
        )

        if args.output_dir:
            # output multiple files, one per challenge
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, challenge in enumerate(bundle["challenges"]):
                filename = f"challenge_{i:03d}_{challenge['layer'].replace('.', '_')}.json"
                filepath = output_dir / filename
                json_str = json.dumps(challenge, indent=2 if args.pretty else None)
                filepath.write_text(json_str)
                print(f"Wrote {filepath}", file=sys.stderr)

            print(f"\nWrote {len(bundle['challenges'])} challenges to {output_dir}/", file=sys.stderr)

        elif args.bundle:
            # output as bundle
            json_str = json.dumps(bundle, indent=2 if args.pretty else None)

            if args.output:
                output_path = Path(args.output)
                output_path.write_text(json_str)
                print(f"\nWrote bundle with {len(bundle['challenges'])} challenges to {output_path}", file=sys.stderr)
            else:
                print(json_str)

        else:
            # output single challenge (first one, or the only one)
            if len(bundle["challenges"]) > 1:
                print(
                    f"Note: Generated {len(bundle['challenges'])} challenges, "
                    "outputting first one only. Use --bundle or --output-dir for all.",
                    file=sys.stderr,
                )

            challenge = bundle["challenges"][0]
            json_str = json.dumps(challenge, indent=2 if args.pretty else None)

            if args.output:
                output_path = Path(args.output)
                output_path.write_text(json_str)
                print(f"\nWrote challenge to {output_path}", file=sys.stderr)
            else:
                print(json_str)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
