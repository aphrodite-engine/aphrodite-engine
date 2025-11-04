import argparse
import copy
import gc

import torch
from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear
from marlin import Layer as MarlinLayer
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str)
parser.add_argument("--save-path", type=str)
parser.add_argument("--do-generation", action="store_true")


def _validate_compatibility(model):
    if not hasattr(model.config, "quantization_config"):
        raise ValueError("Must be a quantized model to convert to Marlin Format")
    quantization_config = model.config.quantization_config
    if quantization_config.quant_method != "gptq":
        raise ValueError(
            "Only GPTQ models can be converted to Marlin format. You passed a "
            f"model with quant_method={quantization_config.quant_method}"
        )
    if quantization_config.bits != 4:
        raise ValueError(
            "Only 4 bit quantized models can be converted to Marlin format. "
            f"You passed a model with bits={quantization_config.bits}"
        )
    if quantization_config.group_size != 128:
        raise ValueError(
            "Only group size 128 models can be converted to Marlin format. You "
            f"passed a model with group_size={quantization_config.group_size}"
        )
    if not quantization_config.sym:
        raise ValueError(
            "Only models with symmetric quantization can be converted to "
            "Marlin Format. You passed a model with sym="
            f"{quantization_config.sym}"
        )
    if quantization_config.desc_act:
        raise ValueError(
            "Models with act order quantization cannot be converted to "
            "Marlin Format. You passed a model with desc_act="
            f"{quantization_config.desc_act}"
        )


@torch.no_grad()
def unpack_4bit_to_32bit_signed(qweight, qzeros):
    # Unpack 4-bit values and interpret them as signed integers
    unpacked_weights = torch.zeros(
        (qweight.shape[0] * 8, qweight.shape[1]), dtype=torch.int8, device=qweight.device, requires_grad=False
    )
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8), dtype=torch.int8, device=qzeros.device, requires_grad=False
    )

    for row in range(unpacked_weights.shape[0]):
        i = row % 8
        unpacked_weights[row, :] = (qweight[row // 8, :] >> (4 * i)) & 0xF

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_weights, unpacked_zeros + 1


@torch.no_grad()
def dequantize_weight(layer):
    qweight, qzeros, scales = layer.qweight, layer.qzeros, layer.scales
    unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight.T


@torch.no_grad()
def convert_model(model, verbose=True):
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        if verbose:
            print(f"--- Converting Module: {name}")
        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1 :]

        # Dequantize the weight.
        dequantized_weight = dequantize_weight(module).to(torch.float16)
        linear_module = torch.nn.Linear(
            in_features=dequantized_weight.shape[1],
            out_features=dequantized_weight.shape[0],
            bias=False,
            dtype=torch.float16,
            device="cuda",
        )
        linear_module.weight.data.copy_(dequantized_weight)

        # Create new linear method and copy to model.
        new_module = MarlinLayer(
            infeatures=linear_module.in_features,
            outfeatures=linear_module.out_features,
            groupsize=model.config.quantization_config.group_size,
        )
        new_module.pack(linear_module, scales=copy.deepcopy(module.scales.data.t()))

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del dequantized_weight, module
        torch.cuda.empty_cache()
        gc.collect()

    return model


@torch.no_grad()
def dequantize_model(model, verbose=True):
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue

        if verbose:
            print(f"--- Dequantizing Module: {name}")
        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1 :]

        # Dequantize the weight.
        dequantized_weight = dequantize_weight(module)
        dequantized_weight_cpu = dequantized_weight.to("cpu")

        # Create new linear method and copy to model.
        new_module = torch.nn.Linear(
            in_features=dequantized_weight_cpu.shape[1],
            out_features=dequantized_weight_cpu.shape[0],
            bias=False,
            dtype=torch.float16,
        )
        new_module.weight.data.copy_(dequantized_weight_cpu)
        new_module.scales = torch.nn.Parameter(copy.deepcopy(module.scales.data))

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del dequantized_weight, dequantized_weight_cpu, module
        torch.cuda.empty_cache()

    return model


if __name__ == "__main__":
    args = parser.parse_args()
    model_id = args.model_id
    save_path = args.save_path
    do_generation = args.do_generation

    print("Loading gptq model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Validate that this model is compatible with Marlin.
    print("Validating compatibility...")
    _validate_compatibility(model)

    # Dequantize the Model.
    print("Converting model...")
    model = convert_model(model).to("cpu")

    # Save after updating quantization config.
    print("Saving marlin model...")
    model.config.quantization_config = {
        "group_size": model.config.quantization_config.group_size,
        "quant_method": "marlin",
    }
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if do_generation:
        print("Generating sample text...")
        model.to("cuda")
        prompt = "My favorite song is"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        print(tokenizer.batch_decode(outputs)[0])
