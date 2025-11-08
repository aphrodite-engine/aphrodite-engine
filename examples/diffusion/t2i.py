#!/usr/bin/env python3
"""
Parameterized client for calling Aphrodite text-to-image API
python openai_client.py --bot-task image --width 1024 --height 1024 --seed 42
"""

import argparse
import base64
import io
import json
import random

import gradio as gr
import requests
from PIL import Image


def default(value, default_value):
    return value if value is not None else default_value


# ------------------ Default Parameters ------------------
DEFAULTS = {
    "prompt": "Generate an image: In a colosseum, a woman and a bear engage in combat, illuminated by torchlight. "
    "Rendered in 3D style.",
    "url": "http://0.0.0.0:2242/v1/chat/completions",
    "model": "hunyuan_image3",
    "max_tokens": 256,
    "temperature": 0,
}

# ------------------ Template Selection ------------------
TEMPLATES_PRETRAIN = {
    "image": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|startoftext|>{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "auto": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|startoftext|>{{ message['content'] }}<boi><image_shape_1024>"
        "{% endif %}"
        "{% endfor %}"
    ),
}


# ------------------ Main Logic ------------------
def build_payload(args):
    if args.sequence_template == "pretrain":
        templates = TEMPLATES_PRETRAIN
    else:
        raise NotImplementedError(f"Sequence template {args.sequence_template} not implemented.")

    chat_template = templates[args.bot_task]
    task_extra_kwargs = {
        "diff_infer_steps": args.diff_infer_steps,
        "use_system_prompt": args.use_system_prompt,
        "bot_task": args.bot_task,
    }
    if args.bot_task == "image":
        task_extra_kwargs["image_size"] = f"{default(args.height, 1024)}x{default(args.width, 1024)}"

    max_tokens = args.max_tokens
    if args.bot_task in ["image", "auto"]:
        max_tokens = 1

    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": ""}, {"role": "user", "content": args.prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": args.temperature,
        "seed": default(args.seed, random.randint(1, 10_000_000)),
        "chat_template": chat_template,
        "task_type": "t2i",
        "task_extra_kwargs": task_extra_kwargs,
    }
    return payload


def generate_image(
    prompt,
    url,
    model,
    width,
    height,
    bot_task,
    diff_infer_steps,
    use_system_prompt,
    system_prompt,
    seed,
    temperature,
):
    class Args:
        def __init__(self):
            self.sequence_template = "pretrain"
            self.prompt = prompt
            self.url = url
            self.model = model
            self.width = width
            self.height = height
            self.bot_task = bot_task
            self.diff_infer_steps = diff_infer_steps
            self.use_system_prompt = use_system_prompt
            self.system_prompt = system_prompt
            self.seed = seed
            self.temperature = temperature
            self.max_tokens = 256

    args = Args()
    payload = build_payload(args)
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(args.url, data=json.dumps(payload), headers=headers, timeout=10000)
        if resp.status_code != 200:
            return None, f"Error: {resp.status_code}\n{resp.text}"

        data = resp.json()
        base64_image = data["image"]

        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        return image, "Image generated successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"


def build_gradio_ui():
    with gr.Blocks(title="Aphrodite Text-to-Image", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Aphrodite Text-to-Image Generator
            
            Generate images from text prompts using the Aphrodite API.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=4,
                    value=DEFAULTS["prompt"],
                )

                with gr.Accordion("Advanced Settings", open=False):
                    url_input = gr.Textbox(
                        label="API URL",
                        value=DEFAULTS["url"],
                    )
                    model_input = gr.Textbox(
                        label="Model",
                        value=DEFAULTS["model"],
                    )

                    with gr.Row():
                        width_input = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024,
                        )
                        height_input = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024,
                        )

                    bot_task_input = gr.Dropdown(
                        label="Task Type",
                        choices=["image", "auto", "think", "recaption"],
                        value="image",
                        info=(
                            "'image' for direct generation, 'auto' for text generation, "
                            "'think' for think->re-write->image, 'recaption' for re-write->image"
                        ),
                    )

                    diff_infer_steps_input = gr.Slider(
                        label="Diffusion Inference Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                    )

                    use_system_prompt_input = gr.Dropdown(
                        label="System Prompt",
                        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        value="None",
                    )

                    system_prompt_input = gr.Textbox(
                        label="Custom System Prompt",
                        placeholder="Enter custom system prompt (only used when System Prompt is 'custom')",
                        lines=2,
                        visible=False,
                    )

                    use_system_prompt_input.change(
                        lambda x: gr.update(visible=(x == "custom")),
                        inputs=[use_system_prompt_input],
                        outputs=[system_prompt_input],
                    )

                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed",
                            value=None,
                            info="Leave empty for random seed",
                        )
                        temperature_input = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=DEFAULTS["temperature"],
                        )

                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                image_output = gr.Image(label="Generated Image", type="pil")
                status_output = gr.Textbox(label="Status", interactive=False)

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                url_input,
                model_input,
                width_input,
                height_input,
                bot_task_input,
                diff_infer_steps_input,
                use_system_prompt_input,
                system_prompt_input,
                seed_input,
                temperature_input,
            ],
            outputs=[image_output, status_output],
        )

        gr.Examples(
            examples=[
                [
                    "A serene landscape with mountains in the background, "
                    "a lake in the foreground, sunset lighting, photorealistic",
                    1024,
                    1024,
                ],
                [
                    "A futuristic cityscape at night with neon lights, cyberpunk style, highly detailed",
                    1024,
                    1024,
                ],
                [
                    "A cute cat wearing a wizard hat, fantasy art style, vibrant colors",
                    1024,
                    1024,
                ],
            ],
            inputs=[prompt_input, width_input, height_input],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Aphrodite text-to-image API")
    parser.add_argument("--sequence_template", choices=["pretrain", "instruct"], default="pretrain")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--prompt", default=DEFAULTS["prompt"])
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        default="None",
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
        "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
        "three predefined system prompts; 'custom' means using custom system prompt. When "
        "using 'custom', --system-prompt must be provided. Defaults to loading from model "
        "generation config.",
    )
    parser.add_argument(
        "--system-prompt", type=str, default="", help="Custom system prompt. Used when --use-system-prompt is 'custom'."
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        default="image",
        choices=["image", "auto", "think", "recaption"],
        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
        "generation; 'think' for think->re-write->image; 'recaption' for re-write->image. "
        "Defaults to loading from model generation config.",
    )
    parser.add_argument("--url", default=DEFAULTS["url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--max_tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio UI")
    parser.add_argument("--gradio-host", type=str, default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--gradio-port", type=int, default=7860, help="Gradio server port")
    parser.add_argument(
        "--gradio-share",
        type=str,
        default=None,
        help="Create a public URL. Set to 'True' for auto-generated URL, or provide a custom share URL.",
    )

    args = parser.parse_args()

    if args.gradio:
        demo = build_gradio_ui()
        share = None
        if args.gradio_share is not None:
            share = True if args.gradio_share.lower() == "true" else args.gradio_share
        demo.queue().launch(server_name=args.gradio_host, server_port=args.gradio_port, share=share)
        return

    payload = build_payload(args)
    headers = {"Content-Type": "application/json"}

    resp = requests.post(args.url, data=json.dumps(payload), headers=headers, timeout=10000)
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Error:", resp.text)
        return

    data = resp.json()
    base64_image = data["image"]

    if "," in base64_image:
        base64_image = base64_image.split(",")[1]

    image_data = base64.b64decode(base64_image)
    with open("output.png", "wb") as f:
        f.write(image_data)
    print("Image saved as output.png")


if __name__ == "__main__":
    main()
