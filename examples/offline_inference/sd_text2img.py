#!/usr/bin/env python3
"""
Example script for Stable Diffusion text-to-image generation using Aphrodite.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python sd_text2img.py \
        --prompts "A cyberpunk cat, 8k render" \
        --height 512 --width 512 --steps 30 --cfg 7.5 --seed 42
"""

import argparse
import os
import time
from typing import List

from aphrodite.endpoints.llm import LLM
from aphrodite.common.sampling_params import SamplingParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stable Diffusion Text-to-Image Generation")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Pretrained model name or path")
    parser.add_argument("--runner", type=str, default="sd_pipeline",
                       help="Runner type for SD pipeline")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--dtype", type=str, default="float16",
                       help="Model dtype (float16, bfloat16, float32)")
    
    # Generation arguments
    parser.add_argument("--prompts", type=str, nargs="+", 
                       default=["A beautiful landscape with mountains and lakes"],
                       help="Text prompts for image generation")
    parser.add_argument("--negative-prompts", type=str, nargs="*", default=None,
                       help="Negative prompts for CFG")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num-images", type=int, default=1,
                       help="Number of images per prompt")
    parser.add_argument("--eta", type=float, default=0.0,
                       help="DDIM eta parameter")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./sd_outputs",
                       help="Output directory for generated images")
    parser.add_argument("--output-format", type=str, default="png",
                       choices=["png", "jpg", "jpeg"],
                       help="Output image format")
    parser.add_argument("--save-latents", action="store_true",
                       help="Save intermediate latents")
    
    # Performance arguments
    parser.add_argument("--enforce-eager", action="store_true",
                       help="Enforce eager execution")
    parser.add_argument("--max-model-len", type=int, default=77,
                       help="Maximum model length for text encoder")
    
    return parser.parse_args()


def main():
    """Main function for SD text-to-image generation."""
    args = parse_args()
    
    print("🎨 Stable Diffusion Text-to-Image Generation")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize LLM with SD pipeline
    print(f"Loading Stable Diffusion model: {args.model}")
    print(f"Using device: {args.device}, dtype: {args.dtype}")
    
    llm = LLM(
        model=args.model,
        runner=args.runner,
        device=args.device,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )
    
    print("✅ Model loaded successfully")
    
    # Prepare generation parameters
    generation_params = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
        "num_images_per_prompt": args.num_images,
        "eta": args.eta,
        "output_type": "pil",
    }
    
    if args.seed is not None:
        generation_params["generator_seed"] = args.seed
    
    # Add negative prompts if provided
    if args.negative_prompts:
        if len(args.negative_prompts) == 1:
            # Use same negative prompt for all
            generation_params["negative_prompt"] = args.negative_prompts * len(args.prompts)
        elif len(args.negative_prompts) == len(args.prompts):
            # One negative prompt per prompt
            generation_params["negative_prompt"] = args.negative_prompts
        else:
            print("⚠️ Number of negative prompts must be 1 or match number of prompts")
            return
    
    print(f"\n📝 Generation Parameters:")
    print(f"   Prompts: {args.prompts}")
    if args.negative_prompts:
        print(f"   Negative prompts: {args.negative_prompts}")
    print(f"   Image size: {args.width}x{args.height}")
    print(f"   Steps: {args.steps}")
    print(f"   CFG scale: {args.cfg}")
    print(f"   Images per prompt: {args.num_images}")
    if args.seed is not None:
        print(f"   Seed: {args.seed}")
    
    # Generate images
    print(f"\n🎯 Generating images...")
    start_time = time.time()
    
    try:
        # Create sampling params (SD pipeline will extract generation params)
        sampling_params = SamplingParams(
            **generation_params
        )
        
        # Generate images
        outputs = llm.generate(
            prompts=args.prompts,
            sampling_params=sampling_params,
        )
        
        generation_time = time.time() - start_time
        
        print(f"✅ Generation completed in {generation_time:.2f} seconds")
        
        # Save generated images
        total_images = 0
        for i, output in enumerate(outputs):
            prompt = args.prompts[i]
            images = output.outputs[0].images  # Get generated images
            
            for j, image in enumerate(images):
                # Create filename
                safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_prompt = safe_prompt.replace(' ', '_')
                
                if args.seed is not None:
                    filename = f"sd_{i:03d}_{j:03d}_{safe_prompt}_seed{args.seed}.{args.output_format}"
                else:
                    filename = f"sd_{i:03d}_{j:03d}_{safe_prompt}.{args.output_format}"
                
                filepath = os.path.join(args.output_dir, filename)
                
                # Save image
                image.save(filepath)
                print(f"💾 Saved: {filepath}")
                total_images += 1
        
        print(f"\n🎉 Successfully generated and saved {total_images} images!")
        print(f"📁 Output directory: {args.output_dir}")
        
        # Performance stats
        images_per_second = total_images / generation_time
        print(f"📊 Performance: {images_per_second:.2f} images/second")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
