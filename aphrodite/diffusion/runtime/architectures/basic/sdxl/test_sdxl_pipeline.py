#!/usr/bin/env python3
"""
Test suite for SDXL pipeline architecture.
Test-driven development: tests first, then implementation.
"""

import os

from aphrodite.diffusion.runtime.architectures.basic.sdxl.sdxl_pipeline import SDXLPipeline
from aphrodite.diffusion.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)


def test_sdxl_pipeline_structure():
    """Test SDXL pipeline structure and required modules."""
    print("=" * 50)
    print("SDXL Pipeline Architecture Test Suite")
    print("=" * 50)

    # Initialize distributed environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12358")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        enable_cfg_parallel=False,
        dp_size=1,
    )

    print("\n1. Testing SDXL pipeline class structure...")
    try:
        # Check pipeline class exists
        assert hasattr(SDXLPipeline, "pipeline_name")
        assert SDXLPipeline.pipeline_name == "StableDiffusionXLPipeline"

        # Check required modules
        assert hasattr(SDXLPipeline, "_required_config_modules")
        required_modules = SDXLPipeline._required_config_modules

        print("   ✓ Pipeline class structure correct!")
        print(f"   Pipeline name: {SDXLPipeline.pipeline_name}")
        print(f"   Required modules: {required_modules}")

        # Verify required modules for SDXL
        expected_modules = [
            "text_encoder",  # Dual CLIP (SDXLClipModel)
            "tokenizer",  # SDXL tokenizer
            "unet",  # SDXL UNet
            "vae",  # Standard AutoencoderKL
            "scheduler",  # Diffusion scheduler
        ]

        for module in expected_modules:
            assert module in required_modules, f"Missing required module: {module}"

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n2. Testing pipeline class methods...")
    try:
        # Check that create_pipeline_stages method exists
        assert hasattr(SDXLPipeline, "create_pipeline_stages")
        assert callable(SDXLPipeline.create_pipeline_stages)

        # Check that it's an instance method (not a class method)
        import inspect

        sig = inspect.signature(SDXLPipeline.create_pipeline_stages)
        assert "server_args" in sig.parameters

        print("   ✓ Pipeline methods structure correct!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n3. Testing pipeline inheritance...")
    try:
        from aphrodite.diffusion.runtime.pipelines import ComposedPipelineBase

        # Check inheritance
        assert issubclass(SDXLPipeline, ComposedPipelineBase)

        print("   ✓ Pipeline inheritance correct!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n4. Testing EntryClass registration...")
    try:
        from aphrodite.diffusion.runtime.architectures.basic.sdxl.sdxl_pipeline import EntryClass

        assert EntryClass == SDXLPipeline
        print("   ✓ EntryClass registered correctly!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_sdxl_pipeline_structure()
    exit(0 if success else 1)
