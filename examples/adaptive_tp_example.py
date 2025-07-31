#!/usr/bin/env python3
"""
Adaptive Tensor Parallelism Example

This example demonstrates how to use Aphrodite's adaptive tensor parallelism
feature with different strategies and configurations.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

from aphrodite import AsyncAphrodite, AsyncEngineArgs


async def run_adaptive_tp_example():
    """Run examples of adaptive tensor parallelism with different strategies."""
    
    # Use a small model for demonstration
    model_name = "microsoft/DialoGPT-small"  # ~117M parameters, good for testing
    
    print("🚀 Adaptive Tensor Parallelism Examples")
    print("=" * 50)
    
    # Example 1: Memory-based adaptive TP
    print("\n📊 Example 1: Memory-based Adaptive TP")
    print("This automatically distributes work based on available GPU memory")
    
    try:
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=2,  # Use 2 GPUs
            adaptive_tp_strategy="memory",
            adaptive_tp_expected_cache_tokens=4096,
            max_model_len=512,  # Keep it small for demo
            enforce_eager=True  # Disable CUDA graphs for simpler demo
        )
        
        print(f"Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Tensor Parallel Size: 2")
        print(f"  Strategy: memory-based")
        print(f"  Expected Cache Tokens: 4096")
        
        async_engine = AsyncAphrodite.from_engine_args(engine_args)
        
        # Test with a simple prompt
        prompt = "Hello, how are you today?"
        print(f"\nPrompt: '{prompt}'")
        
        results = []
        async for output in async_engine.generate(prompt, max_tokens=50):
            results.append(output)
        
        if results:
            print(f"Response: '{results[-1].outputs[0].text}'")
            print("✅ Memory-based adaptive TP working successfully!")
        
        # Clean up
        await async_engine.stop_background_loop()
        del async_engine
        
    except Exception as e:
        print(f"❌ Memory-based example failed: {e}")
        print("This may be expected if you don't have 2 GPUs or sufficient memory")
    
    # Example 2: Manual ratio adaptive TP
    print("\n🎯 Example 2: Manual Ratio Adaptive TP")
    print("This uses user-specified ratios for work distribution")
    
    try:
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=2,
            adaptive_tp_strategy="manual",
            adaptive_tp_memory_ratios=[3.0, 1.0],  # 3:1 ratio
            max_model_len=512,
            enforce_eager=True
        )
        
        print(f"Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Tensor Parallel Size: 2")
        print(f"  Strategy: manual ratios [3.0, 1.0]")
        print(f"  GPU 0 gets 75% of work, GPU 1 gets 25%")
        
        async_engine = AsyncAphrodite.from_engine_args(engine_args)
        
        prompt = "What is artificial intelligence?"
        print(f"\nPrompt: '{prompt}'")
        
        results = []
        async for output in async_engine.generate(prompt, max_tokens=50):
            results.append(output)
        
        if results:
            print(f"Response: '{results[-1].outputs[0].text}'")
            print("✅ Manual ratio adaptive TP working successfully!")
        
        # Clean up
        await async_engine.stop_background_loop()
        del async_engine
        
    except Exception as e:
        print(f"❌ Manual ratio example failed: {e}")
    
    # Example 3: Balanced (standard) TP for comparison
    print("\n⚖️  Example 3: Balanced (Standard) TP")
    print("This is the traditional equal distribution")
    
    try:
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,  # Single GPU for simplicity
            adaptive_tp_strategy="balanced",  # Default
            max_model_len=512,
            enforce_eager=True
        )
        
        print(f"Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Tensor Parallel Size: 1")
        print(f"  Strategy: balanced (standard TP)")
        
        async_engine = AsyncAphrodite.from_engine_args(engine_args)
        
        prompt = "Tell me about machine learning."
        print(f"\nPrompt: '{prompt}'")
        
        results = []
        async for output in async_engine.generate(prompt, max_tokens=50):
            results.append(output)
        
        if results:
            print(f"Response: '{results[-1].outputs[0].text}'")
            print("✅ Balanced TP working successfully!")
        
        # Clean up
        await async_engine.stop_background_loop()
        del async_engine
        
    except Exception as e:
        print(f"❌ Balanced example failed: {e}")


def show_cli_examples():
    """Show CLI usage examples."""
    print("\n🔧 CLI Usage Examples")
    print("=" * 30)
    
    examples = [
        {
            "name": "Memory-based adaptive TP",
            "cmd": "aphrodite serve microsoft/DialoGPT-small --tensor-parallel-size 2 --adaptive-tp-strategy memory"
        },
        {
            "name": "Manual ratio adaptive TP",
            "cmd": "aphrodite serve microsoft/DialoGPT-small --tensor-parallel-size 3 --adaptive-tp-strategy manual --adaptive-tp-memory-ratios 4.0 2.0 1.0"
        },
        {
            "name": "Memory strategy with custom cache size",
            "cmd": "aphrodite serve microsoft/DialoGPT-small --adaptive-tp-strategy memory --adaptive-tp-expected-cache-tokens 8192"
        },
        {
            "name": "Advanced configuration",
            "cmd": "aphrodite serve microsoft/DialoGPT-small --tensor-parallel-size 4 --adaptive-tp-strategy memory --adaptive-tp-min-chunk-size 64"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(f"   {example['cmd']}")
    
    print(f"\n💡 Pro Tips:")
    print(f"   • Use 'memory' strategy for heterogeneous GPU setups")
    print(f"   • Use 'manual' strategy for fine-tuned workload distribution")  
    print(f"   • Use 'balanced' strategy for homogeneous setups (default)")
    print(f"   • Monitor GPU memory usage to optimize ratios")


async def main():
    """Main function to run all examples."""
    print("🎯 Aphrodite Adaptive Tensor Parallelism Demo")
    print("This example demonstrates the new adaptive TP feature")
    print("that enables efficient inference on heterogeneous GPU setups.")
    
    # Show CLI examples first
    show_cli_examples()
    
    # Ask user if they want to run the async examples
    print(f"\nWould you like to run the async API examples? (requires GPUs)")
    response = input("Enter 'y' to run examples, any other key to skip: ")
    
    if response.lower() == 'y':
        await run_adaptive_tp_example()
    else:
        print("Skipping async examples. See CLI examples above!")
    
    print(f"\n🎉 Adaptive TP demo completed!")
    print(f"For more information, see the documentation and examples.")


if __name__ == "__main__":
    # Check if we're in an async context already
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # We're already in an async context, just await
            import asyncio
            loop = asyncio.get_event_loop()
            loop.create_task(main())
        else:
            raise 