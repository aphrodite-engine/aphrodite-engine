"""
Adaptive tensor parallelism utilities for memory-aware tensor splitting.

This module provides utilities for detecting GPU memory and intelligently
splitting tensors across heterogeneous GPU configurations.
"""

import subprocess
import torch
import os
import json
from typing import Dict, List, Tuple, Optional
from loguru import logger


def get_visible_devices() -> Optional[List[int]]:
    """Get the list of visible CUDA devices from environment."""
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices is None:
        return None
    try:
        return [int(dev) for dev in visible_devices.split(',')]
    except ValueError:
        logger.warning(f"Invalid CUDA_VISIBLE_DEVICES format: {visible_devices}")
        return None


def get_nvidia_gpu_memory(visible_devices: Optional[List[int]] = None) -> Dict[int, Dict[str, int]]:
    """
    Get GPU memory information for NVIDIA GPUs using nvidia-smi.
    
    Args:
        visible_devices: List of visible device indices, or None for all devices
        
    Returns:
        Dictionary mapping device index to memory info (total, used, free in MB)
    """
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,memory.total,memory.used,memory.free',
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True, check=True)
        
        gpu_memory = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            
            try:
                index, total, used, free = map(int, parts)
                if visible_devices is None or index in visible_devices:
                    # Map to sequential indices if using CUDA_VISIBLE_DEVICES
                    mapped_index = visible_devices.index(index) if visible_devices else index
                    gpu_memory[mapped_index] = {
                        'total': total,
                        'used': used, 
                        'free': free
                    }
            except ValueError:
                logger.warning(f"Failed to parse nvidia-smi line: {line}")
                continue
                
        return gpu_memory
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"nvidia-smi not available or failed: {e}")
        return {}


def get_amd_gpu_memory() -> Dict[int, Dict[str, int]]:
    """
    Get GPU memory information for AMD GPUs using rocm-smi.
    
    Returns:
        Dictionary mapping device index to memory info (total, used, free in MB)
    """
    try:
        result = subprocess.run([
            'rocm-smi',
            '--showmeminfo',
            'vram',
            '--json'
        ], capture_output=True, text=True, check=True)
        
        data = json.loads(result.stdout)
        gpu_memory = {}
        
        for gpu in data.get('card', []):
            try:
                index = int(gpu['card_id'])
                total = int(gpu['vram_total']) // (1024 * 1024)  # Convert to MB
                used = int(gpu['vram_used']) // (1024 * 1024)
                free = int(gpu['vram_free']) // (1024 * 1024)
                
                gpu_memory[index] = {
                    'total': total,
                    'used': used,
                    'free': free
                }
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse ROCm GPU info: {e}")
                continue
                
        return gpu_memory
        
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"rocm-smi not available or failed: {e}")
        return {}


def fallback_memory_detection() -> Dict[int, Dict[str, int]]:
    """
    Fallback GPU memory detection using PyTorch CUDA APIs.
    
    Returns:
        Dictionary mapping device index to memory info (total, used, free in MB)
    """
    if not torch.cuda.is_available():
        return {}
    
    gpu_memory = {}
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        try:
            # Get memory info for this device
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            cached_memory = torch.cuda.memory_reserved(i)
            
            total_mb = total_memory // (1024 * 1024)
            used_mb = max(allocated_memory, cached_memory) // (1024 * 1024)
            free_mb = total_mb - used_mb
            
            gpu_memory[i] = {
                'total': total_mb,
                'used': used_mb,
                'free': free_mb
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info for GPU {i}: {e}")
            continue
    
    return gpu_memory


def get_all_gpu_memory() -> Dict[int, Dict[str, int]]:
    """
    Get GPU memory information for all available GPUs.
    
    Tries NVIDIA first, then AMD, then falls back to PyTorch.
    
    Returns:
        Dictionary mapping device index to memory info (total, used, free in MB)
    """
    visible_devices = get_visible_devices()
    
    # Try NVIDIA first
    gpu_memory = get_nvidia_gpu_memory(visible_devices)
    if gpu_memory:
        logger.info(f"Detected {len(gpu_memory)} NVIDIA GPU(s) with memory info")
        return gpu_memory
    
    # Try AMD
    gpu_memory = get_amd_gpu_memory()
    if gpu_memory:
        logger.info(f"Detected {len(gpu_memory)} AMD GPU(s) with memory info")
        return gpu_memory
    
    # Fallback to PyTorch
    gpu_memory = fallback_memory_detection()
    if gpu_memory:
        logger.info(f"Using PyTorch fallback for {len(gpu_memory)} GPU(s)")
        return gpu_memory
    
    logger.error("Unable to detect GPU memory information")
    return {}


def adaptive_integer_split(
    total: int, 
    memory_ratios: List[float], 
    minimum: int = 1
) -> List[int]:
    """
    Split an integer proportionally according to memory ratios while ensuring
    exact sum and integer precision.
    
    Args:
        total: Total value to split
        memory_ratios: Relative ratios for each partition
        minimum: Minimum value per partition (partitions below this become 0)
        
    Returns:
        List of integer portions that sum exactly to total
    """
    if not memory_ratios:
        return []
    
    if total <= 0:
        return [0] * len(memory_ratios)
    
    # Filter out zero ratios
    active_indices = [i for i, ratio in enumerate(memory_ratios) if ratio > 0]
    if not active_indices:
        return [0] * len(memory_ratios)
    
    # Calculate initial proportional split
    sum_ratios = sum(memory_ratios[i] for i in active_indices)
    portions = [0] * len(memory_ratios)
    
    for i in active_indices:
        portions[i] = int(total * memory_ratios[i] / sum_ratios)
    
    # Distribute remaining items to indices with highest fractional remainders
    remaining = total - sum(portions)
    if remaining > 0:
        remainders = []
        for i in active_indices:
            fractional_part = (total * memory_ratios[i] / sum_ratios) - portions[i]
            remainders.append((fractional_part, i))
        
        # Sort by fractional remainder (descending)
        remainders.sort(reverse=True)
        
        for j in range(remaining):
            if j < len(remainders):
                _, idx = remainders[j]
                portions[idx] += 1
    
    # Apply minimum constraint
    adjustment_needed = 0
    for i in range(len(portions)):
        if portions[i] > 0 and portions[i] < minimum:
            adjustment_needed += minimum - portions[i]
            portions[i] = 0
    
    # Redistribute adjustment from largest portions
    if adjustment_needed > 0:
        # Find indices with non-zero portions, sorted by size
        non_zero_indices = [(portions[i], i) for i in range(len(portions)) if portions[i] > 0]
        non_zero_indices.sort(reverse=True)
        
        for adjustment in range(adjustment_needed):
            if non_zero_indices:
                # Take from the largest available portion
                _, idx = non_zero_indices[adjustment % len(non_zero_indices)]
                if portions[idx] > minimum:
                    portions[idx] -= 1
    
    # Ensure exact sum (final adjustment)
    actual_sum = sum(portions)
    if actual_sum != total:
        diff = total - actual_sum
        # Add/subtract from the largest partition
        max_idx = max(range(len(portions)), key=lambda i: portions[i])
        portions[max_idx] = max(0, portions[max_idx] + diff)
    
    return portions


def estimate_cache_memory_mb(
    num_layers: int,
    num_kv_heads: int, 
    head_dim: int,
    max_seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2  # fp16 default
) -> int:
    """
    Estimate KV cache memory requirements in MB.
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value heads
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)
        
    Returns:
        Estimated cache memory in MB
    """
    # KV cache: 2 (K+V) * layers * heads * head_dim * seq_len * batch_size * dtype_bytes
    cache_bytes = (2 * num_layers * num_kv_heads * head_dim * 
                   max_seq_len * batch_size * dtype_bytes)
    return cache_bytes // (1024 * 1024)


def calculate_memory_based_splits(
    total_items: int,
    gpu_memory_info: Dict[int, Dict[str, int]],
    num_devices: int,
    reserved_memory_mb: int = 1024,  # Reserve 1GB per GPU
    cache_memory_mb: int = 0
) -> List[int]:
    """
    Calculate tensor splits based on available GPU memory.
    
    Args:
        total_items: Total number of items to split (e.g., attention heads)
        gpu_memory_info: GPU memory information from get_all_gpu_memory()
        num_devices: Number of devices to split across
        reserved_memory_mb: Memory to reserve per GPU (MB)
        cache_memory_mb: Expected cache memory per GPU (MB)
        
    Returns:
        List of splits for each device
    """
    if num_devices <= 0:
        return []
    
    # Calculate available memory for each device
    memory_ratios = []
    for i in range(num_devices):
        if i in gpu_memory_info:
            free_memory = gpu_memory_info[i]['free']
            available = max(0, free_memory - reserved_memory_mb - cache_memory_mb)
            memory_ratios.append(available)
        else:
            # Device not found, assign zero
            memory_ratios.append(0)
    
    # If no devices have available memory, fall back to equal split
    if sum(memory_ratios) == 0:
        logger.warning("No available memory detected, falling back to equal split")
        equal_split = total_items // num_devices
        remainder = total_items % num_devices
        return [equal_split + (1 if i < remainder else 0) for i in range(num_devices)]
    
    return adaptive_integer_split(total_items, memory_ratios, minimum=1)


def validate_splits(splits: List[int], total_expected: int) -> bool:
    """
    Validate that splits sum to the expected total.
    
    Args:
        splits: List of split sizes
        total_expected: Expected total sum
        
    Returns:
        True if splits are valid
    """
    actual_sum = sum(splits)
    if actual_sum != total_expected:
        logger.error(f"Split validation failed: sum={actual_sum}, expected={total_expected}")
        return False
    
    # Check for negative values
    if any(s < 0 for s in splits):
        logger.error(f"Split validation failed: negative values in {splits}")
        return False
    
    return True 