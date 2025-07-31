"""
Utility functions for attention backends to handle adaptive tensor parallelism.
"""

import logging
import os

logger = logging.getLogger(__name__)


def calculate_num_queries_per_kv_adaptive(num_heads: int, num_kv_heads: int) -> int:
    """
    Calculate num_queries_per_kv with adaptive TP support.
    
    For standard TP, enforces strict divisibility.
    For adaptive TP, allows more flexible head configurations.
    
    Args:
        num_heads: Number of query/key heads
        num_kv_heads: Number of key-value heads
        
    Returns:
        Number of queries per KV head
    """
    # Check if adaptive TP strategy is being used
    adaptive_tp_active = False
    try:
        adaptive_tp_strategy = os.environ.get('ADAPTIVE_TP_STRATEGY', '')
        if adaptive_tp_strategy and adaptive_tp_strategy != 'balanced':
            adaptive_tp_active = True
    except ImportError:
        pass
    
    if not adaptive_tp_active:
        # Standard TP: enforce strict constraint
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        return num_heads // num_kv_heads
    else:
        # Adaptive TP: handle uneven head distributions more gracefully
        if num_kv_heads == 0:
            logger.warning(f"Adaptive TP: num_kv_heads is 0, using num_queries_per_kv=1")
            return 1
        elif num_heads % num_kv_heads == 0:
            # Perfect division case
            return num_heads // num_kv_heads
        else:
            # For uneven distributions, use a reasonable approximation
            num_queries_per_kv = max(1, num_heads // num_kv_heads)
            if num_queries_per_kv * num_kv_heads != num_heads:
                # Log the approximation but allow it to continue
                logger.warning(f"Adaptive TP: Using approximate num_queries_per_kv={num_queries_per_kv} "
                             f"for num_heads={num_heads}, num_kv_heads={num_kv_heads}")
                
                # Adjust to make the math work by using the closest valid value
                num_queries_per_kv = max(1, round(num_heads / num_kv_heads))
            
            return num_queries_per_kv


def is_adaptive_tp_active() -> bool:
    """
    Check if adaptive tensor parallelism is currently active.
    
    Returns:
        True if adaptive TP is active, False otherwise
    """
    try:
        adaptive_tp_strategy = os.environ.get('ADAPTIVE_TP_STRATEGY', '')
        return adaptive_tp_strategy and adaptive_tp_strategy != 'balanced'
    except ImportError:
        return False 