"""
Adaptive tensor parallel state management.

This module provides the context and state management for adaptive tensor
parallelism, enabling memory-aware distribution of tensors across heterogeneous GPUs.
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger

from aphrodite.distributed.adaptive_utils import (
    get_all_gpu_memory,
    calculate_memory_based_splits, 
    estimate_cache_memory_mb,
    validate_splits
)


@dataclass
class TensorSplit:
    """Represents a tensor split across devices."""
    device_id: int
    start_idx: int
    end_idx: int
    size: int
    
    def __post_init__(self):
        self.size = self.end_idx - self.start_idx
        
    def __repr__(self):
        return f"TensorSplit(dev={self.device_id}, range=[{self.start_idx}:{self.end_idx}], size={self.size})"


class AdaptiveTPContext:
    """Context for managing adaptive tensor parallelism."""
    
    def __init__(
        self,
        model_config: Any,
        tensor_parallel_size: int,
        strategy: str = "balanced",
        memory_ratios: Optional[List[float]] = None,
        min_chunk_size: int = 32,
        expected_cache_tokens: Optional[int] = None
    ):
        """
        Initialize adaptive TP context.
        
        Args:
            model_config: Model configuration object
            tensor_parallel_size: Number of tensor parallel processes
            strategy: "memory", "balanced", or "manual"
            memory_ratios: Manual ratios when strategy="manual"
            min_chunk_size: Minimum chunk size for splitting
            expected_cache_tokens: Expected cache size for memory planning
        """
        self.model_config = model_config
        self.tensor_parallel_size = tensor_parallel_size
        self.strategy = strategy
        self.memory_ratios = memory_ratios or []
        self.min_chunk_size = min_chunk_size
        self.expected_cache_tokens = expected_cache_tokens
        
        # GPU memory information
        self.gpu_memory_info = {}
        self.attention_splits: List[TensorSplit] = []
        self.mlp_input_splits: List[TensorSplit] = []
        self.mlp_output_splits: List[TensorSplit] = []
        self.vocab_splits: List[TensorSplit] = []
        
        # Initialize splits
        self._detect_gpu_memory()
        self._calculate_tensor_splits()
    
    def _detect_gpu_memory(self):
        """Detect GPU memory information."""
        try:
            self.gpu_memory_info = get_all_gpu_memory()
            if self.gpu_memory_info:
                total_memory = sum(info['total'] for info in self.gpu_memory_info.values())
                free_memory = sum(info['free'] for info in self.gpu_memory_info.values())
                logger.info(f"Detected {len(self.gpu_memory_info)} GPUs: "
                           f"total={total_memory//1024:.1f}GB, free={free_memory//1024:.1f}GB")
            else:
                logger.warning("Could not detect GPU memory, falling back to balanced mode")
                self.strategy = "balanced"
        except Exception as e:
            logger.warning(f"GPU memory detection failed: {e}, falling back to balanced mode")
            self.strategy = "balanced"
    
    def _calculate_tensor_splits(self):
        """Calculate tensor splits for different components."""
        # Get model dimensions
        num_attention_heads = getattr(self.model_config, 'num_attention_heads', 0)
        num_kv_heads = getattr(self.model_config, 'num_key_value_heads', num_attention_heads)
        intermediate_size = getattr(self.model_config, 'intermediate_size', 0)
        hidden_size = getattr(self.model_config, 'hidden_size', 0)
        vocab_size = getattr(self.model_config, 'vocab_size', 0)
        
        # Calculate attention head splits
        if num_attention_heads > 0:
            self.attention_splits = self._calculate_splits(
                total_items=num_attention_heads,
                name="attention_heads"
            )
        
        # Calculate MLP splits (intermediate dimension)
        if intermediate_size > 0:
            # Split intermediate dimension into chunks for column-parallel layers
            num_chunks = max(1, intermediate_size // self.min_chunk_size)
            self.mlp_input_splits = self._calculate_splits(
                total_items=num_chunks,
                name="mlp_intermediate_chunks"
            )
            
            # Row-parallel layer splits (hidden dimension)
            hidden_chunks = max(1, hidden_size // self.min_chunk_size)
            self.mlp_output_splits = self._calculate_splits(
                total_items=hidden_chunks, 
                name="mlp_hidden_chunks"
            )
        
        # Calculate vocabulary splits
        if vocab_size > 0:
            vocab_chunks = max(1, (vocab_size + 31) // 32)  # Round up to 32
            self.vocab_splits = self._calculate_splits(
                total_items=vocab_chunks,
                name="vocab_chunks"
            )
    
    def _calculate_splits(
        self, 
        total_items: int, 
        name: str
    ) -> List[TensorSplit]:
        """Calculate splits for a given number of items."""
        if total_items == 0:
            return []
        
        if self.strategy == "balanced":
            # Equal split (current behavior)
            items_per_device = total_items // self.tensor_parallel_size
            remainder = total_items % self.tensor_parallel_size
            
            splits = []
            start = 0
            for i in range(self.tensor_parallel_size):
                size = items_per_device + (1 if i < remainder else 0)
                if size > 0:
                    splits.append(TensorSplit(i, start, start + size, size))
                    start += size
                    
        elif self.strategy == "memory":
            # Memory-based split
            if not self.gpu_memory_info:
                logger.warning(f"No GPU memory info for {name}, falling back to balanced")
                return self._calculate_splits(total_items, "balanced_fallback")
            
            # Estimate cache memory per device
            cache_memory_mb = 0
            if self.expected_cache_tokens:
                head_dim = getattr(self.model_config, 'head_dim', 
                                 getattr(self.model_config, 'hidden_size', 4096) // 
                                 getattr(self.model_config, 'num_attention_heads', 32))
                num_layers = getattr(self.model_config, 'num_hidden_layers', 32)
                num_kv_heads = getattr(self.model_config, 'num_key_value_heads',
                                     getattr(self.model_config, 'num_attention_heads', 32))
                
                cache_memory_mb = estimate_cache_memory_mb(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim, 
                    max_seq_len=self.expected_cache_tokens,
                    batch_size=1
                ) // self.tensor_parallel_size  # Distribute across devices
            
            split_sizes = calculate_memory_based_splits(
                total_items=total_items,
                gpu_memory_info=self.gpu_memory_info,
                num_devices=self.tensor_parallel_size,
                cache_memory_mb=cache_memory_mb
            )
            
            splits = []
            start = 0
            for i, size in enumerate(split_sizes):
                if size > 0:
                    splits.append(TensorSplit(i, start, start + size, size))
                    start += size
                    
        elif self.strategy == "manual":
            # Manual ratios
            if not self.memory_ratios or len(self.memory_ratios) != self.tensor_parallel_size:
                logger.warning(f"Invalid memory ratios for {name}, falling back to balanced")
                return self._calculate_splits(total_items, "balanced_fallback")
            
            from aphrodite.distributed.adaptive_utils import adaptive_integer_split
            split_sizes = adaptive_integer_split(total_items, self.memory_ratios)
            
            splits = []
            start = 0
            for i, size in enumerate(split_sizes):
                if size > 0:
                    splits.append(TensorSplit(i, start, start + size, size))
                    start += size
        else:
            raise ValueError(f"Unknown adaptive TP strategy: {self.strategy}")
        
        # Validate splits
        total_split = sum(split.size for split in splits)
        if not validate_splits([split.size for split in splits], total_items):
            logger.error(f"Invalid splits for {name}: {splits}")
            raise ValueError(f"Split validation failed for {name}")
        
        logger.info(f"Calculated {name} splits: {splits}")
        return splits
    
    def get_attention_split(self, device_id: int) -> Optional[TensorSplit]:
        """Get attention head split for a specific device."""
        for split in self.attention_splits:
            if split.device_id == device_id:
                return split
        return None
    
    def get_mlp_input_split(self, device_id: int) -> Optional[TensorSplit]:
        """Get MLP input split for a specific device.""" 
        for split in self.mlp_input_splits:
            if split.device_id == device_id:
                return split
        return None
    
    def get_mlp_output_split(self, device_id: int) -> Optional[TensorSplit]:
        """Get MLP output split for a specific device."""
        for split in self.mlp_output_splits:
            if split.device_id == device_id:
                return split
        return None
    
    def get_vocab_split(self, device_id: int) -> Optional[TensorSplit]:
        """Get vocabulary split for a specific device."""
        for split in self.vocab_splits:
            if split.device_id == device_id:
                return split
        return None
    
    def get_all_splits(self) -> Dict[str, List[TensorSplit]]:
        """Get all calculated splits."""
        return {
            'attention': self.attention_splits,
            'mlp_input': self.mlp_input_splits,
            'mlp_output': self.mlp_output_splits,
            'vocab': self.vocab_splits
        }
    
    def print_split_summary(self):
        """Print a summary of all splits."""
        logger.info("=== Adaptive TP Split Summary ===")
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"Tensor Parallel Size: {self.tensor_parallel_size}")
        
        if self.gpu_memory_info:
            logger.info("GPU Memory Info:")
            for dev_id, info in self.gpu_memory_info.items():
                logger.info(f"  GPU {dev_id}: {info['free']//1024:.1f}GB free / {info['total']//1024:.1f}GB total")
        
        splits = self.get_all_splits()
        for split_type, split_list in splits.items():
            if split_list:
                logger.info(f"{split_type.title()} Splits:")
                for split in split_list:
                    logger.info(f"  {split}")
        logger.info("=" * 35) 