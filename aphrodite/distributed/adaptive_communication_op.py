"""
Adaptive communication operations for variable-size tensor parallelism.

This module provides communication primitives that can handle uneven tensor splits
across heterogeneous GPU configurations.
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.distributed
from loguru import logger

from aphrodite.distributed.parallel_state import get_tp_group
from aphrodite.distributed.adaptive_parallel_state import TensorSplit


def adaptive_tensor_model_parallel_all_gather(
    input_: torch.Tensor,
    splits: List[TensorSplit],
    dim: int = -1
) -> torch.Tensor:
    """
    All-gather operation for adaptive tensor parallelism with variable-size chunks.
    
    Args:
        input_: Local tensor chunk from this device
        splits: List of tensor splits defining how the tensor is distributed
        dim: Dimension along which to gather (default: last dimension)
        
    Returns:
        Complete tensor gathered from all devices
    """
    if not splits:
        return input_
    
    tp_group = get_tp_group()
    current_rank = tp_group.rank_in_group
    
    # Find our split
    our_split = None
    for split in splits:
        if split.device_id == current_rank:
            our_split = split
            break
    
    if our_split is None:
        raise ValueError(f"No split found for rank {current_rank}")
    
    # If only one device, return input as-is
    if len(splits) == 1:
        return input_
    
    # Prepare list to store gathered tensors
    gathered_tensors = []
    
    # Handle negative dimension
    if dim < 0:
        dim = input_.dim() + dim
    
    # All-gather each chunk
    for split in splits:
        if split.device_id == current_rank:
            # This is our chunk - use our input
            gathered_tensors.append(input_)
        else:
            # Create tensor to receive data from other device
            shape = list(input_.shape)
            # Adjust the gathering dimension size based on the split
            our_size = our_split.size
            other_size = split.size
            
            if our_size > 0:
                shape[dim] = int(shape[dim] * other_size / our_size)
            else:
                shape[dim] = other_size
            
            recv_tensor = torch.empty(shape, dtype=input_.dtype, device=input_.device)
            
            # Use point-to-point communication
            if current_rank < split.device_id:
                # Send our data, then receive theirs
                torch.distributed.send(input_, dst=split.device_id, group=tp_group.device_group)
                torch.distributed.recv(recv_tensor, src=split.device_id, group=tp_group.device_group)
            else:
                # Receive theirs, then send our data
                torch.distributed.recv(recv_tensor, src=split.device_id, group=tp_group.device_group)
                torch.distributed.send(input_, dst=split.device_id, group=tp_group.device_group)
            
            gathered_tensors.append(recv_tensor)
    
    # Concatenate all chunks along the specified dimension
    return torch.cat(gathered_tensors, dim=dim)


def adaptive_tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    splits: Optional[List[TensorSplit]] = None
) -> torch.Tensor:
    """
    All-reduce operation for adaptive tensor parallelism.
    
    For most cases, all-reduce doesn't need to know about splits since we're
    summing contributions. However, we may need to handle weighting in the future.
    
    Args:
        input_: Tensor to reduce across all devices
        splits: Optional split information (for future weighted reductions)
        
    Returns:
        Reduced tensor
    """
    # For now, use standard all-reduce since we're summing contributions
    tp_group = get_tp_group()
    return tp_group.all_reduce(input_)


def adaptive_tensor_model_parallel_gather(
    input_: torch.Tensor,
    splits: List[TensorSplit],
    dst: int = 0,
    dim: int = -1
) -> Optional[torch.Tensor]:
    """
    Gather operation for adaptive tensor parallelism.
    
    Args:
        input_: Local tensor chunk
        splits: List of tensor splits
        dst: Destination rank (only this rank will have the result)
        dim: Dimension along which to gather
        
    Returns:
        Complete tensor (only on dst rank), None on other ranks
    """
    tp_group = get_tp_group()
    current_rank = tp_group.rank_in_group
    
    if current_rank == dst:
        # We are the destination - gather from all ranks
        return adaptive_tensor_model_parallel_all_gather(input_, splits, dim)
    else:
        # We are not the destination - send our data
        torch.distributed.send(input_, dst=dst, group=tp_group.device_group)
        return None


def adaptive_split_tensor_for_tp(
    tensor: torch.Tensor,
    splits: List[TensorSplit],
    dim: int = -1
) -> torch.Tensor:
    """
    Split a tensor according to adaptive TP splits and return the chunk for current device.
    
    Args:
        tensor: Complete tensor to split
        splits: List of tensor splits defining the distribution
        dim: Dimension along which to split
        
    Returns:
        Tensor chunk for the current device
    """
    tp_group = get_tp_group()
    current_rank = tp_group.rank_in_group
    
    # Find our split
    our_split = None
    for split in splits:
        if split.device_id == current_rank:
            our_split = split
            break
    
    if our_split is None:
        raise ValueError(f"No split found for rank {current_rank}")
    
    # Handle negative dimension
    if dim < 0:
        dim = tensor.dim() + dim
    
    # Split the tensor and return our chunk
    start_idx = our_split.start_idx
    end_idx = our_split.end_idx
    
    # Create slice for the specified dimension
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start_idx, end_idx)
    
    return tensor[tuple(slices)].contiguous()


def broadcast_adaptive_tensor_splits(
    splits: List[TensorSplit],
    src: int = 0
) -> List[TensorSplit]:
    """
    Broadcast tensor split information to all ranks in the TP group.
    
    Args:
        splits: Tensor splits (only valid on src rank)
        src: Source rank that has the split information
        
    Returns:
        Tensor splits broadcasted to all ranks
    """
    tp_group = get_tp_group()
    current_rank = tp_group.rank_in_group
    
    if current_rank == src:
        # We are the source - broadcast the splits
        # Convert splits to a format suitable for broadcasting
        split_data = []
        for split in splits:
            split_data.extend([split.device_id, split.start_idx, split.end_idx, split.size])
        
        # Create tensor for broadcasting
        data_tensor = torch.tensor(split_data, dtype=torch.int64, device=f"cuda:{current_rank}")
        torch.distributed.broadcast(data_tensor, src=src, group=tp_group.device_group)
        return splits
    else:
        # We are not the source - receive the splits
        # First, get the number of splits
        num_splits = len(tp_group.ranks)  # Assume one split per rank initially
        data_size = num_splits * 4  # 4 integers per split
        
        data_tensor = torch.zeros(data_size, dtype=torch.int64, device=f"cuda:{current_rank}")
        torch.distributed.broadcast(data_tensor, src=src, group=tp_group.device_group)
        
        # Convert back to TensorSplit objects
        splits = []
        data = data_tensor.cpu().tolist()
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                device_id, start_idx, end_idx, size = data[i:i+4]
                splits.append(TensorSplit(device_id, start_idx, end_idx, size))
        
        return splits


class AdaptiveTPCommunicationGroup:
    """
    Communication group manager for adaptive tensor parallelism.
    
    This class provides a high-level interface for managing communication
    operations with adaptive tensor splits.
    """
    
    def __init__(self, splits: Dict[str, List[TensorSplit]]):
        """
        Initialize communication group with split information.
        
        Args:
            splits: Dictionary mapping split types to their configurations
        """
        self.splits = splits
        self.tp_group = get_tp_group()
        
    def all_gather_attention(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """All-gather for attention tensors."""
        return adaptive_tensor_model_parallel_all_gather(
            input_, self.splits.get('attention', []), dim
        )
    
    def all_gather_mlp_input(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """All-gather for MLP input tensors."""
        return adaptive_tensor_model_parallel_all_gather(
            input_, self.splits.get('mlp_input', []), dim
        )
    
    def all_gather_vocab(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """All-gather for vocabulary tensors."""
        return adaptive_tensor_model_parallel_all_gather(
            input_, self.splits.get('vocab', []), dim
        )
    
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """All-reduce operation (same for adaptive and regular TP)."""
        return adaptive_tensor_model_parallel_all_reduce(input_)
    
    def split_attention_tensor(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split attention tensor for current device."""
        return adaptive_split_tensor_for_tp(
            tensor, self.splits.get('attention', []), dim
        )
    
    def split_mlp_input_tensor(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split MLP input tensor for current device."""
        return adaptive_split_tensor_for_tp(
            tensor, self.splits.get('mlp_input', []), dim
        )
    
    def split_vocab_tensor(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Split vocabulary tensor for current device."""
        return adaptive_split_tensor_for_tp(
            tensor, self.splits.get('vocab', []), dim
        )
    
    def get_current_rank_splits(self) -> Dict[str, Optional[TensorSplit]]:
        """Get tensor splits for the current rank."""
        current_rank = get_tp_group().rank_in_group  # Get fresh rank each time
        rank_splits = {}
        
        for split_type, split_list in self.splits.items():
            rank_split = None
            for split in split_list:
                if split.device_id == current_rank:
                    rank_split = split
                    break
            rank_splits[split_type] = rank_split
        
        return rank_splits
    
    def __repr__(self):
        return f"AdaptiveTPCommunicationGroup(splits={list(self.splits.keys())})" 