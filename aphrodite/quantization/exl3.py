from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from aphrodite import _custom_ops as ops
from aphrodite.modeling.layers.linear import LinearBase, LinearMethodBase
from aphrodite.modeling.utils import set_weight_attrs
from aphrodite.quantization import QuantizationMethods
from aphrodite.quantization.base_config import QuantizationConfig

# Try to import EXL3 CUDA operations - fallback if not available
try:
    _EXL3_KERNELS_AVAILABLE = hasattr(ops, 'exl3_gemm')
except (ImportError, AttributeError):
    _EXL3_KERNELS_AVAILABLE = False


class EXL3Config(QuantizationConfig):
    """Config class for EXL3 quantization.
    
    EXL3 is based on the QTIP quantization method from Cornell RelaxML,
    using trellis-based encoding with Hadamard transformations.
    """

    def __init__(
        self,
        bits: float,
        head_bits: int = 6,
        calibration: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.calibration = calibration or {}
        
        # Validate bits per weight
        if not (1.0 <= bits <= 8.0):
            raise ValueError(
                f"EXL3 bits per weight must be between 1.0 and 8.0, "
                f"got {bits}")

    def __repr__(self) -> str:
        return (f"EXL3Config(bits={self.bits}, "
                f"head_bits={self.head_bits})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # EXL3 requires modern GPU for optimal performance
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quantization_config.json",
            "config.json"
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EXL3Config":
        # Handle both quantization_config.json and config.json formats
        quant_config = config.get("quantization_config", config)
        
        bits = cls.get_from_keys(quant_config, ["bits"])
        head_bits = cls.get_from_keys_or(quant_config, ["head_bits"], 6)
        calibration = cls.get_from_keys_or(quant_config, ["calibration"], None)
        
        return cls(bits=bits, head_bits=head_bits, calibration=calibration)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["EXL3LinearMethod"]:
        if isinstance(layer, LinearBase):
            return EXL3LinearMethod(self)
        return None


class EXL3LinearMethod(LinearMethodBase):
    """Linear method for EXL3 quantization.
    
    Implements the EXL3 trellis-based quantization with Hadamard transforms
    for efficient GPU inference.
    """

    def __init__(self, quant_config: EXL3Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # Calculate tile dimensions for EXL3 format
        # EXL3 uses 16x16 tiles for optimal GPU memory access
        tiles_k = (input_size_per_partition + 15) // 16
        tiles_n = (output_size_per_partition + 15) // 16
        
        # K parameter determines bits per weight via trellis depth
        # This is derived from the target bits in the config
        K = max(1, int(self.quant_config.bits))

        # Main quantized weight tensor (trellis format)
        # Initialize with proper shape but will be loaded from checkpoint
        trellis = Parameter(
            torch.empty(tiles_k, tiles_n, K * 16, dtype=torch.int16),
            requires_grad=False
        )
        set_weight_attrs(trellis, {
            "input_dim": 0,
            "output_dim": 1,
            "weight_loader": self._make_weight_loader("trellis", weight_loader, layer)
        })

        # Input channel Hadamard sign factors (unpacked format preferred)
        suh = Parameter(
            torch.empty(input_size_per_partition, dtype=torch.half),
            requires_grad=False
        )
        set_weight_attrs(suh, {
            "input_dim": 0,
            "weight_loader": self._make_weight_loader("suh", weight_loader, layer)
        })

        # Output channel Hadamard sign factors (unpacked format preferred)
        svh = Parameter(
            torch.empty(output_size_per_partition, dtype=torch.half),
            requires_grad=False
        )
        set_weight_attrs(svh, {
            "output_dim": 0,
            "weight_loader": self._make_weight_loader("svh", weight_loader, layer)
        })

        # Optional packed versions (for legacy compatibility)
        su = Parameter(
            torch.empty((input_size_per_partition + 15) // 16, dtype=torch.int16),
            requires_grad=False
        )
        set_weight_attrs(su, {
            "input_dim": 0,
            "weight_loader": self._make_weight_loader("su", weight_loader, layer)
        })

        sv = Parameter(
            torch.empty((output_size_per_partition + 15) // 16, dtype=torch.int16),
            requires_grad=False
        )
        set_weight_attrs(sv, {
            "output_dim": 0,
            "weight_loader": self._make_weight_loader("sv", weight_loader, layer)
        })

        # Experimental multipliers (optional)
        mcg = Parameter(
            torch.tensor(0, dtype=torch.int32),
            requires_grad=False
        )
        set_weight_attrs(mcg, {"weight_loader": self._make_weight_loader("mcg", weight_loader, layer)})

        mul1 = Parameter(
            torch.tensor(0, dtype=torch.int32),
            requires_grad=False
        )
        set_weight_attrs(mul1, {"weight_loader": self._make_weight_loader("mul1", weight_loader, layer)})

        # Store tensor shapes and metadata
        layer.K = K
        layer.tiles_k = tiles_k
        layer.tiles_n = tiles_n
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        
        # Initialize multiplier values (will be updated during weight loading)
        layer.mcg_mult = 0
        layer.mul1_mult = 0

        # Register all parameters
        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)
        layer.register_parameter("su", su) 
        layer.register_parameter("sv", sv)
        layer.register_parameter("mcg", mcg)
        layer.register_parameter("mul1", mul1)

    def _make_weight_loader(self, tensor_name: str, default_loader, layer):
        """Create a weight loader that handles EXL3-specific tensor loading."""
        def exl3_weight_loader(param: Parameter, loaded_weight: torch.Tensor, shard_id: Optional[str] = None):
            # Handle different tensor name patterns from EXL3 checkpoints
            if tensor_name in ["trellis", "suh", "svh", "su", "sv", "mcg", "mul1"]:
                # Handle multiplier extraction for mcg and mul1
                if tensor_name == "mcg" and loaded_weight.numel() > 0:
                    # Extract the multiplier value and store it on the layer
                    layer.mcg_mult = loaded_weight.view(torch.uint32).item()
                elif tensor_name == "mul1" and loaded_weight.numel() > 0:
                    # Extract the multiplier value and store it on the layer
                    layer.mul1_mult = loaded_weight.view(torch.uint32).item()
                
                # Ensure the loaded weight matches expected shape and dtype
                if loaded_weight.shape != param.shape:
                    # Handle potential shape mismatches due to padding
                    if tensor_name == "trellis":
                        # Trellis tensor might have different K values
                        if len(loaded_weight.shape) == 3:
                            param.data[:loaded_weight.shape[0], 
                                     :loaded_weight.shape[1], 
                                     :loaded_weight.shape[2]].copy_(loaded_weight)
                            return
                    elif tensor_name in ["suh", "svh"]:
                        # Sign factors might be shorter due to actual vs padded dimensions
                        if loaded_weight.numel() <= param.numel():
                            param.data[:loaded_weight.numel()].copy_(loaded_weight.flatten())
                            return
                
                # Standard copy for matching shapes
                if loaded_weight.dtype != param.dtype:
                    loaded_weight = loaded_weight.to(param.dtype)
                param.data.copy_(loaded_weight)
            else:
                # Fallback to default loader
                if shard_id is not None:
                    default_loader(param, loaded_weight, shard_id)
                else:
                    default_loader(param, loaded_weight)
        
        return exl3_weight_loader

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply EXL3 quantized linear transformation."""
        
        # Get quantization parameters
        trellis = layer.trellis
        suh = getattr(layer, 'suh', None)
        svh = getattr(layer, 'svh', None)
        su = getattr(layer, 'su', None)
        sv = getattr(layer, 'sv', None)
        mcg = getattr(layer, 'mcg', None)
        mul1 = getattr(layer, 'mul1', None)

        # Use pre-extracted multiplier values from the layer (set during weight loading)
        mcg_mult = getattr(layer, 'mcg_mult', 0)
        mul1_mult = getattr(layer, 'mul1_mult', 0)

        # Unpack signs if needed (fallback for legacy format)
        if suh is None and su is not None:
            suh = self._unpack_signs(su)
        if svh is None and sv is not None:
            svh = self._unpack_signs(sv)

        if suh is None or svh is None:
            raise ValueError("EXL3 layer missing required sign factors (suh/svh)")

        # Reshape input for processing
        input_shape = x.shape
        x_reshaped = x.view(-1, x.shape[-1])
        batch_size = x_reshaped.shape[0]

        # Try to use optimized CUDA kernel if available
        if _EXL3_KERNELS_AVAILABLE and x_reshaped.is_cuda:
            # Create empty sign tensors if not provided
            if suh.numel() == 0:
                suh = torch.empty(0, dtype=torch.half, device=x_reshaped.device)
            if svh.numel() == 0:
                svh = torch.empty(0, dtype=torch.half, device=x_reshaped.device)
                
            output = ops.exl3_gemm(
                x_reshaped.to(torch.float16),  # Convert to half precision
                trellis,
                suh,
                svh,
                mcg_mult,
                mul1_mult
            )
            # Convert back to input dtype if needed
            if output.dtype != x.dtype:
                output = output.to(x.dtype)
        else:
            # Use fallback reconstruction method
            output = self._apply_fallback(x_reshaped, trellis, suh, svh,
                                        layer.K, mcg_mult, mul1_mult)

        # Add bias if present
        if bias is not None:
            output = output + bias

        # Reshape output to match input shape
        result = output.view(input_shape[:-1] + (layer.output_size_per_partition,))
        
        return result

    def _unpack_signs(self, packed_signs: torch.Tensor) -> torch.Tensor:
        """Unpack bit-packed sign factors to float16."""
        # Convert packed int16 to individual sign bits
        device = packed_signs.device
        packed_signs = packed_signs.view(torch.uint16).to(torch.int32)
        
        # Extract individual bits and convert to signs
        masks = (1 << torch.arange(16, device=device)).unsqueeze(0)
        expanded = (packed_signs.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        
        # Convert boolean to sign values (-1 or +1)
        signs = torch.where(expanded, 
                           torch.tensor(-1.0, dtype=torch.half, device=device),
                           torch.tensor(1.0, dtype=torch.half, device=device))
        
        return signs.contiguous()

    def _apply_fallback(self, x: torch.Tensor, trellis: torch.Tensor,
                       suh: torch.Tensor, svh: torch.Tensor,
                       K: int, mcg_mult: int, mul1_mult: int) -> torch.Tensor:
        """Fallback implementation using weight reconstruction."""
        # This is a simplified fallback - in practice you'd want the optimized kernel
        
        # Reconstruct the weight matrix from trellis format
        weight = self._reconstruct_weight(trellis, suh, svh, K, mcg_mult, mul1_mult)
        
        # Standard matrix multiplication
        return torch.mm(x, weight)

    def _reconstruct_weight(self, trellis: torch.Tensor, suh: torch.Tensor,
                           svh: torch.Tensor, K: int, mcg_mult: int, mul1_mult: int) -> torch.Tensor:
        """Reconstruct full precision weight from EXL3 format."""
        
        tiles_k, tiles_n, trellis_depth = trellis.shape
        actual_in_features = suh.shape[0]
        actual_out_features = svh.shape[0]
        
        # Create weight matrix with actual dimensions
        weight = torch.zeros(
            (actual_in_features, actual_out_features),
            dtype=torch.float16,
            device=trellis.device
        )
        
        # Decode trellis using 3INST procedural codebook
        for tile_k in range(tiles_k):
            for tile_n in range(tiles_n):
                # Process each 16x16 tile
                tile_start_k = tile_k * 16
                tile_start_n = tile_n * 16
                tile_end_k = min(tile_start_k + 16, actual_in_features)
                tile_end_n = min(tile_start_n + 16, actual_out_features)
                
                # Decode the trellis data for this tile
                for row in range(tile_end_k - tile_start_k):
                    for col in range(tile_end_n - tile_start_n):
                        # Calculate trellis index
                        elem_idx = (row * 16 + col) % (K * 16)
                        
                        # Get quantized value
                        quant_val = trellis[tile_k, tile_n, elem_idx]
                        
                        # Decode using 3INST procedural codebook
                        if mcg_mult != 0:
                            # MCG mode
                            decoded = self._decode_3inst_mcg(quant_val.item(), mcg_mult)
                        elif mul1_mult != 0:
                            # MUL1 mode  
                            decoded = self._decode_3inst_mul1(quant_val.item(), mul1_mult)
                        else:
                            # Default mode
                            decoded = self._decode_3inst_default(quant_val.item())
                        
                        # Store decoded value
                        k_idx = tile_start_k + row
                        n_idx = tile_start_n + col
                        if k_idx < actual_in_features and n_idx < actual_out_features:
                            weight[k_idx, n_idx] = decoded
        
        # Apply Hadamard sign transforms
        # Input signs (suh) transform the rows, output signs (svh) transform the columns
        suh_expanded = suh[:actual_in_features].unsqueeze(1)  # [in_features, 1]
        svh_expanded = svh[:actual_out_features].unsqueeze(0)  # [1, out_features]
        
        # Apply sign factors
        weight = weight * suh_expanded * svh_expanded
        
        return weight

    def _decode_3inst_default(self, x: int) -> float:
        """Default 3INST procedural codebook decoding."""
        # Convert to unsigned 16-bit
        x = x & 0xFFFF
        # Default MCG multiplier
        x = (x * 89226354) & 0xFFFFFFFF
        x = (x + 64248484) & 0xFFFFFFFF
        # Convert to float and normalize
        return float(x) / float(0xFFFFFFFF) * 2.0 - 1.0

    def _decode_3inst_mcg(self, x: int, mult: int) -> float:
        """MCG mode 3INST procedural codebook decoding."""
        # Convert to unsigned 16-bit
        x = x & 0xFFFF
        # MCG mode with custom multiplier
        x = (x * mult) & 0xFFFFFFFF
        # Convert to float and normalize
        return float(x) / float(0xFFFFFFFF) * 2.0 - 1.0

    def _decode_3inst_mul1(self, x: int, mult: int) -> float:
        """MUL1 mode 3INST procedural codebook decoding."""
        # Convert to unsigned 16-bit
        x = x & 0xFFFF
        # MUL1 mode with custom multiplier
        x = (x * mult) & 0xFFFFFFFF
        # Scale and offset (simplified version)
        return float(x) * 6.77e-6 - 10.39 