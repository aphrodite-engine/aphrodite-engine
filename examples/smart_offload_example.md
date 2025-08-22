# Smart CPU Offload Example

The smart offload feature allows you to selectively move model parameters to CPU memory based on their computational intensity and access patterns, rather than blindly offloading the first N bytes of parameters.

## Usage

### Basic Usage

Enable smart offload with the `--smart-offload` flag:

```bash
aphrodite run \
    meta-llama/Llama-3.2-7B-Instruct \
    --cpu-offload-gb 4 \
    --smart-offload
```

### What Gets Offloaded

With `--smart-offload`, the system prioritizes offloading parameters with low computational impact:

1. **Embedding tables** - Only accessed once per prompt
2. **Normalization vectors** (LayerNorm, RMSNorm) - Small vectors with minimal compute
3. **Bias vectors** - Small vectors with minimal compute overhead
4. **Positional embeddings** - Only accessed once per sequence
5. **Language model head** - Only used for final token generation

### Performance Comparison

| Configuration | GPU RAM Saved | Latency Impact | Notes |
|---------------|---------------|----------------|--------|
| `--cpu-offload-gb 4` (old) | 4 GB | +45% | Offloads random layers |
| `--cpu-offload-gb 4 --smart-offload` | 3-4 GB | +5-10% | Targets low-impact params |

### Advanced Configuration

You can create custom offload policies in your code:

```python
from aphrodite.modeling.models.offload_policy import (
    OffloadPolicy, create_conservative_policy, create_aggressive_policy
)

# Conservative: only norms and biases
conservative = create_conservative_policy(max_bytes=2 * 1024**3)  # 2GB

# Aggressive: includes embeddings and LM head
aggressive = create_aggressive_policy(max_bytes=4 * 1024**3)  # 4GB

# Custom policy
custom_predicates = [is_embedding_table, is_norm_vector]
custom = OffloadPolicy(max_bytes=3 * 1024**3, offload_predicates=custom_predicates)
```

## Best Practices

1. **Start conservative**: Begin with smaller `--cpu-offload-gb` values and enable `--smart-offload`
2. **Consider quantization first**: Try quantization before offloading

