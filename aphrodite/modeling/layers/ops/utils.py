import triton

MAX_FUSED_SIZE : int = 65536
next_power_of_2 = triton.next_power_of_2


# Calculate the optimal block size and number of warps for the layernorm kernel
# borrowed from https://github.com/unslothai/unsloth/blob/038e6d4c8d40207a87297ab3aaf787c19b1006d1/unsloth/kernels/utils.py#L49-L59
def calculate_settings(n : int) -> tuple[int, int]:
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >=  8192:
        num_warps = 16
    elif BLOCK_SIZE >=  2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps
pass