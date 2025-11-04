import math
from contextlib import suppress
from pathlib import Path

import scipy
import torch
from safetensors.torch import load_file

with suppress(ImportError):
    import aphrodite._hadamard_C as hadamard_C

HADA_TENSORS = load_file(Path(__file__).resolve().parent / "hadamard.safetensors")


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx._hadamard_transform_scale = scale  # pylint: disable=protected-access
        return hadamard_C.fast_hadamard_transform(x, scale)


def hadamard_transform(x, scale=1.0):
    return HadamardTransformFn.apply(x, scale)


def int2mask(i, int_map):
    return ((i & int_map) > 0).int()


def mask2int(mask, int_map):
    return (int_map.unsqueeze(0) * mask.int()).sum(dim=-1)


def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return (
        torch.tensor(
            [
                [3, 1, 1, 1, 3, 3, 3, 3],
                [1, 3, 1, 1, 3, 3, 3, 3],
                [1, 1, 3, 1, 3, 3, 3, 3],
                [1, 1, 1, 3, 3, 3, 3, 3],
                [3, 3, 3, 1, 3, 3, 1, 1],
                [3, 3, 3, 1, 3, 1, 3, 1],
                [3, 3, 3, 1, 1, 3, 3, 1],
                [3, 3, 3, 1, 3, 1, 1, 3],
                [3, 3, 3, 1, 1, 3, 1, 3],
                [3, 3, 3, 1, 1, 1, 3, 3],
                [3, 3, 1, 3, 3, 3, 1, 1],
                [3, 3, 1, 3, 3, 1, 3, 1],
                [3, 3, 1, 3, 1, 3, 3, 1],
                [3, 3, 1, 3, 3, 1, 1, 3],
                [3, 3, 1, 3, 1, 3, 1, 3],
                [3, 3, 1, 3, 1, 1, 3, 3],
                [3, 1, 3, 3, 3, 3, 1, 1],
                [3, 1, 3, 3, 3, 1, 3, 1],
                [3, 1, 3, 3, 1, 3, 3, 1],
                [3, 1, 3, 3, 3, 1, 1, 3],
                [3, 1, 3, 3, 1, 3, 1, 3],
                [1, 3, 3, 3, 1, 1, 3, 3],
                [1, 3, 3, 3, 3, 3, 1, 1],
                [1, 3, 3, 3, 3, 1, 3, 1],
                [1, 3, 3, 3, 1, 3, 3, 1],
                [1, 3, 3, 3, 3, 1, 1, 3],
                [1, 3, 3, 3, 1, 3, 1, 3],
                [1, 1, 3, 3, 1, 3, 3, 3],
                [3, 3, 1, 1, 3, 3, 3, 1],
            ]
        )
        / 2
    )


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = d8.sum(dim=-1) % 2 == 0
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 1, 3, 4, 6, 5, 7]]
    cba[:, 7] *= 1 - 2 * (cba.sum(1) % 2)
    cba = cba * 4
    cba = cba.to(torch.int64)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 8))
    return acc


def next_power_of_2(n):
    if n == 0:
        return 1
    return 2 ** math.ceil(math.log(n, 2))


def get_power_of_2(n):
    """Returns the highest power of 2 that divides n."""
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k, n


def get_hadK(n, use_rand=True):
    exp, base = get_power_of_2(n)
    if base == 1:
        return None, 1, n
    if use_rand:
        rand_mat = torch.tensor(scipy.stats.special_ortho_group.rvs(base)).to(torch.float32)
        return rand_mat, base, n

    # Use hadamad only and add padding if cannot find one
    pad_n = next_power_of_2(n)
    if exp < 2 or str(base * 4) not in HADA_TENSORS:
        return None, 1, pad_n
    base_mat = HADA_TENSORS[str(base * 4)] / math.sqrt(base * 4)
    return base_mat, base * 4, n


def matmul_hadU_cuda(X, hadK, K, n, scale=None, transpose=False):
    if n != X.shape[-1]:
        X = torch.nn.functional.pad(X, (0, n - X.shape[-1]))

    had_scale = 1 / math.sqrt(n // K) if scale is None else scale / math.sqrt(n // K)
    if K == 1:
        return hadamard_transform(X.contiguous(), scale=had_scale)

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.view(-1, K, n // K)  # pylint: disable=redefined-builtin
    input = hadamard_transform(input.contiguous(), scale=had_scale)
    input = hadK @ input
    return input.reshape(X.shape)


def matmul_hadUt_cuda(X, hadK, K, n, scale=None):
    return matmul_hadU_cuda(X, hadK, K, n, scale=scale, transpose=True)
