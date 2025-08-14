#!/usr/bin/env bash
set -euo pipefail

# Compile Vulkan compute shaders to SPIR-V without rebuilding the whole project.
# Outputs to kernels/vulkan/spv and sets APHRODITE_VK_SPV_DIR for convenience.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHADERS_DIR="${ROOT_DIR}/kernels/vulkan/shaders"
OUT_DIR="${ROOT_DIR}/kernels/vulkan/spv"

GLSLC_BIN="${GLSLC:-glslc}"
TARGET_ENV="${VK_TARGET_ENV:-vulkan1.2}"

if ! command -v "${GLSLC_BIN}" >/dev/null 2>&1; then
  echo "error: glslc not found. Install the Vulkan SDK or set GLSLC=/path/to/glslc" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

compile() {
  local name="$1"; shift
  local src="$1"; shift
  echo "Compiling ${name}.spv from ${src}"
  "${GLSLC_BIN}" -fshader-stage=compute --target-env="${TARGET_ENV}" \
    -I"${SHADERS_DIR}" "$@" -o "${OUT_DIR}/${name}.spv" "${src}"
}

# Activations (GLU-style)
compile swiglu_f32  "${SHADERS_DIR}/swiglu.comp" -DA_TYPE=float     -DD_TYPE=float     -DFLOAT_TYPE=float
compile swiglu_f16  "${SHADERS_DIR}/swiglu.comp" -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1 -DENABLE_F16_ARITH=1
compile geglu_f32   "${SHADERS_DIR}/geglu.comp"  -DA_TYPE=float     -DD_TYPE=float     -DFLOAT_TYPE=float
compile geglu_f16   "${SHADERS_DIR}/geglu.comp"  -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1 -DENABLE_F16_ARITH=1

# (Optional) Elementwise GELU variants if needed later
# compile gelu_f32       "${SHADERS_DIR}/gelu.comp"       -DA_TYPE=float     -DD_TYPE=float
# compile gelu_erf_f32   "${SHADERS_DIR}/gelu_erf.comp"   -DA_TYPE=float     -DD_TYPE=float
# compile gelu_quick_f32 "${SHADERS_DIR}/gelu_quick.comp" -DA_TYPE=float     -DD_TYPE=float
# compile gelu_f16       "${SHADERS_DIR}/gelu.comp"       -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1
# compile gelu_erf_f16   "${SHADERS_DIR}/gelu_erf.comp"   -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1
# compile gelu_quick_f16 "${SHADERS_DIR}/gelu_quick.comp" -DA_TYPE=float16_t -DD_TYPE=float16_t -DRTE16=1

# Norms
compile rms_norm_f32 "${SHADERS_DIR}/rms_norm.comp" -DA_TYPE=float -DB_TYPE=float -DD_TYPE=float -DFLOAT_TYPE=float
compile norm_f32     "${SHADERS_DIR}/norm.comp"     -DA_TYPE=float -DD_TYPE=float -DFLOAT_TYPE=float

# RoPE (both layouts)
compile rope_neox_f32 "${SHADERS_DIR}/rope_neox.comp" -DA_TYPE=float -DD_TYPE=float -DFLOAT_TYPE=float
compile rope_norm_f32 "${SHADERS_DIR}/rope_norm.comp" -DA_TYPE=float -DD_TYPE=float -DFLOAT_TYPE=float

echo "SPIR-V output directory: ${OUT_DIR}"
echo "export APHRODITE_VK_SPV_DIR=\"${OUT_DIR}\""


