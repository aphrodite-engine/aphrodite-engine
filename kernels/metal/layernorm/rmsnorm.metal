#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_kernel(
    device const float* input     [[ buffer(0) ]],
    device const float* weight    [[ buffer(1) ]],
    device float* output          [[ buffer(2) ]],
    constant uint& hidden_size    [[ buffer(3) ]],
    constant float& epsilon       [[ buffer(4) ]],
    uint gid                      [[ thread_position_in_grid ]]) {

    const uint offset = gid * hidden_size;

    // Step 1: compute RMS
    float sum_squares = 0.0;
    for (uint i = 0; i < hidden_size; ++i) {
        float val = input[offset + i];
        sum_squares += val * val;
    }

    float rms = sqrt(sum_squares / hidden_size + epsilon);

    // Step 2: normalize and scale
    for (uint i = 0; i < hidden_size; ++i) {
        float val = input[offset + i];
        output[offset + i] = (val / rms) * weight[i];
    }
}