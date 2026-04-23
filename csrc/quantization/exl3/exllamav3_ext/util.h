#pragma once

#include <chrono>

#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// Some decluttering macros
//
// TORCH_CHECK_DTYPE(x, T):                     assert x is dtype T
// TORCH_CHECK_DTYPE_OPT(x, T):                 assert x is dtype T, unless x is None
// TORCH_CHECK_FLOAT_HALF(x):                   assert x is either kFloat or kHalf
// TORCH_CHECK_SHAPES(x, i, y, j, scale):       assert x.size(i) == y.size(j) * scale
// TORCH_CHECK_SHAPES_OPT(x, i, y, j, scale):   assert x.size(i) == y.size(j) * scale, unless x is None
// TORCH_CHECK_SHAPES_FULL(x, y):               assert x and y are same shape
// TORCH_CHECK_NUMEL(x, y):                     assert x and y have same number of elements
// TORCH_CHECK_DIV(x, i, divisor):              assert x.size(i) is divisible by divisor
// TORCH_CHECK_DIM(x, D):                       assert x has D dimensions
// TORCH_CHECK_DIM_OPT(x, D):                   assert x has D dimensions, unless x is None
// TORCH_CHECK_SIZE(x, i, s):                   assert x.size(i) == s
// OPTPTR(x):                                   x.data_ptr() or nullptr if x is None

#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == at::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype) TORCH_CHECK((!__x.has_value()) || (__x).value().dtype() == at::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_FLOAT_HALF(__x) TORCH_CHECK((__x).dtype() == at::kHalf || (__x).dtype() == at::kFloat,  #__x " is incorrect datatype, must be kHalf or kFloat")
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((!(__x).has_value()) || (__x).value().size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_FULL(__x, __y) TORCH_CHECK((__x).sizes() == (__y).sizes(), #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_NUMEL(__x, __y) TORCH_CHECK((__x).numel() == (__y).numel(), #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_DIV(__x, __dim_x, __div) TORCH_CHECK((__x).size(__dim_x) % __div == 0, #__x " dimension " #__dim_x " must be divisible by " #__div)
#define TORCH_CHECK_DIM(__x, __dims) TORCH_CHECK((__x).dim() == __dims, #__x " must have " #__dims " dimensions")
#define TORCH_CHECK_DIM_OPT(__x, __dims) TORCH_CHECK((!__x.has_value()) || (__x).value().dim() == __dims, #__x " must have " #__dims " dimensions")
#define TORCH_CHECK_SIZE(__x, __dim_x, __s) TORCH_CHECK((__x).size(__dim_x) == (__s), #__x " dimension " #__dim_x " is incorrect size")
#define OPTPTR(__x) (__x.has_value() ? __x.value().data_ptr() : nullptr)

// Debug stuff

#define DBGS(__x) printf("%s\n", __x)
#define DBGI(__x) \
    printf("%s: %i\n", #__x, __x)
#define DBGI2(__x, __y) \
    printf("%s, %s: %i, %i\n", #__x, #__y, __x, __y)
#define DBGI3(__x, __y, __z) \
    printf("%s, %s, %s: %i, %i, %i\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGI4(__x, __y, __z, __w) \
    printf("%s, %s, %s, %s: %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, __x, __y, __z, __w)
#define DBGI5(__x, __y, __z, __w, __v) \
    printf("%s, %s, %s, %s, %s: %i, %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, #__v, __x, __y, __z, __w, __v)
#define DBGI6(__x, __y, __z, __w, __v, __u) \
    printf("%s, %s, %s, %s, %s, %s: %i, %i, %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, #__v, #__u, __x, __y, __z, __w, __v, __u)
#define DBGI7(__x, __y, __z, __w, __v, __u, __t) \
    printf("%s, %s, %s, %s, %s, %s, %s: %i, %i, %i, %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, #__v, #__u, #__t, __x, __y, __z, __w, __v, __u, __t)
#define DBGI8(__x, __y, __z, __w, __v, __u, __t, __s) \
    printf("%s, %s, %s, %s, %s, %s, %s, %s: %i, %i, %i, %i, %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, #__v, #__u, #__t, #__s, __x, __y, __z, __w, __v, __u, __t, __s)
#define DBGI9(__x, __y, __z, __w, __v, __u, __t, __s, __r) \
    printf("%s, %s, %s, %s, %s, %s, %s, %s, %s: %i, %i, %i, %i, %i, %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, #__v, #__u, #__t, #__s, #__r, __x, __y, __z, __w, __v, __u, __t, __s, __r)
#define DBGI10(__x, __y, __z, __w, __v, __u, __t, __s, __r, __q) \
    printf("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s: %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n", #__x, #__y, #__z, #__w, #__v, #__u, #__t, #__s, #__r, #__q, __x, __y, __z, __w, __v, __u, __t, __s, __r, __q)
#define DBGX(__x) printf("%s: %x\n", #__x, __x)
#define DBGX2(__x, __y) printf("%s, %s: %x, %x\n", #__x, #__y, __x, __y)
#define DBGX3(__x, __y, __z) printf("%s, %s, %s: %x, %x, %x\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGIX(__x, __y) printf("%s, %s: %i, %x\n", #__x, #__y, __x, __y)
#define DBGIX2(__x, __y, __z) printf("%s, %s, %s: %i, %x, %x\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGIF(__x, __y) printf("%s, %s: %i, %f\n", #__x, #__y, __x, __y)
#define DBGIF2(__x, __y, __z) printf("%s, %s, %s: %i, %f, %f\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGF(__x) printf("%s: %f\n", #__x, __x)
#define DBGF2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __x, __y)
#define DBGF3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGF4(__x, __y, __z, __w) printf("%s, %s, %s, %s: %f, %f, %f, %f\n", #__x, #__y, #__z, #__w, __x, __y, __z, __w)
#define DBGH(__x) printf("%s: %f\n", #__x, __half2float(__x))
#define DBGH2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __half2float(__x), __half2float(__y))
#define DBGH3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __half2float(__x), __half2float(__y), __half2float(__z))
#define DBGIH(__x, __y) printf("%s, %s: %i, %f\n", #__x, #__y, __x, __half2float(__y))
#define DBGIH2(__x, __y, __z) printf("%s, %s, %s: %i, %f, %f\n", #__x, #__y, #__z, __x, __half2float(__y), __half2float(__z))
#define DBGI2H2(__x, __y, __z, __w) printf("%s, %s, %s, %s: %i, %i, %f, %f\n", #__x, #__y, #__z, #__w, __x, __y, __half2float(__z), __half2float(__w))
#define DBGIH3(__x, __y, __z, __w) printf("%s, %s, %s, %s: %i, %f, %f, %f\n", #__x, #__y, #__z, #__w, __x, __half2float(__y), __half2float(__z), __half2float(__w))
#define DBGIH4(__x, __y, __z, __w, __v) printf("%s, %s, %s, %s, %s: %i, %f, %f, %f, %f\n", #__x, #__y, #__z, #__w, #__v, __x, __half2float(__y), __half2float(__z), __half2float(__w), __half2float(__v))
#define DBGA(__x) printf("%s: %016llx\n", #__x, __x)
#define DBGIA(__x, __y) printf("%s, %s: %i, %016llx\n", #__x, #__y, __x, __y)
#define DBGI2A(__x, __y, __z) printf("%s, %s, %s: %i, %i, %016llx\n", #__x, #__y, #__z, __x, __y, __z)

#define TIME_START \
    auto start = std::chrono::high_resolution_clock::now()

#define TIME_STOP \
    do { \
        auto stop = std::chrono::high_resolution_clock::now(); \
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
        DBGI(duration_us); \
    } while (false)

/*
Compile-time for loop. Supports template instancing. Example usage:

int kernel_arg = select_kernel_somehow();

// Not nice
if (kernel_arg == 2)
    launch_kernel_instance<2><<< ... >>>( ... )
if (kernel_arg == 3)
    launch_kernel_instance<3><<< ... >>>( ... )
if (kernel_arg == 4)
    launch_kernel_instance<4><<< ... >>>( ... )
if (kernel_arg == 6)
    launch_kernel_instance<6><<< ... >>>( ... )
if (kernel_arg == 8)
    launch_kernel_instance<8><<< ... >>>( ... )

// Nice?
static_for_pack<2, 3, 4, 6, 8>([&](auto ic)
{
    constexpr int i = decltype(ic)::value;
    if (kernel_arg == i)
        launch_kernel_instance<i><<< ... >>>( ... )
});

// Ultimately much cleaner
#define __(i, j) quant_cache_paged_kernel<i, j>
constexpr auto quant_cache_paged_kernel_instances = std::array
{
    std::array{ __(2, 2), __(2, 3), __(2, 4), __(2, 5), __(2, 6), __(2, 7), __(2, 8) },
    std::array{ __(3, 2), __(3, 3), __(3, 4), __(3, 5), __(3, 6), __(3, 7), __(3, 8) },
    std::array{ __(4, 2), __(4, 3), __(4, 4), __(4, 5), __(4, 6), __(4, 7), __(4, 8) },
    std::array{ __(5, 2), __(5, 3), __(5, 4), __(5, 5), __(5, 6), __(5, 7), __(5, 8) },
    std::array{ __(6, 2), __(6, 3), __(6, 4), __(6, 5), __(6, 6), __(6, 7), __(6, 8) },
    std::array{ __(7, 2), __(7, 3), __(7, 4), __(7, 5), __(7, 6), __(7, 7), __(7, 8) },
    std::array{ __(8, 2), __(8, 3), __(8, 4), __(8, 5), __(8, 6), __(8, 7), __(8, 8) }
};
#undef __
*/

// This breaks with nesting on VC++ older than 17.13 (late 2024 preview)
template <int... Values, class F>
constexpr void static_for_pack(F&& f)
{
    (f(std::integral_constant<int, Values>{}), ...);
}
