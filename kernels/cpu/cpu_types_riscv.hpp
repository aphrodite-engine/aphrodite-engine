#include <riscv_vector.h>

typedef vfloat16m1_t fixed_vfloat16m1_t
  __attribute__((riscv_rvv_vector_bits(128)));
typedef vfloat16m2_t fixed_vfloat16m2_t
  __attribute__((riscv_rvv_vector_bits(256)));
typedef vfloat32m1_t fixed_vfloat32m1_t
  __attribute__((riscv_rvv_vector_bits(128)));
typedef vfloat32m2_t fixed_vfloat32m2_t
  __attribute__((riscv_rvv_vector_bits(256)));
typedef vfloat32m4_t fixed_vfloat32m4_t
  __attribute__((riscv_rvv_vector_bits(512)));


namespace vec_op {


#ifdef RISCV_BF16_SUPPORT
  #define APHRODITE_DISPATCH_CASE_FLOATING_TYPES(...)    \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#else
  #define APHRODITE_DISPATCH_CASE_FLOATING_TYPES(...)    \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)
#endif

#ifndef CPU_OP_GUARD
  #define CPU_KERNEL_GUARD_IN(NAME)
  #define CPU_KERNEL_GUARD_OUT(NAME)
#else
  #define CPU_KERNEL_GUARD_IN(NAME) \
    std::cout << #NAME << " invoked." << std::endl;
  #define CPU_KERNEL_GUARD_OUT(NAME) \
    std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
  template <typename T, T... indexes, typename F>
  constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
    (f(std::integral_constant<T, indexes>{}), ...);
  };
  };  // namespace

template <typename T, T count, typename F,
typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; };
};

struct FP32Vec8;
struct FP32Vec16;

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  fixed_vfloat16m1_t reg;

  explicit FP16Vec8(const void* ptr)
    : reg(__riscv_vle16_v_f16m1(static_cast<const _Float16*>(ptr),
                                VEC_ELEM_NUM)) {};

  explicit FP16Vec8(const FP32Vec8& vec)

  void save(void* ptr) const {
    __riscv_vse16_v_f16m1(static_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  fixed_vfloat16m2_t reg;

  explicit FP16Vec16(const void* ptr)
    : reg(__riscv_vle16_v_f16m2(static_cast<const _Float16*>(ptr),
                                VEC_ELEM_NUM)) {};

  explicit FP16Vec16(const FP32Vec16& vec)

  void save(void* ptr) const {
    __riscv_vse16_v_f16m2(static_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }

  void save(void* ptr, const int elem_num) const {
    vuint16m2_t index = __riscv_vid_v_u16m2(elem_num);
    vbloo8_t mask = __riscv_vmsltu_vx_u16m2_b8(index, elem_num, VEC_ELEM_NUM);
    __riscv_vse16_v_f16m2_m(mask, reinterpret_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }
};


#ifdef RISCV_BF16_SUPPORT
typedef vfloat16m1_t fixed_vfloat16m1_t;
  __attribute__((riscv_rvv_vector_bits(128)));

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  fixed_vfloat16m1_t reg;

  explicit BF16Vec8(const void* ptr)
    : reg(*reinterpret_cast<const __riscv_bfloat16*>(ptr)) {};

  explicit BF16Vec8(fixed_vfloat16m1_t data) : reg(data) {};

  explicit BF16Vec8(const FP16Vec8&);

  explicit BF16Vec8(fixed_vfloat32m1_t v)
    : reg(__riscv_vfncvtbf16_f_f_w_bf16m1(v, VEC_ELEM_NUM)) {};

  void save(void* ptr) const {
    *reinterpret_cast<__riscv_bfloat16*>(ptr) = reg;
  };
};

typedef vbfloat16m4_t fixed_vbfloat16m4_t
  __attribute__((riscv_rvv_vector_bits(512)));

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  fixed_vbfloat16m4_t reg;

  explicit BF16Vec32(const void* ptr)
    : reg(*reinterpret_cast<const __riscv_bfloat16*>(ptr)) {};

  explicit BF16Vec32(fixed_vbfloat16m4_t data) : reg(data) {};

  explicit BF16Vec32(const FP16Vec16&);

  explicit BF16Vec32(fixed_vfloat32m2_t v)

}  // namespace vec_op