//! Implementations of Pessimize for x86 and x86_64

use super::Pessimize;
use core::arch::asm;
#[cfg(not(target_arch = "x86_64"))]
use core::arch::x86 as target_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as target_arch;
#[cfg(target_feature = "sse")]
use target_arch::{__m128, __m128d, __m128i};
#[cfg(target_feature = "avx")]
use target_arch::{__m256, __m256d, __m256i};

// Implementation of Pessimize for values without pointer semantics
macro_rules! pessimize_values {
    ($reg:ident, $($t:ty),*) => {
        $(
            #[allow(asm_sub_register)]
            impl Pessimize for $t {
                #[inline(always)]
                fn hide(mut self) -> Self {
                    unsafe {
                        asm!("/* {0} */", inout($reg) self, options(preserves_flags, nostack, nomem));
                    }
                    self
                }

                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        asm!("/* {0} */", in($reg) *self, options(preserves_flags, nostack, nomem))
                    }
                }
            }
        )*
    };
}
//
pessimize_values!(reg, i16, u16, i32, u32, isize, usize);
pessimize_values!(reg_byte, i8, u8);
//
// 64-bit values normally go to GP registers on x86_64, but since 32-bit has
// no 64-bit GP registers, we try to use XMM registers instead if available
#[cfg(target_arch = "x86_64")]
pessimize_values!(reg, i64, u64);
#[cfg(all(not(target_arch = "x86_64"), target_feature = "sse"))]
pessimize_values!(xmm_reg, i64, u64);
//
// Given that CPUs without SSE should now all be extinct and that compilers try
// to use SSE whenever at all possible, we assume that float data should go
// to SSE registers if possible and to the x87 stack otherwise (on old 32-bit).
#[cfg(target_feature = "sse")]
pessimize_values!(xmm_reg, f32, f64);
#[cfg(not(target_feature = "sse"))]
pessimize_values!(st, f32, f64);
//
#[cfg(target_feature = "sse")]
pessimize_values!(xmm_reg, __m128, __m128d, __m128i);
//
#[cfg(target_feature = "avx")]
pessimize_values!(ymm_reg, __m256, __m256d, __m256i);

// TODO: Add nightly support for AVX-512 (including masks, which will
//       require an architecture-specific extension) and BF16 vectors
// TODO: Add nightly support for portable_simd types

// TODO: Remember to also run CI in AVX mode, and ideally in 32-bit mode too
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_feature = "sse")]
    #[test]
    fn xmm() {
        use target_arch::{_mm_andnot_si128, _mm_cmpeq_epi8, _mm_movemask_epi8, _mm_setzero_si128};

        let zeros = unsafe { _mm_setzero_si128() };
        let ones = unsafe { _mm_andnot_si128(zeros, zeros) };
        let all_eq = |x, y| unsafe {
            let cmp = _mm_cmpeq_epi8(x, y);
            _mm_movemask_epi8(cmp) == 0xFFFF
        };
        assert!(all_eq(zeros, zeros));
        assert!(all_eq(ones, zeros));

        let test = |x: __m128i| {
            let x2 = x;
            x.assume_read();
            assert!(all_eq(x, x2));
            assert!(all_eq(x.hide(), x2));
        };

        test(zeros);
        test(ones);
    }

    #[cfg(target_feature = "avx")]
    #[test]
    fn ymm() {
        use target_arch::{
            _mm256_andnot_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8, _mm256_setzero_si256,
        };

        let zeros = unsafe { _mm256_setzero_si256() };
        let ones = unsafe { _mm256_andnot_si256(zeros, zeros) };
        let all_eq = |x, y| unsafe {
            let cmp = _mm256_cmpeq_epi8(x, y);
            _mm256_movemask_epi8(cmp) == !0
        };
        assert!(all_eq(zeros, zeros));
        assert!(all_eq(ones, zeros));

        let test = |x: __m256i| {
            let x2 = x;
            x.assume_read();
            assert!(all_eq(x, x2));
            assert!(all_eq(x.hide(), x2));
        };

        test(zeros);
        test(ones);
    }
}
