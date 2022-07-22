//! Implementations of Unoptimize and UnoptimizeRef for x86 and x86_64

use super::{Unoptimize, UnoptimizeRef};
use core::arch::asm;
#[cfg(not(target_arch = "x86_64"))]
use core::arch::x86 as target_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as target_arch;
#[cfg(target_feature = "sse")]
use target_arch::{__m128, __m128d, __m128i};
#[cfg(target_feature = "avx")]
use target_arch::{__m256, __m256d, __m256i};

// Implementation of Unoptimize for values that belong to a general-purpose register
macro_rules! unoptimize_general_values {
    ($($t:ty),*) => {
        $(
            impl Unoptimize for $t {
                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        asm!("/* {0:r} */", in(reg) *self, options(preserves_flags, nostack, nomem))
                    }
                }

                #[inline(always)]
                fn black_box(mut self) -> Self {
                    unsafe {
                        asm!("/* {0:r} */", inout(reg) self, options(preserves_flags, nostack, nomem));
                    }
                    self
                }
            }
        )*
    };
}
//
unoptimize_general_values!(i16, u16, i32, u32, i64, u64, isize, usize);
//
// Most x86 calling conventions do not use XMM registers for FP values, so we
// use a GP register for FP values on 32-bit x86 irrespective of SSE support.
#[cfg(not(target = "x86_64"))]
unoptimize_general_values!(f32, f64);

// Implementation of Unoptimize for values that require use of special registers
macro_rules! unoptimize_special_values {
    ($reg:ident, $($t:ty),*) => {
        $(
            impl Unoptimize for $t {
                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        asm!("/* {0} */", in($reg) *self, options(preserves_flags, nostack, nomem))
                    }
                }

                #[inline(always)]
                fn black_box(mut self) -> Self {
                    unsafe {
                        asm!("/* {0} */", inout($reg) self, options(preserves_flags, nostack, nomem));
                    }
                    self
                }
            }
        )*
    };
}
//
unoptimize_special_values!(reg_byte, i8, u8);
//
#[cfg(target_feature = "sse")]
unoptimize_special_values!(xmm_reg, __m128, __m128d, __m128i);
//
// x86_64 mandates SSE support and most x86_64 calling conventions use XMM
// registers for FP values, so we an XMM register there.
#[cfg(target = "x86_64")]
unoptimize_special_values!(xmm_reg, f32, f64);
//
#[cfg(target_feature = "avx")]
unoptimize_special_values!(ymm_reg, __m256, __m256d, __m256i);

// TODO: Add nightly support for AVX-512 (including masks) and BF16
// TODO: Add nightly support for portable_simd types

// Implementation of Unoptimize and UnoptimizeRef for pointers
macro_rules! unoptimize_pointers {
    ($($t:ty),*) => {
        $(
            impl<T: Sized> Unoptimize for $t {
                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        asm!("/* {0:r} */", in(reg) *self, options(preserves_flags, nostack, readonly))
                    }
                }

                #[inline(always)]
                fn black_box(mut self) -> Self {
                    unsafe {
                        asm!("/* {0} */", inout(reg) self, options(preserves_flags, nostack, nomem));
                    }
                    self
                }
            }

            impl<T: Sized> UnoptimizeRef for $t {
                #[inline(always)]
                fn assume_accessed(&self) {
                    unsafe {
                        asm!("/* {0:r} */", in(reg) *self, options(preserves_flags, nostack))
                    }
                }
            }
        )*
    };
}
//
unoptimize_pointers!(*const T, *mut T);

// TODO: Remember to also run CI in AVX mode
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
            assert!(all_eq(x.black_box(), x2));
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
            assert!(all_eq(x.black_box(), x2));
        };

        test(zeros);
        test(ones);
    }
}
