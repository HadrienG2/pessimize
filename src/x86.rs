//! Implementations of Pessimize for x86 and x86_64

use super::Pessimize;
use core::arch::asm;
#[allow(unused)]
#[cfg(not(target_arch = "x86_64"))]
use core::arch::x86 as target_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as target_arch;
#[cfg(target_feature = "sse")]
use target_arch::__m128;
#[cfg(target_feature = "avx2")]
use target_arch::__m256i;
#[cfg(target_feature = "sse2")]
use target_arch::{__m128d, __m128i};
#[cfg(target_feature = "avx")]
use target_arch::{__m256, __m256d};

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
pessimize_values!(reg_byte, i8, u8);
pessimize_values!(reg, i16, u16, i32, u32, isize, usize);
//
// 64-bit values normally go to GP registers on x86_64, but since 32-bit has
// no 64-bit GP registers, we try to use XMM registers instead if available
#[cfg(target_arch = "x86_64")]
pessimize_values!(reg, i64, u64);
#[cfg(all(not(target_arch = "x86_64"), target_feature = "sse2"))]
pessimize_values!(xmm_reg, i64, u64);
//
// Given that CPUs without SSE should now all be extinct and that compilers try
// to use SSE whenever at all possible, we assume that float data should go
// to SSE registers if possible and to GP registers otherwise (on old 32-bit).
#[cfg(target_feature = "sse")]
pessimize_values!(xmm_reg, f32);
#[cfg(not(target_feature = "sse"))]
pessimize_values!(reg, f32);
//
// Ditto for double precision, but we need SSE2 for that
#[cfg(target_feature = "sse2")]
pessimize_values!(xmm_reg, f64);
//
#[cfg(target_feature = "sse")]
pessimize_values!(xmm_reg, __m128);
//
#[cfg(target_feature = "sse2")]
pessimize_values!(xmm_reg, __m128d, __m128i);
//
#[cfg(target_feature = "avx")]
pessimize_values!(ymm_reg, __m256, __m256d);
//
#[cfg(target_feature = "avx2")]
pessimize_values!(ymm_reg, __m256i);

// TODO: Add nightly support for AVX-512 (including masks, which will
//       require an architecture-specific extension) and BF16 vectors

// Support safe_arch types if enabled
#[cfg(any(feature = "safe_arch", test))]
mod safe_arch {
    #[allow(unused)]
    use crate::Pessimize;
    #[cfg(target_feature = "sse")]
    use safe_arch::m128;
    #[cfg(target_feature = "avx2")]
    use safe_arch::m256i;
    #[cfg(target_feature = "sse2")]
    use safe_arch::{m128d, m128i};
    #[cfg(target_feature = "avx")]
    use safe_arch::{m256, m256d};

    #[allow(unused)]
    macro_rules! pessimize_safe_arch {
        ($($t:ty),*) => {
            $(
                impl Pessimize for $t {
                    #[inline(always)]
                    fn hide(self) -> Self {
                        Self(self.0.hide())
                    }

                    #[inline(always)]
                    fn assume_read(&self) {
                        self.0.assume_read()
                    }
                }
            )*
        };
    }

    #[cfg(target_feature = "sse")]
    pessimize_safe_arch!(m128);

    #[cfg(target_feature = "sse2")]
    pessimize_safe_arch!(m128d, m128i);

    #[cfg(target_feature = "avx")]
    pessimize_safe_arch!(m256, m256d);

    #[cfg(target_feature = "avx2")]
    pessimize_safe_arch!(m256i);
}

// TODO: Add nightly support for portable_simd types

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use crate::{
        tests::{test_unoptimized_value, test_value_type},
        Pessimize,
    };
    use std::fmt::Debug;

    #[allow(unused)]
    fn test_simd<
        Scalar: Copy + Default,
        const LANES: usize,
        T: Copy + Debug + Default + From<[Scalar; LANES]> + PartialEq + Pessimize,
    >(
        min: Scalar,
        max: Scalar,
    ) {
        test_value_type(T::from([min; LANES]), T::from([max; LANES]));
    }

    #[cfg(target_feature = "sse")]
    #[test]
    fn sse() {
        use safe_arch::m128;
        test_simd::<f32, 4, m128>(f32::MIN, f32::MAX);
    }

    #[cfg(target_feature = "sse")]
    #[test]
    #[ignore]
    fn sse_optim() {
        use safe_arch::m128;
        test_unoptimized_value::<m128>();
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn sse2() {
        use safe_arch::{m128d, m128i};
        test_simd::<f64, 2, m128d>(f64::MIN, f64::MAX);
        test_simd::<i8, 16, m128i>(i8::MIN, i8::MAX);
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    #[ignore]
    fn sse2_optim() {
        use safe_arch::{m128d, m128i};
        test_unoptimized_value::<m128d>();
        test_unoptimized_value::<m128i>();
    }

    #[cfg(target_feature = "avx")]
    #[test]
    fn avx() {
        use safe_arch::{m256, m256d};
        test_simd::<f32, 8, m256>(f32::MIN, f32::MAX);
        test_simd::<f64, 4, m256d>(f64::MIN, f64::MAX);
    }

    #[cfg(target_feature = "avx")]
    #[test]
    #[ignore]
    fn avx_optim() {
        use safe_arch::{m256, m256d};
        test_unoptimized_value::<m256>();
        test_unoptimized_value::<m256d>();
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn avx2() {
        use safe_arch::m256i;
        test_simd::<i8, 32, m256i>(i8::MIN, i8::MAX);
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    #[ignore]
    fn avx2_optim() {
        use safe_arch::m256i;
        test_unoptimized_value::<m256i>();
    }
}
