//! Implementations of Pessimize for x86 and x86_64

use super::Pessimize;
use core::arch::asm;
#[allow(unused)]
#[cfg(not(target_arch = "x86_64"))]
use core::arch::x86 as target_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as target_arch;
#[cfg(any(target_feature = "sse", doc))]
use target_arch::__m128;
#[cfg(any(target_feature = "avx2", doc))]
use target_arch::__m256i;
#[cfg(all(
    feature = "nightly",
    any(all(target_feature = "avx512vl", target_feature = "bf16"), doc)
))]
use target_arch::{__m128bh, __m256bh};
#[cfg(any(target_feature = "sse2", doc))]
use target_arch::{__m128d, __m128i};
#[cfg(any(target_feature = "avx", doc))]
use target_arch::{__m256, __m256d};

// Implementation of Pessimize for values without pointer semantics
macro_rules! pessimize_values {
    ($($reg:ident: ($($t:ty),*)),*) => {
        $($(
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
        )*)*
    };
}
//
pessimize_values!(reg_byte: (i8, u8), reg: (i16, u16, i32, u32, isize, usize));
//
// 64-bit values normally go to GP registers on x86_64, but since 32-bit has
// no 64-bit GP registers, we try to use XMM registers instead if available
#[cfg(target_arch = "x86_64")]
pessimize_values!(reg: (i64, u64));
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))]
#[cfg(all(not(target_arch = "x86_64"), any(target_feature = "sse2", doc)))]
pessimize_values!(xmm_reg: (i64, u64));
//
// Given that CPUs without SSE should now all be extinct and that compilers try
// to use SSE whenever at all possible, we assume that float data should go
// to SSE registers if possible and to GP registers otherwise (on old 32-bit).
#[cfg(any(target_feature = "sse", doc))]
pessimize_values!(xmm_reg: (f32));
#[cfg(all(not(target_feature = "sse"), not(doc)))]
pessimize_values!(reg: (f32));
//
// Ditto for double precision, but we need SSE2 for that
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))]
#[cfg(any(target_feature = "sse2", doc))]
pessimize_values!(xmm_reg: (f64));
//
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse")))]
#[cfg(any(target_feature = "sse", doc))]
pessimize_values!(xmm_reg: (__m128));
//
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))]
#[cfg(any(target_feature = "sse2", doc))]
pessimize_values!(xmm_reg: (__m128d, __m128i));
//
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "avx")))]
#[cfg(any(target_feature = "avx", doc))]
pessimize_values!(ymm_reg: (__m256, __m256d));
//
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "avx2")))]
#[cfg(any(target_feature = "avx2", doc))]
pessimize_values!(ymm_reg: (__m256i));
//
/// AVX-512 specific functionality
#[cfg_attr(
    feature = "nightly",
    doc(cfg(all(feature = "nightly", target_feature = "avx512f")))
)]
#[cfg(all(feature = "nightly", any(target_feature = "avx512f", doc)))]
pub mod avx512 {
    use super::*;
    #[cfg(any(target_feature = "bf16", doc))]
    use target_arch::__m512bh;
    use target_arch::{__m512, __m512d, __m512i};

    // Basic register type support
    #[cfg_attr(feature = "nightly", doc(cfg(target_feature = "avx512f")))]
    pessimize_values!(zmm_reg: (__m512, __m512d, __m512i));
    //
    #[cfg_attr(
        feature = "nightly",
        doc(cfg(all(target_feature = "avx512f", target_feature = "bf16")))
    )]
    #[cfg(any(target_feature = "bf16", doc))]
    pessimize_values!(zmm_reg: (__m512bh));
    //
    #[cfg_attr(
        feature = "nightly",
        doc(cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512vl",
            target_feature = "bf16"
        )))
    )]
    #[cfg(any(all(target_feature = "avx512vl", target_feature = "bf16"), doc))]
    pessimize_values!(xmm_reg: (__m128bh), ymm_reg: (__m256bh));

    /// Like Pessimize, but assumes the input data is resident in an AVX-512
    /// mask register (kreg), which will reduce overhead if this is true.
    pub trait PessimizeMask {
        /// See `Pessimize::hide` and `PessimizeMask` documentation
        fn hide_mask(self) -> Self;

        /// See `Pessimize::assume_read` and `PessimizeMask` documentation
        fn assume_read_mask(&self);
    }

    macro_rules! pessimize_mask {
        ($($t:ty),*) => {
            $(
                impl PessimizeMask for $t {
                    #[inline(always)]
                    fn hide_mask(mut self) -> Self {
                        unsafe {
                            asm!("/* {0} */", inout(kreg) self, options(preserves_flags, nostack, nomem));
                        }
                        self
                    }

                    #[inline(always)]
                    fn assume_read_mask(&self) {
                        unsafe {
                            asm!("/* {0} */", in(kreg) *self, options(preserves_flags, nostack, nomem))
                        }
                    }
                }
            )*
        };
    }
    //
    pessimize_mask!(i8, u8, i16, u16);
    //
    #[cfg_attr(feature = "nightly", doc(cfg(target_feature = "avx512bw")))]
    #[cfg(any(target_feature = "avx512bw", doc))]
    pessimize_mask!(i32, u32, i64, u64);
}

// Support safe_arch types if enabled
#[cfg(any(feature = "safe_arch", test))]
mod safe_arch {
    #[allow(unused)]
    use crate::Pessimize;
    #[cfg(any(target_feature = "sse", doc))]
    use safe_arch::m128;
    #[cfg(any(target_feature = "avx2", doc))]
    use safe_arch::m256i;
    #[cfg(any(target_feature = "sse2", doc))]
    use safe_arch::{m128d, m128i};
    #[cfg(any(target_feature = "avx", doc))]
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

    #[cfg_attr(
        feature = "nightly",
        doc(cfg(all(feature = "safe_arch", target_feature = "sse")))
    )]
    #[cfg(any(target_feature = "sse", doc))]
    pessimize_safe_arch!(m128);

    #[cfg_attr(
        feature = "nightly",
        doc(cfg(all(feature = "safe_arch", target_feature = "sse2")))
    )]
    #[cfg(any(target_feature = "sse2", doc))]
    pessimize_safe_arch!(m128d, m128i);

    #[cfg_attr(
        feature = "nightly",
        doc(cfg(all(feature = "safe_arch", target_feature = "avx")))
    )]
    #[cfg(any(target_feature = "avx", doc))]
    pessimize_safe_arch!(m256, m256d);

    #[cfg_attr(
        feature = "nightly",
        doc(cfg(all(feature = "safe_arch", target_feature = "avx2")))
    )]
    #[cfg(any(target_feature = "avx2", doc))]
    pessimize_safe_arch!(m256i);
}

// TODO: Add nightly support for portable_simd types

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use crate::{
        tests::{test_simd, test_unoptimized_value},
        Pessimize,
    };

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

    // FIXME: Add tests for AVX-512 and BF16
}
