//! Implementations of Pessimize for x86 and x86_64

use super::Pessimize;
use core::arch::asm;
#[allow(unused)]
#[cfg(target_arch = "x86")]
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
// FIXME: Clean up cfg(doc()) handling, then generalize to other arches
macro_rules! pessimize_values {
    ( $m:meta { $($reg:ident: ($($t:ty),*)),* }) => {
        $($(
            #[allow(asm_sub_register)]
            #[$m]
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
pessimize_values!(allow(missing_docs) { reg_byte: (i8, u8), reg: (i16, u16, i32, u32, isize, usize) });
//
// 64-bit values normally go to GP registers on x86_64, but since 32-bit has
// no 64-bit GP registers, we try to use XMM registers instead if available
#[cfg(target_arch = "x86_64")]
pessimize_values!(allow(missing_docs) { reg: (i64, u64) });
#[cfg(all(target_arch = "x86", any(target_feature = "sse2", doc)))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))
    {
        xmm_reg: (i64, u64)
    }
);
//
// Given that CPUs without SSE should now all be extinct and that compilers try
// to use SSE whenever at all possible, we assume that float data should go
// to SSE registers if possible and to GP registers otherwise (on old 32-bit).
#[cfg(any(target_feature = "sse", doc))]
pessimize_values!(
    allow(missing_docs)
    {
        xmm_reg: (f32)
    }
);
#[cfg(all(not(target_feature = "sse"), not(doc)))]
pessimize_values!(
    allow(missing_docs)
    {
        reg: (f32)
    }
);
//
// Ditto for double precision, but we need SSE2 for that
#[cfg(any(target_feature = "sse2", doc))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))
    {
        xmm_reg: (f64)
    }
);
//
#[cfg(any(target_feature = "sse", doc))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse")))
    {
        xmm_reg: (__m128)
    }
);
//
#[cfg(any(target_feature = "sse2", doc))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))
    {
        xmm_reg: (__m128d, __m128i)
    }
);
//
#[cfg(any(target_feature = "avx", doc))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "avx")))
    {
        ymm_reg: (__m256, __m256d)
    }
);
//
#[cfg(any(target_feature = "avx2", doc))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "avx2")))
    {
        ymm_reg: (__m256i)
    }
);
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
    pessimize_values!(
        doc(cfg(target_feature = "avx512f"))
        {
            zmm_reg: (__m512, __m512d, __m512i)
        }
    );
    //
    #[cfg(any(target_feature = "bf16", doc))]
    pessimize_values!(
        doc(cfg(all(target_feature = "avx512f", target_feature = "bf16")))
        {
            zmm_reg: (__m512bh)
        }
    );
    //
    #[cfg(any(all(target_feature = "avx512vl", target_feature = "bf16"), doc))]
    pessimize_values!(
        doc(cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512vl",
            target_feature = "bf16"
        )))
        { xmm_reg: (__m128bh), ymm_reg: (__m256bh) }
    );

    /// Like Pessimize, but assumes the input data is resident in an AVX-512
    /// mask register (kreg), which will reduce overhead if this is true.
    pub trait PessimizeMask {
        /// See `Pessimize::hide` and `PessimizeMask` documentation
        fn hide_mask(self) -> Self;

        /// See `Pessimize::assume_read` and `PessimizeMask` documentation
        fn assume_read_mask(&self);
    }

    macro_rules! pessimize_mask {
        ($m:meta { $($t:ty),* }) => {
            $(
                #[$m]
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
    pessimize_mask!(allow(missing_docs) { i8, u8, i16, u16 });
    //
    #[cfg(any(target_feature = "avx512bw", doc))]
    pessimize_mask!(doc(cfg(target_feature = "avx512bw")) { i32, u32, i64, u64 });
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
        ($m:meta { $($t:ty),* }) => {
            $(
                #[$m]
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

    #[cfg(any(target_feature = "sse", doc))]
    pessimize_safe_arch!(
        cfg_attr(
            feature = "nightly",
            doc(cfg(all(feature = "safe_arch", target_feature = "sse")))
        )
        { m128 }
    );

    #[cfg(any(target_feature = "sse2", doc))]
    pessimize_safe_arch!(
        cfg_attr(
            feature = "nightly",
            doc(cfg(all(feature = "safe_arch", target_feature = "sse2")))
        )
        { m128d, m128i });

    #[cfg(any(target_feature = "avx", doc))]
    pessimize_safe_arch!(
        cfg_attr(
            feature = "nightly",
            doc(cfg(all(feature = "safe_arch", target_feature = "avx")))
        )
        { m256, m256d }
    );

    #[cfg(any(target_feature = "avx2", doc))]
    pessimize_safe_arch!(
        cfg_attr(
            feature = "nightly",
            doc(cfg(all(feature = "safe_arch", target_feature = "avx2")))
        )
        { m256i }
    );
}

// Support portable_simd if enabled
#[cfg(feature = "nightly")]
mod portable_simd {
    #[allow(unused)]
    use super::*;
    #[allow(unused)]
    use core::simd::Simd;

    #[allow(unused)]
    macro_rules! pessimize_portable_simd {
        ($m:meta { $( $inner:ident: ($($t:ty),*) ),* }) => {
            $($(
                #[$m]
                impl Pessimize for $t {
                    #[inline(always)]
                    fn hide(self) -> Self {
                        // FIXME: This probably works, but it would be nicer if
                        //        portable_simd provided a conversion from
                        //        architectural SIMD types.
                        unsafe {
                            core::mem::transmute($inner::from(self).hide())
                        }
                    }

                    #[inline(always)]
                    fn assume_read(&self) {
                        $inner::from(*self).assume_read()
                    }
                }
            )*)*
        };
    }

    #[cfg(any(target_feature = "sse", doc))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "sse")))
        { __m128: (Simd<f32, 4>) }
    );

    #[cfg(any(target_feature = "sse2", doc))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "sse2")))
        {
            __m128d: (Simd<f64, 2>),
            __m128i:
                (
                    Simd<i8, 16>,
                    Simd<u8, 16>,
                    Simd<i16, 8>,
                    Simd<u16, 8>,
                    Simd<i32, 4>,
                    Simd<u32, 4>,
                    Simd<i64, 2>,
                    Simd<u64, 2>
                )
        }
    );
    //
    #[cfg(all(target_arch = "x86", any(target_feature = "sse2", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_arch = "x86", target_feature = "sse2")))
        { __m128i: (Simd<isize, 4>, Simd<usize, 4>) }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "sse2", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_arch = "x86_64", target_feature = "sse2")))
        { __m128i: (Simd<isize, 2>, Simd<usize, 2>) }
    );

    #[cfg(any(target_feature = "avx", doc))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "avx")))
        { __m256: (Simd<f32, 8>), __m256d: (Simd<f64, 4>) }
    );

    #[cfg(any(target_feature = "avx2", doc))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "avx2")))
        {
            __m256i:
                (
                    Simd<i8, 32>,
                    Simd<u8, 32>,
                    Simd<i16, 16>,
                    Simd<u16, 16>,
                    Simd<i32, 8>,
                    Simd<u32, 8>,
                    Simd<i64, 4>,
                    Simd<u64, 4>
                )
        }
    );
    //
    #[cfg(all(target_arch = "x86", any(target_feature = "avx2", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_arch = "x86", target_feature = "avx2")))
        { __m256i: (Simd<isize, 8>, Simd<usize, 8>) }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "avx2", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_arch = "x86_64", target_feature = "avx2")))
        { __m256i: (Simd<isize, 4>, Simd<usize, 4>) }
    );

    #[cfg(any(target_feature = "avx512f", doc))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "avx512f")))
        {
            __m512: (Simd<f32, 16>),
            __m512d: (Simd<f64, 8>),
            __m512i: (Simd<i32, 16>, Simd<u32, 16>, Simd<i64, 8>, Simd<u64, 8>)
        }
    );
    //
    #[cfg(all(target_arch = "x86", any(target_feature = "avx512f", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_arch = "x86", target_feature = "avx512f")))
        { __m512i: (Simd<isize, 16>, Simd<usize, 16>) }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "avx512f", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(
            feature = "nightly",
            target_arch = "x86_64",
            target_feature = "avx512f"
        )))
        { __m512i: (Simd<isize, 8>, Simd<usize, 8>) }
    );

    #[cfg(any(all(target_feature = "avx512f", target_feature = "avx512bw"), doc))]
    pessimize_portable_simd!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )))
        { __m512i: (Simd<i8, 64>, Simd<u8, 64>, Simd<i16, 32>, Simd<u16, 32>) }
    );

    // TODO: Add masks for AVX-512 via ToBitMask and PessimizeMask
}

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
    // FIXME: Add tests for portable_simd
}
