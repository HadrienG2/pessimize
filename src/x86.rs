//! Implementations of Pessimize for x86 and x86_64

use super::{pessimize_values, Pessimize};
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
    any(all(target_feature = "avx512vl", target_feature = "avx512bf16"), doc)
))]
use target_arch::{__m128bh, __m256bh};
#[cfg(any(target_feature = "sse2", doc))]
use target_arch::{__m128d, __m128i};
#[cfg(any(target_feature = "avx", doc))]
use target_arch::{__m256, __m256d};

pessimize_values!(allow(missing_docs) { reg_byte: (i8, u8), reg: (i16, u16, i32, u32, isize, usize) });

// 64-bit values normally go to GP registers on x86_64, but since 32-bit has
// no 64-bit GP registers, we try to use XMM registers instead if available
#[cfg(target_arch = "x86_64")]
pessimize_values!(allow(missing_docs) { reg: (i64, u64) });
//
#[cfg(all(target_arch = "x86", any(target_feature = "sse2", doc)))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))
    {
        xmm_reg: (i64, u64)
    }
);

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
//
#[cfg(all(not(target_feature = "sse"), not(doc)))]
pessimize_values!(
    allow(missing_docs)
    {
        reg: (f32)
    }
);

// Ditto for double precision, but we need SSE2 for that
#[cfg(any(target_feature = "sse2", doc))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "sse2")))
    {
        xmm_reg: (f64)
    }
);

// Then come SIMD registers
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

/// AVX-512 specific functionality
#[cfg_attr(
    feature = "nightly",
    doc(cfg(all(feature = "nightly", target_feature = "avx512f")))
)]
#[cfg(all(feature = "nightly", any(target_feature = "avx512f", doc)))]
pub mod avx512 {
    use super::*;
    #[cfg(any(target_feature = "avx512bf16", doc))]
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
    #[cfg(any(target_feature = "avx512bf16", doc))]
    pessimize_values!(
        doc(cfg(target_feature = "avx512bf16"))
        {
            zmm_reg: (__m512bh)
        }
    );
    //
    #[cfg(any(all(target_feature = "avx512vl", target_feature = "avx512bf16"), doc))]
    pessimize_values!(
        doc(cfg(all(
            target_feature = "avx512vl",
            target_feature = "avx512bf16"
        )))
        { xmm_reg: (__m128bh), ymm_reg: (__m256bh) }
    );

    /// AVX-512 mask register pessimization
    ///
    /// By wrapping a primitive integer into this tuple struct, you access an
    /// alternate Pessimize implementation that asserts this integer should be
    /// kept resident in AVX-512 mask registers, not general purpose registers.
    /// This will avoid register moves  if that's how it is actually used.
    pub struct Mask<T>(pub T);
    //
    macro_rules! pessimize_mask {
        (
            $doc_cfg:meta
            { $($mask_impl:ty),* }
        ) => {
            $(
                #[$doc_cfg]
                unsafe impl Pessimize for Mask<$mask_impl> {
                    #[inline(always)]
                    fn hide(mut self) -> Self {
                        unsafe {
                            asm!("/* {0} */", inout(kreg) self.0, options(preserves_flags, nostack, nomem));
                        }
                        self
                    }

                    #[inline(always)]
                    fn assume_read(&self) {
                        unsafe {
                            asm!("/* {0} */", in(kreg) self.0, options(preserves_flags, nostack, nomem))
                        }
                    }
                }
            )*
        };
    }
    //
    pessimize_mask!(doc(cfg(target_feature = "avx512f")) { i8, u8, i16, u16 });
    //
    #[cfg(any(target_feature = "avx512bw", doc))]
    pessimize_mask!(
        doc(cfg(target_feature = "avx512bw"))
        { i32, u32, i64, u64 }
    );
}

// Support safe_arch types if enabled
#[cfg(any(feature = "safe_arch", test))]
mod safe_arch {
    #[allow(unused)]
    use crate::{consume, hide, Pessimize};
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
        (
            $doc_cfg:meta
            { $($safe_arch_type:ty),* }
        ) => {
            $(
                #[$doc_cfg]
                unsafe impl Pessimize for $safe_arch_type {
                    #[inline(always)]
                    fn hide(self) -> Self {
                        Self(hide(self.0))
                    }

                    #[inline(always)]
                    fn assume_read(&self) {
                        consume(self.0)
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
    use crate::{consume, hide, pessimize_portable_simd, x86::*, Pessimize};
    #[allow(unused)]
    use core::simd::{Mask, Simd, ToBitMask};
    #[cfg(any(target_feature = "avx512f", doc))]
    use target_arch::{__m512, __m512d, __m512i};

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
        doc(cfg(all(feature = "nightly", target_feature = "sse2")))
        { __m128i: (Simd<isize, 4>, Simd<usize, 4>) }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "sse2", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "sse2")))
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
        doc(cfg(all(feature = "nightly", target_feature = "avx2")))
        { __m256i: (Simd<isize, 8>, Simd<usize, 8>) }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "avx2", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "avx2")))
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
        doc(cfg(all(feature = "nightly", target_feature = "avx512f")))
        { __m512i: (Simd<isize, 16>, Simd<usize, 16>) }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "avx512f", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512f"
        )))
        { __m512i: (Simd<isize, 8>, Simd<usize, 8>) }
    );

    #[cfg(any(all(target_feature = "avx512f", target_feature = "avx512bw"), doc))]
    pessimize_portable_simd!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512bw"
        )))
        { __m512i: (Simd<i8, 64>, Simd<u8, 64>, Simd<i16, 32>, Simd<u16, 32>) }
    );

    // Generate Pessimize implementation for a portable_simd Mask type
    #[cfg(any(target_feature = "avx512f", doc))]
    macro_rules! pessimize_portable_mask {
        (
            $doc_cfg:meta
            { $($mask_type:ty),* }
        ) => {
            $(
                #[$doc_cfg]
                unsafe impl Pessimize for $mask_type {
                    #[inline(always)]
                    fn hide(self) -> Self {
                        Self::from_bitmask(
                            hide(avx512::Mask(self.to_bitmask())).0
                        )
                    }

                    #[inline(always)]
                    fn assume_read(&self) {
                        consume(avx512::Mask(self.to_bitmask()))
                    }
                }
            )*
        };
    }

    #[cfg(any(target_feature = "avx512f", doc))]
    pessimize_portable_mask!(
        doc(cfg(all(feature = "nightly", target_feature = "avx512f")))
        { Mask<i32, 16>, Mask<i64, 8> }
    );
    //
    #[cfg(all(target_arch = "x86", any(target_feature = "avx512f", doc)))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512f"
        )))
        { Mask<isize, 16> }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "avx512f", doc)))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512f"
        )))
        { Mask<isize, 8> }
    );

    #[cfg(any(target_feature = "avx512bw", doc))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512bw"
        )))
        { Mask<i8, 64>, Mask<i16, 32> }
    );

    #[cfg(any(target_feature = "avx512vl", doc))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512vl"
        )))
        {
            Mask<i8, 16>,
            Mask<i16, 8>, Mask<i16, 16>,
            Mask<i32, 4>, Mask<i32, 8>,
            Mask<i64, 2>, Mask<i64, 4>
        }
    );
    //
    #[cfg(all(target_arch = "x86", any(target_feature = "avx512vl", doc)))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512vl"
        )))
        { Mask<isize, 4>, Mask<isize, 8> }
    );
    //
    #[cfg(all(target_arch = "x86_64", any(target_feature = "avx512vl", doc)))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512vl"
        )))
        { Mask<isize, 2>, Mask<isize, 4> }
    );

    #[cfg(any(all(target_feature = "avx512bw", target_feature = "avx512vl"), doc))]
    pessimize_portable_mask!(
        doc(cfg(all(
            feature = "nightly",
            target_feature = "avx512bw"
            target_feature = "avx512vl"
        )))
        { Mask<i8, 32> }
    );
}

#[allow(unused)]
#[cfg(test)]
mod tests {
    #[cfg(feature = "nightly")]
    use crate::tests::test_portable_simd;
    use crate::{
        tests::{test_simd, test_unoptimized_value_type},
        Pessimize,
    };
    #[cfg(feature = "nightly")]
    use std::{
        fmt::Debug,
        simd::{LaneCount, Mask, MaskElement, Simd, SupportedLaneCount},
    };

    #[cfg(feature = "nightly")]
    fn test_portable_simd_mask<T: Debug + MaskElement + PartialEq, const LANES: usize>()
    where
        LaneCount<LANES>: SupportedLaneCount,
        Mask<T, LANES>: Pessimize,
    {
        test_simd::<bool, LANES, Mask<T, LANES>>(false, true)
    }

    #[cfg(feature = "nightly")]
    macro_rules! portable_simd_tests {
        ( $( ( $elem:ty, $lanes:expr ) ),* ) => {
            $( test_portable_simd::<$elem, $lanes>(<$elem>::MIN, <$elem>::MAX); )*
        }
    }

    #[cfg(feature = "nightly")]
    macro_rules! portable_mask_tests {
        ( $( ( $elem:ty, $lanes:expr ) ),* ) => {
            $( test_portable_simd_mask::<$elem, $lanes>(); )*
        }
    }

    #[cfg(feature = "nightly")]
    macro_rules! portable_simd_tests_optim {
        ( $( ( $elem:ty, $lanes:expr ) ),* ) => {
            $( test_unoptimized_value_type::<Simd<$elem, $lanes>>(); )*
        }
    }

    #[cfg(feature = "nightly")]
    macro_rules! portable_mask_tests_optim {
        ( $( ( $elem:ty, $lanes:expr ) ),* ) => {
            $( test_unoptimized_value_type::<Mask<$elem, $lanes>>(); )*
        }
    }

    #[cfg(target_feature = "sse")]
    #[test]
    fn sse() {
        use safe_arch::m128;
        test_simd::<f32, 4, m128>(f32::MIN, f32::MAX);
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests!((f32, 4));
            #[cfg(target_feature = "avx512vl")]
            portable_mask_tests!((i32, 4));
        }
    }

    #[cfg(target_feature = "sse")]
    #[test]
    #[ignore]
    fn sse_optim() {
        use safe_arch::m128;
        test_unoptimized_value_type::<m128>();
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests_optim!((f32, 4));
            #[cfg(target_feature = "avx512vl")]
            portable_mask_tests_optim!((i32, 4));
        }
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    fn sse2() {
        use safe_arch::{m128d, m128i};
        test_simd::<f64, 2, m128d>(f64::MIN, f64::MAX);
        test_simd::<i8, 16, m128i>(i8::MIN, i8::MAX);
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests!(
                (i8, 16),
                (u8, 16),
                (i16, 8),
                (u16, 8),
                (i32, 4),
                (u32, 4),
                (i64, 2),
                (u64, 2)
            );
            #[cfg(target_arch = "x86")]
            portable_simd_tests!((isize, 4), (usize, 4));
            #[cfg(target_arch = "x86_64")]
            portable_simd_tests!((isize, 2), (usize, 2));
            #[cfg(target_feature = "avx512vl")]
            {
                portable_mask_tests!((i8, 16), (i16, 8), (i64, 2));
                #[cfg(target_arch = "x86")]
                portable_mask_tests!((isize, 4));
                #[cfg(target_arch = "x86_64")]
                portable_mask_tests!((isize, 2));
            }
        }
    }

    #[cfg(target_feature = "sse2")]
    #[test]
    #[ignore]
    fn sse2_optim() {
        use safe_arch::{m128d, m128i};
        test_unoptimized_value_type::<m128d>();
        test_unoptimized_value_type::<m128i>();
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests_optim!(
                (i8, 16),
                (u8, 16),
                (i16, 8),
                (u16, 8),
                (i32, 4),
                (u32, 4),
                (i64, 2),
                (u64, 2)
            );
            #[cfg(target_arch = "x86")]
            portable_simd_tests_optim!((isize, 4), (usize, 4));
            #[cfg(target_arch = "x86_64")]
            portable_simd_tests_optim!((isize, 2), (usize, 2));
            #[cfg(target_feature = "avx512vl")]
            {
                portable_mask_tests_optim!((i8, 16), (i16, 8), (i64, 2));
                #[cfg(target_arch = "x86")]
                portable_mask_tests_optim!((isize, 4));
                #[cfg(target_arch = "x86_64")]
                portable_mask_tests_optim!((isize, 2));
            }
        }
    }

    #[cfg(target_feature = "avx")]
    #[test]
    fn avx() {
        use safe_arch::{m256, m256d};
        test_simd::<f32, 8, m256>(f32::MIN, f32::MAX);
        test_simd::<f64, 4, m256d>(f64::MIN, f64::MAX);
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests!((f32, 8), (f64, 4));
            #[cfg(target_feature = "avx512vl")]
            portable_mask_tests!((i32, 8), (i64, 4));
        }
    }

    #[cfg(target_feature = "avx")]
    #[test]
    #[ignore]
    fn avx_optim() {
        use safe_arch::{m256, m256d};
        test_unoptimized_value_type::<m256>();
        test_unoptimized_value_type::<m256d>();
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests_optim!((f32, 8), (f64, 4));
            #[cfg(target_feature = "avx512vl")]
            portable_mask_tests_optim!((i32, 8), (i64, 4));
        }
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn avx2() {
        use safe_arch::m256i;
        test_simd::<i8, 32, m256i>(i8::MIN, i8::MAX);
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests!(
                (i8, 32),
                (u8, 32),
                (i16, 16),
                (u16, 16),
                (i32, 8),
                (u32, 8),
                (i64, 4),
                (u64, 4)
            );
            #[cfg(target_arch = "x86")]
            portable_simd_tests!((isize, 8), (usize, 8));
            #[cfg(target_arch = "x86_64")]
            portable_simd_tests!((isize, 4), (usize, 4));
            #[cfg(target_feature = "avx512vl")]
            {
                portable_mask_tests!((i16, 16));
                #[cfg(target_arch = "x86")]
                portable_mask_tests!((isize, 8));
                #[cfg(target_arch = "x86_64")]
                portable_mask_tests!((isize, 4));
                #[cfg(target_feature = "avx512bw")]
                portable_mask_tests!((i8, 32));
            }
        }
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    #[ignore]
    fn avx2_optim() {
        use safe_arch::m256i;
        test_unoptimized_value_type::<m256i>();
        #[cfg(feature = "nightly")]
        {
            portable_simd_tests_optim!(
                (i8, 32),
                (u8, 32),
                (i16, 16),
                (u16, 16),
                (i32, 8),
                (u32, 8),
                (i64, 4),
                (u64, 4)
            );
            #[cfg(target_arch = "x86")]
            portable_simd_tests_optim!((isize, 8), (usize, 8));
            #[cfg(target_arch = "x86_64")]
            portable_simd_tests_optim!((isize, 4), (usize, 4));
            #[cfg(target_feature = "avx512vl")]
            {
                portable_mask_tests_optim!((i16, 16));
                #[cfg(target_arch = "x86")]
                portable_mask_tests_optim!((isize, 8));
                #[cfg(target_arch = "x86_64")]
                portable_mask_tests_optim!((isize, 4));
                #[cfg(target_feature = "avx512bw")]
                portable_mask_tests_optim!((i8, 32));
            }
        }
    }

    #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
    mod avx512 {
        use super::*;

        #[test]
        fn avx512f() {
            portable_simd_tests!(
                (i32, 16),
                (u32, 16),
                (f32, 16),
                (i64, 8),
                (u64, 8),
                (f64, 8)
            );
            portable_mask_tests!((i32, 16), (i64, 8));
            #[cfg(target_arch = "x86")]
            {
                portable_simd_tests!((isize, 16), (usize, 16));
                portable_mask_tests!((isize, 16));
            }
            #[cfg(target_arch = "x86_64")]
            {
                portable_simd_tests!((isize, 8), (usize, 8));
                portable_mask_tests!((isize, 8));
            }
        }

        #[test]
        #[ignore]
        fn avx512f_optim() {
            portable_simd_tests_optim!(
                (i32, 16),
                (u32, 16),
                (f32, 16),
                (i64, 8),
                (u64, 8),
                (f64, 8)
            );
            portable_mask_tests_optim!((i32, 16), (i64, 8));
            #[cfg(target_arch = "x86")]
            {
                portable_simd_tests_optim!((isize, 16), (usize, 16));
                portable_mask_tests_optim!((isize, 16));
            }
            #[cfg(target_arch = "x86_64")]
            {
                portable_simd_tests_optim!((isize, 8), (usize, 8));
                portable_mask_tests_optim!((isize, 8));
            }
        }

        #[cfg(target_feature = "avx512bw")]
        #[test]
        fn avx512bw() {
            portable_simd_tests!((i8, 64), (u8, 64), (i16, 32), (u16, 32));
            portable_mask_tests!((i8, 64), (i16, 32));
        }

        #[cfg(target_feature = "avx512bw")]
        #[test]
        #[ignore]
        fn avx512bw_optim() {
            portable_simd_tests_optim!((i8, 64), (u8, 64), (i16, 32), (u16, 32));
            portable_mask_tests_optim!((i8, 64), (i16, 32));
        }

        // TODO: Add avx512bf16 tests once there are enough operations to allow
        //       for it (at least conversion from BF16 back to F32).
    }
}
