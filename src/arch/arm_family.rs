//! Implementations of Pessimize for arm and aarch64

use crate::pessimize_asm_values;
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
use crate::pessimize_tuple_structs;
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
use core::arch::aarch64::{
    float64x1_t, float64x1x2_t, float64x1x3_t, float64x1x4_t, float64x2_t, float64x2x2_t,
    float64x2x3_t, float64x2x4_t,
};

pessimize_asm_values!(allow(missing_docs) { reg: (i8, u8, i16, u16, i32, u32, isize, usize) });

// 64-bit values normally go to GP registers on aarch64, but since 32-bit has
// no 64-bit GP registers, we try to use dreg instead if available
#[cfg(target_arch = "aarch64")]
pessimize_asm_values!(allow(missing_docs) { reg: (i64, u64) });
#[cfg(all(
    target_arch = "arm",
    any(all(target_feature = "vfp2", target_feature = "d32"), doc)
))]
pessimize_asm_values!(
    doc(cfg(all(target_feature = "vfp2", target_feature = "d32")))
    { dreg: (i64, u64) }
);

// On AArch64 with NEON, using NEON registers for f32 and f64 is standard
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_asm_values!(allow(missing_docs) { vreg: (f32, f64) });
// On AArch64 without NEON, f32 and f64 are passed via GP registers instead
#[cfg(all(target_arch = "aarch64", not(target_feature = "neon"), not(doc)))]
pessimize_asm_values!(allow(missing_docs) { reg: (f32, f64) });
// On 32-bit ARM with VFP2, using sregs and dregs for f32 and f64 is standard
#[cfg(all(target_arch = "arm", any(target_feature = "vfp2", doc)))]
pessimize_asm_values!(allow(missing_docs) { sreg: (f32) });
#[cfg(all(
    target_arch = "arm",
    any(all(target_feature = "vfp2", target_feature = "d32"), doc)
))]
pessimize_asm_values!(
    doc(cfg(all(target_feature = "vfp2", target_feature = "d32")))
    { dreg: (f64) }
);
// On 32-bit ARM without VFP2, f32 is passed via GP registers
#[cfg(all(target_arch = "arm", not(target_feature = "vfp2"), not(doc)))]
pessimize_asm_values!(allow(missing_docs) { reg: (f32) });

// SIMD types
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_asm_values!(
    doc(cfg(target_feature = "neon"))
    { vreg: (float64x1_t, float64x2_t) }
);
//
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_tuple_structs!(
    doc(cfg(target_feature = "neon"))
    {
        float64x1x2_t { a: float64x1_t, b: float64x1_t },
        float64x1x3_t { a: float64x1_t, b: float64x1_t, c: float64x1_t },
        float64x1x4_t { a: float64x1_t, b: float64x1_t, c: float64x1_t, d: float64x1_t },
        float64x2x2_t { a: float64x2_t, b: float64x2_t },
        float64x2x3_t { a: float64x2_t, b: float64x2_t, c: float64x2_t },
        float64x2x4_t { a: float64x2_t, b: float64x2_t, c: float64x2_t, d: float64x2_t }
    }
);
//
// TODO: Add nightly support for SIMD types which are currently gated by stdsimd
//       (+ portable_simd support and associated tests)

// Support portable_simd if enabled
#[allow(unused)]
#[cfg(feature = "nightly")]
mod portable_simd {
    use crate::{arch::arm_family::*, pessimize_into_from};
    use core::simd::Simd;

    #[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
    pessimize_into_from!(
        doc(cfg(all(feature = "nightly", target_feature = "neon")))
        {
            float64x1_t: (Simd<f64, 1>),
            float64x2_t: (Simd<f64, 2>)
        }
    );
}

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pessimize_newtypes,
        tests::{test_simd, test_unoptimized_value_type},
    };

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    mod neon {
        use super::*;
        use core::arch::aarch64;

        #[test]
        fn neon() {
            test_simd::<f64, 1, TestableF64x1>(f64::MIN, f64::MAX);
            test_simd::<f64, 2, TestableF64x2>(f64::MIN, f64::MAX);
            test_simd::<f64, 2, TestableF64x1x2>(f64::MIN, f64::MAX);
            test_simd::<f64, 3, TestableF64x1x3>(f64::MIN, f64::MAX);
            test_simd::<f64, 4, TestableF64x1x4>(f64::MIN, f64::MAX);
            test_simd::<f64, 4, TestableF64x2x2>(f64::MIN, f64::MAX);
            test_simd::<f64, 6, TestableF64x2x3>(f64::MIN, f64::MAX);
            test_simd::<f64, 8, TestableF64x2x4>(f64::MIN, f64::MAX);
            #[cfg(feature = "nightly")]
            {
                test_simd::<f64, 1, Simd<f64, 1>>(f64::MIN, f64::MAX);
                test_simd::<f64, 2, Simd<f64, 2>>(f64::MIN, f64::MAX);
            }
        }

        #[test]
        #[ignore]
        fn neon_optim() {
            test_unoptimized_value_type::<TestableF64x1>();
            test_unoptimized_value_type::<TestableF64x2>();
            test_unoptimized_value_type::<TestableF64x1x2>();
            test_unoptimized_value_type::<TestableF64x1x3>();
            test_unoptimized_value_type::<TestableF64x1x4>();
            test_unoptimized_value_type::<TestableF64x2x2>();
            test_unoptimized_value_type::<TestableF64x2x3>();
            test_unoptimized_value_type::<TestableF64x2x4>();
            #[cfg(feature = "nightly")]
            {
                test_unoptimized_value_type::<Simd<f64, 1>>();
                test_unoptimized_value_type::<Simd<f64, 2>>();
            }
        }

        // Minimal NEON abstraction layer to be able to reuse main tests
        macro_rules! abstract_float64xN_t {
            (
                $(
                    ($name:ident, $inner:ident, $lanes:expr, $load:ident, $store:ident)
                ),*
            ) => {
                $(
                    #[derive(Clone, Copy, Debug)]
                    struct $name($inner);

                    impl From<[f64; $lanes]> for $name {
                        #[inline]
                        fn from(x: [f64; $lanes]) -> Self {
                            // Round trip to $inner ensures SIMD alignment
                            unsafe {
                                let x = core::mem::transmute::<[f64; $lanes], $inner>(x);
                                Self(aarch64::$load((&x) as *const $inner as *const f64))
                            }
                        }
                    }

                    impl Default for $name {
                        #[inline]
                        fn default() -> Self {
                            Self::from([0.0; $lanes])
                        }
                    }

                    impl PartialEq for $name {
                        #[inline]
                        fn eq(&self, other: &Self) -> bool {
                            let value = |x: &Self| -> [f64; $lanes] {
                                // Round trip to Self ensures SIMD alignment
                                let mut result = Self::from([0.0; $lanes]);
                                unsafe {
                                    aarch64::$store(
                                        (&mut result) as *mut Self as *mut f64,
                                        x.0,
                                    );
                                    core::mem::transmute::<Self, [f64; $lanes]>(result)
                                }
                            };
                            value(self) == value(other)
                        }
                    }

                    pessimize_newtypes!( allow(missing_docs) { $name{ $inner } } );
                )*
            };
        }
        abstract_float64xN_t!(
            (TestableF64x1, float64x1_t, 1, vld1_f64, vst1_f64),
            (TestableF64x2, float64x2_t, 2, vld1q_f64, vst1q_f64),
            (TestableF64x1x2, float64x1x2_t, 2, vld1_f64_x2, vst1_f64_x2),
            (TestableF64x1x3, float64x1x3_t, 3, vld1_f64_x3, vst1_f64_x3),
            (TestableF64x1x4, float64x1x4_t, 4, vld1_f64_x4, vst1_f64_x4),
            (
                TestableF64x2x2,
                float64x2x2_t,
                4,
                vld1q_f64_x2,
                vst1q_f64_x2
            ),
            (
                TestableF64x2x3,
                float64x2x3_t,
                6,
                vld1q_f64_x3,
                vst1q_f64_x3
            ),
            (
                TestableF64x2x4,
                float64x2x4_t,
                8,
                vld1q_f64_x4,
                vst1q_f64_x4
            )
        );
    }
}
