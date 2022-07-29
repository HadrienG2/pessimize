//! Implementations of Pessimize for arm and aarch64

use super::{pessimize_values, Pessimize};
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
use core::arch::aarch64::{float64x1_t, float64x2_t};

pessimize_values!(allow(missing_docs) { reg: (i8, u8, i16, u16, i32, u32, isize, usize) });

// 64-bit values normally go to GP registers on aarch64, but since 32-bit has
// no 64-bit GP registers, we try to use dreg instead if available
#[cfg(target_arch = "aarch64")]
pessimize_values!(allow(missing_docs) { reg: (i64, u64) });
#[cfg(all(target_arch = "arm", any(target_feature = "vfp2", doc)))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "vfp2")))
    { dreg: (i64, u64) }
);

// On AArch64 with NEON, using NEON registers for f32 and f64 is standard
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_values!(allow(missing_docs) { vreg: (f32, f64) });
// On AArch64 without NEON, f32 and f64 are passed via GP registers instead
#[cfg(all(target_arch = "aarch64", not(target_feature = "neon"), not(doc)))]
pessimize_values!(allow(missing_docs) { reg: (f32, f64) });
// On 32-bit ARM with VFP2, using sregs and dregs for f32 and f64 is standard
#[cfg(all(target_arch = "arm", any(target_feature = "vfp2", doc)))]
pessimize_values!(allow(missing_docs) { sreg: (f32) });
#[cfg(all(target_arch = "arm", any(target_feature = "vfp2", doc)))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "vfp2")))
    { dreg: (f64) }
);
// On 32-bit ARM without VFP2, f32 is passed via GP registers
#[cfg(all(target_arch = "arm", not(target_feature = "vfp2"), not(doc)))]
pessimize_values!(allow(missing_docs) { reg: (f32) });

// SIMD types
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_values!(
    cfg_attr(feature = "nightly", doc(cfg(target_feature = "neon")))
    { vreg: (float64x1_t, float64x2_t) }
);

// TODO: Add nightly support for SIMD types which are currently gated by stdsimd
//       (+ portable_simd support and associated tests)

// Support portable_simd if enabled
#[cfg(feature = "nightly")]
mod portable_simd {
    #[allow(unused)]
    use crate::{arm::*, pessimize_portable_simd, Pessimize};
    #[allow(unused)]
    use core::simd::Simd;

    #[cfg(target_arch = "aarch64", all(any(target_feature = "neon", doc)))]
    pessimize_portable_simd!(
        doc(cfg(all(feature = "nightly", target_feature = "neon")))
        {
            float64x1_t: (Simd<f64, 1>),
            float64x2_t: (Simd<f64, 2>)
        }
    );
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    #[allow(unused)]
    use crate::{
        consume, hide,
        tests::{test_simd, test_unoptimized_value_type},
    };

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    mod neon {
        use super::*;
        use core::arch::aarch64;

        #[test]
        fn neon() {
            test_simd::<f64, 1, F64x1>(f64::MIN, f64::MAX);
            test_simd::<f64, 2, F64x2>(f64::MIN, f64::MAX);
            #[cfg(target_feature = "nightly")]
            {
                test_simd::<f64, 1, Simd<f64, 1>>(f64::MIN, f64::MAX);
                test_simd::<f64, 2, Simd<f64, 2>>(f64::MIN, f64::MAX);
            }
        }

        #[test]
        #[ignore]
        fn neon_optim() {
            test_unoptimized_value_type::<F64x1>();
            test_unoptimized_value_type::<F64x2>();
            #[cfg(target_feature = "nightly")]
            {
                test_unoptimized_value_type::<Simd<f64, 1>>();
                test_unoptimized_value_type::<Simd<f64, 2>>();
            }
        }

        // Minimal NEON abstraction layer to be able to reuse main tests
        macro_rules! abstract_float64xN_t {
            ($name:ident, $inner:ty, $lanes:expr, $load:ident, $store:ident) => {
                #[derive(Clone, Copy, Debug)]
                struct $name($inner);

                impl From<[f64; $lanes]> for $name {
                    fn from(x: [f64; $lanes]) -> Self {
                        // FIXME: Check alignment requirements, not sure if this is right
                        Self(unsafe { aarch64::$load((&x) as *const [f64; $lanes] as *const f64) })
                    }
                }

                impl Default for $name {
                    fn default() -> Self {
                        Self::from([0.0; $lanes])
                    }
                }

                impl PartialEq for $name {
                    fn eq(&self, other: &Self) -> bool {
                        let value = |x: &Self| -> [f64; $lanes] {
                            let mut result = [0.0; $lanes];
                            unsafe {
                                // FIXME: Check alignment requirements, not sure if this is right
                                aarch64::$store((&mut x) as *mut [f64; $lanes] *mut f64, x.0)
                            }
                            result
                        };
                        value(self) == value(other)
                    }
                }

                unsafe impl Pessimize for $name {
                    fn hide(self) -> Self {
                        Self(hide(self.0))
                    }

                    fn assume_read(&self) {
                        consume(self.0)
                    }
                }
            };
        }
        abstract_float64xN_t!(F64x1, float64x1_t, 1, vld1_f64, vst1_f64);
        abstract_float64xN_t!(F64x2, float64x2_t, 2, vld1q_f64, vst1q_f64);
    }
}
