//! Implementations of Pessimize for arm and aarch64

use super::Pessimize;
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
use core::arch::aarch64::{float64x1_t, float64x2_t};
use core::arch::asm;

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
pessimize_values!(reg, i8, u8, i16, u16, i32, u32, isize, usize);
//
// 64-bit values normally go to GP registers on aarch64, but since 32-bit has
// no 64-bit GP registers, we try to use dreg instead if available
#[cfg(target_arch = "aarch64")]
pessimize_values!(reg, i64, u64);
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "vfp2")))]
#[cfg(all(not(target_arch = "aarch64"), any(target_feature = "vfp2", doc)))]
pessimize_values!(dreg, i64, u64);
//
// On AArch64 with NEON, using NEON registers for f32 and f64 is standard
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_values!(vreg, f32, f64);
// On AArch64 without NEON, f32 and f64 are passed via GP registers instead
#[cfg(all(target_arch = "aarch64", not(target_feature = "neon"), not(doc)))]
pessimize_values!(reg, f32, f64);
// On 32-bit ARM with VFP2, using sregs and dregs for f32 and f64 is standard
#[cfg(all(not(target_arch = "aarch64"), any(target_feature = "vfp2", doc)))]
pessimize_values!(sreg, f32);
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "vfp2")))]
#[cfg(all(not(target_arch = "aarch64"), any(target_feature = "vfp2", doc)))]
pessimize_values!(dreg, f64);
// On 32-bit ARM without VFP2, f32 is passed via GP registers
#[cfg(all(not(target_arch = "aarch64"), not(target_feature = "vfp2"), not(doc)))]
pessimize_values!(reg, f32);
//
#[cfg_attr(feature = "nightly", doc(cfg(target_feature = "neon")))]
#[cfg(all(target_arch = "aarch64", any(target_feature = "neon", doc)))]
pessimize_values!(vreg, float64x1_t, float64x2_t);

// TODO: Add nightly support for SIMD types which are currently gated by stdsimd
// TODO: Add nightly support for portable_simd types

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    #[allow(unused)]
    use crate::tests::{test_simd, test_unoptimized_value};

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    mod neon {
        use super::*;
        use core::arch::aarch64;

        #[test]
        fn neon() {
            test_simd::<f64, 1, F64x1>(f64::MIN, f64::MAX);
            test_simd::<f64, 2, F64x2>(f64::MIN, f64::MAX);
        }

        #[test]
        #[ignore]
        fn neon_optim() {
            test_unoptimized_value::<F64x1>();
            test_unoptimized_value::<F64x2>();
        }

        // Minimal NEON abstraction layer to be able to reuse main tests
        macro_rules! abstract_float64xN_t {
            ($name:ident, $inner:ty, $lanes:expr, $load:ident, $store:ident) => {
                #[derive(Clone, Copy, Debug)]
                struct $name($inner);

                impl From<[f64; $lanes]> for $name {
                    fn from(x: [f64; $lanes]) -> Self {
                        Self(unsafe { aarch64::$load(&x as *const [f64; $lanes] as *const f64) })
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
                                aarch64::$store(&mut result as *mut [f64; $lanes] as *mut f64, x.0)
                            }
                            result
                        };
                        value(self) == value(other)
                    }
                }

                impl Pessimize for $name {
                    fn hide(self) -> Self {
                        Self(self.0.hide())
                    }

                    fn assume_read(&self) {
                        self.0.assume_read()
                    }
                }
            };
        }
        abstract_float64xN_t!(F64x1, float64x1_t, 1, vld1_f64, vst1_f64);
        abstract_float64xN_t!(F64x2, float64x2_t, 2, vld1q_f64, vst1q_f64);
    }
}
