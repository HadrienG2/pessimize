//! Implementations of Pessimize for arm and aarch64

use super::Pessimize;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
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
#[cfg(all(not(target_arch = "aarch64"), target_feature = "vfp2"))]
pessimize_values!(dreg, i64, u64);
//
// On AArch64 with NEON, using NEON registers for f32 and f64 is standard
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pessimize_values!(vreg, f32, f64);
// On AArch64 without NEON, f32 and f64 are passed via GP registers instead
#[cfg(all(target_arch = "aarch64", not(target_feature = "neon")))]
pessimize_values!(reg, f32, f64);
// On 32-bit ARM with VFP2, using sregs and dregs for f32 and f64 is standard
#[cfg(all(not(target_arch = "aarch64"), target_feature = "vfp2"))]
pessimize_values!(sreg, f32);
#[cfg(all(not(target_arch = "aarch64"), target_feature = "vfp2"))]
pessimize_values!(dreg, f64);
// On 32-bit ARM without VFP2, f32 is passed via GP registers
#[cfg(all(not(target_arch = "aarch64"), not(target_feature = "vfp2")))]
pessimize_values!(reg, f32);
//
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pessimize_values!(vreg, float64x1_t, float64x2_t);

// TODO: Add nightly support for SIMD types which are currently gated by stdsimd
// TODO: Add nightly support for portable_simd types

// FIXME: Add tests in the same spirit as the x86 ones
