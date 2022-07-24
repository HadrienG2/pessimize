//! Implementations of Pessimize for RISC-V

use super::Pessimize;
use core::arch::asm;

// Implementation of Pessimize for values without pointer semantics
macro_rules! pessimize_values {
    ($reg:ident, $($t:ty),*) => {
        $(
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
#[cfg(target_arch = "riscv64")]
pessimize_values!(reg, i64, u64);
//
// If the F feature set is available, 32-bit floats are normally passed by freg
#[cfg(target_feature = "f")]
pessimize_values!(freg, f32);
// Otherwise, they are passed by reg
#[cfg(not(target_feature = "f"))]
pessimize_values!(reg, f32);
// Same idea for f64 with respect to the D feature set
#[cfg(target_feature = "d")]
pessimize_values!(freg, f64);
#[cfg(all(target_arch = "riscv64", not(target_feature = "d")))]
pessimize_values!(reg, f64);
