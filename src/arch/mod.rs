//! Hardware-specific functionality

// Each architecture-specific module is tasked to implement Pessimize for
// primitive integers, floats and SIMD vector types.
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm_family;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv_family;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[doc(hidden)]
pub mod x86_family;
// TODO: On nightly, support more arches via asm_experimental_arch

// For arch with specific definitions, export under std-like name
#[cfg(target_arch = "x86")]
pub use x86_family as x86;
#[cfg(target_arch = "x86_64")]
pub use x86_family as x86_64;
