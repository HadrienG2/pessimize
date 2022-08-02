//! Implementations of Pessimize for RISC-V

use crate::pessimize_values;

pessimize_values!(
    allow(missing_docs)
    { reg: (i8, u8, i16, u16, i32, u32, isize, usize) }
);
#[cfg(target_arch = "riscv64")]
pessimize_values!(allow(missing_docs) { reg: (i64, u64) });

// If the F feature set is available, 32-bit floats are normally passed by freg
#[cfg(any(target_feature = "f", doc))]
pessimize_values!(allow(missing_docs) { freg: (f32) });
// Otherwise, they are passed by reg
#[cfg(all(not(target_feature = "f"), not(doc)))]
pessimize_values!(allow(missing_docs) { reg: (f32) });

// Same idea for f64 with respect to the D feature set
#[cfg(any(target_feature = "d", doc))]
pessimize_values!(
    doc(cfg(target_feature = "d"))
    { freg: (f64) }
);
#[cfg(all(target_arch = "riscv64", not(target_feature = "d"), not(doc)))]
pessimize_values!(allow(missing_docs) { reg: (f64) });
