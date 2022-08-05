//! Pessimize implementations for core::alloc

use crate::pessimize_copy;
use core::alloc::Layout;
#[cfg(feature = "std")]
use std::alloc::System;

pessimize_copy!(
    allow(missing_docs)
    {
        (usize, usize): (
            Layout: (
                |self_: Layout| (self_.size(), self_.align()),
                |(size, align)| Self::from_size_align_unchecked(size, align)
            )
        )
    }
);

#[cfg(feature = "std")]
crate::pessimize_zsts!(doc(cfg(feature = "std")) { System: System });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value, test_value};

    #[test]
    fn layout() {
        test_value(Layout::from_size_align(0, 1).unwrap());
        let last_usize_pow2 = usize::MAX - (usize::MAX >> 1);
        let prev_usize_pow2 = last_usize_pow2 >> 1;
        test_value(Layout::from_size_align(prev_usize_pow2, prev_usize_pow2).unwrap());
    }

    #[test]
    #[ignore]
    fn layout_optim() {
        test_unoptimized_value(Layout::from_size_align(0, 1).unwrap());
    }

    // FIXME: Can't test the System impl since System doesn't implement the right traits
}
