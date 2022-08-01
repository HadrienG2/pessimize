//! Pessimize implementations for core::alloc

use crate::{impl_assume_accessed, impl_with_pessimize, BorrowPessimize, PessimizeCast};
use core::alloc::Layout;

// Correct because Pessimize operations do nothing
unsafe impl PessimizeCast for Layout {
    type Pessimized = (usize, usize);

    #[inline(always)]
    fn into_pessimize(self) -> (usize, usize) {
        (self.size(), self.align())
    }

    #[inline(always)]
    unsafe fn from_pessimize((size, align): (usize, usize)) -> Self {
        Self::from_size_align_unchecked(size, align)
    }
}
//
impl BorrowPessimize for Layout {
    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
        impl_with_pessimize(self, f)
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        impl_assume_accessed(self, |r| *r)
    }
}

// NOTE: Not implementing Pessimize for std::alloc::System since 1/it is not
//       easily available to no_std crates and 2/as a zero-sized type, it does
//       not optimize out nicely anyway.

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
}
