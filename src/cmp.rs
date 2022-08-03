//! Pessimize implementations for core::cmp

use crate::{assume_accessed, BorrowPessimize, Pessimize, PessimizeCast};
use core::cmp::Reverse;

// Trivially correct
unsafe impl<T: Pessimize> PessimizeCast for Reverse<T> {
    type Pessimized = T;

    #[inline(always)]
    fn into_pessimize(self) -> T {
        self.0
    }

    #[inline(always)]
    unsafe fn from_pessimize(x: T) -> Self {
        Self(x)
    }
}
//
impl<T: Pessimize> BorrowPessimize for Reverse<T> {
    type BorrowedPessimize = T;

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&T)) {
        f(&self.0)
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        assume_accessed::<T>(&mut self.0)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    #[test]
    fn cell() {
        test_value_type(Reverse(isize::MIN), Reverse(isize::MAX));
    }

    #[test]
    #[ignore]
    fn cell_optim() {
        test_unoptimized_value_type::<Reverse<isize>>();
    }
}
