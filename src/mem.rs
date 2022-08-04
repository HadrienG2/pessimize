//! Implementations of Pessimize for core::mem

use crate::{assume_accessed, BorrowPessimize, Pessimize, PessimizeCast};
use core::mem::ManuallyDrop;

unsafe impl<T: Pessimize> PessimizeCast for ManuallyDrop<T> {
    type Pessimized = T;

    #[inline(always)]
    fn into_pessimize(self) -> T {
        Self::into_inner(self)
    }

    #[inline(always)]
    unsafe fn from_pessimize(inner: T) -> Self {
        Self::new(inner)
    }
}
//
impl<T: Pessimize> BorrowPessimize for ManuallyDrop<T> {
    type BorrowedPessimize = T;

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&T)) {
        f(self)
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        assume_accessed::<T>(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    #[test]
    fn manually_drop() {
        test_value_type(ManuallyDrop::new(isize::MIN), ManuallyDrop::new(isize::MAX));
    }

    #[test]
    #[ignore]
    fn manually_drop_optim() {
        test_unoptimized_value_type::<ManuallyDrop<isize>>();
    }
}
