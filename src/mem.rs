//! Implementations of Pessimize for core::mem

use crate::{assume_accessed, pessimize_cast, BorrowPessimize, Pessimize};
use core::mem::ManuallyDrop;

pessimize_cast!(
    allow(missing_docs)
    {
        T : (
            |T: (Pessimize)| ManuallyDrop<T> : (
                Self::into_inner,
                Self::new
            )
        )
    }
);
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
