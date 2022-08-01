//! Pessimize implementations for core::fmt

use crate::{impl_assume_accessed, impl_with_pessimize, BorrowPessimize, PessimizeCast};
use core::fmt::Error;

// Trivially correct since Error is a ZST
unsafe impl PessimizeCast for Error {
    type Pessimized = ();

    #[inline(always)]
    fn into_pessimize(self) -> () {
        ()
    }

    #[inline(always)]
    unsafe fn from_pessimize((): ()) -> Self {
        Self
    }
}
//
impl BorrowPessimize for Error {
    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
        impl_with_pessimize(self, f)
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        impl_assume_accessed(self, |r| *r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_value;

    #[test]
    fn error() {
        test_value(Error);
    }

    // NOTE: hide() does not act as an optimization barrier for ZSTs like Error
}
