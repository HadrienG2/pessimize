//! Pessimize implementations for core::iter

use crate::{
    impl_assume_accessed_via_extract_pessimized, pessimize_once_like, BorrowPessimize, Pessimize,
    PessimizeCast,
};
use core::iter::{Empty, Once, Repeat};

// As a zero-sized type, Empty is trivially pessimizable
unsafe impl<T> PessimizeCast for Empty<T> {
    type Pessimized = ();

    #[inline(always)]
    fn into_pessimize(self) {}

    #[inline(always)]
    unsafe fn from_pessimize((): ()) -> Self {
        core::iter::empty()
    }
}
//
impl<T> BorrowPessimize for Empty<T> {
    type BorrowedPessimize = *const Self;

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        f(&(self as *const Self))
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        impl_assume_accessed_via_extract_pessimized(self, |_self| ())
    }
}

// std::io::repeat behaves like std::iter::once
pessimize_once_like!(
    allow(missing_docs)
    {
        |T: (Pessimize)| Once<T>: (
            T,
            |self_: &mut Self| self_.next().unwrap(),
            core::iter::once
        ),
        |T: (Clone, Pessimize)| Repeat<T>: (
            T,
            |self_: &mut Self| self_.next().unwrap(),
            core::iter::repeat
        )
    }
);

// FIXME: Can't test iterators as they don't implement the right traits
