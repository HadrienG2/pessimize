//! Implementations of Pessimize for core::pin

use crate::{assume_accessed, BorrowPessimize, Pessimize, PessimizeCast};
use core::{
    marker::Unpin,
    ops::{Deref, DerefMut},
    pin::Pin,
};

// We can only support `Unpin` types because `PessimizeCast::into_pessimize` is
// a safe function, and with `!Unpin` types this safe function would allow
// people to violate Pin's contract by moving data out of the Pin's target.
//
// NOTE: Can't use pessimize_cast! because making it support ?Sized is too hard
//
unsafe impl<T: ?Sized + Unpin, P: Deref<Target = T> + Pessimize> PessimizeCast for Pin<P> {
    type Pessimized = P;

    #[inline]
    fn into_pessimize(self) -> P {
        Self::into_inner(self)
    }

    #[inline]
    unsafe fn from_pessimize(p: P) -> Self {
        Self::new(p)
    }
}
//
impl<T: ?Sized + Unpin, P: Deref<Target = T> + DerefMut + Pessimize> BorrowPessimize for Pin<P>
where
    *const T: Pessimize,
{
    type BorrowedPessimize = *const T;

    #[inline]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        f(&(self.deref() as *const T))
    }

    #[inline]
    fn assume_accessed_impl(&mut self) {
        assume_accessed(&mut (self.deref_mut() as *const T))
    }
}

// NOTE: Pin is tested in the ptr module, alongside other pointer types
