//! Pessimize implementations for core::iter

use crate::{assume_accessed, assume_accessed_imut, consume, pessimize_once_like, Pessimize};
use core::iter::{Empty, Once, Repeat};

// As a zero-sized type, Empty is trivially pessimizable
unsafe impl<T> Pessimize for Empty<T> {
    #[inline(always)]
    fn hide(self) -> Self {
        core::iter::empty()
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume::<&Self>(self);
    }

    #[inline(always)]
    fn assume_accessed(mut self: &mut Self) {
        assume_accessed::<&mut Self>(&mut self);
        *self = core::iter::empty()
    }

    #[inline(always)]
    fn assume_accessed_imut(&self) {
        assume_accessed_imut::<&Self>(&self);
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
