//! Pessimize implementations for core::iter

use crate::{pessimize_once_like, pessimize_zsts, Pessimize};
use core::iter::{Empty, Once, Repeat};

pessimize_zsts!(
    allow(missing_docs)
    {
        |T| Empty<T>: core::iter::empty()
    }
);

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
