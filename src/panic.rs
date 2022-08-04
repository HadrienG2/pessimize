//! Implementations of Pessimize for core::panic

use crate::{pessimize_newtypes, Pessimize};
use core::panic::AssertUnwindSafe;

pessimize_newtypes!(
    allow(missing_docs)
    {
        |T: (Pessimize)| AssertUnwindSafe<T>{ T }
    }
);

// FIXME: Can't test AssertUnwindSafe as it doesn't implement the right traits
