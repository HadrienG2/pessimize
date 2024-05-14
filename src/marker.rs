//! Implementations of Pessimize for core::marker

use crate::pessimize_zsts;
use core::marker::{PhantomData, PhantomPinned};

pessimize_zsts!(
    allow(missing_docs)
    {
        |T| PhantomData<T> : PhantomData,
        PhantomPinned : PhantomPinned
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_pinned_value, test_value};

    #[test]
    fn phantom_data() {
        test_value::<PhantomData<isize>>(PhantomData);
    }

    #[test]
    fn phantom_pinned() {
        test_pinned_value(PhantomPinned);
    }

    // There are no phantom_data_optim and phantom_pinned_optim tests because
    // Pessimize does not act as an optimization barrier for stateless ZSTs like
    // phantom types.
}
