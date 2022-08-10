//! Implementations of Pessimize for core::marker

use crate::pessimize_zsts;
use core::marker::{PhantomData, PhantomPinned};

pessimize_zsts!(
    allow(missing_docs)
    {
        |T| PhantomData<T> : (PhantomData, false),
        PhantomPinned : (PhantomPinned, false)
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_pinned_value, test_unoptimized_zst, test_value};

    #[test]
    fn phantom_data() {
        test_value::<PhantomData<isize>>(PhantomData);
    }

    #[test]
    #[ignore]
    fn phantom_data_optim() {
        test_unoptimized_zst::<PhantomData<isize>>();
    }

    #[test]
    fn phantom_pinned() {
        test_pinned_value(PhantomPinned);
    }

    #[test]
    #[ignore]
    fn phantom_pinned_optim() {
        test_unoptimized_zst::<PhantomPinned>();
    }
}
