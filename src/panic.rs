//! Implementations of Pessimize for core::panic

use crate::{pessimize_newtypes, Pessimize};
use core::panic::AssertUnwindSafe;

pessimize_newtypes!(
    allow(missing_docs)
    {
        |T: (Pessimize)| AssertUnwindSafe<T>{ T }
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pessimize_newtypes,
        tests::{test_unoptimized_value_type, test_value_type},
    };

    #[derive(Debug, Default)]
    struct TestableAssertUnwindSafe(AssertUnwindSafe<isize>);
    //
    impl TestableAssertUnwindSafe {
        fn new(inner: isize) -> Self {
            Self(AssertUnwindSafe(inner))
        }
    }
    //
    impl Clone for TestableAssertUnwindSafe {
        fn clone(&self) -> Self {
            Self::new((self.0).0)
        }
    }
    //
    impl PartialEq for TestableAssertUnwindSafe {
        fn eq(&self, other: &Self) -> bool {
            (self.0).0 == (other.0).0
        }
    }
    //
    pessimize_newtypes!( allow(missing_docs) { TestableAssertUnwindSafe{ AssertUnwindSafe<isize> } } );
    //
    #[test]
    fn assert_unwind_safe() {
        test_value_type(
            TestableAssertUnwindSafe::new(isize::MIN),
            TestableAssertUnwindSafe::new(isize::MAX),
        );
    }
    //
    #[test]
    #[ignore]
    fn assert_unwind_safe_optim() {
        test_unoptimized_value_type::<TestableAssertUnwindSafe>();
    }
}
