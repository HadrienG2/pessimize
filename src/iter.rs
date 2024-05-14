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

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::{pessimize_newtypes, tests::test_value};

    #[derive(Clone, Debug, Default)]
    struct TestableEmpty(Empty<u8>);
    //
    impl PartialEq for TestableEmpty {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    //
    pessimize_newtypes!( allow(missing_docs) { TestableEmpty{ Empty<u8> } } );
    //
    #[test]
    fn empty() {
        test_value(TestableEmpty::default());
    }
    //
    // There is no empty_optim test because Pessimize does not act as an
    // optimization barrier for stateless ZSTs like Empty.

    // Can't test Once and Repeat because there is no reliable way to extract
    // the inner value from &self, which would be needed to implement PartialEq
    // via a newtype.
}
