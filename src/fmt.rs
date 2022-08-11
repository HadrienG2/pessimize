//! Pessimize implementations for core::fmt

use crate::pessimize_zsts;
use core::fmt::Error;

pessimize_zsts!(allow(missing_docs) { Error: Error });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_value;

    #[test]
    fn error() {
        test_value(Error);
    }

    // NOTE: There is no error_optim test because Pessimize does not act as an
    //       optimization barrier for stateless ZSTs like fmt::Error.
}
