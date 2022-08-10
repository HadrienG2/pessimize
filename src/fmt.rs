//! Pessimize implementations for core::fmt

use crate::pessimize_zsts;
use core::fmt::Error;

pessimize_zsts!(allow(missing_docs) { Error: (Error, false) });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_zst, test_value};

    #[test]
    fn error() {
        test_value(Error);
    }

    #[test]
    #[ignore]
    fn error_optim() {
        test_unoptimized_zst::<Error>();
    }
}
