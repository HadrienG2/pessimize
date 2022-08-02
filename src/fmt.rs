//! Pessimize implementations for core::fmt

use crate::pessimize_zsts;
use core::fmt::Error;

// Trivially correct since Error is a ZST
pessimize_zsts!(allow(missing_docs) { Error: Error });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_value;

    #[test]
    fn error() {
        test_value(Error);
    }

    // NOTE: hide() does not act as an optimization barrier for ZSTs like Error
}
