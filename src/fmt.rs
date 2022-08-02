//! Pessimize implementations for core::fmt

use crate::pessimize_zst;
use core::fmt::Error;

// Trivially correct since Error is a ZST
pessimize_zst!(Error, Error, allow(missing_docs));

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
