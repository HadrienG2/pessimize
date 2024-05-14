//! Implementations of Pessimize for core::ffi

use crate::pessimize_collections;
use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(all(not(unix), target_os = "wasi"))]
use std::os::wasi::ffi::{OsStrExt, OsStringExt};

// No need to implement Pessimize for OsStr because...
//
// 1. OsStr can only be manipulated by reference
// 2. We already have an impl of Pessimize for all references

// On Unix and WASI, OsString is a glorified Vec<u8>
pessimize_collections!(
    doc(cfg(all(feature = "std", any(unix, target_os = "wasi"))))
    {
        (Vec<u8>, (*const u8, usize, usize)) : (
            OsString : (
                Self::into_vec,
                Self::from_vec,
                |self_: &Self| (
                    self_.as_bytes() as *const [u8] as *const u8,
                    self_.as_bytes().len(),
                    self_.capacity()
                )
            )
        )
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value_type, test_value_type};
    use std::str::FromStr;

    #[test]
    fn os_string() {
        test_value_type(
            OsString::from_str("a").unwrap(),
            OsString::from_str("Z").unwrap(),
        );
    }

    #[test]
    #[ignore]
    fn os_string_optim() {
        test_unoptimized_value_type::<OsString>();
    }
}
