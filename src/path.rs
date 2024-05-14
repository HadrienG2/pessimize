//! Implementations of Pessimize for std::path

use crate::pessimize_collections;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(all(not(unix), target_os = "wasi"))]
use std::os::wasi::ffi::OsStrExt;
use std::{ffi::OsString, path::PathBuf};

// No need to implement Pessimize for Path because...
//
// 1. Path can only be manipulated by reference
// 2. We already have an impl of Pessimize for all references

// PathBuf is a glorified OsString, so if we can pessimize the former...
pessimize_collections!(
    doc(cfg(all(feature = "std", any(unix, target_os = "wasi"))))
    {
        (OsString, (*const u8, usize, usize)) : (
            PathBuf : (
                |self_: Self| self_.into_os_string(),
                Self::from,
                |self_: &Self| (
                    self_.as_os_str().as_bytes() as *const [u8] as *const u8,
                    self_.as_os_str().as_bytes().len(),
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
    fn path_buf() {
        test_value_type(
            PathBuf::from_str("a").unwrap(),
            PathBuf::from_str("Z").unwrap(),
        );
    }

    #[test]
    #[ignore]
    fn path_buf_optim() {
        test_unoptimized_value_type::<PathBuf>();
    }
}
