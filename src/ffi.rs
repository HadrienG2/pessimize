//! Implementations of Pessimize for core::ffi

use crate::{impl_assume_accessed, BorrowPessimize, PessimizeCast};
use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(all(not(unix), target_os = "wasi"))]
use std::os::wasi::ffi::{OsStrExt, OsStringExt};

// NOTE: No need to implement Pessimize for OsStr because...
//       1/OsStr can only be manipulated by reference
//       2/We already have an impl of Pessimize for all references

// Trivially correct, almost an offload to Vec<u8>: Pessimize implementation
#[cfg_attr(
    feature = "nightly",
    doc(cfg(all(feature = "std", any(unix, target_os = "wasi"))))
)]
unsafe impl PessimizeCast for OsString {
    type Pessimized = (*const u8, usize, usize);

    #[inline(always)]
    fn into_pessimize(self) -> Self::Pessimized {
        self.into_vec().into_pessimize()
    }

    #[inline(always)]
    unsafe fn from_pessimize(x: Self::Pessimized) -> Self {
        Self::from_vec(Vec::<u8>::from_pessimize(x))
    }
}
//
impl BorrowPessimize for OsString {
    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
        let pessimized = (
            self.as_bytes() as *const [u8] as *const u8,
            self.as_bytes().len(),
            self.capacity(),
        );
        f(&pessimized)
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        impl_assume_accessed(self, core::mem::take)
    }
}

#[cfg(test)]
pub(crate) mod tests {
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
