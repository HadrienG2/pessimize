//! Pessimize implementations for std::fs

use crate::{assume_accessed, BorrowPessimize, PessimizeCast};
#[cfg(any(unix, windows))]
use std::fs::File;
#[cfg(unix)]
use std::fs::Permissions;
#[cfg(unix)]
use std::os::unix::{
    fs::PermissionsExt,
    io::{AsRawFd, FromRawFd, IntoRawFd, RawFd},
};
#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle};

macro_rules! pessimize_file {
    ($fd:ty, $as_fd:ident, $into_fd:ident, $from_fd:ident) => {
        #[cfg_attr(
            feature = "nightly",
            doc(cfg(all(feature = "std", any(unix, windows))))
        )]
        unsafe impl PessimizeCast for File {
            type Pessimized = $fd;

            #[inline(always)]
            fn into_pessimize(self) -> RawFd {
                self.$into_fd()
            }

            #[inline(always)]
            unsafe fn from_pessimize(x: RawFd) -> Self {
                Self::$from_fd(x)
            }
        }
        //
        #[cfg_attr(
            feature = "nightly",
            doc(cfg(all(feature = "std", any(unix, windows))))
        )]
        impl BorrowPessimize for File {
            type BorrowedPessimize = $fd;

            #[inline(always)]
            fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
                let fd = self.$as_fd();
                f(&fd)
            }

            #[inline(always)]
            fn assume_accessed_impl(&mut self) {
                let mut fd = self.$as_fd();
                assume_accessed::<$fd>(&mut fd);
                unsafe { (self as *mut Self).write(Self::from_pessimize(fd)) }
            }
        }
    };
}
//
#[cfg(unix)]
pessimize_file!(RawFd, as_raw_fd, into_raw_fd, from_raw_fd);
//
#[cfg(windows)]
pessimize_file!(RawHandle, as_raw_handle, into_raw_handle, from_raw_handle);

#[cfg(unix)]
mod unix {
    use super::*;

    #[cfg_attr(feature = "nightly", doc(cfg(all(feature = "std", unix))))]
    unsafe impl PessimizeCast for Permissions {
        type Pessimized = u32;

        #[inline(always)]
        fn into_pessimize(self) -> u32 {
            self.mode()
        }

        #[inline(always)]
        unsafe fn from_pessimize(x: u32) -> Self {
            Self::from_mode(x)
        }
    }
    //
    #[cfg_attr(feature = "nightly", doc(cfg(all(feature = "std", unix))))]
    impl BorrowPessimize for Permissions {
        type BorrowedPessimize = u32;

        #[inline(always)]
        fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
            let mode = self.mode();
            f(&mode)
        }

        #[inline(always)]
        fn assume_accessed_impl(&mut self) {
            let mut mode = self.mode();
            assume_accessed::<u32>(&mut mode);
            self.set_mode(mode);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value, test_value};

    // FIXME: Can't test File as it doesn't implement the right traits

    fn make_permissions() -> Permissions {
        std::fs::metadata(file!()).unwrap().permissions()
    }

    #[test]
    fn permissions() {
        test_value(make_permissions());
    }

    #[test]
    #[ignore]
    fn permissions_optim() {
        test_unoptimized_value(make_permissions());
    }
}
