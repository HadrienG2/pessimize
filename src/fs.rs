//! Pessimize implementations for std::fs

use crate::pessimize_extractible;
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
    ($fd:ty, $as_fd:path, $into_fd:path, $from_fd:path) => {
        pessimize_extractible!(
            doc(cfg(all(feature = "std", any(unix, windows))))
            {
                $fd : (
                    File : ($into_fd, $from_fd, $as_fd)
                )
            }
        );
    };
}
//
#[cfg(unix)]
pessimize_file!(
    RawFd,
    AsRawFd::as_raw_fd,
    IntoRawFd::into_raw_fd,
    FromRawFd::from_raw_fd
);
//
#[cfg(windows)]
pessimize_file!(
    RawHandle,
    AsRawHandle::as_raw_handle,
    IntoRawHandle::into_raw_handle,
    FromRawHandle::from_raw_handle
);

#[cfg(unix)]
mod unix {
    use super::*;

    pessimize_extractible!(
        doc(cfg(all(feature = "std", unix)))
        {
            u32 : (
                Permissions : (|self_: Self| self_.mode(), PermissionsExt::from_mode, PermissionsExt::mode)
            )
        }
    );
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
