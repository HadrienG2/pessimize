//! Pessimize implementations for std::fs

use crate::pessimize_extractible;
use std::fs::File;
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
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
    use std::{fs::Permissions, os::unix::fs::PermissionsExt};

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
    use crate::{
        pessimize_newtypes,
        tests::{test_unoptimized_value, test_value},
    };
    use std::mem::ManuallyDrop;

    #[derive(Debug)]
    struct TestableFile(ManuallyDrop<File>);
    //
    impl TestableFile {
        fn new() -> Self {
            Self(ManuallyDrop::new(tempfile::tempfile().unwrap()))
        }
    }
    //
    impl Clone for TestableFile {
        fn clone(&self) -> Self {
            // This is safe because ManuallyDrop avoids double closure
            #[cfg(unix)]
            unsafe {
                Self(ManuallyDrop::new(File::from_raw_fd(self.0.as_raw_fd())))
            }
            #[cfg(windows)]
            unsafe {
                Self(ManuallyDrop::new(File::from_raw_handle(
                    self.0.as_raw_handle(),
                )))
            }
        }
    }
    //
    impl PartialEq for TestableFile {
        fn eq(&self, other: &Self) -> bool {
            // Going for simple fd equality as there's no way even a defective
            // Pessimize impl for File would perform filesystem IO.
            #[cfg(unix)]
            {
                self.0.as_raw_fd() == other.0.as_raw_fd()
            }
            #[cfg(windows)]
            {
                self.0.as_raw_handle() == other.0.as_raw_handle()
            }
        }
    }
    //
    pessimize_newtypes!( allow(missing_docs) { TestableFile{ ManuallyDrop<File> } } );
    //
    #[test]
    fn file() {
        test_value(TestableFile::new())
    }
    //
    #[test]
    #[ignore]
    fn file_optim() {
        test_unoptimized_value(TestableFile::new())
    }

    #[cfg(unix)]
    mod unix {
        use super::*;
        use std::fs::Permissions;

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
}
