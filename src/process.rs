//! Implementations of Pessimize for std::process

use crate::{assume_accessed, pessimize_cast, pessimize_copy, BorrowPessimize};
use std::{
    os::unix::process::ExitStatusExt,
    process::{ExitStatus, Output},
};

pessimize_copy!(
    doc(cfg(all(any(feature = "std", test), unix)))
    {
        i32 : (
            ExitStatus : (Self::into_raw, Self::from_raw)
        )
    }
);

pessimize_cast!(
    doc(cfg(all(any(feature = "std", test), unix)))
    {
        (ExitStatus, Vec<u8>, Vec<u8>) : (
            Output : (
                |Self { status, stdout, stderr }| (status, stdout, stderr),
                |(status, stdout, stderr)| Self { status, stdout, stderr }
            )
        )
    }
);
//
#[cfg_attr(feature = "nightly", doc(cfg(all(any(feature = "std", test), unix))))]
impl BorrowPessimize for Output {
    type BorrowedPessimize = (ExitStatus, *const u8, usize, usize, *const u8, usize, usize);

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        f(&(
            self.status,
            self.stdout.as_ptr(),
            self.stdout.len(),
            self.stdout.capacity(),
            self.stderr.as_ptr(),
            self.stderr.len(),
            self.stderr.capacity(),
        ))
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        assume_accessed(&mut self.status);
        assume_accessed(&mut self.stdout);
        assume_accessed(&mut self.stderr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value, test_value};

    #[test]
    fn exit_status() {
        test_value(ExitStatus::from_raw(i32::MIN));
        test_value(ExitStatus::from_raw(0));
        test_value(ExitStatus::from_raw(i32::MAX));
    }

    #[test]
    #[ignore]
    fn exit_status_optim() {
        test_unoptimized_value(ExitStatus::from_raw(0));
    }

    #[test]
    fn output() {
        test_value(Output {
            status: ExitStatus::from_raw(i32::MIN),
            stdout: vec![1],
            stderr: vec![2],
        });
        test_value(Output {
            status: ExitStatus::from_raw(0),
            stdout: vec![12],
            stderr: vec![34],
        });
        test_value(Output {
            status: ExitStatus::from_raw(i32::MAX),
            stdout: vec![],
            stderr: vec![u8::MAX],
        });
    }

    #[test]
    #[ignore]
    fn output_optim() {
        test_unoptimized_value(Output {
            status: ExitStatus::from_raw(0),
            stdout: vec![],
            stderr: vec![],
        });
    }
}
