//! Implementations of Pessimize for core::sync

/// Implementations of Pessimize for core::sync::atomic
mod atomic {
    use crate::{assume_accessed, pessimize_cast, BorrowPessimize};
    use core::sync::atomic;

    macro_rules! pessimize_atomics {
        (
            $(
                $inner:ty : $outer:ident $(<$param:ident>)?
            ),*
        ) => {
            use atomic::{ $($outer),* };
            //
            pessimize_cast!(
                allow(missing_docs)
                {
                    $(
                        $inner : (
                            $(|$param|)? $outer $(<$param>)? : (Self::into_inner, Self::new)
                        )
                    ),*
                }
            );
            //
            $(
                impl $(<$param>)? BorrowPessimize for $outer $(<$param>)? {
                    type BorrowedPessimize = *const Self;

                    #[inline(always)]
                    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
                        // While it is possible to extract an inner value from
                        // &self, doing so is not something that LLVM can be
                        // reliably expected to optimize out at the moment, so
                        // must pessimize by pointer.
                        f(&(self as *const Self))
                    }

                    #[inline(always)]
                    fn assume_accessed_impl(&mut self) {
                        assume_accessed::<$inner>(self.get_mut())
                    }
                }
            )*
        };
    }
    //
    #[cfg(target_has_atomic = "8")]
    pessimize_atomics!(bool: AtomicBool, i8: AtomicI8, u8: AtomicU8);
    #[cfg(target_has_atomic = "16")]
    pessimize_atomics!(i16: AtomicI16, u16: AtomicU16);
    #[cfg(target_has_atomic = "32")]
    pessimize_atomics!(i32: AtomicI32, u32: AtomicU32);
    #[cfg(target_has_atomic = "64")]
    pessimize_atomics!(i64: AtomicI64, u64: AtomicU64);
    #[cfg(target_has_atomic = "ptr")]
    pessimize_atomics!(isize: AtomicIsize, usize: AtomicUsize, *mut T: AtomicPtr<T>);

    #[allow(non_snake_case)]
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cell::tests::test_unoptimized_cell;

        macro_rules! test_int_atomics {
            (
                $(
                    ($inner:ty, $outer:ident, $optim_test:ident)
                ),*
            ) => {
                $(
                    // FIMXE: Can't test AtomicXyz yet as it doesn't implement the right traits
                    /* #[test]
                    fn $outer() {
                        test_value_type($outer::new($inner::MIN), $outer::new($inner::MAX));
                    } */

                    #[test]
                    #[ignore]
                    fn $optim_test() {
                        // FIMXE: Can't test AtomicXyz yet as it doesn't implement the right traits
                        // test_unoptimized_value_type::<$outer>();
                        test_unoptimized_cell($outer::new(<$inner>::default()), $outer::get_mut);
                    }
                )*
            };
        }
        //
        #[cfg(target_has_atomic = "8")]
        test_int_atomics!(
            (bool, AtomicBool, AtomicBool_optim),
            (i8, AtomicI8, AtomicI8_optim),
            (u8, AtomicU8, AtomicU8_optim)
        );
        #[cfg(target_has_atomic = "16")]
        test_int_atomics!(
            (i16, AtomicI16, AtomicI16_optim),
            (u16, AtomicU16, AtomicU16_optim)
        );
        #[cfg(target_has_atomic = "32")]
        test_int_atomics!(
            (i32, AtomicI32, AtomicI32_optim),
            (u32, AtomicU32, AtomicU32_optim)
        );
        #[cfg(target_has_atomic = "64")]
        test_int_atomics!(
            (i64, AtomicI64, AtomicI64_optim),
            (u64, AtomicU64, AtomicU64_optim)
        );
        #[cfg(target_has_atomic = "ptr")]
        test_int_atomics!(
            (isize, AtomicIsize, AtomicIsize_optim),
            (usize, AtomicUsize, AtomicUsize_optim)
        );

        // FIMXE: Can't test AtomicPtr yet as it doesn't implement the right traits
        /* #[test]
        fn AtomicPtr() {
            test_value_type(AtomicPtr::<()>::new(core::ptr::null_mut()));
        } */

        #[test]
        #[ignore]
        fn AtomicPtr_optim() {
            // FIMXE: Can't test AtomicBool yet as it doesn't implement the right traits
            // test_unoptimized_value_type::<AtomicPtr<()>>();
            test_unoptimized_cell(
                AtomicPtr::<()>::new(core::ptr::null_mut()),
                AtomicPtr::get_mut,
            );
        }
    }
}
