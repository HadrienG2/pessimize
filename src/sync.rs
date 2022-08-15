//! Implementations of Pessimize for core::sync

/// Implementations of Pessimize for core::sync::atomic
mod atomic {
    use crate::{assume_accessed, pessimize_cast, BorrowPessimize};
    use core::sync::atomic;

    macro_rules! pessimize_atomics {
        (
            $doc_cfg:meta
            {
                $(
                    $inner:ty : $outer:ident $(<$param:ident>)?
                ),*
            }
        ) => {
            use atomic::{ $($outer),* };
            //
            pessimize_cast!(
                $doc_cfg
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
                #[cfg_attr(feature = "nightly", $doc_cfg)]
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
    pessimize_atomics!(
        doc(cfg(target_has_atomic = "8"))
        {
            bool: AtomicBool,
            i8: AtomicI8,
            u8: AtomicU8
        }
    );
    #[cfg(target_has_atomic = "16")]
    pessimize_atomics!(
        doc(cfg(target_has_atomic = "16"))
        {
            i16: AtomicI16,
            u16: AtomicU16
        }
    );
    #[cfg(target_has_atomic = "32")]
    pessimize_atomics!(
        doc(cfg(target_has_atomic = "32"))
        {
            i32: AtomicI32,
            u32: AtomicU32
        }
    );
    // Need a 64-bit Pessimize impl as well as 64-bit atomics. Taking a little
    // shortcut by taking the pessimistic assumption that you need a 64-bit
    // arch to be able to hold 64-bit integers into registers.
    #[cfg(all(target_has_atomic = "64", target_pointer_width = "64"))]
    pessimize_atomics!(
        doc(cfg(all(target_has_atomic = "64", target_pointer_width = "64")))
        {
            i64: AtomicI64,
            u64: AtomicU64
        }
    );
    #[cfg(target_has_atomic = "ptr")]
    pessimize_atomics!(
        doc(cfg(target_has_atomic = "ptr"))
        {
            isize: AtomicIsize,
            usize: AtomicUsize,
            *mut T: AtomicPtr<T>
        }
    );

    #[allow(non_snake_case)]
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{
            cell::tests::test_unoptimized_cell,
            pessimize_newtypes,
            tests::{test_unoptimized_value_type, test_value, test_value_type},
        };
        use std::sync::atomic::Ordering;

        macro_rules! make_testable_atomics {
            (
                $(
                    ($inner:ty, $outer:ty, $testable:ident)
                ),*
            ) => {
                $(
                    #[derive(Debug, Default)]
                    struct $testable($outer);

                    impl $testable {
                        fn new(inner: $inner) -> Self {
                            Self(<$outer>::new(inner))
                        }

                        fn load(&self) -> $inner {
                            self.0.load(Ordering::Relaxed)
                        }
                    }

                    impl Clone for $testable {
                        fn clone(&self) -> Self {
                            $testable::new(self.load())
                        }
                    }

                    impl PartialEq for $testable {
                        fn eq(&self, other: &Self) -> bool {
                            self.load() == other.load()
                        }
                    }

                    pessimize_newtypes!( allow(missing_docs) { $testable{ $outer } } );
                )*
            };
        }
        //
        macro_rules! test_scalar_atomics {
            (
                $(
                    ($inner:ty, $outer:ty, $testable:ident, $basic_test:ident, $optim_test:ident, $min:expr, $max:expr)
                ),*
            ) => {
                make_testable_atomics!(
                    $(
                        ($inner, $outer, $testable)
                    ),*
                );
                $(
                    #[test]
                    fn $basic_test() {
                        test_value_type($testable::new($min), $testable::new($max));
                    }

                    #[test]
                    #[ignore]
                    fn $optim_test() {
                        test_unoptimized_value_type::<$testable>();
                        test_unoptimized_cell(<$outer>::new(<$inner>::default()), <$outer>::get_mut);
                    }
                )*
            };
        }
        //
        macro_rules! test_int_atomics {
            (
                $(
                    ($inner:ty, $outer:ident, $testable:ident, $basic_test:ident, $optim_test:ident)
                ),*
            ) => {
                test_scalar_atomics!(
                    $(
                        ($inner, $outer, $testable, $basic_test, $optim_test, <$inner>::MIN, <$inner>::MAX)
                    ),*
                );
            };
        }
        //
        #[cfg(target_has_atomic = "8")]
        test_scalar_atomics!((
            bool,
            AtomicBool,
            TestableAtomicBool,
            atomic_bool,
            atomic_bool_optim,
            false,
            true
        ));
        #[cfg(target_has_atomic = "8")]
        test_int_atomics!(
            (i8, AtomicI8, TestableAtomicI8, atomic_i8, atomic_i8_optim),
            (u8, AtomicU8, TestableAtomicU8, atomic_u8, atomic_u8_optim)
        );
        #[cfg(target_has_atomic = "16")]
        test_int_atomics!(
            (
                i16,
                AtomicI16,
                TestableAtomicI16,
                atomic_i16,
                atomic_i16_optim
            ),
            (
                u16,
                AtomicU16,
                TestableAtomicU16,
                atomic_u16,
                atomic_u16_optim
            )
        );
        #[cfg(target_has_atomic = "32")]
        test_int_atomics!(
            (
                i32,
                AtomicI32,
                TestableAtomicI32,
                atomic_i32,
                atomic_i32_optim
            ),
            (
                u32,
                AtomicU32,
                TestableAtomicU32,
                atomic_u32,
                atomic_u32_optim
            )
        );
        #[cfg(all(target_has_atomic = "64", target_pointer_width = "64"))]
        test_int_atomics!(
            (
                i64,
                AtomicI64,
                TestableAtomicI64,
                atomic_i64,
                atomic_i64_optim
            ),
            (
                u64,
                AtomicU64,
                TestableAtomicU64,
                atomic_u64,
                atomic_u64_optim
            )
        );
        #[cfg(target_has_atomic = "ptr")]
        test_int_atomics!(
            (
                isize,
                AtomicIsize,
                TestableAtomicIsize,
                atomic_isize,
                atomic_isize_optim
            ),
            (
                usize,
                AtomicUsize,
                TestableAtomicUsize,
                atomic_usize,
                atomic_usize_optim
            )
        );

        make_testable_atomics!((*mut (), AtomicPtr<()>, TestableAtomicPtr));
        //
        #[test]
        fn atomic_ptr() {
            test_value(TestableAtomicPtr::default());
        }
        //
        #[test]
        #[ignore]
        fn atomic_ptr_optim() {
            test_unoptimized_value_type::<TestableAtomicPtr>();
            test_unoptimized_cell(
                AtomicPtr::<()>::new(core::ptr::null_mut()),
                AtomicPtr::get_mut,
            );
        }
    }
}
