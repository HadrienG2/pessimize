//! Implementation of Pessimize for alloc::boxed

use crate::{assume_accessed, assume_globals_accessed, BorrowPessimize, Pessimize, PessimizeCast};
use std_alloc::boxed::Box;

#[cfg_attr(feature = "nightly", doc(cfg(feature = "alloc")))]
unsafe impl<T: ?Sized> PessimizeCast for Box<T>
where
    *const T: Pessimize,
{
    type Pessimized = *mut T;

    #[inline(always)]
    fn into_pessimize(self) -> *mut T {
        Box::into_raw(self)
    }

    #[inline(always)]
    unsafe fn from_pessimize(x: *mut T) -> Self {
        assume_globals_accessed();
        Box::from_raw(x)
    }
}
//
#[cfg_attr(feature = "nightly", doc(cfg(feature = "alloc")))]
impl<T: ?Sized> BorrowPessimize for Box<T>
where
    *const T: Pessimize,
{
    type BorrowedPessimize = *const T;

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        let inner: &T = self.as_ref();
        f(&(inner as *const T))
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        // This reborrow would allow us to mutate, which is needed for
        // `assume_accessed` to reliably work.
        let mut inner: &mut T = self.as_mut();
        assume_accessed::<&mut T>(&mut inner);

        // With an &mut Box, one can trigger an allocation, and thus mutate
        // global allocator state
        assume_globals_accessed();

        // Safe because we avoid drop and the pointer effectively wasn't modified
        let inner_ptr = inner as *mut T;
        unsafe { (self as *mut Self).write(Box::from_raw(inner_ptr)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    // --- Boxed value ---

    #[test]
    fn thin_box() {
        test_value_type(Box::new(isize::MIN), Box::new(isize::MAX));
    }

    #[test]
    #[ignore]
    fn thin_box_optim() {
        test_unoptimized_value_type::<Box<isize>>();
    }

    // --- Fat box and fat pointer types ---

    #[cfg(feature = "nightly")]
    mod dst {
        use super::*;
        use crate::{
            ptr::tests::{test_all_pointers, test_unoptimized_ptrs},
            tests::{test_unoptimized_value, BIG},
        };

        fn make_boxed_slice(value: isize) -> Box<[isize]> {
            vec![value; BIG].into()
        }

        #[test]
        fn boxed_slice() {
            // Test boxed slices as owned values
            test_value_type(make_boxed_slice(isize::MIN), make_boxed_slice(isize::MAX));

            // Test slice pointers, using boxed slices as owned storage
            for inner in [isize::MIN, 0, isize::MAX] {
                test_all_pointers::<[isize], _>(make_boxed_slice(inner))
            }
        }

        #[test]
        #[ignore]
        fn boxed_slice_optim() {
            test_unoptimized_value(make_boxed_slice(0));
            test_unoptimized_ptrs::<[isize], _>(make_boxed_slice(0));
        }

        fn make_boxed_str(c: char) -> Box<str> {
            String::from_iter([c].into_iter()).into()
        }

        #[test]
        fn boxed_str() {
            // Test boxed slices as owned values
            test_value_type(make_boxed_str('\0'), make_boxed_str(char::MAX));

            // Test slice pointers, using boxed slices as owned storage
            for inner in ['\0', char::MAX] {
                test_all_pointers::<str, _>(make_boxed_str(inner))
            }
        }

        #[test]
        #[ignore]
        fn boxed_str_optim() {
            test_unoptimized_value(make_boxed_str('\0'));
            test_unoptimized_ptrs::<str, _>(make_boxed_str('\0'));
        }

        // FIXME: Box<dyn Trait> cannot easily be tested because we need
        //        PartialEq for testing and dyn PartialEq is illegal. Find a way
        //        around this, e.g. a trait implemented for PartialEq types
    }
}
