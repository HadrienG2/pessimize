//! Implementation of Pessimize for alloc::boxed

use crate::{assume_accessed, BorrowPessimize, Pessimize, PessimizeCast};
use std_alloc::boxed::Box;

// Box<T> is effectively a NonNull<T> with RAII, so if the NonNull impl is
// correct, this impl is correct.
#[cfg_attr(feature = "nightly", doc(cfg(feature = "alloc")))]
unsafe impl<T: ?Sized> PessimizeCast for Box<T>
where
    *const T: Pessimize,
{
    type Pessimized = *const T;

    #[inline(always)]
    fn into_pessimize(self) -> *const T {
        Box::into_raw(self) as *const T
    }

    #[inline(always)]
    unsafe fn from_pessimize(x: *const T) -> Self {
        Box::from_raw(x as *mut T)
    }
}
//
#[cfg_attr(feature = "nightly", doc(cfg(feature = "alloc")))]
impl<T: ?Sized> BorrowPessimize for Box<T>
where
    *const T: Pessimize,
{
    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
        let inner: &T = self.as_ref();
        f(&(inner as *const T))
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        // This reborrow would allow us to mutate, which is needed for
        // `assume_accessed` to reliably work.
        let inner: &mut T = self.as_mut();
        let mut inner_ptr = inner as *const T;
        assume_accessed(&mut inner_ptr);
        // Safe because we avoid drop and the pointer effectively wasn't modified
        unsafe { (self as *mut Self).write(Self::from_pessimize(inner_ptr)) }
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