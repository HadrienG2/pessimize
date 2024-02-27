//! Implementations of Pessimize for std::vec

use crate::pessimize_collections;
use core::mem::ManuallyDrop;
#[cfg(not(any(feature = "std", test)))]
use std_alloc::vec::Vec;

// Vec<T> is basically a thin NonNull<T>, a length and a capacity
pessimize_collections!(
    doc(cfg(feature = "alloc"))
    {
        ((*mut T, usize, usize), (*const T, usize, usize)) : (
            |T| Vec<T> : (
                |self_: Self| {
                    let mut v = ManuallyDrop::new(self_);
                    (v.as_mut_ptr(), v.len(), v.capacity())
                },
                |(ptr, length, capacity)| {
                    Self::from_raw_parts(ptr, length, capacity)
                },
                |self_: &Self| {
                    (self_.as_ptr(), self_.len(), self_.capacity())
                }
            )
        )
    }
);

#[cfg(test)]
mod tests {
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    #[test]
    fn string() {
        test_value_type(vec![isize::MIN], vec![isize::MAX]);
    }

    #[test]
    #[ignore]
    fn string_optim() {
        test_unoptimized_value_type::<Vec<isize>>();
    }
}
