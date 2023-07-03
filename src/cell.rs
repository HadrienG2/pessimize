//! Pessimize implementations for core::cell

use crate::{
    assume_accessed, assume_accessed_imut, consume, hide, pessimize_cast, BorrowPessimize,
    Pessimize,
};
use core::cell::{Cell, UnsafeCell};

// TODO: Once specialization is available, provide a default implementation for
//       all Cell<T> that goes through as_ptr().
// NOTE: Can't use PessimizeCast/BorrowPessimize because need to propagate
//       changes back in `assume_accessed_imut`
unsafe impl<T: Copy + Pessimize> Pessimize for Cell<T> {
    #[inline]
    fn hide(self) -> Self {
        Cell::new(hide(self.into_inner()))
    }

    #[inline]
    fn assume_read(&self) {
        consume::<T>(self.get())
    }

    #[inline]
    fn assume_accessed(&mut self) {
        assume_accessed::<T>(self.get_mut())
    }

    #[inline]
    fn assume_accessed_imut(&self) {
        let inner = self.get();
        assume_accessed_imut::<T>(&inner);
        self.set(inner);
    }
}

pessimize_cast!(
    allow(missing_docs)
    {
        T : (
            |T: (Pessimize)| UnsafeCell<T> : (
                Self::into_inner,
                Self::new
            )
        )
    }
);
//
impl<T: Pessimize> BorrowPessimize for UnsafeCell<T> {
    type BorrowedPessimize = *mut T;

    #[inline]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        // Can't safely get an &T out of an &UnsafeCell<T>, must use by-ptr barrier
        f(&self.get());
    }

    #[inline]
    fn assume_accessed_impl(&mut self) {
        assume_accessed::<T>(self.get_mut())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        pessimize_newtypes,
        tests::{assert_unoptimized, test_unoptimized_value_type, test_value_type},
    };

    pub fn test_unoptimized_cell<T: Clone + PartialEq, CellT: Pessimize>(
        mut cell: CellT,
        mut get_mut: impl FnMut(&mut CellT) -> &mut T,
    ) {
        let old_value = get_mut(&mut cell).clone();
        assert_unoptimized(cell, |mut cell| {
            assume_accessed_imut::<&CellT>(&&cell);
            consume(*get_mut(&mut cell) == old_value);
            cell
        });
    }

    #[test]
    fn cell() {
        test_value_type(Cell::new(isize::MIN), Cell::new(isize::MAX));
    }

    #[test]
    #[ignore]
    fn cell_optim() {
        test_unoptimized_value_type::<Cell<isize>>();
        test_unoptimized_cell(Cell::new(0isize), Cell::get_mut);
    }

    #[derive(Debug, Default)]
    struct TestableUnsafeCell(UnsafeCell<isize>);
    //
    impl TestableUnsafeCell {
        fn new(i: isize) -> Self {
            Self(UnsafeCell::new(i))
        }
    }
    //
    impl Clone for TestableUnsafeCell {
        fn clone(&self) -> Self {
            Self::new(unsafe { *self.0.get() })
        }
    }
    //
    impl PartialEq for TestableUnsafeCell {
        fn eq(&self, other: &Self) -> bool {
            unsafe { *self.0.get() == *other.0.get() }
        }
    }
    //
    pessimize_newtypes!( allow(missing_docs) { TestableUnsafeCell{ UnsafeCell<isize> } } );
    //
    #[test]
    fn unsafe_cell() {
        test_value_type(
            TestableUnsafeCell::new(isize::MIN),
            TestableUnsafeCell::new(isize::MAX),
        );
    }
    //
    #[test]
    #[ignore]
    fn unsafe_cell_optim() {
        test_unoptimized_value_type::<TestableUnsafeCell>();
        test_unoptimized_cell(UnsafeCell::new(0isize), UnsafeCell::get_mut);
    }
}
