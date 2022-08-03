//! Pessimize implementations for core::cell

use crate::{
    assume_accessed, assume_accessed_imut, consume, hide, BorrowPessimize, Pessimize, PessimizeCast,
};
use core::cell::{Cell, UnsafeCell};

// TODO: Once specialization is available, provide a default implementation for
//       all Cell<T> that goes through as_ptr().
unsafe impl<T: Copy + Pessimize> Pessimize for Cell<T> {
    #[inline(always)]
    fn hide(self) -> Self {
        Cell::new(hide(self.into_inner()))
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume::<T>(self.get())
    }

    #[inline(always)]
    fn assume_accessed(&mut self) {
        assume_accessed::<T>(self.get_mut())
    }

    #[inline(always)]
    fn assume_accessed_imut(&self) {
        let inner = self.get();
        assume_accessed_imut::<T>(&inner);
        self.set(inner);
    }
}

unsafe impl<T: Pessimize> PessimizeCast for UnsafeCell<T> {
    type Pessimized = T;

    #[inline(always)]
    fn into_pessimize(self) -> T {
        self.into_inner()
    }

    #[inline(always)]
    unsafe fn from_pessimize(x: T) -> Self {
        Self::new(x)
    }
}
//
impl<T: Pessimize> BorrowPessimize for UnsafeCell<T> {
    type BorrowedPessimize = *mut T;

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        f(&self.get());
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        assume_accessed::<T>(self.get_mut())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::{assert_unoptimized, test_unoptimized_value_type, test_value_type};

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

    // FIXME: Can't test UnsafeCell<T> as a value type since it doesn't impl the
    //        required traits. Whatever solution is chosen will likely also
    //        enable testing of Box<dyn Trait>.
    #[test]
    #[ignore]
    fn unsafe_cell_optim() {
        // FIXME: Add a test_unoptimized_value_type
        test_unoptimized_cell(UnsafeCell::new(0isize), UnsafeCell::get_mut);
    }
}
