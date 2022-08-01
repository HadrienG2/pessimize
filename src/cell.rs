//! Pessimize implementations for core::cell

use crate::{assume_accessed, assume_accessed_imut, consume, hide, Pessimize};
use core::cell::Cell;

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

// TODO: UnsafeCell, tests

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::{assert_unoptimized, test_unoptimized_value_type, test_value_type};

    pub fn test_unoptimized_cell<T: Clone + PartialEq, CellT: Pessimize>(
        mut cell: CellT,
        mut get_mut: impl FnMut(&mut CellT) -> &mut T,
    ) {
        let old_x = get_mut(&mut cell).clone();
        assert_unoptimized(cell, |mut cell| {
            assume_accessed_imut::<&CellT>(&&cell);
            consume(*get_mut(&mut cell) == old_x);
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
}
