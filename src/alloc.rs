//! Pessimize implementations for core::alloc

use crate::pessimize_copy;
use core::alloc::Layout;

pessimize_copy!(
    allow(missing_docs)
    {
        (usize, usize): (
            Layout: (
                |self_: Layout| (self_.size(), self_.align()),
                |(size, align)| Self::from_size_align_unchecked(size, align)
            )
        )
    }
);

#[cfg(any(feature = "std", test))]
mod std_feature {
    use crate::{assume_globals_accessed, Pessimize};
    use std::alloc::System;

    // NOTE: Need a manual Pessimize implementation due to use of global state
    #[cfg_attr(feature = "nightly", doc(cfg(feature = "std")))]
    unsafe impl Pessimize for System {
        #[inline(always)]
        fn hide(self) -> Self {
            assume_globals_accessed();
            self
        }

        #[inline(always)]
        fn assume_read(&self) {}

        #[inline(always)]
        fn assume_accessed(&mut self) {
            assume_globals_accessed();
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            assume_globals_accessed();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pessimize_newtypes,
        tests::{test_unoptimized_stateful_zst, test_unoptimized_value, test_value},
    };
    use std::alloc::System;

    #[test]
    fn layout() {
        test_value(Layout::from_size_align(0, 1).unwrap());
        let last_usize_pow2 = usize::MAX - (usize::MAX >> 1);
        let prev_usize_pow2 = last_usize_pow2 >> 1;
        test_value(Layout::from_size_align(prev_usize_pow2, prev_usize_pow2).unwrap());
    }
    //
    #[test]
    #[ignore]
    fn layout_optim() {
        test_unoptimized_value(Layout::from_size_align(0, 1).unwrap());
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct TestableSystem(System);
    //
    impl PartialEq for TestableSystem {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    //
    pessimize_newtypes!( allow(missing_docs) { TestableSystem{ System } } );
    //
    #[test]
    fn system() {
        test_value(TestableSystem(System));
    }
    //
    #[test]
    #[ignore]
    fn system_optim() {
        test_unoptimized_stateful_zst::<System>();
    }
}
