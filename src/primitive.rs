//! Implementations of Pessimize for primitive types

use crate::{consume, hide, Pessimize};

// Implementation of Pessimize for bool based on that for u8
unsafe impl Pessimize for bool {
    #[allow(clippy::transmute_int_to_bool)]
    #[inline(always)]
    fn hide(self) -> Self {
        // Safe because hide() is guaranteed to be the identity function
        unsafe { core::mem::transmute(hide(self as u8)) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume((*self) as u8)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    // --- Boolean ---

    #[test]
    fn bool() {
        test_value_type::<bool>(false, true);
    }

    #[test]
    #[ignore]
    fn bool_optim() {
        test_unoptimized_value_type::<bool>();
    }
}
