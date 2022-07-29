//! Implementations of Pessimize for some primitive types
//!
//! Integers, floats and SIMD types are taken care of by the arch module, since
//! the CPU register in which they should end up is arch-specific.

use crate::{consume, hide, Pessimize};

// Implementation of Pessimize for bool based on that for u8
unsafe impl Pessimize for bool {
    #[allow(clippy::transmute_int_to_bool)]
    #[inline(always)]
    fn hide(self) -> Self {
        // Safe because hide() is guaranteed to be the identity function
        unsafe { core::mem::transmute(hide(u8::from(self))) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume(u8::from(*self))
    }
}

// Implementation of Pessimize for char based on that for u32
unsafe impl Pessimize for char {
    #[inline(always)]
    fn hide(self) -> Self {
        // Safe because hide() is guaranteed to be the identity function
        unsafe { char::from_u32_unchecked(hide(u32::from(self))) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume(u32::from(*self))
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    // --- Numeric types ---

    macro_rules! primitive_tests {
        ( $( ( $t:ident, $t_optim_test_name:ident ) ),* ) => {
            $(
                // Basic test that can run in debug and release mode
                #[test]
                fn $t() {
                    test_value_type::<$t>($t::MIN, $t::MAX);
                }

                // Advanced test that only makes sense in release mode
                #[test]
                #[ignore]
                fn $t_optim_test_name() {
                    test_unoptimized_value_type::<$t>();
                }
            )*
        };
    }
    //
    primitive_tests!(
        (i8, i8_optim),
        (u8, u8_optim),
        (i16, i16_optim),
        (u16, u16_optim),
        (i32, i32_optim),
        (u32, u32_optim),
        (isize, isize_optim),
        (usize, usize_optim),
        (f32, f32_optim)
    );
    //
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "arm", target_feature = "vfp2"),
        target_arch = "riscv64",
        all(target_arch = "x86", target_feature = "sse2"),
        target_arch = "x86_64",
    ))]
    primitive_tests!((i64, i64_optim), (u64, u64_optim));
    //
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "arm", target_feature = "vfp2"),
        all(target_arch = "riscv32", target_feature = "d"),
        target_arch = "riscv64",
        all(target_arch = "x86", target_feature = "sse2"),
        target_arch = "x86_64",
    ))]
    primitive_tests!((f64, f64_optim));

    // --- char ---

    #[test]
    fn char() {
        test_value_type('\0', char::MAX);
    }

    #[test]
    #[ignore]
    fn char_optim() {
        test_unoptimized_value_type::<char>();
    }

    // --- bool ---

    #[test]
    fn bool() {
        test_value_type(false, true);
    }

    #[test]
    #[ignore]
    fn bool_optim() {
        test_unoptimized_value_type::<bool>();
    }
}
