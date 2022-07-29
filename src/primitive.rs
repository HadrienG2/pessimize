//! Implementations of Pessimize for some primitive types
//!
//! Integers, floats and SIMD types are taken care of by the arch module, since
//! the CPU register in which they should end up is arch-specific.

use crate::{assume_read, consume, hide, Pessimize};

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

// Implementation of Pessimize for unit & small tuples of Pessimize values
//
// Larger tuples would spill to memory anyway, so the default implementation
// that spills to memory provided by the default_impl feature would be adequate.
//
macro_rules! pessimize_tuple {
    ($($args:ident),*) => {
        // This is safe because a tuple is just the sum of its parts
        #[allow(non_snake_case)]
        unsafe impl<$($args: Pessimize),*> Pessimize for ($($args,)*) {
            #[allow(clippy::unused_unit)]
            #[inline(always)]
            fn hide(self) -> Self {
                let ($($args,)*) = self;
                ( $(hide($args),)* )
            }

            #[inline(always)]
            fn assume_read(&self) {
                let ($(ref $args,)*) = self;
                $( assume_read($args) );*
            }
        }
    };
}
pessimize_tuple!();
pessimize_tuple!(A1);
pessimize_tuple!(A1, A2);
pessimize_tuple!(A1, A2, A3);
pessimize_tuple!(A1, A2, A3, A4);
pessimize_tuple!(A1, A2, A3, A4, A5);
pessimize_tuple!(A1, A2, A3, A4, A5, A6);
pessimize_tuple!(A1, A2, A3, A4, A5, A6, A7);
pessimize_tuple!(A1, A2, A3, A4, A5, A6, A7, A8);

// Although the logic used above for tuples could, in principle, be used to
// implement Pessimize for small arrays, we do not do so because the proper
// CPU register where values should be moved for minimal-cost optimization
// barriers depends on whether SIMD is used, which we do not know.
//
// We do make an exception for [T; 1], since that is unfit for SIMD anyway.
//
// This is safe because [T; 1] is just a T in disguise.
//
unsafe impl<T: Pessimize> Pessimize for [T; 1] {
    #[inline(always)]
    fn hide(self) -> Self {
        let [x] = self;
        [hide(x)]
    }

    #[inline(always)]
    fn assume_read(&self) {
        assume_read(&self[0])
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    // --- Numeric types and single-element arrays ---

    macro_rules! primitive_tests {
        ( $( ( $t:ident, $t_optim_test_name:ident ) ),* ) => {
            $(
                // Basic test that can run in debug and release mode
                #[test]
                fn $t() {
                    test_value_type::<$t>($t::MIN, $t::MAX);
                    test_value_type::<[$t; 1]>([$t::MIN], [$t::MAX]);
                }

                // Advanced test that only makes sense in release mode
                #[test]
                #[ignore]
                fn $t_optim_test_name() {
                    test_unoptimized_value_type::<$t>();
                    test_unoptimized_value_type::<[$t; 1]>();
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

    // --- tuples ---

    #[test]
    fn tuple() {
        test_value_type((u8::MIN, usize::MIN), (u8::MAX, usize::MAX));
    }

    #[test]
    #[ignore]
    fn tuple_optim() {
        test_unoptimized_value_type::<(u8, usize)>();
    }
}
