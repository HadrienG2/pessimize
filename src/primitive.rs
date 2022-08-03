//! Implementations of Pessimize for some primitive types
//!
//! Integers, floats and SIMD types are taken care of by the arch module, since
//! the CPU register in which they should end up is arch-specific.

#![allow(clippy::transmute_int_to_bool)]

use crate::{
    assume_accessed, assume_accessed_imut, assume_read, hide, pessimize_copy, BorrowPessimize,
    Pessimize, PessimizeCast,
};

pessimize_copy!(
    allow(missing_docs)
    {
        u8: (
            bool: (
                Self::into,
                // Safe because the inner u8 is not modified by Pessimize ops
                core::mem::transmute
            )
        ),
        u32: (
            char: (
                Self::into,
                char::from_u32_unchecked
            )
        )
    }
);

// Implementation of Pessimize for unit & small tuples of Pessimize values
//
// Larger tuples would spill to memory anyway, so the default implementation
// that spills to memory provided by the default_impl feature would be adequate.
//
macro_rules! pessimize_tuple {
    ($($args:ident),*) => {
        // NOTE: Can't use PessimizeCast/BorrowPessimize here because need to
        //       apply optimization barrier to multiple inner types.
        #[allow(non_snake_case)]
        unsafe impl<$($args: Pessimize),*> Pessimize for ($($args,)*) {
            #[allow(clippy::unused_unit)]
            #[inline(always)]
            fn hide(self) -> Self {
                let ($($args,)*) = self;
                ( $(hide::<$args>($args),)* )
            }

            #[inline(always)]
            fn assume_read(&self) {
                let ($(ref $args,)*) = self;
                $( assume_read::<$args>($args) );*
            }

            #[inline(always)]
            fn assume_accessed(&mut self) {
                let ($(ref mut $args,)*) = self;
                $( assume_accessed::<$args>($args) );*
            }

            #[inline(always)]
            fn assume_accessed_imut(&self) {
                let ($(ref $args,)*) = self;
                $( assume_accessed_imut::<$args>($args) );*
            }
        }
    };
}
//
macro_rules! pessimize_tuples_up_to {
    () => {
        pessimize_tuple!();
    };
    ($arg:ident $(, $other_args:ident)*) => {
        pessimize_tuple!($arg $(, $other_args)*);
        pessimize_tuples_up_to!($($other_args),*);
    }
}
//
pessimize_tuples_up_to!(A1, A2, A3, A4, A5, A6, A7, A8);

// Although the logic used above for tuples could, in principle, be used to
// implement Pessimize for small arrays, we do not do so because the proper
// CPU register where values should be moved for minimal-cost optimization
// barriers depends on whether SIMD is used, which we do not know.
//
// We do make an exception for [T; 1], since that is unfit for SIMD anyway.
//
unsafe impl<T: Pessimize> PessimizeCast for [T; 1] {
    type Pessimized = T;

    #[inline(always)]
    fn into_pessimize(self) -> T {
        let [x] = self;
        x
    }

    #[inline(always)]
    unsafe fn from_pessimize(x: T) -> Self {
        [x]
    }
}
//
impl<T: Pessimize> BorrowPessimize for [T; 1] {
    type BorrowedPessimize = T;

    #[inline(always)]
    fn with_pessimize(&self, f: impl FnOnce(&T)) {
        let [ref x] = self;
        f(x)
    }

    #[inline(always)]
    fn assume_accessed_impl(&mut self) {
        let [ref mut x] = self;
        assume_accessed(x);
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
