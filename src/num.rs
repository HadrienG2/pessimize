//! Implementations of Pessimize for core::num

use crate::{pessimize_copy, Pessimize};
use core::num::{
    NonZeroI16, NonZeroI32, NonZeroI8, NonZeroIsize, NonZeroU16, NonZeroU32, NonZeroU8,
    NonZeroUsize, Wrapping,
};
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "vfp2"),
    target_arch = "riscv64",
    all(target_arch = "x86", target_feature = "sse2"),
    target_arch = "x86_64",
))]
use core::num::{NonZeroI64, NonZeroU64};

macro_rules! pessimize_nonzero {
    ( $( $inner:ty => $outer:ty ),* ) => {
        pessimize_copy!(
            allow(missing_docs)
            {
                $(
                    $inner : ($outer : (Self::into, Self::new_unchecked))
                ),*
            }
        );
    };
}
//
pessimize_nonzero!(
    i8 => NonZeroI8,
    u8 => NonZeroU8,
    i16 => NonZeroI16,
    u16 => NonZeroU16,
    i32 => NonZeroI32,
    u32 => NonZeroU32,
    isize => NonZeroIsize,
    usize => NonZeroUsize
);
//
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "vfp2"),
    target_arch = "riscv64",
    all(target_arch = "x86", target_feature = "sse2"),
    target_arch = "x86_64",
))]
pessimize_nonzero!(
    i64 => NonZeroI64,
    u64 => NonZeroU64
);

pessimize_copy!(
    allow(missing_docs)
    {
        T: (
            // It's okay to demand a Copy bound here since all current Ts for
            // which Wrapping<T> is useful implement Copy
            |T: (Copy, Pessimize)| Wrapping<T>: (|Self(inner)| inner, Self)
        )
    }
);

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{
        test_unoptimized_value, test_unoptimized_value_type, test_value, test_value_type,
    };

    macro_rules! num_tests {
        (
            $(
                (
                    $t:ident,
                    $non_zero_t:ident,
                    $non_zero_t_optim_test:ident,
                    $wrapping_t_test:ident,
                    $wrapping_t_optim_test:ident
                )
            ),*
        ) => {
            $(
                #[test]
                fn $non_zero_t() {
                    test_value($non_zero_t::new(1).unwrap());
                    test_value($non_zero_t::new($t::MAX).unwrap());
                }

                #[test]
                #[ignore]
                fn $non_zero_t_optim_test() {
                    test_unoptimized_value($non_zero_t::new($t::MAX).unwrap());
                }

                #[test]
                fn $wrapping_t_test() {
                    test_value_type(Wrapping($t::MIN), Wrapping($t::MAX));
                }

                #[test]
                #[ignore]
                fn $wrapping_t_optim_test() {
                    test_unoptimized_value_type::<Wrapping<$t>>();
                }
            )*
        };
    }
    //
    num_tests!(
        (i8, NonZeroI8, NonZeroI8_optim, WrappingI8, WrappingI8_optim),
        (u8, NonZeroU8, NonZeroU8_optim, WrappingU8, WrappingU8_optim),
        (
            i16,
            NonZeroI16,
            NonZeroI16_optim,
            WrappingI16,
            WrappingI16_optim
        ),
        (
            u16,
            NonZeroU16,
            NonZeroU16_optim,
            WrappingU16,
            WrappingU16_optim
        ),
        (
            i32,
            NonZeroI32,
            NonZeroI32_optim,
            WrappingI32,
            WrappingI32_optim
        ),
        (
            u32,
            NonZeroU32,
            NonZeroU32_optim,
            WrappingU32,
            WrappingU32_optim
        ),
        (
            isize,
            NonZeroIsize,
            NonZeroIsize_optim,
            WrappingIsize,
            WrappingIsize_optim
        ),
        (
            usize,
            NonZeroUsize,
            NonZeroUsize_optim,
            WrappingUsize,
            WrappingUsize_optim
        )
    );
    //
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "arm", target_feature = "vfp2"),
        target_arch = "riscv64",
        all(target_arch = "x86", target_feature = "sse2"),
        target_arch = "x86_64",
    ))]
    num_tests!(
        (
            i64,
            NonZeroI64,
            NonZeroI64_optim,
            WrappingI64,
            WrappingI64_optim
        ),
        (
            u64,
            NonZeroU64,
            NonZeroU64_optim,
            WrappingU64,
            WrappingU64_optim
        )
    );
}
