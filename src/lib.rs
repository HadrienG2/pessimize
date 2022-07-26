//! This crate aims to implement minimally costly optimization barriers for
//! every architecture that has `asm!()` support (currently x86(_64),
//! 32-bit ARM, AArch64 and RISC-V).
//!
//! You can use these barriers to prevent the compiler from optimizing out
//! selected redundant or unnecessary computations in situations where such
//! optimization is undesirable, like microbenchmarking.
//!
//! For a fully implemented architecture, the barriers will be implemented for
//! - Primitive integers (iN and uN, including isize and usize but excluding
//!   128-bit integers)
//! - Primitive floats (f32 and f64)
//! - Thin pointers and references (`&T`-like other than `&[T]` or `&dyn T`,
//!   including function pointers)
//! - SIMD vector types (with optional support for `safe_arch` and `core::simd`
//!   via feature flags)
//!
//! Some legacy and embedded architectures will not support 64-bit primitive
//! types. The rule of thumb is, if your target CPU can fit primitive type T in
//! a single architectural register, then that type should implement Pessimize.
//!
//! Any type which is not directly supported can still be subjected to an
//! optimization barrier by taking a reference to it and subjecting that
//! reference to an optimization barrier, at the cost of causing the value to
//! be spilled to memory.
//!
//! For pointer-like entities, optimization barriers other than `hide` will
//! have the side-effect of causing the compiler to assume that global and
//! thread-local variable might have been accessed using similar semantics as
//! the pointer itself. This will reduce applicable compiler optimizations for
//! such variables, so use of `hide` should be preferred when global or
//! thread-local variables are used.
//!
//! In general, `assume_read` and `assume_accessed` can have more surprising
//! behavior than `hide` (see their documentation for details), so you
//! should strive to do what you want with `hide` if possible, and only
//! reach for `assume_read` and `assume_accessed` where the extra expressive
//! power of these primitives is truly needed.
//!
//! You should consider use of this crate over `core::hint::black_box`, or
//! third party cousins thereof, because...
//! - It works on stable Rust
//! - It has a better-defined API contract with stronger guarantees (unlike
//!   `black_box`, for which "do nothing" is a valid implementation).
//! - It exposes finer-grained operations, which clarify your code's intent and
//!   reduce harmful side-effects.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(feature = "default_impl", allow(incomplete_features))]
#![cfg_attr(feature = "default_impl", feature(specialization))]
#![cfg_attr(
    feature = "nightly",
    feature(doc_cfg, stdsimd, portable_simd, ptr_metadata)
)]
#![deny(missing_docs)]

// Each architecture-specific module is tasked to implement Pessimize for
// primitive integers, floats and SIMD vector types.
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;
// TODO: On nightly, support more arches via asm_experimental_arch

use core::arch::asm;

/// Optimization barriers for supported values
///
/// This trait is implemented for both value and reference types, which can
/// lead to unexpected method syntax semantics (you expected to call the
/// `Pessimize` impl of `T`, and you actually called that of `&T`). As a result,
/// it is strongly recommended to use the optimization barriers via the free
/// functions provided at the crate root, rather than via method syntax.
///
/// # Safety
///
/// Unsafe code may rely on hide() behaving as an identity function (returning
/// the input value unaltered).
///
pub unsafe trait Pessimize {
    /// See `pessimize::hide()` for documentation
    fn hide(self) -> Self;

    /// See `pessimize::assume_read()` for documentation
    fn assume_read(&self);
}

/// Optimization barriers for pointers and references
///
/// This trait is only implemented for thin pointers and references on stable,
/// but also supports fat pointers (slices and trait objects) on nightly.
///
/// Calling this trait on a reference does not have the same semantics as
/// calling it on a reference of reference, which can lead to unexpected method
/// syntax semantics (you expected to call the `Pessimize` impl of `&T`, and you
/// actually called that of `&&T`). As a result, it is strongly recommended to
/// use the optimization barriers via the free functions provided at the crate
/// root, rather than via method syntax.
///
pub trait PessimizeRef {
    /// See `pessimize::assume_accessed()` for documentation
    fn assume_accessed(&mut self);

    /// See `pessimize::assume_accessed_imut()` for documentation
    fn assume_accessed_imut(&self);
}

/// Re-emit the input value as its output (identity function), but force the
/// compiler to assume that it is a completely different value.
///
/// If you want to re-do the exact same computation in a loop, you can pass
/// its inputs through this barrier to prevent the compiler from optimizing
/// out the redundant computations.
///
/// If you need a `hide` alternative for a variable `x` that does not
/// implement `Pessimize`, you can use `*hide(&x)`, at the cost of forcing
/// all data reachable via x which is currently cached in registers to be
/// spilled to memory and reloaded when needed later on.
///
/// If you are familiar with the unstable `core::hint::black_box` function or
/// analogs in benchmarking libraries like Criterion, please note that although
/// this function has a similar API signature, it does not have the same
/// semantics and cannot be used as a direct replacement. For example,
/// `black_box(&mut x)` should have the effect of `assume_accessed(&mut x)` in
/// this crate's vocabulary, whereas `hide` does not enforce any compiler
/// assumptions concerning the original value, it just turns it into another
/// value that looks unrelated in the eye of the compiler.
///
#[inline(always)]
pub fn hide<T: Pessimize>(x: T) -> T {
    Pessimize::hide(x)
}

/// Force the compiler to assume that a value, and data transitively
/// reachable via that value (for pointers/refs), is being used if Rust
/// rules allow for it.
///
/// You can apply this barrier to unused computation results in order to
/// prevent the compiler from optimizing out the associated computations.
///
/// If you need an `assume_read` alternative for a variable `x` that does not
/// implement `Pessimize`, you can use `assume_read(&x)`, at the cost of
/// forcing any data from x which is currently cached in registers to be
/// spilled into memory.
///
/// The `assume_read` implementation of `*const T` and `*mut T` may not work
/// as expected if an `&mut T` reference to the same data exists somewhere,
/// because dereferencing the pointer in that situation would be undefined
/// behavior, which by definition does not exist in the eye of the compiler.
///
/// For pointer types, this operation may sometimes be pessimized into a
/// full `assume_accessed()` optimization barrier, as a result of rustc not
/// leveraging the underlying `readonly` optimization hint. It is hoped that
/// future versions of rustc will take stronger notice of that hint.
///
#[inline(always)]
pub fn assume_read<T: Pessimize>(x: &T) {
    Pessimize::assume_read(x)
}

/// Variant of `assume_read` which is more ergonomic in the common case where
/// a `Pessimize` value is Copy or will not be needed anymore.
#[inline(always)]
pub fn consume<T: Pessimize>(x: T) {
    assume_read(&x);
}

/// Force the compiler to assume that any data transitively reachable via a
/// pointer/reference has been read, and modified if Rust rules allow for it.
///
/// This will cause all target data which is currently cached in registers
/// to be spilled to memory and reloaded when needed later on.
///
/// The compiler is allowed to assume that data which is only reachable via
/// an &-reference and does not have interior mutability semantics cannot be
/// modified, so you should not expect this pattern to work:
///
/// ```
/// # use pessimize::assume_accessed;
/// let x = 42;
/// let mut r = &x;
/// assume_accessed(&mut r);
/// // Compiler may still infer that x and *r are both 42 here
/// ```
///
/// Instead, if you have a shared reference to something and need the
/// compiler to assume that it is a shared reference to something completely
/// different, use `hide` to obscure the shared reference's target.
///
/// ```
/// # use pessimize::hide;
/// let x = 42;
/// let mut r = &x;
/// r = hide(r);
/// // Compiler still knows that x is 42 but cannot infer that *r is 42 here
/// ```
///
/// Similar considerations apply to the use of `assume_accessed` on a `*const T`
/// or `*mut T` in the presence of an `&T` or `&mut T` to the same target, where
/// the compiler may or may not manage to infer that these pointers cannot be
/// used to modify or read their targets where that would be undefined behavior.
///
#[inline(always)]
pub fn assume_accessed<R: PessimizeRef>(r: &mut R) {
    PessimizeRef::assume_accessed(r)
}

/// Variant of `assume_accessed` for internally mutable types
///
/// You should only use this variant on internally mutable types (Cell,
/// RefCell, Mutex, AtomicXyz...), otherwise you will instantly fall victim
/// of the "shared reference mutation is UB" edge case mentioned in the
/// documentation of `assume_accessed`.
///
#[inline(always)]
pub fn assume_accessed_imut<R: PessimizeRef>(r: &R) {
    PessimizeRef::assume_accessed_imut(r)
}

/// Default implementation of Pessimize when no better one is available
#[cfg(feature = "default_impl")]
#[doc(cfg(all(feature = "nightly", feature = "default_impl")))]
unsafe impl<T> Pessimize for T {
    #[inline(always)]
    default fn hide(mut self) -> Self {
        assume_accessed(&mut ((&mut self) as *mut T));
        self
    }

    #[inline(always)]
    default fn assume_read(&self) {
        assume_read(&(self as *const T))
    }
}

// Implementation of Pessimize for bool based on that for u8
unsafe impl Pessimize for bool {
    #[allow(clippy::transmute_int_to_bool)]
    #[inline(always)]
    fn hide(self) -> Self {
        // This is safe because hide() returns the same u8, which is a valid bool
        unsafe { core::mem::transmute(hide(self as u8)) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume(*self as u8)
    }
}

/// Implementation of Pessimize for values without pointer semantics
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_values {
    (
        $doc_cfg:meta
        {
            $(
                $reg:ident: ( $($value_type:ty),* )
            ),*
        }
    ) => {
        $($(
            #[allow(asm_sub_register)]
            #[$doc_cfg]
            unsafe impl Pessimize for $value_type {
                #[inline(always)]
                fn hide(mut self) -> Self {
                    unsafe {
                        asm!("/* {0} */", inout($reg) self, options(preserves_flags, nostack, nomem));
                    }
                    self
                }

                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        asm!("/* {0} */", in($reg) *self, options(preserves_flags, nostack, nomem))
                    }
                }
            }
        )*)*
    };
}

/// Implementation of Pessimize for Simd types from portable_simd
#[allow(unused)]
#[cfg(feature = "nightly")]
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_portable_simd {
    (
        $doc_cfg:meta
        {
            $(
                $inner:ident: ( $($simd_type:ty),* )
            ),*
        }
    ) => {
        $($(
            #[$doc_cfg]
            unsafe impl Pessimize for $simd_type {
                #[inline(always)]
                fn hide(self) -> Self {
                    // FIXME: This probably works, but it would be nicer if
                    //        portable_simd provided a conversion from
                    //        architectural SIMD types.
                    unsafe {
                        core::mem::transmute($crate::hide($inner::from(self)))
                    }
                }

                #[inline(always)]
                fn assume_read(&self) {
                    $crate::consume($inner::from(*self))
                }
            }
        )*)*
    };
}

// Implementation of Pessimize and PessimizeRef for pointers
#[allow(asm_sub_register)]
#[inline(always)]
fn hide_thin_ptr<T: Sized>(mut x: *const T) -> *const T {
    unsafe {
        asm!("/* {0} */", inout(reg) x, options(preserves_flags, nostack, nomem));
    }
    x
}
//
#[allow(asm_sub_register)]
#[inline(always)]
fn assume_read_thin_ptr<T: Sized>(x: *const T) {
    unsafe { asm!("/* {0} */", in(reg) x, options(preserves_flags, nostack, readonly)) }
}
//
#[allow(asm_sub_register)]
#[inline(always)]
fn assume_accessed_thin_ptr<T: Sized>(x: *mut T) {
    unsafe { asm!("/* {0} */", in(reg) x, options(preserves_flags, nostack)) }
}
//
#[cfg(not(feature = "nightly"))]
macro_rules! pessimize_thin_pointers {
    ($pointee:ident: ($($ptr:ty),*)) => {
        $(
            unsafe impl<$pointee: Sized> Pessimize for $ptr {
                #[inline(always)]
                fn hide(self) -> Self {
                    hide_thin_ptr(self as *const $pointee) as Self
                }

                #[inline(always)]
                fn assume_read(&self) {
                    assume_read_thin_ptr(*self as *const $pointee)
                }
            }

            impl<$pointee: Sized> PessimizeRef for $ptr {
                #[inline(always)]
                fn assume_accessed(&mut self) {
                    assume_accessed_thin_ptr(*self as *mut $pointee)
                }

                #[inline(always)]
                fn assume_accessed_imut(&self) {
                    assume_accessed_thin_ptr(*self as *mut $pointee)
                }
            }
        )*
    };
}
#[cfg(not(feature = "nightly"))]
pessimize_thin_pointers!(T: (*const T, *mut T));
//
#[cfg(feature = "nightly")]
mod pessimize_all_pointers {
    use super::*;
    use core::ptr::{self, DynMetadata, Pointee};

    unsafe impl<T: ?Sized> Pessimize for DynMetadata<T> {
        #[inline(always)]
        fn hide(self) -> Self {
            unsafe { core::mem::transmute(hide(core::mem::transmute::<_, usize>(self))) }
        }

        #[inline(always)]
        fn assume_read(&self) {
            unsafe { consume(core::mem::transmute::<_, usize>(self)) }
        }
    }

    unsafe impl<T: Pointee + ?Sized> Pessimize for *const T
    where
        T::Metadata: Pessimize,
    {
        #[inline(always)]
        fn hide(self) -> Self {
            let (thin, metadata) = self.to_raw_parts();
            ptr::from_raw_parts(hide_thin_ptr(thin), hide(metadata))
        }

        #[inline(always)]
        fn assume_read(&self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_read_thin_ptr(thin);
            consume(metadata);
        }
    }

    impl<T: core::ptr::Pointee + ?Sized> PessimizeRef for *const T
    where
        T::Metadata: Pessimize,
    {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_accessed_thin_ptr(thin as *mut ());
            // At this point in time, the metadata of a pointer cannot be a
            // pointer to a mutable object (only (), usize or DynMetadata that
            // points to a table of read-only type metadata), so this is fine.
            consume(metadata);
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_accessed_thin_ptr(thin as *mut ());
            // At this point in time, the metadata of a pointer cannot be a
            // pointer to a mutable object (only (), usize or DynMetadata that
            // points to a table of read-only type metadata), so this is fine.
            consume(metadata);
        }
    }

    unsafe impl<T: Pointee + ?Sized> Pessimize for *mut T
    where
        T::Metadata: Pessimize,
    {
        #[inline(always)]
        fn hide(self) -> Self {
            hide(self as *const T) as *mut T
        }

        #[inline(always)]
        fn assume_read(&self) {
            consume((*self) as *const T)
        }
    }

    impl<T: core::ptr::Pointee + ?Sized> PessimizeRef for *mut T
    where
        T::Metadata: Pessimize,
    {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            assume_accessed(&mut (*self as *const T))
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            assume_accessed_imut(&(*self as *const T))
        }
    }
}

// Implementation of Pessimize and PessimizeRef for references
macro_rules! pessimize_references {
    ($($t:ty),*) => {
        $(
            unsafe impl<'a, T: ?Sized> Pessimize for $t
                where *const T: Pessimize
            {
                #[allow(clippy::transmute_ptr_to_ref)]
                #[inline(always)]
                fn hide(self) -> Self {
                    unsafe {
                        // While this may sound like a questionable operation
                        // for &mut T, as it may lead to the transient existence
                        // of two &mut to the same data, it is actually not UB
                        // according to the current Unsafe Code Guidelines
                        // consensus, which is that **using** the two
                        // coexisting references is what causes UB.
                        core::mem::transmute(hide(self as *const T))
                    }
                }

                #[inline(always)]
                fn assume_read(&self) {
                    assume_read::<*const T>(&((*self) as *const T))
                }
            }

            impl<'a, T: ?Sized> PessimizeRef for $t
                where *const T: PessimizeRef
            {
                #[inline(always)]
                fn assume_accessed(&mut self) {
                    assume_accessed::<*const T>(&mut ((*self) as *const T))
                }

                #[inline(always)]
                fn assume_accessed_imut(&self) {
                    assume_accessed_imut::<*const T>(&((*self) as *const T))
                }
            }
        )*
    };
}
//
pessimize_references!(&'a T, &'a mut T);

// Implementation of Pessimize for function pointers
macro_rules! pessimize_fn {
    ($res:ident $( , $args:ident )* ) => {
        unsafe impl< $res $( , $args )* > Pessimize for fn( $($args),* ) -> $res {
            #[inline(always)]
            fn hide(self) -> Self {
                unsafe { core::mem::transmute(
                    hide(self as *const ())
                ) }
            }

            #[inline(always)]
            fn assume_read(&self) {
                consume((*self) as *const ())
            }
        }
    }
}
pessimize_fn!(R);
pessimize_fn!(R, A1);
pessimize_fn!(R, A1, A2);
pessimize_fn!(R, A1, A2, A3);
pessimize_fn!(R, A1, A2, A3, A4);
pessimize_fn!(R, A1, A2, A3, A4, A5);
pessimize_fn!(R, A1, A2, A3, A4, A5, A6);
pessimize_fn!(R, A1, A2, A3, A4, A5, A6, A7);
pessimize_fn!(R, A1, A2, A3, A4, A5, A6, A7, A8);

// Implementation of Pessimize for small tuples of Pessimize values
//
// Larger tuples would spill to memory anyway, so the default implementation
// that spills to memory provided by the default_impl feature would be adequate.
//
macro_rules! pessimize_tuple {
    ($($args:ident),*) => {
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
                $( assume_read(&$args) );*
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
// implement Pessimize for small arrays, we do not do so because it would
// force individual scalars to be moved to GP registers, which pessimizes SIMD

// TODO: Provide a Derive macro to derive Pessimize for a small struct, with a
//       warning that it will do more harm than good on a larger struct

// TODO: Set up CI in the spirit of test-everything.sh

// FIXME: Test new nightly types: fat pointers and tuples of Pessimize values
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "nightly")]
    use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
    use std::{
        fmt::Debug,
        time::{Duration, Instant},
    };

    // === Tests asserting that the barriers don't modify anything ===
    // ===    (should be run on both debug and release builds)     ===

    unsafe fn test_thin_pointer<Value: Clone + Debug + PartialEq>(
        mut p: impl Pessimize + PessimizeRef + UnsafeDeref<Target = Value>,
        expected_target: Value,
    ) {
        assume_read(&p);
        assert_eq!(*p.unsafe_deref(), expected_target);
        assume_accessed(&mut p);
        assert_eq!(*p.unsafe_deref(), expected_target);
        assume_accessed_imut(&p);
        assert_eq!(*p.unsafe_deref(), expected_target);
        assert_eq!(*(hide(p).unsafe_deref()), expected_target);
    }

    fn test_all_thin_pointers(mut x: impl Copy + Debug + PartialEq) {
        let old_x = x;
        unsafe {
            test_thin_pointer(&x as *const _, old_x);
            test_thin_pointer(&mut x as *mut _, old_x);
            test_thin_pointer(&x, old_x);
            test_thin_pointer(&mut x, old_x);
        }
    }

    fn test_value(x: impl Copy + Debug + PartialEq + Pessimize) {
        let old_x = x;
        assume_read(&x);
        assert_eq!(x, old_x);
        assert_eq!(hide(x), old_x);
        test_all_thin_pointers(x);
    }

    fn test_value_type<T: Copy + Debug + Default + PartialEq + Pessimize>(min: T, max: T) {
        test_value(min);
        test_value(T::default());
        test_value(max);
    }

    #[allow(unused)]
    pub fn test_simd<
        Scalar: Copy + Default,
        const LANES: usize,
        T: Copy + Debug + Default + From<[Scalar; LANES]> + PartialEq + Pessimize,
    >(
        min: Scalar,
        max: Scalar,
    ) {
        test_value_type(T::from([min; LANES]), T::from([max; LANES]));
    }

    #[allow(unused)]
    #[cfg(feature = "nightly")]
    pub fn test_portable_simd<
        Scalar: Debug + Default + PartialEq + SimdElement,
        const LANES: usize,
    >(
        min: Scalar,
        max: Scalar,
    ) where
        LaneCount<LANES>: SupportedLaneCount,
        Simd<Scalar, LANES>: Pessimize,
    {
        test_simd::<Scalar, LANES, Simd<Scalar, LANES>>(min, max)
    }

    // === Tests asserting that the barriers prevent optimization ===
    // ===         (should only be run on release builds)         ===

    // --- Basic harness ---

    /// Maximum realistic operation processing frequency
    /// This can be Nx higher than the clock rate on an N-way superscalar CPU
    const MAX_PROCESSING_FREQ: u64 = 100_000_000_000;

    /// Maximum expected clock granularity
    /// The system clock is expected to always be able to measure this duration
    const MIN_DURATION: Duration = Duration::from_millis(2);

    /// Minimum number of loop iterations for which a duration greater than or
    /// equal to MIN_DURATION should be measured
    const MIN_ITERATIONS: u64 =
        2 * MAX_PROCESSING_FREQ * MIN_DURATION.as_nanos() as u64 / 1_000_000_000;

    fn assert_unoptimized(mut op: impl FnMut()) {
        let start = Instant::now();
        for _ in 0..MIN_ITERATIONS {
            op();
        }
        assert!(start.elapsed() >= MIN_DURATION);
    }

    // --- Tests for values with native Pessimize support ---

    fn test_unoptimized_eq<T: Copy + Default + PartialEq + Pessimize>() {
        let x = T::default();
        let y = T::default();
        assert_unoptimized(|| consume(hide(x) == hide(y)));
    }
    //
    fn test_unoptimized_load_via_hide<T: Copy + Default + Pessimize>() {
        let x = T::default();
        let r = &x;
        assert_unoptimized(|| consume(*hide(r)));
    }
    //
    fn test_unoptimized_load_via_assume_accessed<T: Copy + Default + Pessimize>() {
        let mut x = T::default();
        let mut r = &mut x;
        assert_unoptimized(|| {
            assume_accessed(&mut r);
            consume(*r);
        });
    }
    //
    pub fn test_unoptimized_value<T: Copy + Default + PartialEq + Pessimize>() {
        test_unoptimized_eq::<T>();
        test_unoptimized_load_via_hide::<T>();
        test_unoptimized_load_via_assume_accessed::<T>();
    }

    // === Generate a test suite for all primitive types ===

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
                    test_unoptimized_value::<$t>();
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

    // --- bool ---

    // Basic test that can run in debug and release mode
    #[test]
    fn bool() {
        test_value_type::<bool>(false, true);
    }

    // Advanced test that only makes sense in release mode
    #[test]
    #[ignore]
    fn bool_optim() {
        test_unoptimized_value::<bool>();
    }

    // === Test suite for types that only implement Pessimize by reference ===

    // --- Array too big to fit in a register ---

    // What is considered too big (in units of isize)
    const BIG: usize = 32;

    // Should be run on both debug and release builds
    #[test]
    fn non_native() {
        test_all_thin_pointers([isize::MIN; BIG]);
        test_all_thin_pointers([0; 1024]);
        test_all_thin_pointers([isize::MAX; BIG]);
        #[cfg(feature = "default_impl")]
        test_value_type::<[isize; BIG]>([isize::MIN; BIG], [isize::MAX; BIG]);
    }

    // Should only be run on release builds
    #[test]
    #[ignore]
    fn non_native_optim() {
        // Copy optimization inhibition using hide
        let src = [0isize; BIG];
        let mut dst = [0isize; BIG];
        assert_unoptimized(|| {
            dst = *hide(&src);
            consume(&dst);
        });

        // Copy optimization inhibition using assume_accessed
        let mut src = [0isize; BIG];
        let mut dst = [0isize; BIG];
        assert_unoptimized(|| {
            #[allow(clippy::unnecessary_mut_passed)]
            assume_accessed(&mut (&mut src));
            dst = src;
            consume(&dst);
        });

        // Standard tests if the default impl is enabled
        #[cfg(feature = "default_impl")]
        test_unoptimized_value::<[isize; BIG]>();
    }

    // --- Function pointers ---

    const MIN: isize = isize::MIN;
    fn min() -> isize {
        MIN
    }

    const ZERO: isize = 0;
    fn zero() -> isize {
        ZERO
    }

    const MAX: isize = isize::MAX;
    fn max() -> isize {
        MAX
    }

    fn test_function_pointer(fptr: fn() -> isize, expected_result: isize) {
        assume_read(&fptr);
        assert_eq!(fptr(), expected_result);
        assert_eq!(hide(fptr)(), expected_result);
    }

    fn test_function_pointer_optim(fptr: fn() -> isize) {
        assert_unoptimized(|| {
            let new_fptr = hide(fptr);
            consume(new_fptr())
        })
    }

    // Should be run on both debug and release builds
    #[test]
    fn function_pointer() {
        test_function_pointer(min, MIN);
        test_function_pointer(zero, ZERO);
        test_function_pointer(max, MAX);
    }

    // Should only be run on release builds
    #[test]
    #[ignore]
    fn function_pointer_optim() {
        test_function_pointer_optim(min);
        test_function_pointer_optim(zero);
        test_function_pointer_optim(max);
    }

    // === Uninteresting helpers ===

    // Abstraction layer to handle references and pointers homogeneously
    trait UnsafeDeref {
        type Target: ?Sized;
        unsafe fn unsafe_deref(&self) -> &Self::Target;
    }
    //
    impl<'a, T: ?Sized> UnsafeDeref for &'a T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            self
        }
    }
    //
    impl<'a, T: ?Sized> UnsafeDeref for &'a mut T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            self
        }
    }
    //
    impl<T: ?Sized> UnsafeDeref for *const T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &**self
        }
    }
    //
    impl<T: ?Sized> UnsafeDeref for *mut T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &**self
        }
    }
}
