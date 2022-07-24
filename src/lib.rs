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
#![deny(missing_docs)]

// Each architecture-specific module is tasked to implement Pessimize for
// primitive integers, floats and SIMD vector types.
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

use core::arch::asm;

/// Optimization barriers for supported values
///
/// Implemented for all the types described in the crate documentation
///
pub trait Pessimize {
    /// Re-emit the input value as its output (identity function), but force the
    /// compiler to assume that it is a completely different value.
    ///
    /// If you want to re-do the exact same computation in a loop, you can pass
    /// its inputs through this barrier to prevent the compiler from optimizing
    /// out the redundant computations.
    ///
    /// If you need a `hide` alternative for a variable `x` that does not
    /// implement `Pessimize`, you can use `*((&x).hide())`, at the cost
    /// of forcing all data reachable via `x` which is currently cached in
    /// registers to be spilled to memory and reloaded when needed later on.
    ///
    fn hide(self) -> Self;

    /// Force the compiler to assume that a value, and data transitively
    /// reachable via that value (for pointers/refs), is being used if Rust
    /// rules allow for it.
    ///
    /// You can apply this barrier to unused computation results in order to
    /// prevent the compiler from optimizing out the associated computations.
    ///
    /// If you need an `assume_read` alternative for a variable `x` that does
    /// not implement `Pessimize`, you can use `(&x).assume_read()`, at the
    /// cost of forcing any data from `x` which is currently cached in registers
    /// to be spilled into memory.
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
    fn assume_read(&self);
}

/// Optimization barriers for thin pointers and references
///
/// Implemented only for `&T`, `&mut T`, `*const T` and `*mut T` where `T: Sized`.
///
pub trait PessimizeRef {
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
    /// # use pessimize::PessimizeRef;
    /// let x = 42;
    /// let r = &x;
    /// r.assume_accessed();
    /// // Compiler may still infer that x and *r are both 42 here
    /// ```
    ///
    /// Instead, if you have a shared reference to something and need the
    /// compiler to assume that it is a shared reference to something completely
    /// different, use `hide` to obscure the shared reference's target.
    ///
    /// ```
    /// # use pessimize::Pessimize;
    /// let x = 42;
    /// let mut r = &x;
    /// r = r.hide();
    /// // Compiler still knows that x is 42 but cannot infer that *r is 42 here
    /// ```
    ///
    /// Similar considerations apply to the use of `assume_accessed` on a
    /// `*const T` or `*mut T` in the presence of an `&T` or `&mut T` to the
    /// same target, where the compiler may or may not manage to infer that
    /// these pointers cannot be used to modify or read their targets where that
    /// would be undefined behavior.
    ///
    fn assume_accessed(&self);
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
    x.hide()
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
    x.assume_read()
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
/// let r = &x;
/// assume_accessed(&r);
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
pub fn assume_accessed<R: PessimizeRef>(r: &R) {
    r.assume_accessed()
}

// Implementation of Pessimize for bool based on that for u8
impl Pessimize for bool {
    #[inline(always)]
    fn hide(self) -> Self {
        // This is safe because hide() returns the same u8, which is a valid bool
        unsafe { core::mem::transmute((self as u8).hide()) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        (*self as u8).assume_read()
    }
}

// Implementation of Pessimize and PessimizeRef for pointers
macro_rules! pessimize_pointers {
    ($($t:ty),*) => {
        $(
            #[allow(asm_sub_register)]
            impl<T: Sized> Pessimize for $t {
                #[inline(always)]
                fn hide(mut self) -> Self {
                    unsafe {
                        asm!("/* {0} */", inout(reg) self, options(preserves_flags, nostack, nomem));
                    }
                    self
                }

                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        asm!("/* {0} */", in(reg) *self, options(preserves_flags, nostack, readonly))
                    }
                }
            }

            #[allow(asm_sub_register)]
            impl<T: Sized> PessimizeRef for $t {
                #[inline(always)]
                fn assume_accessed(&self) {
                    unsafe {
                        asm!("/* {0} */", in(reg) *self, options(preserves_flags, nostack))
                    }
                }
            }
        )*
    };
}
//
pessimize_pointers!(*const T, *mut T);

// Implementation of Pessimize and PessimizeRef for references
macro_rules! pessimize_references {
    ($($t:ty),*) => {
        $(
            impl<'a, T: Sized> Pessimize for $t {
                #[inline(always)]
                fn hide(self) -> Self {
                    unsafe {
                        // While this may sound like a questionable operation
                        // for &mut T, as it may lead to the transient existence
                        // of two &mut to the same data, it is actually not UB
                        // according to the current Unsafe Code Guidelines
                        // consensus, which is that **using** the two
                        // coexisting references is what causes UB.
                        core::mem::transmute((self as *const T).hide())
                    }
                }

                #[inline(always)]
                fn assume_read(&self) {
                    (*self as *const T).assume_read()
                }
            }

            impl<'a, T: Sized> PessimizeRef for $t {
                #[inline(always)]
                fn assume_accessed(&self) {
                    (*self as *const T).assume_accessed()
                }
            }
        )*
    };
}
//
pessimize_references!(&'a T, &'a mut T);

// TODO: Set up CI in the spirit of test-everything.sh

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{
        fmt::Debug,
        time::{Duration, Instant},
    };

    // === Tests asserting that the barriers don't modify anything ===
    // ===    (should be run on both debug and release builds)     ===

    unsafe fn test_pointer<Value: Clone + Debug + PartialEq>(
        p: impl Pessimize + PessimizeRef + UnsafeDeref<Target = Value>,
        expected_target: Value,
    ) {
        p.assume_read();
        assert_eq!(*p.unsafe_deref(), expected_target);
        p.assume_accessed();
        assert_eq!(*p.unsafe_deref(), expected_target);
        assert_eq!(*(p.hide().unsafe_deref()), expected_target);
    }

    fn test_all_pointers(mut x: impl Copy + Debug + PartialEq) {
        let old_x = x.clone();
        unsafe {
            test_pointer(&x as *const _, old_x);
            test_pointer(&mut x as *mut _, old_x);
            test_pointer(&x, old_x);
            test_pointer(&mut x, old_x);
        }
    }

    fn test_value(x: impl Pessimize + Copy + Debug + PartialEq) {
        let old_x = x.clone();
        x.assume_read();
        assert_eq!(x, old_x);
        assert_eq!(x.hide(), old_x);
        test_all_pointers(x);
    }

    pub fn test_value_type<T: Copy + Debug + Default + PartialEq + Pessimize>(min: T, max: T) {
        test_value(min);
        test_value(T::default());
        test_value(max);
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
        assert_unoptimized(|| (hide(x) == hide(y)).assume_read());
    }
    //
    fn test_unoptimized_load_via_hide<T: Copy + Default + Pessimize>() {
        let x = T::default();
        let r = &x;
        assert_unoptimized(|| (*hide(r)).assume_read());
    }
    //
    fn test_unoptimized_load_via_assume_accessed<T: Copy + Default + Pessimize>() {
        let mut x = T::default();
        let r = &mut x;
        assert_unoptimized(|| {
            r.assume_accessed();
            (*r).assume_read()
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
        test_all_pointers([isize::MIN; BIG]);
        test_all_pointers([0; 1024]);
        test_all_pointers([isize::MAX; BIG]);
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
            (&dst).assume_read();
        });

        // Copy optimization inhibition using assume_accessed
        let mut src = [0isize; BIG];
        let mut dst = [0isize; BIG];
        assert_unoptimized(|| {
            (&mut src).assume_accessed();
            dst = src;
            (&dst).assume_read();
        });
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

    fn test_function_pointer(fptr: &impl Fn() -> isize, expected_result: isize) {
        (fptr).assume_read();
        assert_eq!(fptr(), expected_result);
        (fptr).assume_accessed();
        assert_eq!(fptr(), expected_result);
        assert_eq!(super::hide(fptr)(), expected_result);
    }

    fn test_function_pointer_optim(fptr: &impl Fn() -> isize) {
        assert_unoptimized(|| {
            let new_fptr = hide(fptr);
            new_fptr().assume_read()
        })
    }

    // Should be run on both debug and release builds
    #[test]
    fn function_pointer() {
        test_function_pointer(&min, MIN);
        test_function_pointer(&zero, ZERO);
        test_function_pointer(&max, MAX);
    }

    // Should only be run on release builds
    #[test]
    #[ignore]
    fn function_pointer_optim() {
        test_function_pointer_optim(&min);
        test_function_pointer_optim(&zero);
        test_function_pointer_optim(&max);
    }

    // === Uninteresting helpers ===

    // Abstraction layer to handle references and pointers homogeneously
    trait UnsafeDeref {
        type Target;
        unsafe fn unsafe_deref(&self) -> &Self::Target;
    }
    //
    impl<'a, T> UnsafeDeref for &'a T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &**self
        }
    }
    //
    impl<'a, T> UnsafeDeref for &'a mut T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &**self
        }
    }
    //
    impl<T> UnsafeDeref for *const T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &**self
        }
    }
    //
    impl<T> UnsafeDeref for *mut T {
        type Target = T;
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &**self
        }
    }
}
