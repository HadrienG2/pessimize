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
//! - SIMD vector types (with support for `std::simd` on nightly)
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
//! You should favor use of this crate over `core::hint::black_box`, or
//! third party cousins thereof, because...
//! - It works on stable Rust
//! - It has a better-defined API contract with stronger guarantees (unlike
//!   `black_box`, for which "do nothing" is a valid implementation).
//! - It exposes finer-grained operations, which clarify your code's intent and
//!   reduce harmful side-effects.

#![cfg_attr(not(test), no_std)]
#![deny(missing_docs)]

// Each architecture-specific module is tasked to implement Pessimize for
// primitive integers and floats, raw pointers, and SIMD vector types.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
// TODO: Add support for 32-bit ARM, AArch64 and RISC-V (check calling
//       conventions to know where floats should be put)

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
/// let x = 42;
/// let r = &x;
/// assume_accessed(r);
/// // Compiler may still infer that x and *r are both 42 here
/// ```
///
/// Instead, if you have a shared reference to something and need the
/// compiler to assume that it is a shared reference to something completely
/// different, use `hide` to obscure the shared reference's target.
///
/// ```
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
    r.assume_accessed()
}

// Implementation of Pessimize and PessimizeRef for references
macro_rules! pessimize_references {
    ($($t:ty),*) => {
        $(
            impl<'a, T: Sized> Pessimize for $t {
                #[inline(always)]
                fn hide(self) -> Self {
                    unsafe {
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

// FIXME: Inspect effect on assembly, then add benchmark-like tests
//        automatically testing the optimization inhibition effect

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    fn test_value(x: impl Pessimize + Copy + Debug + PartialEq) {
        let old_x = x;
        x.assume_read();
        assert_eq!(x, old_x);
        assert_eq!(super::hide(x), old_x);
    }

    #[test]
    fn values() {
        test_value(i8::MIN);
        test_value(i8::MAX);
        test_value(u8::MIN);
        test_value(u8::MAX);
        test_value(i16::MIN);
        test_value(i16::MAX);
        test_value(u16::MIN);
        test_value(u16::MAX);
        test_value(i32::MIN);
        test_value(i32::MAX);
        test_value(u32::MIN);
        test_value(u32::MAX);
        test_value(i64::MIN);
        test_value(i64::MAX);
        test_value(u64::MIN);
        test_value(u64::MAX);
        test_value(isize::MIN);
        test_value(isize::MAX);
        test_value(usize::MIN);
        test_value(usize::MAX);
        test_value(f32::MIN);
        test_value(f32::MAX);
        test_value(f64::MIN);
        test_value(f64::MAX);
    }

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

    const VALUE: usize = usize::MAX;

    unsafe fn test_pointer(p: impl Pessimize + PessimizeRef + UnsafeDeref<Target = usize>) {
        p.assume_read();
        assert_eq!(*p.unsafe_deref(), VALUE);
        p.assume_accessed();
        assert_eq!(*p.unsafe_deref(), VALUE);
        let p = super::hide(p);
        assert_eq!(*p.unsafe_deref(), VALUE);
    }

    fn function() -> usize {
        VALUE
    }

    #[test]
    fn pointers() {
        let mut value = VALUE;
        unsafe {
            test_pointer(&value as *const _);
            test_pointer(&mut value as *mut _);
            test_pointer(&value);
            test_pointer(&mut value);
        }
        (&function).assume_read();
        assert_eq!(function(), VALUE);
        (&function).assume_accessed();
        assert_eq!(function(), VALUE);
        let function2 = super::hide(&function);
        assert_eq!(function2(), VALUE);
    }
}
