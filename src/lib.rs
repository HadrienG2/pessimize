//! This crate aims to implement minimally costly optimization barriers for
//! every architecture that has asm!() support (currently x86(_64), 32-bit ARM,
//! AArch64 and RISC-V).
//!
//! You can use these barriers to prevent the compiler from optimizing out
//! selected redundant or unnecessary computations in situations where such
//! optimization is undesirable, like microbenchmarking.
//!
//! For a fully implemented architecture, the barriers will be implemented for
//! - Primitive integers (iN and uN except for 128-bit, isize and usize)
//! - Primitive floats (f32 and f64)
//! - Thin pointers and references (basically &T-like other than &[T] or &dyn T)
//! - Function pointers
//! - SIMD vector types (with support for std::simd on nightly)
//!
//! Any type which is not directly supported can still be subjected to an
//! optimization barrier by taking a reference to it and subjecting that
//! reference to an optimization barrier, at the cost of causing the value to
//! be spilled to memory.
//!
//! For pointer-like entities, optimization barriers other than black_box will
//! have the side-effect of causing the compiler to assume that global and
//! thread-local variable might have been accessed using similar semantics as
//! the pointer itself. This will reduce applicable compiler optimizations for
//! such variables. If you use those, you may want to only use black_box.

#![cfg_attr(not(test), no_std)]
#![deny(missing_docs)]

// Each architecture-specific module is tasked to implement Unoptimize for
// primitive integers and floats, raw pointers, and SIMD vector types.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
// TODO: Add support for 32-bit ARM, AArch64 and RISC-V (check calling
//       conventions to know where floats should be put)

/// Optimization barriers with value semantics
///
/// Implemented for all the types described in the crate documentation
///
pub trait Unoptimize {
    /// Force the compiler to assume that a value, and data transitively
    /// reachable via that value if it is a pointer, is being read by something.
    ///
    /// Apply this barrier to unused computation results in order to prevent the
    /// compiler from optimizing out the associated computations.
    ///
    fn assume_read(&self);

    /// Re-emit the input value as its output (identity function), but force the
    /// compiler to assume that it is a completely different value.
    ///
    /// If you want to re-do the exact same computation in a loop, pass its
    /// inputs through this barrier to prevent the compiler from optimizing out
    /// the redundant computations.
    ///
    fn black_box(self) -> Self;
}

/// Optimization barriers for thin pointers and references
///
/// Implemented only for &T, &mut T, *const T and *mut T where T: Sized.
///
pub trait UnoptimizeRef {
    /// Force the compiler to assume that any data transitively reachable via a
    /// pointer/reference has been read, and modified if Rust allows for it.
    ///
    /// The compiler will not assume modification of data which is reachable
    /// via an &-reference and does not have interior mutability semantics.
    ///
    /// You can use `(&mut x).assume_accessed(); x` as an alternative to
    /// `black_box(x)` when x does not implement Unoptimize, at the cost of
    /// forcing the spilling of x's contents to memory if they are currently
    /// resident in CPU registers.
    ///
    fn assume_accessed(&self);
}

/// Force the compiler to assume that a value, and data transitively
/// reachable via that value, is being read by something.
///
/// Apply this barrier to unused computation results in order to prevent the
/// compiler from optimizing out the associated computations.
///
#[inline(always)]
pub fn assume_read<T: Unoptimize>(x: &T) {
    x.assume_read()
}

/// Re-emit the input value as its output (identity function), but force the
/// compiler to assume that it is a completely different value.
///
/// If you want to re-do the exact same computation in a loop, pass its
/// inputs through this barrier to prevent the compiler from optimizing out
/// the redundant computations.
///
#[inline(always)]
pub fn black_box<T: Unoptimize>(x: T) -> T {
    x.black_box()
}

/// Force the compiler to assume that any data transitively reachable via a
/// pointer or reference has been read, and modified if Rust allows for it.
///
/// The compiler will not assume modification of data which is only reachable
/// via an &-reference and does not have interior mutability semantics.
///
/// You can use `assume_accessed(&mut x); x` as an alternative to `black_box(x)`
/// when x does not implement Unoptimize, at the cost of forcing the spilling of
/// x's contents to memory if they are currently resident in CPU registers.
///
#[inline(always)]
pub fn assume_accessed<R: UnoptimizeRef>(r: &mut R) {
    r.assume_accessed()
}

// Implementation of Unoptimize and UnoptimizeRef for references
macro_rules! unoptimize_references {
    ($($t:ty),*) => {
        $(
            impl<'a, T: Sized> Unoptimize for $t {
                #[inline(always)]
                fn assume_read(&self) {
                    (*self as *const T).assume_read()
                }

                #[inline(always)]
                fn black_box(self) -> Self {
                    unsafe {
                        core::mem::transmute((self as *const T).black_box())
                    }
                }
            }

            impl<'a, T: Sized> UnoptimizeRef for $t {
                #[inline(always)]
                fn assume_accessed(&self) {
                    (*self as *const T).assume_accessed()
                }
            }
        )*
    };
}
//
unoptimize_references!(&'a T, &'a mut T);

// FIXME: Inspect effect on assembly, then add benchmark-like tests
//        automatically testing the optimization inhibition effect

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    fn test_value(x: impl Unoptimize + Copy + Debug + PartialEq) {
        let old_x = x;
        x.assume_read();
        assert_eq!(x, old_x);
        assert_eq!(super::black_box(x), old_x);
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

    unsafe fn test_pointer(p: impl Unoptimize + UnoptimizeRef + UnsafeDeref<Target = usize>) {
        p.assume_read();
        assert_eq!(*p.unsafe_deref(), VALUE);
        p.assume_accessed();
        assert_eq!(*p.unsafe_deref(), VALUE);
        let p = super::black_box(p);
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
        let function2 = super::black_box(&function);
        assert_eq!(function2(), VALUE);
    }
}
