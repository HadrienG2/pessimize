//! This crate aims to implement minimally costly optimization barriers for
//! every architecture that has `asm!()` support (currently x86(_64),
//! 32-bit ARM, AArch64 and RISC-V).
//!
//! You can use these barriers to prevent the compiler from optimizing out
//! selected redundant or unnecessary computations in situations where such
//! optimization is undesirable, like microbenchmarking.
//!
//! The barriers will be implemented for any type from core/std that either...
//! - Can be shoved into a CPU register (or a small set thereof), with a
//!   "natural" target register dictated by normal ABI calling conventions.
//! - Can be losslessly converted back and forth to a set of values that have
//!   the above property, at no runtime cost.
//!
//! Any type which is not directly supported can still be subjected to an
//! optimization barrier by taking a reference to it and subjecting that
//! reference to an optimization barrier, at the cost of causing the value to
//! be spilled to memory. If the `default_impl` feature is enabled, the crate
//! will provide a default `Pessimize` impl that does this for you.
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

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

// TODO: mod alloc
pub mod arch;
// TODO: mod boxed
// TODO: mod cell
// TODO: mod ffi
// TODO: mod fmt (for fmt::Error)
// TODO: mod fs (for File, at least)
// TODO: mod io (for Error, at least)
// TODO: mod marker (PhantomData and PhantomPinned)
// TODO: mod mem (ManuallyDrop)
// TODO: mod net (IpvNAddr, SocketAddrVN)
// TODO: mod num (NonZeroXyz, Wrapping)
// TODO: mod ops (RangeXyz)
// TODO: mod path
// TODO: mod pin
mod primitive;
// TODO: mod process (ExistStatus with ExitStatusExt)
mod ptr;
// TODO: mod string
// TODO: mod sync (at least atomics)
// TODO: mod task
// TODO: mod time
// TODO: mod vec

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
/// the input value unaltered) and `assume_xyz()` not altering anything even if
/// the type is internally mutable.
///
pub unsafe trait Pessimize {
    /// See `pessimize::hide()` for documentation
    fn hide(self) -> Self;

    /// See `pessimize::assume_read()` for documentation
    fn assume_read(&self);

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
/// On pointers/references, it will have the side effect of spilling target data
/// resident in CPU registers to memory, although the in-register copies remain
/// valid and can be reused later on without reloading.
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
/// This operation only makes sense on pointers/references, or values that
/// contain them. On others, it is equivalent to `assume_read()`. That's because
/// we assume that the pointer's target has changed, not the pointer itself
/// (except for fat pointer types where pointer metadata is target-dependent).
///
/// At the optimizer level, `assume_accessed()` will cause all target data which
/// is currently cached in registers to be spilled to memory and invalidated.
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
pub fn assume_accessed<R: Pessimize>(r: &mut R) {
    Pessimize::assume_accessed(r)
}

/// Variant of `assume_accessed` for internally mutable types
///
/// You should only use this variant on pointers/references to internally
/// mutable types (Cell, RefCell, Mutex, AtomicXyz...), or values that contain
/// them. Otherwise you will instantly fall victim of the "shared reference
/// mutation is UB" edge case mentioned in the docs of `assume_accessed()`.
///
/// For example, calling `assume_accessed_imut()` on a slice pointer will not
/// assume that the slice length has changed, since slice length does not have
/// internal mutability semantics.
///
#[inline(always)]
pub fn assume_accessed_imut<R: Pessimize>(r: &R) {
    Pessimize::assume_accessed_imut(r)
}

/// Convert Self back and forth to a type that implements Pessimize
///
/// While only a small number of Pessimize types are supported in hardware, many
/// standard types can be losslessly converted to a lower-level type (or tuple
/// of types) that implement Pessimize and back at no runtime cost.
///
/// This trait exposes that capability under a common abstraction vocabulary,
/// for the purpose of automatically implementing `Pessimize`.
///
/// # Safety
///
/// By implementing this trait, you guarantee that someone using it as
/// documented will not trigger Undefined Behavior, since the safe `Pessimize`
/// trait will be automatically implemented on top of it
///
pub unsafe trait PessimizeCast {
    /// Associated Pessimize type
    type Pessimized: Pessimize;

    /// Convert to Pessimized
    fn into_pessimize(self) -> Self::Pessimized;

    /// Convert back from Pessimized
    ///
    /// # Safety
    ///
    /// This operation should be safe for the intended use case of converting
    /// `Self` to `Pessimized` using `into_pessimize()`, invoking `Pessimize`
    /// trait operations on the resulting value, and optionally converting the
    /// `Pessimized` value back to `Self` afterwards via `from_pessimize()`.
    ///
    /// The final `from_pessimize()` operation of this round trip should be
    /// performed in the same scope where the initial `into_pessimize()`
    /// operation was called, or a child scope thereof.
    ///
    /// Even if `Pessimized` is `Clone`, it is strongly advised to treat it as
    /// a `!Clone` value: don't clone it, and stop using it after converting it
    /// back to `Self`. Otherwise, _suprising_ (but safe) behavior may occur.
    ///
    /// Nothing else is guaranteed to work. To give a few examples...
    ///
    /// - `Self` may contain references to the surrounding stack frame, so even
    ///   if `Pessimized` is `'static`, letting a `Pessimized` escape the scope
    ///   in which `into_pessimize()` was called before converting it back to
    //    `Self` using `from_pessimize()` is unsafe.
    /// - `Self` may contain `!Clone` data like &mut references, so even if
    ///   `Pessimized` is `Clone`, converting two clones of a single
    ///   `Pessimized` value back into `Self` is unsafe. In fact, even using the
    ///   `Pessimize` implementation after converting one of the clones to
    ///   `Self` is not guaranteed to produce the desired optimization barrier.
    /// - `Self` may be `!Send`, so even if `Pessimized` is `Send`, sending a
    ///   `Pessimized` value to another thread before calling `from_pessimize()`
    ///   on that separate thread is unsafe.
    /// - Even if two types share the same `Pessimized` representation, abusing
    ///   the `PessimizeCast` trait to perform a cast operation, like casting a
    ///   reference to another reference with different mutability or lifetime,
    ///   is unsafe.
    ///
    unsafe fn from_pessimize(x: Self::Pessimized) -> Self;
}

/// Process a type as its Pessimize dual by shared reference
pub trait BorrowPessimize: PessimizeCast {
    /// Process shared self as a shared Pessimize value
    fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized));

    /// Process unique self as a unique Pessimize value
    ///
    /// Be sure to propagate the updates performed to Self::Pessimized into
    /// the original &mut self reference at the end.
    ///
    /// # Safety
    ///
    /// `f` should only perform operations allowed before `from_pessimize`.
    ///
    unsafe fn with_pessimize_mut(&mut self, f: impl FnOnce(&mut Self::Pessimized));
}

/// Implementation of BorrowPessimize::with_pessimize for Copy types
// TODO: Use specializable BorrowPessimize impl once available on stable
#[inline(always)]
pub fn with_pessimize_copy<T: Copy + PessimizeCast>(self_: &T, f: impl FnOnce(&T::Pessimized)) {
    let pessimize = T::into_pessimize(*self_);
    f(&pessimize)
}

/// Implementation of BorrowPessimize::with_pessimize_mut for types where there
/// is a cheap way to get a T from an &mut T (typically Copy or core::mem::take)
///
/// # Safety
///
/// `f` must only contain operations allowed before `T::from_pessimize`.
///
#[inline(always)]
pub unsafe fn with_pessimize_mut_impl<T: PessimizeCast>(
    r: &mut T,
    f: impl FnOnce(&mut T::Pessimized),
    extract: impl FnOnce(&mut T) -> T,
) {
    let x: T = extract(r);
    let mut pessimize = T::into_pessimize(x);
    f(&mut pessimize);
    *r = T::from_pessimize(pessimize);
}

// Given a BorrowPessimize impl, we can automatically implement Pessimize
unsafe impl<T: BorrowPessimize> Pessimize for T {
    #[inline(always)]
    fn hide(self) -> Self {
        // Safe because `from_pessimize` is from the same scope and `hide` is
        // allowed before `from_pessimize`.
        unsafe { Self::from_pessimize(hide(self.into_pessimize())) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        Self::with_pessimize(self, assume_read)
    }

    #[inline(always)]
    fn assume_accessed(&mut self) {
        // Safe because `assume_accessed` is allowed before `from_pessimize`
        unsafe {
            Self::with_pessimize_mut(self, assume_accessed);
        }
    }

    #[inline(always)]
    fn assume_accessed_imut(&self) {
        Self::with_pessimize(self, assume_accessed_imut)
    }
}

/// Default implementation of Pessimize when no better one is available
///
/// Uses the implementation of `Pessimize` for references, which will cause any
/// state currently cached in CPU registers to be spilled to memory and reloaded
/// if used again.
///
#[cfg(feature = "default_impl")]
#[doc(cfg(all(feature = "nightly", feature = "default_impl")))]
mod default_impl {
    use super::*;

    unsafe impl<T> Pessimize for T {
        #[inline(always)]
        default fn hide(mut self) -> Self {
            let mut r: &mut Self = &mut self;
            assume_accessed::<&mut T>(&mut r);
            self
        }

        #[inline(always)]
        default fn assume_read(&self) {
            consume::<&T>(self)
        }

        #[inline(always)]
        default fn assume_accessed(mut self: &mut Self) {
            assume_accessed::<&mut T>(&mut self);
        }

        #[inline(always)]
        default fn assume_accessed_imut(&self) {
            assume_accessed_imut::<&T>(&self)
        }
    }
}

/// Implementation of Pessimize for values without pointer semantics
///
/// To be used by arch-specific modules to implement Pessimize for primitive and
/// arch-specific SIMD types.
///
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
            unsafe impl $crate::Pessimize for $value_type {
                #[inline(always)]
                fn hide(mut self) -> Self {
                    unsafe {
                        core::arch::asm!("/* {0} */", inout($reg) self, options(preserves_flags, nostack, nomem, pure));
                    }
                    self
                }

                #[inline(always)]
                fn assume_read(&self) {
                    unsafe {
                        core::arch::asm!("/* {0} */", in($reg) *self, options(preserves_flags, nostack, nomem))
                    }
                }

                #[inline(always)]
                fn assume_accessed(&mut self) {
                    Self::assume_read(self)
                }

                #[inline(always)]
                fn assume_accessed_imut(&self) {
                    Self::assume_read(self)
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
            unsafe impl $crate::PessimizeCast for $simd_type {
                type Pessimized = $inner;

                #[inline(always)]
                fn into_pessimize(self) -> $inner {
                    self.into()
                }

                #[inline(always)]
                unsafe fn from_pessimize(x: $inner) -> Self {
                    Self::from(x)
                }
            }
            //
            #[$doc_cfg]
            impl $crate::BorrowPessimize for $simd_type {
                #[inline(always)]
                fn with_pessimize(&self, f: impl FnOnce(&$inner)) {
                    $crate::with_pessimize_copy(self, f)
                }

                #[inline(always)]
                unsafe fn with_pessimize_mut(&mut self, f: impl FnOnce(&mut $inner)) {
                    $crate::with_pessimize_mut_impl(self, f, core::mem::take)
                }
            }
        )*)*
    };
}

// Although all Rust collections are basically pointers with extra metadata, we
// may only implement Pessimize for them when all the metadata is exposed and
// there is a way to build a collection back from all the raw parts
//
// TODO: Once allocator_api is stable, support collections with custom
//       allocators by applying optimization barriers to the allocator as well.
//       Right now, doing so would require either duplicating the tricky
//       collection code (one version with allocators and one version without)
//       or dropping collection support on stable, neither of which sound
//       satisfactory given that custom allocators are expected to be niche.
//
#[cfg(any(feature = "alloc", test))]
mod alloc_feature {
    use super::*;
    use alloc::{boxed::Box, string::String, vec::Vec};
    use core::mem::ManuallyDrop;

    // Box<T> is effectively a NonNull<T> with RAII, so if the NonNull impl is
    // correct, this impl is correct.
    unsafe impl<T: ?Sized> PessimizeCast for Box<T>
    where
        *const T: Pessimize,
    {
        type Pessimized = *const T;

        #[inline(always)]
        fn into_pessimize(self) -> *const T {
            Box::into_raw(self) as *const T
        }

        #[inline(always)]
        unsafe fn from_pessimize(x: *const T) -> Self {
            Box::from_raw(x as *mut T)
        }
    }
    //
    impl<T: ?Sized> BorrowPessimize for Box<T>
    where
        *const T: Pessimize,
    {
        #[inline(always)]
        fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
            let inner: &T = self.as_ref();
            f(&(inner as *const T))
        }

        #[inline(always)]
        unsafe fn with_pessimize_mut(&mut self, f: impl FnOnce(&mut Self::Pessimized)) {
            let inner: &mut T = self.as_mut();
            let mut inner_ptr = inner as *const T;
            f(&mut inner_ptr);
            (self as *mut Self).write(Self::from_pessimize(inner_ptr))
        }
    }

    // Vec<T> is basically a thin NonNull<T>, a length and a capacity
    unsafe impl<T> PessimizeCast for Vec<T> {
        type Pessimized = (*const T, usize, usize);

        #[inline(always)]
        fn into_pessimize(self) -> Self::Pessimized {
            let mut v = ManuallyDrop::new(self);
            (v.as_mut_ptr() as *const T, v.len(), v.capacity())
        }

        #[inline(always)]
        unsafe fn from_pessimize((ptr, length, capacity): Self::Pessimized) -> Self {
            Vec::from_raw_parts(ptr as *mut T, length, capacity)
        }
    }
    //
    impl<T> BorrowPessimize for Vec<T> {
        #[inline(always)]
        fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
            let pessimized = (self.as_ptr(), self.len(), self.capacity());
            f(&pessimized)
        }

        #[inline(always)]
        unsafe fn with_pessimize_mut(&mut self, f: impl FnOnce(&mut Self::Pessimized)) {
            with_pessimize_mut_impl(self, f, core::mem::take)
        }
    }

    // String is basically a Vec<u8> with an UTF-8 validity invariant
    unsafe impl PessimizeCast for String {
        type Pessimized = (*const u8, usize, usize);

        #[inline(always)]
        fn into_pessimize(self) -> Self::Pessimized {
            self.into_bytes().into_pessimize()
        }

        #[inline(always)]
        unsafe fn from_pessimize(x: Self::Pessimized) -> Self {
            String::from_utf8_unchecked(Vec::<u8>::from_pessimize(x))
        }
    }
    //
    impl BorrowPessimize for String {
        #[inline(always)]
        fn with_pessimize(&self, f: impl FnOnce(&Self::Pessimized)) {
            let pessimized = (self.as_ptr(), self.len(), self.capacity());
            f(&pessimized)
        }

        #[inline(always)]
        unsafe fn with_pessimize_mut(&mut self, f: impl FnOnce(&mut Self::Pessimized)) {
            with_pessimize_mut_impl(self, f, core::mem::take)
        }
    }
}

// TODO: Implement Pessimize for internally mutable types with
//       no hidden state (UnsafeCell, Cell, AtomicXyz).
// NOTE: hide() is easy: just call into_inner(), call the contained value's
//       Pessimize::hide() impl, and get back to the original type via new().
//       But assume_read() is tricky: in the current state of Rust's mutability
//       rules, it is dangerous to create an &T to the inside of an internally
//       mutable type without knowing about the existence of concurrent refs to
//       said data. Consider going via the consume(&self) route in the
//       beginning, with a TODO suggesting acquisition of a shared reference to
//       the insides once more clearly allowed by UCG

// TODO: Impl + tests: NonZero, char, PhantomData, PhantomPinned, ManuallyDrop,
//       IpVxAddr, SocketAddrVx, Wrapping<T>, Range*, AssertUnwindSafe, OsStr,
//       Path, PathBuf, Pin, task::{RawWaker, Waker, Context}, time::{Duration, Instant}...

// TODO: Provide a Derive macro to derive Pessimize for a small struct, with a
//       warning that it will do more harm than good on a larger struct

// TODO: Set up CI in the spirit of test-everything.sh

// FIXME: Test new types: str (nightly), trait objects (nightly),
//        Box<T> where *const T: Pessimize (alloc),
//        Vec<T> (alloc), String (alloc) and internally mutable types with no
//        internal state (remember to check assume_accessed_imut for those)
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ptr::tests::{test_all_pointers, test_unoptimized_ptrs};
    #[cfg(feature = "nightly")]
    use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
    use std::{
        fmt::Debug,
        time::{Duration, Instant},
    };

    // === Tests asserting that the barriers don't modify anything ===
    // ===    (should be run on both debug and release builds)     ===

    // Test that, for a given value, Pessimize seems to work
    fn test_value<T: Clone + Debug + PartialEq + Pessimize>(x: T) {
        let old_x = x.clone();
        assume_read(&x);
        assert_eq!(x, old_x);
        assert_eq!(hide(x.clone()), old_x);
        test_all_pointers::<T, _>(x.clone());
        consume(x);
    }

    // Run test_value on the minimal, default and maximal value of a type
    pub fn test_value_type<T: Clone + Debug + Default + PartialEq + Pessimize>(min: T, max: T) {
        test_value(min);
        test_value(T::default());
        test_value(max);
    }

    // Run test_value_type for an encapsulated SIMD type
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

    // Run test_value_type for a portable_simd::Simd type
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

    /// Maximum degree of instruction-level parallelism for any instruction
    const MAX_SUPERSCALAR_WAYS: u64 = 4;

    /// Maximum realistic number of empty loop iterations per second
    ///
    /// At the time of writing, CPU boost clocks commonly go above 5 GHz and
    /// 8-9 GHz has been observed in overclocking records 10 years ago, it seems
    /// unlikely that these records will be beaten anytime soon so 10 GHz is a
    /// good clock frequency upper bound.
    ///
    /// While most processors can only process 1 conditional jump per second, it
    /// is possible to go all the way up to the maximal ILP integer increment
    /// rate through loop unrolling.
    ///
    const MAX_LOOP_FREQ: u64 = MAX_SUPERSCALAR_WAYS * 10_000_000_000;

    /// Maximum expected clock granularity
    /// The system clock is expected to always be able to measure this duration
    const MIN_DURATION: Duration = Duration::from_millis(2);

    /// Minimum number of loop iterations for which a loop duration greater than
    /// or equal to MIN_DURATION should be measured
    const MIN_ITERATIONS: u64 = 2 * MAX_LOOP_FREQ * MIN_DURATION.as_nanos() as u64 / 1_000_000_000;

    // Measure the time it takes to run something in a loop
    fn time_loop(mut op: impl FnMut(u64)) -> Duration {
        let start = Instant::now();
        for iter in 0..MIN_ITERATIONS {
            op(iter);
        }
        start.elapsed()
    }

    // Measure time to run an empty loop (counter increment and nothing else),
    // check that it is within expectations.
    fn checked_empty_loop_duration() -> Duration {
        // Any architecture with 64-bit pointers can store 64-bit integers in
        // registers and thus has native u64: Pessimize. On other arches, we
        // consume a reference to the loop counter, which is always valid but
        // causes it to be spilled to memory.
        #[cfg(target_pointer_width = "64")]
        let elapsed = time_loop(consume);
        #[cfg(not(target_pointer_width = "64"))]
        let elapsed = time_loop(|iter| consume(&iter));
        assert!(elapsed >= MIN_DURATION);
        elapsed
    }

    // Make sure that an operation was not optimized out
    //
    // An input value is initially provided. This input is passed down to the
    // operation, and whatever the operation emits will be the next input. The
    // output of the last operation is emitted as the output of the test. This
    // allows running tests with values that cannot or should not be cloned.
    //
    pub fn assert_unoptimized<T>(input: T, mut op: impl FnMut(T) -> T) -> T {
        // Run the operation in a loop
        let mut opt = Some(input);
        let elapsed = time_loop(|_iter| {
            let mut input = opt.take().unwrap();
            // For each loop iteration, we perform more of the requested
            // operation than can be performed in a single CPU cycle. Therefore,
            // if the operation is not optimized out, each loop iteration should
            // take at least one more CPU cycle.
            for _ in 0..=MAX_SUPERSCALAR_WAYS {
                input = op(input);
            }
            opt = Some(input);
        });

        // Immediately check empty loop iteration speed to evaluate clock rate,
        // which can vary depending on the operation we're doing.
        // Since a loop iteration takes at most 1 CPU clock cycle on modern
        // CPUs, with >1 extra cycle, the loop should be at least 2x slower.
        let elapsed_empty = checked_empty_loop_duration();
        let ratio = elapsed.as_secs_f64() / elapsed_empty.as_secs_f64();
        assert!(ratio > 1.9);

        // Next, deduce the actual rate at which operations are being executed
        // and compare that to the rate at which empty loop iterations execute.
        eprintln!(
            "Operation pessimized (running at {:.1}x empty iteration speed of {:.1} GHz)",
            (MAX_SUPERSCALAR_WAYS + 1) as f64 / ratio,
            MIN_ITERATIONS as f64 / elapsed_empty.as_nanos() as f64
        );

        // Finally, return the last output for the next benchmark test
        opt.take().unwrap()
    }

    // --- Tests for values with native Pessimize support ---

    fn test_unoptimized_value<T: Clone + PartialEq + Pessimize>(x: T) {
        let old_x = x.clone();
        assert_unoptimized(x, |mut x| {
            x = hide(x);
            consume(x == old_x);
            x
        });
    }
    //
    pub fn test_unoptimized_value_type<T: Clone + Default + PartialEq + Pessimize>() {
        test_unoptimized_value(T::default());
    }

    // === Tests for types implemented here ===

    // --- "Big" array with no native Pessimize implementation ---

    // What is considered too big (in units of isize)
    // 2 is enough as we don't currently pessimize larger arrays
    const BIG: usize = 2;

    // Should be run on both debug and release builds
    #[test]
    fn non_native() {
        #[cfg(feature = "default_impl")]
        test_value_type::<[isize; BIG]>([isize::MIN; BIG], [isize::MAX; BIG]);
        #[cfg(not(feature = "default_impl"))]
        for inner in [isize::MIN, 0, isize::MAX] {
            test_all_pointers::<[isize; BIG], _>([inner; BIG]);
        }
    }

    // Should only be run on release builds
    #[test]
    #[ignore]
    fn non_native_optim() {
        #[cfg(feature = "default_impl")]
        test_unoptimized_value_type::<[isize; BIG]>();
        #[cfg(not(feature = "default_impl"))]
        test_unoptimized_ptrs::<[isize; BIG], _>([0isize; BIG]);
    }

    // --- Fat pointer types ---

    #[cfg(feature = "nightly")]
    mod dst {
        use super::*;

        fn make_boxed_slice(value: isize) -> Box<[isize]> {
            vec![value; BIG].into()
        }

        // Should be run on both debug and release builds
        #[test]
        fn boxed_slice() {
            // Test boxed slices as owned values
            test_value_type::<Box<[isize]>>(
                make_boxed_slice(isize::MIN),
                make_boxed_slice(isize::MAX),
            );

            // Test slice pointers, using boxed slices as owned storage
            for inner in [isize::MIN, 0, isize::MAX] {
                test_all_pointers::<[isize], _>(make_boxed_slice(inner))
            }
        }

        // Should only be run on release builds
        #[test]
        #[ignore]
        fn boxed_slice_optim() {
            test_unoptimized_value(make_boxed_slice(0));
            test_unoptimized_ptrs::<[isize], _>(make_boxed_slice(0));
        }

        // TODO: Do the same with str and dyn Trait
    }

    // TODO: Test thin Box and Vec
}
