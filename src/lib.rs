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
//! - Thin pointers and references (`&T`-like other than `&[T]` `&str` and
//!   `&dyn T`, including function pointers), and fat pointers too on nightly
//! - SIMD vector types (with optional support for `safe_arch` and `core::simd`
//!   via feature flags)
//! - Small tuples of these types.
//!
//! Some legacy and embedded architectures will not support 64-bit primitive
//! types. The rule of thumb is, if your target CPU can fit primitive type T in
//! architectural registers, then that type should implement Pessimize.
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

// Each architecture-specific module is tasked to implement Pessimize for
// primitive integers, floats and SIMD vector types.
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;
// TODO: On nightly, support more arches via asm_experimental_arch

use core::ptr::NonNull;

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
/// the input value unaltered) and assume_read not altering anything even if
/// the type is internally mutable.
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
/// # Safety
///
/// Unsafe code may rely on assume_accessed and assume_accessed_imut not
/// altering anything even if the type is internally mutable.
///
pub unsafe trait PessimizeRef: Pessimize {
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
/// For example, calling assume_accessed_imut on a slice pointer will not assume
/// that the slice length has changed, since slice length does not have
/// internal mutability semantics.
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
        assume_accessed(&mut &mut self);
        self
    }

    #[inline(always)]
    default fn assume_read(&self) {
        consume(self)
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
            unsafe impl Pessimize for $value_type {
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
            }
        )*)*
    };
}

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
                    $crate::hide($inner::from(self)).into()
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
        core::arch::asm!("/* {0} */", inout(reg) x, options(preserves_flags, nostack, nomem, pure));
    }
    x
}
//
#[allow(asm_sub_register)]
#[inline(always)]
fn assume_read_thin_ptr<T: Sized>(x: *const T) {
    unsafe { core::arch::asm!("/* {0} */", in(reg) x, options(preserves_flags, nostack, readonly)) }
}
//
#[allow(asm_sub_register)]
#[inline(always)]
fn assume_accessed_thin_ptr<T: Sized>(x: *mut T) {
    unsafe { core::arch::asm!("/* {0} */", in(reg) x, options(preserves_flags, nostack)) }
}
//
#[cfg(not(feature = "nightly"))]
mod thin_pointers {
    use super::*;

    unsafe impl<T: Sized> Pessimize for *const T {
        #[inline(always)]
        fn hide(self) -> Self {
            hide_thin_ptr(self)
        }

        #[inline(always)]
        fn assume_read(&self) {
            assume_read_thin_ptr(*self)
        }
    }

    unsafe impl<T: Sized> PessimizeRef for *const T {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            assume_accessed_thin_ptr(*self as *mut T)
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            assume_accessed_thin_ptr(*self as *mut T)
        }
    }
}
//
#[cfg(feature = "nightly")]
mod all_pointers {
    use super::*;
    use core::ptr::{self, DynMetadata, Pointee};

    unsafe impl<T: ?Sized> Pessimize for DynMetadata<T> {
        #[inline(always)]
        fn hide(self) -> Self {
            // Safe because hide() is guarenteed to be the identity function
            unsafe { core::mem::transmute(hide(core::mem::transmute::<_, usize>(self))) }
        }

        #[inline(always)]
        fn assume_read(&self) {
            // Safe because DynMetadata is a pointer, which is convertible to usize
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

    unsafe impl<T: core::ptr::Pointee + ?Sized> PessimizeRef for *const T
    where
        T::Metadata: Pessimize,
    {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_accessed_thin_ptr(thin as *mut ());
            // Correct because hide is identity and assume_accessed doesn't modify
            *self = ptr::from_raw_parts(thin, hide(metadata))
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_accessed_thin_ptr(thin as *mut ());
            consume(metadata);
        }
    }
}
//
unsafe impl<T: ?Sized> Pessimize for *mut T
where
    *const T: Pessimize,
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
//
unsafe impl<T: ?Sized> PessimizeRef for *mut T
where
    *const T: PessimizeRef,
{
    #[inline(always)]
    fn assume_accessed(&mut self) {
        // Safe because *mut T is effectively the same type as *const T and
        // Rust does not have C-style strict aliasing rules.
        unsafe { assume_accessed::<*const T>(core::mem::transmute(self)) }
    }

    #[inline(always)]
    fn assume_accessed_imut(&self) {
        assume_accessed_imut(&(*self as *const T))
    }
}
//
unsafe impl<T: ?Sized> Pessimize for NonNull<T>
where
    *mut T: Pessimize,
{
    #[inline(always)]
    fn hide(self) -> Self {
        // Safe because hide is guarenteed to be the identity function
        unsafe { NonNull::new_unchecked(hide(self.as_ptr())) }
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume(self.as_ptr())
    }
}
//
unsafe impl<T: ?Sized> PessimizeRef for NonNull<T>
where
    *mut T: PessimizeRef,
{
    #[inline(always)]
    fn assume_accessed(&mut self) {
        let mut ptr: *mut T = self.as_ptr();
        assume_accessed(&mut ptr);
        // Safe because assume_accessed doesn't change anything
        *self = unsafe { NonNull::new_unchecked(ptr) };
    }

    #[inline(always)]
    fn assume_accessed_imut(&self) {
        assume_accessed_imut(&self.as_ptr())
    }
}

// Implementation of Pessimize and PessimizeRef for references
macro_rules! pessimize_references {
    (
        $(
            ($ref_t:ty, $from_nonnull:ident)
        ),*
    ) => {
        $(
            unsafe impl<'a, T: ?Sized> Pessimize for $ref_t
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
                        // This won't happen here since we drop the original
                        // &mut w/o using it before returning the new one.
                        core::mem::transmute(hide(self as *const T))
                    }
                }

                #[inline(always)]
                fn assume_read(&self) {
                    let inner: &T = &**self;
                    consume(inner as *const T)
                }
            }

            unsafe impl<'a, T: ?Sized> PessimizeRef for $ref_t
                where *const T: PessimizeRef
            {
                #[allow(clippy::unnecessary_mut_passed)]
                #[inline(always)]
                fn assume_accessed(&mut self) {
                    let mut ptr: NonNull<T> = (*self).into();
                    assume_accessed(&mut ptr);
                    // Safe because assume_accessed doesn't change anything
                    *self = unsafe { NonNull::$from_nonnull(&mut ptr) };
                }

                #[inline(always)]
                fn assume_accessed_imut(&self) {
                    assume_accessed_imut(&((*self) as *const T))
                }
            }
        )*
    };
}
//
pessimize_references!((&'a T, as_ref), (&'a mut T, as_mut));

// Implementation of Pessimize for function pointers
macro_rules! pessimize_fn {
    ($res:ident $( , $args:ident )* ) => {
        unsafe impl< $res $( , $args )* > Pessimize for fn( $($args),* ) -> $res {
            #[inline(always)]
            fn hide(self) -> Self {
                // Safe because hide is guaranteed to be the identity function
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
// implement Pessimize for small arrays, we do not do so because it would
// force individual scalars to be moved to GP registers, which pessimizes SIMD

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

    // Box<T> is effectively a fancy NonNull<T>
    unsafe impl<T: ?Sized> Pessimize for Box<T>
    where
        *mut T: Pessimize,
    {
        #[inline(always)]
        fn hide(self) -> Self {
            // Safe because hide is the identity function
            unsafe { Box::from_raw(hide(Box::into_raw(self))) }
        }

        #[inline(always)]
        fn assume_read(&self) {
            let inner: &T = self.as_ref();
            consume(inner as *const T as *mut T)
        }
    }
    //
    unsafe impl<T: ?Sized> PessimizeRef for Box<T>
    where
        *mut T: PessimizeRef,
    {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            let inner: &mut T = self.as_mut();
            let mut inner_ptr = inner as *mut T;
            assume_accessed(&mut inner_ptr);
            // Safe because assume_accessed doesn't modify its target
            unsafe { (self as *mut Self).write(Box::from_raw(inner_ptr)) }
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            let inner: &T = self.as_ref();
            assume_accessed_imut(&(inner as *const T as *mut T))
        }
    }

    // Vec<T> is basically a thin NonNull<T>, a length and a capacity
    unsafe impl<T> Pessimize for Vec<T> {
        #[inline(always)]
        fn hide(self) -> Self {
            let mut v = ManuallyDrop::new(self);
            // Safe because self destructor has been inhibited and hide is
            // guaranteed to be the identity function
            unsafe { Vec::from_raw_parts(hide(v.as_mut_ptr()), hide(v.len()), hide(v.capacity())) }
        }

        #[inline(always)]
        fn assume_read(&self) {
            consume(self.as_ptr());
            consume(self.len());
            consume(self.capacity());
        }
    }
    //
    unsafe impl<T> PessimizeRef for Vec<T> {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            let ptr = self.as_mut_ptr();
            assume_accessed_thin_ptr(ptr);
            let length = hide(self.len());
            let capacity = hide(self.capacity());
            // Safe because self's destructor is inhibited through use of write,
            // hide is identity and assume_accessed doesn't modify its target
            unsafe { (self as *mut Self).write(Vec::from_raw_parts(ptr, length, capacity)) }
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            assume_accessed_thin_ptr(self.as_ptr() as *mut T);
            consume(self.len());
            consume(self.capacity());
        }
    }

    // String is basically a Vec<u8> with an UTF-8 validity invariant
    unsafe impl Pessimize for String {
        #[inline(always)]
        fn hide(self) -> Self {
            // Safe because hide is the identity function
            unsafe { String::from_utf8_unchecked(hide(self.into_bytes())) }
        }

        #[inline(always)]
        fn assume_read(&self) {
            consume(self.as_bytes() as *const [u8] as *const u8);
            consume(self.len());
            consume(self.capacity());
        }
    }
    //
    unsafe impl PessimizeRef for String {
        #[inline(always)]
        fn assume_accessed(&mut self) {
            // Safe because assume_accessed does not modify anything
            assume_accessed(unsafe { self.as_mut_vec() })
        }

        #[inline(always)]
        fn assume_accessed_imut(&self) {
            assume_read(self)
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

// FIXME: Test new types: str (nightly), trait objects (nightly) and tuples of
//        Pessimize values, Box<T> where *const T: Pessimize (alloc),
//        Vec<T> (alloc), String (alloc) and internally mutable types with no
//        internal state (remember to check assume_accessed_imut for those)
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "nightly")]
    use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
    use std::{
        borrow::{Borrow, BorrowMut},
        fmt::Debug,
        time::{Duration, Instant},
    };

    // === Tests asserting that the barriers don't modify anything ===
    // ===    (should be run on both debug and release builds)     ===

    // This may cause UB if the PessimizeRef implementation is wrong (hide is
    // not an identity function, assume_xyz modifies the target), but we try
    // extra hard to detect this situation.
    unsafe fn test_pointer<Value: Debug + PartialEq + ?Sized, Ptr: PtrLike<Target = Value>>(
        mut p: Ptr,
        expected_target: &Value,
    ) {
        let old_p = p.as_const_ptr();
        assert!(p.deep_eq(old_p, expected_target));
        p = hide(p);
        assert!(p.deep_eq(old_p, expected_target));
        assume_read(&p);
        assert!(p.deep_eq(old_p, expected_target));
        assume_accessed(&mut p);
        assert!(p.deep_eq(old_p, expected_target));
        assume_accessed_imut(&p);
        assert!(p.deep_eq(old_p, expected_target));
        consume(p)
    }

    fn test_all_pointers<T: Debug + PartialEq + ?Sized, OwnedT: Clone + Borrow<T> + BorrowMut<T>>(
        mut x: OwnedT,
    ) where
        *const T: PessimizeRef,
    {
        let old_x = x.clone();
        let old_x = old_x.borrow();
        unsafe {
            test_pointer(x.borrow() as *const T, old_x);
            test_pointer(x.borrow_mut() as *mut T, old_x);
            test_pointer(x.borrow(), old_x);
            test_pointer(x.borrow_mut(), old_x);
            test_pointer(NonNull::<T>::from(x.borrow_mut()), old_x);
        }
    }

    fn test_value<T: Clone + Debug + PartialEq + Pessimize>(x: T) {
        let old_x = x.clone();
        assume_read(&x);
        assert_eq!(x, old_x);
        assert_eq!(hide(x.clone()), old_x);
        test_all_pointers::<T, _>(x.clone());
        consume(x);
    }

    fn test_value_type<T: Clone + Debug + Default + PartialEq + Pessimize>(min: T, max: T) {
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
    fn assert_unoptimized<T>(input: T, mut op: impl FnMut(T) -> T) -> T {
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

    // --- Tests for pointers with PessimizeRef support

    // Check that one component of the output of hide() looks different
    //
    // The initial value of the component is provided, along with a way to
    // extract the component for a complete pointer and a way to get back the
    // complete pointer from a component value.
    //
    // # Safety
    //
    // - If Ptr is &mut, there should be no live &mut to the target
    // - extract and rebuild should honor their contract, in the sense that
    //   extract(rebuild(c)) == c && rebuild(extract(p)) == p.
    // - The output of rebuild (which is the original pointer) should be safe to
    //   cast to Ptr via PtrLike::from_const_ptr.
    //
    #[cfg(feature = "nightly")]
    unsafe fn test_unoptimized_ptr_component<
        Value: ?Sized,
        Ptr: PtrLike<Target = Value>,
        PtrComponent: Copy + PartialEq,
    >(
        component: PtrComponent,
        mut extract: impl FnMut(*const Value) -> PtrComponent,
        mut rebuild: impl FnMut(PtrComponent) -> *const Value,
    ) where
        *const Value: PessimizeRef,
    {
        // Check that hide() pretends to return a modified version of the
        // selected pointer component
        assert_unoptimized(component, |component| {
            let old_component = component;
            let const_ptr = rebuild(component);
            let fat_ptr = hide(Ptr::from_const_ptr(const_ptr));
            let component = extract(fat_ptr.to_const_ptr());
            consume(component == old_component);
            component
        });
    }
    //
    fn test_unoptimized_ptr<Value: PartialEq + ?Sized, Ptr: PtrLike<Target = Value>>(
        mut p: Ptr,
        expected_target: &Value,
    ) where
        *const Value: PessimizeRef,
    {
        // assume_accessed() makes the pointee look modified
        let old_p = p.as_const_ptr();
        assert_unoptimized((), |()| {
            assume_accessed(&mut p);
            // This is safe because assume_accessed doesn't modify anything
            consume(unsafe { p.deep_eq(old_p, expected_target) });
        });

        // On stable, we only have thin pointers to care about
        #[cfg(not(feature = "nightly"))]
        {
            // hide() seems to return a different pointer
            assert_unoptimized(p, |mut p| {
                p = hide(p);
                consume(p.as_const_ptr() == old_p);
                p
            });
        }

        // On nightly, we must account for fat pointers
        #[cfg(feature = "nightly")]
        {
            use core::ptr;

            // assert_unoptimized() seems to modify the pointer's metadata
            let extract_metdata = |p: &Ptr| p.as_const_ptr().to_raw_parts().1;
            let metadata = extract_metdata(&p);
            let nontrivial_metadata = core::mem::size_of_val(&metadata) > 0;
            if nontrivial_metadata {
                assert_unoptimized((), |()| {
                    assume_accessed(&mut p);
                    consume(extract_metdata(&p) == metadata);
                });
            }

            // In the output of hide(), both the thin pointer and metadata look
            // unrelated to that of the original pointer.
            //
            // This is safe because..
            // - If p is an &mut, it is moved away by to_const_ptr.
            // - The provided extract and rebuild impls are correct
            // - The pointer emitted by rebuild is basically p.as_const_ptr().
            //
            let thin_ptr = p.to_const_ptr().to_raw_parts().0;
            unsafe {
                test_unoptimized_ptr_component::<Value, Ptr, *const ()>(
                    thin_ptr,
                    |const_ptr| {
                        let (thin_ptr, _metadata) = const_ptr.to_raw_parts();
                        thin_ptr
                    },
                    |thin_ptr| ptr::from_raw_parts(thin_ptr, metadata),
                );
                if nontrivial_metadata {
                    test_unoptimized_ptr_component::<Value, Ptr, _>(
                        metadata,
                        |const_ptr| {
                            let (_thin_ptr, metadata) = const_ptr.to_raw_parts();
                            metadata
                        },
                        |metadata| ptr::from_raw_parts(thin_ptr, metadata),
                    );
                }
            }
        }
    }
    //
    fn test_unoptimized_ptrs<T: PartialEq + ?Sized, OwnedT: Clone + Borrow<T> + BorrowMut<T>>(
        mut x: OwnedT,
    ) where
        *const T: PessimizeRef,
    {
        let old_x = x.clone();
        let old_x = old_x.borrow();
        test_unoptimized_ptr(x.borrow() as *const T, old_x);
        test_unoptimized_ptr(x.borrow_mut() as *mut T, old_x);
        test_unoptimized_ptr(x.borrow(), old_x);
        test_unoptimized_ptr(x.borrow_mut(), old_x);
        test_unoptimized_ptr(NonNull::<T>::from(x.borrow_mut()), old_x);
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
        test_unoptimized_value_type::<bool>();
    }

    // === Test suite for types that only implement Pessimize by reference ===

    // --- Array too big to fit in a register ---

    // What is considered too big (in units of isize)
    // 2 is enough as we don't currently pessimize arrays
    const BIG: usize = 1;

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
        consume(fptr);
    }

    fn test_function_pointer_optim(fptr: fn() -> isize) {
        let expected_result = fptr();
        assert_unoptimized(fptr, |mut fptr| {
            fptr = hide(fptr);
            consume(fptr() == expected_result);
            fptr
        });
        consume(fptr())
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

    // TODO: Test tuples, thin Box and Vec

    // === Uninteresting helpers ===

    // Abstraction layer to handle references and pointers homogeneously
    trait PtrLike: PessimizeRef + Sized {
        type Target: ?Sized;

        // Abstraction of self as *const T
        fn as_const_ptr(&self) -> *const Self::Target;

        // Like as_const_ptr, but translating back is not UB
        fn to_const_ptr(self) -> *const Self::Target {
            self.as_const_ptr()
        }

        // Get back from *const Self::Target to Self after to_const_ptr
        //
        // # Safety
        //
        // Transmuting back from *const T to &mut T is UB if the original
        // reference still exists (more precisely using the ref is UB).
        //
        // Transmuting *const T to a reference with a different lifetime than
        // the original one can also lead to UB.
        //
        // Basically, only use this on a pointer from Self::to_const_ptr,
        // obtained in the same scope or a parent scope, and stop using the
        // *const T as soon as the original pointer is back..
        //
        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self;

        // Abstraction of &**self
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &*self.as_const_ptr()
        }

        // Abstraction of (self as *const T) == (other as *const T)
        fn ptr_eq(&self, other: *const Self::Target) -> bool {
            self.as_const_ptr() == other
        }

        // Check ptr_eq, then target value against expectation
        //
        // Used in scenarios where there is a risk of pointer corruption, in
        // which cas it is not safe to dereference the pointer.
        //
        unsafe fn deep_eq(&self, other: *const Self::Target, expected_target: &Self::Target) -> bool
        where
            Self::Target: PartialEq,
        {
            self.ptr_eq(other) && self.unsafe_deref() == expected_target
        }
    }
    //
    impl<'a, T: ?Sized> PtrLike for &'a T
    where
        *const T: PessimizeRef,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            *self as *const T
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            &*ptr
        }
    }
    //
    impl<'a, T: ?Sized> PtrLike for &'a mut T
    where
        *const T: PessimizeRef,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            *self as *const T
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            &mut *(ptr as *mut T)
        }
    }
    //
    impl<T: ?Sized> PtrLike for *const T
    where
        *const T: PessimizeRef,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            *self as *const T
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            ptr
        }
    }
    //
    impl<T: ?Sized> PtrLike for *mut T
    where
        *const T: PessimizeRef,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            *self as *const T
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            ptr as *mut T
        }
    }
    //
    impl<T: ?Sized> PtrLike for NonNull<T>
    where
        *const T: PessimizeRef,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            self.as_ptr()
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            Self::new_unchecked(ptr as *mut Self::Target)
        }
    }
}
