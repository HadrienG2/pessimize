//! This crate aims to implement minimally costly optimization barriers for
//! every architecture that has `asm!()` support (currently x86(_64),
//! 32-bit ARM, AArch64 and RISC-V, more possible on nightly via the
//! `asm_experimental_arch` unstable feature).
//!
//! You can use these barriers to prevent the compiler from optimizing out
//! selected redundant or unnecessary computations in situations where such
//! optimization is undesirable. The most typical usage scenario is
//! microbenchmarking, but there might also be applications to cryptography or
//! low-level development, where optimization must also be controlled.
//!
//! # Implementations
//!
//! The barriers will be implemented for any type from core/std that either...
//! - Can be shoved into CPU registers, with "natural" target registers
//!   dictated by normal ABI calling conventions.
//! - Can be losslessly converted back and forth to a set of values that have
//!   this property, in a manner that is easily optimized out.
//!
//! Any type which is not directly supported can still be subjected to an
//! optimization barrier by taking a reference to it and subjecting that
//! reference to an optimization barrier, at the cost of causing the value to
//! be spilled to memory. If the nightly `default_impl` feature is enabled, the
//! crate will provide a default `Pessimize` impl that does this for you.
//!
//! You can tell which types implement `Pessimize` on your compiler target by
//! running `cargo doc` and checking the implementor list of `Pessimize` and
//! `BorrowPessimize`.
//!
//! To implement `Pessimize` for your own types, you should consider
//! implementing `PessimizeCast` and `BorrowPessimize`, which make the job a
//! bit easier. `Pessimize` is automatically implemented for any type that
//! implements `BorrowPessimize`.
//!
//! # Semantics
//!
//! For pointer-like entities, optimization barriers other than `hide` can
//! have the side-effect of causing the compiler to assume that global and
//! thread-local variables might have been accessed using similar semantics as
//! the pointer itself. This will reduce applicable compiler optimizations for
//! such variables, so the use of `hide` should be favored whenever global or
//! thread-local variables are used (or you don't know if they are used).
//!
//! In general, barriers other than `hide` have more avenues for surprising
//! behavior (see their documentation for details), so you should strive to do
//! what you want with `hide` if possible, and only reach for other barriers
//! where the extra expressive power of these primitives is truly needed.
//!
//! The documentation of the top-level functions (`hide`, `assume_read`,
//! `consume`, `assume_accessed` and `assume_accessed_imut`) contain more
//! details on the optimization barrier that is being implemented.
//!
//! # When to use this crate
//!
//! You should consider use of this crate over `core::hint::black_box`, or
//! third party cousins thereof, because...
//! - It works on stable Rust
//! - It has a better-defined API contract with stronger guarantees (unlike
//!   `core::hint::black_box`, where "do nothing" is a valid implementation).
//! - It exposes finer-grained operations, which clarify your code's intent and
//!   reduce harmful side-effects.
//!
//! The main drawbacks of this crate's approach being that...
//! - It only works on selected hardware architectures (though they are the ones
//!   on which you are most likely to run benchmarks, and it should get better
//!   over time as more inline assembly architectures get stabilized).
//! - It needs a lot of tricky unsafe code.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(
    feature = "nightly",
    feature(doc_cfg, stdsimd, portable_simd, ptr_metadata)
)]
#![cfg_attr(feature = "default_impl", allow(incomplete_features))]
#![cfg_attr(feature = "default_impl", feature(specialization))]
#![deny(missing_docs)]

// TODO: Once allocator_api is stable, support collections with custom
//       allocators by applying optimization barriers to the allocator as well.
//       Right now, doing so would require either duplicating the tricky
//       collection code (one version with allocators and one version without)
//       or dropping collection support on stable, neither of which sound
//       satisfactory given that custom allocators are expected to be niche.
#[cfg(any(feature = "alloc", test))]
extern crate alloc as std_alloc;

mod alloc;
pub mod arch;
#[cfg(any(feature = "alloc", test))]
mod boxed;
mod cell;
mod cmp;
#[cfg(all(any(feature = "std", test), any(unix, target_os = "wasi")))]
mod ffi;
mod fmt;
#[cfg(any(feature = "std", test))]
mod fs;
#[cfg(any(feature = "std", test))]
mod io;
mod iter;
mod marker;
mod mem;
#[cfg(any(feature = "std", test))]
mod net;
mod num;
mod ops;
mod panic;
#[cfg(all(any(feature = "std", test), any(unix, target_os = "wasi")))]
mod path;
mod pin;
mod primitive;
#[cfg(all(any(feature = "std", test), unix))]
mod process;
mod ptr;
// TODO: mod string (String, take from lib.rs and add tests)
// TODO: mod sync (atomic, reuse cell tests)
// TODO: mod task (Context, RawWaker, RawWakerVTable, Waker)
// TODO: mod time (Duration)
// TODO: mod vec (Vec, take from lib.rs and add tests)

/// Optimization barriers provided by this crate
///
/// This trait is implemented for both value and reference types, which can
/// lead to unexpected method syntax semantics (you expected to call the
/// `Pessimize` impl of `T`, and you actually called that of `&T`). As a result,
/// it is strongly recommended to use the optimization barriers via the free
/// functions provided at the crate root, rather than via method syntax.
///
/// Implementing `Pessimize` requires fairly tricky code. Consider implementing
/// `PessimizeCast` and `BorrowPessimize` instead, which is slightly easier, and
/// will lead to an automatic `Pessimize` implementation.
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
/// Although `Pessimize` is implemented for zero-sized types, `hide()` will not
/// act as an optimization barrier on those types, because there is only one
/// possible return value so the compiler knows that the output value is the
/// same as the input value. Since zero-sized types may only hold state through
/// global and thread-local variables, an `assume_xyz` barrier will be effective
/// at assuming that the state has been read or modified.
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
/// `core::hint::black_box(&mut x)` should have the effect of
/// `pessimize::assume_accessed(&mut x)`, whereas `pessimize::hide(x)` does not
/// enforce any compiler assumptions concerning the input value, it just turns
/// it into another value that looks unrelated in the eye of the compiler.
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

/// Like `assume_read`, but by value
///
/// This is a more ergonomic alternative to `assume_read` in the common case
/// where the input is `Copy` or will not be needed anymore.
///
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

/// Convert `Self` back and forth to a `Pessimize` impl (`Pessimize` impl helper)
///
/// While only a small number of `Pessimize` types are supported by inline
/// assembly, many standard types can be losslessly converted to a lower-level
/// type (or tuple of types) that implement Pessimize and back in such a way
/// that the runtime costs should be optimized out.
///
/// This trait exposes that capability under a common abstraction vocabulary.
/// Combined with the related `BorrowPessimize` trait, it enables implementation
/// of `Pessimize` with increased safety reduced boilerplate.
///
/// # Safety
///
/// By implementing this trait, you guarantee that someone using it as
/// documented will not trigger Undefined Behavior, since the safe `Pessimize`
/// trait can be automatically implemented on top of it
///
pub unsafe trait PessimizeCast {
    /// Pessimize type that can be converted to and from a Self value
    type Pessimized: Pessimize;

    /// Convert Self to Pessimized
    fn into_pessimize(self) -> Self::Pessimized;

    /// Convert back from Pessimized to Self
    ///
    /// # Safety
    ///
    /// A correct implementation of this operation only needs to be safe for the
    /// intended purpose of converting `Self` to `Pessimized` using
    /// `into_pessimize()`, invoking `Pessimize` trait operations on the
    /// resulting value, and optionally converting the `Pessimized` value back
    /// to `Self` afterwards via `from_pessimize()`.
    ///
    /// The final `from_pessimize()` operation of this round trip must be
    /// performed in the same scope where the initial `into_pessimize()`
    /// operation was called, or a child scope thereof.
    ///
    /// Even if `Pessimized` is `Clone`, it is strongly advised to treat the
    /// `Pessimized` value from `into_pessimize()` as a `!Clone` value: don't
    /// clone or copy it, and stop using it after converting it back to `Self`.
    /// Otherwise, _suprising_ (but safe) behavior may occur.
    ///
    /// **No other usage of `from_pessimize()` is safe.** To give a few examples
    /// of incorrect usage of `from_pessimize()`...
    ///
    /// - `Self` may contain references to the surrounding stack frame, so even
    ///   if `Pessimized` is `'static`, letting a `Pessimized` escape the scope
    ///   in which `into_pessimize()` was called before converting it back to
    ///   `Self` is unsafe.
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

/// Extract references to `Pessimize` values from references to `Self` (`Pessimize` impl helper)
pub trait BorrowPessimize: PessimizeCast {
    /// Pessimize type to which a reference can be extracted from &self
    type BorrowedPessimize: Pessimize;

    /// Extract an `&Pessimized` from `&self` and all the provided operation
    /// on it.
    ///
    /// In the common case where `Self: Copy`, you can implement this by calling
    /// `pessimize::impl_with_pessimize_via_copy`.
    ///
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize));

    /// Extract an `&mut Pessimized` from `&mut self`, call `assume_accessed`
    /// on it, and propagate any "changes" back to the original `&mut self`.
    ///
    /// In the common case where there is a cheap way to go from an `&mut Self`
    /// to a `Self` or `Self::Pessimized`, you can implement this by calling
    /// one of the `pessimize::impl_assume_accessed_via_xyz` functions.
    ///
    fn assume_accessed_impl(&mut self);
}

/// Implementation of `BorrowPessimize::with_pessimize` for `Copy` types
// TODO: Use specializable BorrowPessimize impl once available on stable
#[inline(always)]
pub fn impl_with_pessimize_via_copy<T: Copy + PessimizeCast>(
    self_: &T,
    f: impl FnOnce(&T::Pessimized),
) {
    let pessimize = T::into_pessimize(*self_);
    f(&pessimize)
}

/// Implementation of `BorrowPessimize::assume_accessed_impl` for types where
/// there is a way to get a `T::Pessimized` from an `&mut T`
///
/// The `Drop` impl of the value left in `self_` by `extract_pessimized` will
/// not be called, make sure that this does not result in a resource leak.
///
#[inline(always)]
pub fn impl_assume_accessed_via_extract_pessimized<T: PessimizeCast>(
    self_: &mut T,
    extract_pessimized: impl FnOnce(&mut T) -> T::Pessimized,
) {
    let mut pessimize = extract_pessimized(self_);
    assume_accessed(&mut pessimize);
    // Safe because assume_accessed is allowed between into_pessimize and from_pessimize
    unsafe { (self_ as *mut T).write(T::from_pessimize(pessimize)) };
}

/// Implementation of `BorrowPessimize::assume_accessed_impl` for types where
/// there is a cheap way to extract the inner `T` from an `&mut T`
///
/// For `Copy` types, this is a dereference, and for `Default` types where the
/// default value is truly trivial and guaranteed to be optimized out (like
/// `Vec`), this is `core::mem::take`.
///
/// The `Drop` impl of the value left in `self_` by `extract_self` will not be
/// called, make sure that this does not result in a resource leak.
///
// TODO: Remove once pointers are migrated to macros
#[inline(always)]
pub fn impl_assume_accessed_via_extract_self<T: PessimizeCast>(
    self_: &mut T,
    extract_self: impl FnOnce(&mut T) -> T,
) {
    impl_assume_accessed_via_extract_pessimized(self_, |self_| {
        T::into_pessimize(extract_self(self_))
    });
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
        Self::assume_accessed_impl(self)
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
mod default_impl {
    use super::*;

    // NOTE: Can't use PessimizeCast/BorrowPessimize here because need to use
    //       `assume_accessed` in `hide` impl.
    #[doc(cfg(all(feature = "nightly", feature = "default_impl")))]
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

/// Implementation of Pessimize for asm inputs/outputs without pointer semantics
///
/// To be used by arch-specific modules to implement Pessimize for primitive and
/// arch-specific SIMD types.
///
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_asm_values {
    (
        $doc_cfg:meta
        {
            $(
                $reg:ident : ( $($value_type:ty),* )
            ),*
        }
    ) => {
        $($(
            // NOTE: This is one of the primitive Pessimize impls on which the
            //       PessimizeCast/BorrowPessimize stack is built
            #[allow(asm_sub_register)]
            #[cfg_attr(feature = "nightly", $doc_cfg)]
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

/// Implementation of PessimizeCast for types that can be converted to and from
/// a Pessimized impl at low cost
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_cast {
    (
        $doc_cfg:meta
        {
            $(
                $inner:ty : (
                    $(
                        $(
                            | $param:ident $( : ( $trait1:path $(, $traitN:path)* ) )? |
                        )?
                        $outer:ty : ($into:expr, $from:expr)
                    ),*
                )
            ),*
        }
    ) => {
        $($(
            #[cfg_attr(feature = "nightly", $doc_cfg)]
            unsafe impl $(< $param $( : $trait1 $( + $traitN )* )? >)? $crate::PessimizeCast for $outer {
                type Pessimized = $inner;

                #[inline(always)]
                fn into_pessimize(self) -> $inner {
                    $into(self)
                }

                #[inline(always)]
                unsafe fn from_pessimize(inner: $inner) -> Self {
                    $from(inner)
                }
            }
        )*)*
    };
}

/// Implementation of Pessimize for types from which a Pessimize impl can be
/// extracted given nothing but an &self, having round trip conversion functions
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_extractible {
    (
        $doc_cfg:meta
        {
            $(
                $inner:ty : (
                    $(
                        $(
                            | $param:ident $( : ( $trait1:path $(, $traitN:path)* ) )? |
                        )?
                        $outer:ty : ($into:expr, $from:expr, $extract:expr)
                    ),*
                )
            ),*
        }
    ) => {
        $crate::pessimize_cast!(
            $doc_cfg
            {
                $(
                    $inner : (
                        $(
                            $(
                                | $param $( : ( $trait1 $(, $traitN)* ) )? |
                            )?
                            $outer : ($into, $from)
                        ),*
                    )
                ),*
            }
        );
        //
        $($(
            #[cfg_attr(feature = "nightly", $doc_cfg)]
            impl $(< $param $( : $trait1 $( + $traitN )* )? >)? $crate::BorrowPessimize for $outer {
                type BorrowedPessimize = $inner;

                #[inline(always)]
                fn with_pessimize(&self, f: impl FnOnce(&$inner)) {
                    f(&$extract(self))
                }

                #[inline(always)]
                fn assume_accessed_impl(&mut self) {
                    $crate::impl_assume_accessed_via_extract_pessimized(self, |self_: &mut Self| $extract(self_))
                }
            }
        )*)*
    };
}

/// Implementation of Pessimize for Copy types with round trip conversion functions
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_copy {
    (
        $doc_cfg:meta
        {
            $(
                $inner:ty : (
                    $(
                        $(
                            | $param:ident $( : $traits:tt )? |
                        )?
                        $outer:ty : ($into:expr, $from:expr)
                    ),*
                )
            ),*
        }
    ) => {
        $crate::pessimize_extractible!(
            $doc_cfg
            {
                $(
                    $inner : (
                        $(
                            $(
                                | $param $( : $traits )? |
                            )?
                            $outer : ($into, $from, |self_: &Self| $into(*self_))
                        ),*
                    )
                ),*
            }
        );
    };
}

/// Like pessimize_copy, but using a standard Into/From pair for PessimizeCast
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_into_from {
    (
        $doc_cfg:meta
        {
            $(
                $inner:ty : (
                    $(
                        $( | $param:ident $( : $traits:tt )? | )?
                        $outer:ty
                    ),*
                )
            ),*
        }
    ) => {
        $crate::pessimize_copy!(
            $doc_cfg
            {
                $(
                    $inner : (
                        $(
                            $( | $param $( : $traits )? | )?
                            $outer : (Self::into, Self::from)
                        ),*
                    )
                ),*
            }
        );
    };
}

/// Implementation of Pessimize for any stateless zero-sized type
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_zsts {
    (
        $doc_cfg:meta
        {
            $(
                $( | $param:ident $( : $traits:tt )? | )?
                $name:ty : $make:expr
            ),*
        }
    ) => {
        $crate::pessimize_extractible!(
            $doc_cfg
            {
                () : (
                    $(
                        $( | $param $( : $traits )? | )?
                        $name : (
                            |_self| (),
                            |()| $make,
                            |_self| ()
                        )
                    ),*
                )
            }
        );
    };
}

/// Implementation of Pessimize for tuple structs
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_tuple_structs {
    (
        $doc_cfg:meta
        {
            $(
                $( | $param:ident $( : $traits:tt )? | )?
                $outer:ty {
                    $( $name:ident : $inner:ty ),*
                }
            ),*
        }
    ) => {
        $crate::pessimize_copy!(
            $doc_cfg
            {
                $(
                    ( $($inner,)* ) : (
                        $( | $param $( : $traits )? | )?
                        $outer : (
                            |Self( $($name),* )| ( $($name,)* ),
                            |( $($name,)* )| Self( $($name),* )
                        )
                    )
                ),*
            }
        );
    };
}

/// Implementation of Pessimize for T(pub U) style newtypes
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_newtypes {
    (
        $doc_cfg:meta
        {
            $(
                $(
                    | $param:ident $( : ( $trait1:path $(, $traitN:path)* ) )? |
                )?
                $outer:ty { $inner:ident }
            ),*
        }
    ) => {
        $crate::pessimize_cast!(
            $doc_cfg
            {
                $(
                    $inner : (
                        $(
                            | $param $( : ( $trait1 $(, $traitN)* ) )? |
                        )?
                        $outer : (
                            |Self(inner)| inner,
                            |inner| Self(inner)
                        )
                    )
                ),*
            }
        );
        //
        $(
            #[cfg_attr(feature = "nightly", $doc_cfg)]
            impl $(< $param $( : $trait1 $( + $traitN )* )? >)? $crate::BorrowPessimize for $outer {
                type BorrowedPessimize = $inner;

                #[inline(always)]
                fn with_pessimize(&self, f: impl FnOnce(&$inner)) {
                    f(&self.0)
                }

                #[inline(always)]
                fn assume_accessed_impl(&mut self) {
                    $crate::assume_accessed(&mut self.0)
                }
            }
        )*
    };
}

/// Pessimize a type that behaves like core::iter::Once
// FIXME: Merge into main macro hierarchy.
#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_once_like {
    (
        $doc_cfg:meta
        {
            $(
                $(
                    | $param:ident $( : ( $trait1:path $(, $traitN:path)* ) )? |
                )?
                $outer:ty : (
                    $inner:ty,
                    $extract:expr,
                    $make:expr
                )
            ),*
        }
    ) => {
        $crate::pessimize_cast!(
            $doc_cfg
            {
                $(
                    $inner : (
                        $(
                            | $param $( : ( $trait1 $(, $traitN)* ) )? |
                        )?
                        $outer : (|mut self_| $extract(&mut self_), $make)
                    )
                ),*
            }
        );
        //
        $(
            #[cfg_attr(feature = "nightly", $doc_cfg)]
            impl $(< $param $( : $trait1 $( + $traitN )* )? >)? $crate::BorrowPessimize for $outer {
                type BorrowedPessimize = *const Self;

                #[inline(always)]
                fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
                    // Need at least an &mut Self to access the inner value, so
                    // must use by-reference optimization barrier with &Self
                    f(&(self as *const Self))
                }

                #[inline(always)]
                fn assume_accessed_impl(&mut self) {
                    let mut value = $extract(self);
                    $crate::assume_accessed(&mut value);
                    *self = $make(value)
                }
            }
        )*
    };
}

/// Pessimize a type that behaves like a collection (cheap Default impl, owned
/// state differs from borrowed state)#[doc(hidden)]
#[macro_export]
macro_rules! pessimize_collections {
    (
        $doc_cfg:meta
        {
            $(
                ($owned_inner:ty, $borrowed_inner:ty) : (
                    $(
                        $(
                            | $param:ident $( : ( $trait1:path $(, $traitN:path)* ) )? |
                        )?
                        $outer:ty : ($into_owned:expr, $from_owned:expr, $extract_borrowed:expr)
                    ),*
                )
            ),*
        }
    ) => {
        $crate::pessimize_cast!(
            $doc_cfg
            {
                $(
                    $owned_inner : (
                        $(
                            $(
                                | $param $( : ( $trait1 $(, $traitN)* ) )? |
                            )?
                            $outer : ($into_owned, $from_owned)
                        ),*
                    )
                ),*
            }
        );
        //
        $($(
            #[cfg_attr(feature = "nightly", $doc_cfg)]
            impl $(< $param $( : $trait1 $( + $traitN )* )? >)? $crate::BorrowPessimize for $outer {
                type BorrowedPessimize = $borrowed_inner;

                #[inline(always)]
                fn with_pessimize(&self, f: impl FnOnce(&$borrowed_inner)) {
                    f(&$extract_borrowed(self))
                }

                #[inline(always)]
                fn assume_accessed_impl(&mut self) {
                    $crate::impl_assume_accessed_via_extract_pessimized(self, |self_: &mut Self| $into_owned(core::mem::take(self_)))
                }
            }
        )*)*
    };
}

// TODO: Once done with std, go through the crate looking for patterns in impls
//       of Pessimize, PessimizeCast and BorrowPessimize, and factor these out.
//
//       Consider splitting PessimizeCast and BorrowPessimize macros
//
//       Current candidates are...
//       - where bounds (needed by ptr::* and boxed::Box<T>)
//       - assume_accessed needs fancy borrows (needed by boxed::Box<T> and ptr::&[mut] T)
//       - Pointer getter and get_mut (needed by cell::UnsafeCell<T>)
//       - Multiple generic args (needed by ptr::fn())
//       - Making pessimize_once_like implementable atop the main macros
//       - into_inner, new, Deref and DerefMut (mem::ManuallyDrop<T>)

// Although all Rust collections are basically pointers with extra metadata, we
// may only implement Pessimize for them when all the metadata is exposed and
// there is a way to build a collection back from all the raw parts
#[cfg(any(feature = "alloc", test))]
mod alloc_feature {
    use super::*;
    use core::mem::ManuallyDrop;
    use std_alloc::{string::String, vec::Vec};

    pessimize_collections!(
        doc(cfg(feature = "alloc"))
        {
            // Vec<T> is basically a thin NonNull<T>, a length and a capacity
            // FIXME: Migrate to crate::vec, along with associated tests
            ((*mut T, usize, usize), (*const T, usize, usize)) : (
                |T| Vec<T> : (
                    |self_: Self| {
                        let mut v = ManuallyDrop::new(self_);
                        (v.as_mut_ptr(), v.len(), v.capacity())
                    },
                    |(ptr, length, capacity)| {
                        Self::from_raw_parts(ptr, length, capacity)
                    },
                    |self_: &Self| {
                        (self_.as_ptr(), self_.len(), self_.capacity())
                    }
                )
            ),

            // String is basically a Vec<u8> with an UTF-8 validity invariant
            // FIXME: Migrate to crate::string, along with associated tests
            (Vec<u8>, (*const u8, usize, usize)) : (
                String : (
                    Self::into_bytes,
                    Self::from_utf8_unchecked,
                    |self_: &Self| {
                        (self_.as_ptr(), self_.len(), self_.capacity())
                    }
                )
            )
        }
    );
}

// TODO: Provide a Derive macro to derive Pessimize for a small struct, with a
//       warning that it will do more harm than good on a larger struct

// TODO: Set up CI in the spirit of test-everything.sh

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ptr::tests::{test_all_pinned_pointers, test_all_pointers, test_unpinned_pointers};
    #[cfg(feature = "nightly")]
    use std::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
    use std::{
        fmt::Debug,
        marker::Unpin,
        time::{Duration, Instant},
    };

    // === Tests asserting that the barriers don't modify anything ===
    // ===    (should be run on both debug and release builds)     ===

    // Test that, for a given value, Pessimize seems to work
    pub fn test_pinned_value<T: Clone + Debug + PartialEq + Pessimize>(x: T) {
        let old_x = x.clone();
        assume_read(&x);
        assert_eq!(x, old_x);
        assert_eq!(hide(x.clone()), old_x);
        test_all_pinned_pointers::<T, _>(x.clone());
        consume(x);
    }

    // Test that, for a given value, Pessimize seems to work
    pub fn test_value<T: Clone + Debug + PartialEq + Pessimize + Unpin>(x: T) {
        test_unpinned_pointers(x.clone());
        test_pinned_value(x);
    }

    // Run test_value on the minimal, default and maximal value of a type
    pub fn test_value_type<T: Clone + Debug + Default + PartialEq + Pessimize + Unpin>(
        min: T,
        max: T,
    ) {
        test_value(min);
        test_value(T::default());
        test_value(max);
    }

    // Run test_value_type for an encapsulated SIMD type
    #[allow(unused)]
    pub fn test_simd<
        Scalar: Copy + Default,
        const LANES: usize,
        T: Copy + Debug + Default + From<[Scalar; LANES]> + PartialEq + Pessimize + Unpin,
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
        Scalar: Debug + Default + PartialEq + SimdElement + Unpin,
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

    // --- Tests for normal values with native Pessimize support ---

    pub fn test_unoptimized_value<T: Clone + PartialEq + Pessimize>(x: T) {
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

    // --- Tests for ZSTs ---

    pub fn test_unoptimized_zst<T: Default + Pessimize>() {
        use core::cell::Cell;

        // ZSTs don't have inner state to pessimize, but they can provide access
        // to global and thread-local state.
        thread_local! {
            static STATE: Cell<isize> = Cell::new(-42);
        }

        let old_state = -42isize;
        assert_unoptimized(T::default(), |mut x| {
            assume_accessed(&mut x);
            STATE.with(|state| {
                consume(state.get() == old_state);
            });
            x
        });
    }

    // === Tests for types implemented here ===

    // --- "Big" array with no native Pessimize implementation ---

    // What is considered too big (in units of isize)
    // 2 is enough as we don't currently pessimize larger arrays
    pub const BIG: usize = 2;

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
        crate::ptr::tests::test_unoptimized_ptrs::<[isize; BIG], _>([0isize; BIG]);
    }
}
