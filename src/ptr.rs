//! Implementation of Pessimize for pointers and references: `*const T`, `*mut
//! T`, `NonNull<T>`, `&T`, `&mut T` and `fn(Args...) -> Res`

use crate::{
    assume_accessed, impl_assume_accessed_via_extract_self, impl_with_pessimize_via_copy,
    BorrowPessimize, Pessimize, PessimizeCast,
};
use core::ptr::NonNull;

// Basic logic used to implement Pessimize for pointers
#[allow(asm_sub_register)]
#[inline]
fn hide_thin_ptr<T: Sized>(mut x: *const T) -> *const T {
    // Safe because it just captures a value and reemits it as is
    #[allow(unknown_lints, clippy::pointers_in_nomem_asm_block)]
    unsafe {
        core::arch::asm!("/* {0} */", inout(reg) x, options(preserves_flags, nostack, nomem));
    }
    x
}
//
#[allow(asm_sub_register)]
#[inline]
fn assume_read_thin_ptr<T: Sized>(x: *const T) {
    // Safe because it just captures a value and does nothing with it
    unsafe { core::arch::asm!("/* {0} */", in(reg) x, options(preserves_flags, nostack, readonly)) }
}
//
#[allow(asm_sub_register)]
#[inline]
fn assume_accessed_thin_ptr<T: Sized>(x: *mut T) {
    // Safe because it just captures a value and does nothing with it
    unsafe { core::arch::asm!("/* {0} */", in(reg) x, options(preserves_flags, nostack)) }
}

// Implementation of Pessimize for thin *const T and *const [T]
#[cfg(not(feature = "nightly"))]
mod thin_pointers {
    use super::*;
    use core::ptr;

    // Safe because the functions above are implemented as expected
    //
    // This is one of the primitive Pessimize impls on which the
    // PessimizeCast/BorrowPessimize stack is built
    unsafe impl<T: Sized> Pessimize for *const T {
        #[inline]
        fn hide(self) -> Self {
            hide_thin_ptr(self)
        }

        #[inline]
        fn assume_read(&self) {
            assume_read_thin_ptr(*self)
        }

        #[inline]
        fn assume_accessed(&mut self) {
            assume_accessed_thin_ptr(*self as *mut T);
        }

        #[inline]
        fn assume_accessed_imut(&self) {
            assume_accessed_thin_ptr(*self as *mut T);
        }
    }

    // Safe because the functions above are implemented as expected
    //
    // This is one of the primitive Pessimize impls on which the
    // PessimizeCast/BorrowPessimize stack is built
    unsafe impl<T: Sized> Pessimize for *const [T] {
        #[inline]
        fn hide(self) -> Self {
            let (data, len) = slice_to_raw_parts(self);
            ptr::slice_from_raw_parts(hide_thin_ptr(data), crate::hide(len))
        }

        #[inline]
        fn assume_read(&self) {
            let (data, len) = slice_to_raw_parts(*self);
            assume_read_thin_ptr(data.cast::<()>());
            crate::consume(len)
        }

        #[inline]
        fn assume_accessed(&mut self) {
            let (data, mut len) = slice_to_raw_parts(*self);
            assume_accessed_thin_ptr(data.cast::<()>().cast_mut());
            assume_accessed(&mut len);
            // Correct because hide is identity and assume_accessed doesn't modify
            *self = ptr::slice_from_raw_parts(data, len)
        }

        #[inline]
        fn assume_accessed_imut(&self) {
            assume_accessed_thin_ptr(self.cast::<()>().cast_mut());
            // No need to do an optimization barrier on the length, it's not
            // legal to change the length of a slice from &self and no
            // internally mutable state is transitively reachable via the length
        }
    }

    fn slice_to_raw_parts<T>(slice: *const [T]) -> (*const T, usize) {
        (slice.cast::<T>(), slice.len())
    }
}
//
#[cfg(feature = "nightly")]
mod all_pointers {
    use super::*;
    use crate::{assume_accessed_imut, consume, hide};
    use core::ptr::{self, DynMetadata, Pointee};

    // DynMetadata is a pointer to a trait object's vtable
    //
    // Can't use PessimizeCast/BorrowPessimize because this is one of the
    // building blocks on which the `*const ()`, which we would want to delegate
    // to, is built.
    #[doc(cfg(feature = "nightly"))]
    unsafe impl<T: ?Sized> Pessimize for DynMetadata<T> {
        #[inline]
        fn hide(self) -> Self {
            // Safe because `DynMetadata` is effectively defined to be a thin
            // pointer to an opaque type, which can be converted to `*const ()`
            // and back, and `hide_thin_ptr` doesn't modify the pointer.
            unsafe {
                core::mem::transmute::<*const (), Self>(hide_thin_ptr(core::mem::transmute::<
                    Self,
                    *const (),
                >(self)))
            }
        }

        #[inline]
        fn assume_read(&self) {
            // Safe because `DynMetadata` is effectively defined to be a thin
            // pointer to an opaque type, which can be converted to `*const ()`.
            assume_read_thin_ptr(unsafe { core::mem::transmute::<Self, *const ()>(*self) })
        }

        #[inline]
        fn assume_accessed(&mut self) {
            // Need to use `assume_accessed` barrier because someone with access
            // to this vtable could call a method that mutates global state
            assume_accessed_thin_ptr(unsafe { core::mem::transmute::<Self, *mut ()>(*self) })
        }

        #[inline]
        fn assume_accessed_imut(&self) {
            // Need to use `assume_accessed` barrier because someone with access
            // to this vtable could call a method that mutates global state
            assume_accessed_thin_ptr(unsafe { core::mem::transmute::<Self, *mut ()>(*self) })
        }
    }

    // Safe if the toplevel functions & DynMetadata impl operate as expected
    //
    // This is one of the primitive Pessimize impls on which the
    // PessimizeCast/BorrowPessimize stack is built
    #[doc(cfg(feature = "nightly"))]
    unsafe impl<T: Pointee + ?Sized> Pessimize for *const T
    where
        T::Metadata: Pessimize,
    {
        #[inline]
        fn hide(self) -> Self {
            let (thin, metadata) = self.to_raw_parts();
            ptr::from_raw_parts(hide_thin_ptr(thin), hide(metadata))
        }

        #[inline]
        fn assume_read(&self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_read_thin_ptr(thin);
            consume(metadata);
        }

        #[inline]
        fn assume_accessed(&mut self) {
            let (thin, mut metadata) = self.to_raw_parts();
            assume_accessed_thin_ptr(thin as *mut ());
            assume_accessed(&mut metadata);
            // Correct because hide is identity and assume_accessed doesn't modify
            *self = ptr::from_raw_parts(thin, metadata)
        }

        #[inline]
        fn assume_accessed_imut(&self) {
            let (thin, metadata) = self.to_raw_parts();
            assume_accessed_thin_ptr(thin as *mut ());
            assume_accessed_imut(&metadata);
        }
    }
}

// Can't use pessimize_cast! because making it support ?Sized is too hard
unsafe impl<T: ?Sized> PessimizeCast for *mut T
where
    *const T: Pessimize,
{
    type Pessimized = *const T;

    #[inline]
    fn into_pessimize(self) -> *const T {
        self as *const T
    }

    #[inline]
    unsafe fn from_pessimize(x: *const T) -> Self {
        x as *mut T
    }
}
//
impl<T: ?Sized> BorrowPessimize for *mut T
where
    *const T: Pessimize,
{
    type BorrowedPessimize = *const T;

    #[inline]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        impl_with_pessimize_via_copy(self, f)
    }

    #[inline]
    fn assume_accessed_impl(&mut self) {
        impl_assume_accessed_via_extract_self(self, |p: &mut *mut T| *p)
    }
}

// Can't use pessimize_cast! because making it support ?Sized is too hard
unsafe impl<T: ?Sized> PessimizeCast for NonNull<T>
where
    *mut T: Pessimize,
{
    type Pessimized = *mut T;

    #[inline]
    fn into_pessimize(self) -> *mut T {
        self.as_ptr()
    }

    #[inline]
    unsafe fn from_pessimize(x: *mut T) -> Self {
        NonNull::new_unchecked(x)
    }
}
//
impl<T: ?Sized> BorrowPessimize for NonNull<T>
where
    *mut T: Pessimize,
{
    type BorrowedPessimize = *mut T;

    #[inline]
    fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
        impl_with_pessimize_via_copy(self, f)
    }

    #[inline]
    fn assume_accessed_impl(&mut self) {
        impl_assume_accessed_via_extract_self(self, |nn: &mut NonNull<T>| *nn)
    }
}

// Once we have pessimized NonNull, we have pessimized function pointers, which
// are just a special kind of read-only pointer.
macro_rules! pessimize_fn {
    ($res:ident $( , $args:ident )* ) => {
        // Can't use pessimize_cast! macro here as making it support multiple
        // generic arguments would be too hard.
        unsafe impl< $res $( , $args )* > PessimizeCast for fn( $($args),* ) -> $res {
            type Pessimized = *const ();

            #[inline]
            fn into_pessimize(self) -> *const () {
                self as *const ()
            }

            #[inline]
            unsafe fn from_pessimize(x: *const ()) -> Self {
                core::mem::transmute::<*const (), Self>(x)
            }
        }
        //
        impl< $res $( , $args )* > BorrowPessimize for fn( $($args),* ) -> $res {
            type BorrowedPessimize = *const ();

            #[inline]
            fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
                impl_with_pessimize_via_copy(self, f)
            }

            #[inline]
            fn assume_accessed_impl(&mut self) {
                impl_assume_accessed_via_extract_self(self, |r| *r)
            }
        }
    }
}
//
macro_rules! pessimize_fn_up_to {
    ($res:ident) => {
        pessimize_fn!($res);
    };
    ($res:ident, $arg1:ident $(, $other_args:ident)*) => {
        pessimize_fn!($res, $arg1 $(, $other_args)*);
        pessimize_fn_up_to!($res $(, $other_args)*);
    }
}
//
pessimize_fn_up_to!(R, A1, A2, A3, A4, A5, A6, A7, A8);

// References are basically NonNull with a safe dereference and a shared XOR
// unique invariant. We can handle their Pessimize impl using homogeneous logic,
// though the safety proofs are a bit different.
//
// For all references, dereference must stay safe, which is true if we don't
// alter the inner pointer, pointee, or lifetime.
//
// For shared references, we must not create an &mut to the same data nor modify
// it. For unique references, we must consume the original reference (by passing
// it by value to something) before we create a new one. If we do all of that,
// our impl is correct for both reference types.
//
macro_rules! pessimize_references {
    (
        $(
            ($ref_t:ty, $from_nonnull:ident, $reborrow:expr)
        ),*
    ) => {
        $(
            // Can't use pessimize_cast! because making it support ?Sized is too
            // hard
            unsafe impl<'a, T: ?Sized> PessimizeCast for $ref_t
                where NonNull<T>: Pessimize
            {
                type Pessimized = NonNull<T>;

                #[inline]
                fn into_pessimize(self) -> NonNull<T> {
                    NonNull::from(self)
                }

                #[allow(unused_mut)]
                #[inline]
                unsafe fn from_pessimize(mut x: NonNull<T>) -> Self {
                    x.$from_nonnull()
                }
            }
            //
            impl<'a, T: ?Sized> BorrowPessimize for $ref_t
                where NonNull<T>: Pessimize
            {
                type BorrowedPessimize = NonNull<T>;

                #[inline]
                fn with_pessimize(&self, f: impl FnOnce(&Self::BorrowedPessimize)) {
                    let target: &T = &**self;
                    let target_nn: NonNull<T> = NonNull::from(target);
                    f(&target_nn);
                }

                #[allow(unknown_lints, clippy::missing_transmute_annotations)]
                #[inline]
                fn assume_accessed_impl(&mut self) {
                    // First, we create an owned reference to T via reborrow
                    // If *self is an &mut, this invalidates it as long as the
                    // reborrow is live, so we are allowed by Rust rules to
                    // mutate the underlying T through the reborrow. This is
                    // needed for the next optimization barrier to work.
                    let mut reborrow = $reborrow(self);

                    // Now that we have an owned reference to T, we can turn it
                    // into a NonNull, pass it through the
                    // NonNull::assume_accessed optimization barrier, and go
                    // back to a reference, like in the hide() implementation.
                    // This is safe because assume_accessed does nothing.
                    let mut ptr = NonNull::from(reborrow);
                    assume_accessed::<NonNull<T>>(&mut ptr);
                    reborrow = unsafe { ptr.$from_nonnull() };

                    // At this point, "reborrow" contains the reference that we
                    // would like *self to be: the compiler knows that it still
                    // points to the same target, but if it is a fat reference
                    // with metadata (slice length, vtable...), the compiler
                    // thinks that the metadata may be different.
                    //
                    // We want to propagate this "metadata looks different"
                    // optimizer assumption to *self, which we can do by
                    // replacing *self with the reborrowed reference.
                    //
                    // Sadly a simple `*self = reborrow` fails borrow checking:
                    // - It replaces the original self with another reference
                    //   which, from the perspective of the borrow checker,
                    //   has a shorter lifetime.
                    // - It erases the original self, so the reborrow that we
                    //   are trying to write down is invalidated.
                    //
                    // Nevertheless, we happen to know that...
                    // 1/This assignment is a no-op, and no-op can't go wrong
                    //   at the machine level.
                    // 2/Borrow checking does not affect the compiler backend in
                    //   a way that could create new UB at the optimizer level.
                    //
                    // So we extend the reborrow's lifteime to allow for this.
                    let extended: Self = unsafe { core::mem::transmute::<_, Self>(reborrow) };
                    *self = extended;
                }
            }
        )*
    };
}
//
fn reborrow<'local, T: ?Sized>(original: &'local mut &T) -> &'local T {
    original
}
//
fn reborrow_mut<'local, T: ?Sized>(original: &'local mut &mut T) -> &'local mut T {
    original
}
//
pessimize_references!((&'a T, as_ref, reborrow), (&'a mut T, as_mut, reborrow_mut));

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{assume_accessed_imut, assume_read, consume, hide, tests::assert_unoptimized};
    use std::{
        borrow::{Borrow, BorrowMut},
        fmt::Debug,
        ops::Deref,
        pin::Pin,
    };

    // === Tests asserting that the barriers don't modify anything ===
    // ===    (should be run on both debug and release builds)     ===

    // Test the Pessimize impl of one pointer type, which is
    // assumed to be dereferenceable to a certain target
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

    // Given some owned data, test all pointer types
    pub fn test_all_pinned_pointers<
        T: Debug + PartialEq + ?Sized,
        OwnedT: Clone + Borrow<T> + BorrowMut<T>,
    >(
        mut x: OwnedT,
    ) where
        *const T: Pessimize,
    {
        let old_x = x.clone();
        let old_x = old_x.borrow();
        unsafe {
            test_pointer(x.borrow(), old_x);
            test_pointer(x.borrow_mut(), old_x);
            test_pointer(x.borrow() as *const T, old_x);
            test_pointer(x.borrow_mut() as *mut T, old_x);
            test_pointer(NonNull::<T>::from(x.borrow_mut()), old_x);
        }
    }

    // Given some owned data, test all pointer types
    pub fn test_unpinned_pointers<
        T: Debug + PartialEq + ?Sized + Unpin,
        OwnedT: Clone + Borrow<T> + BorrowMut<T>,
    >(
        mut x: OwnedT,
    ) where
        *const T: Pessimize,
    {
        let old_x = x.clone();
        let old_x = old_x.borrow();
        unsafe {
            test_pointer(Pin::new(x.borrow_mut()), old_x);
        }
    }

    // Given some owned data, test all pointer types
    pub fn test_all_pointers<
        T: Debug + PartialEq + ?Sized + Unpin,
        OwnedT: Clone + Borrow<T> + BorrowMut<T>,
    >(
        x: OwnedT,
    ) where
        *const T: Pessimize,
    {
        test_unpinned_pointers(x.clone());
        test_all_pinned_pointers(x);
    }

    // === Tests asserting that the barriers prevent optimization ===
    // ===         (should only be run on release builds)         ===

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
        *const Value: Pessimize,
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

    // Check that one pointer looks different
    fn test_unoptimized_ptr<Value: PartialEq + ?Sized, Ptr: PtrLike<Target = Value>>(
        mut p: Ptr,
        expected_target: &Value,
    ) where
        *const Value: Pessimize,
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
                consume(std::ptr::eq(p.as_const_ptr(), old_p));
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

    // Given an owned value, check that all pointers look different
    pub fn test_unoptimized_ptrs<T: PartialEq + ?Sized, OwnedT: Clone + Borrow<T> + BorrowMut<T>>(
        mut x: OwnedT,
    ) where
        *const T: Pessimize,
    {
        let old_x = x.clone();
        let old_x = old_x.borrow();
        test_unoptimized_ptr(x.borrow() as *const T, old_x);
        test_unoptimized_ptr(x.borrow_mut() as *mut T, old_x);
        test_unoptimized_ptr(x.borrow(), old_x);
        test_unoptimized_ptr(x.borrow_mut(), old_x);
        test_unoptimized_ptr(NonNull::<T>::from(x.borrow_mut()), old_x);
    }

    // === Test concrete data types ===

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

    // === Uninteresting utilities ===

    // Abstraction layer to handle references and pointers homogeneously
    trait PtrLike: Pessimize + Sized {
        type Target: ?Sized;

        // Abstraction of self as *const T
        fn as_const_ptr(&self) -> *const Self::Target;

        // Like as_const_ptr, but translating back is not UB
        #[allow(unused)]
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
        #[allow(unused)]
        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self;

        // Abstraction of &**self
        unsafe fn unsafe_deref(&self) -> &Self::Target {
            &*self.as_const_ptr()
        }

        // Abstraction of (self as *const T) == (other as *const T)
        fn ptr_eq(&self, other: *const Self::Target) -> bool {
            std::ptr::eq(self.as_const_ptr(), other)
        }

        // Check ptr_eq, then target value against expectation
        //
        // Used in scenarios where there is a risk of pointer corruption, in
        // which case it is not safe to dereference the pointer.
        //
        unsafe fn deep_eq(&self, other: *const Self::Target, expected_target: &Self::Target) -> bool
        where
            Self::Target: PartialEq,
        {
            self.ptr_eq(other) && self.unsafe_deref() == expected_target
        }
    }
    //
    impl<T: ?Sized> PtrLike for &T
    where
        *const T: Pessimize,
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
    impl<T: ?Sized> PtrLike for &mut T
    where
        *const T: Pessimize,
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
        *const T: Pessimize,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            *self
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            ptr
        }
    }
    //
    impl<T: ?Sized> PtrLike for *mut T
    where
        *const T: Pessimize,
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
        *const T: Pessimize,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            self.as_ptr()
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            Self::new_unchecked(ptr as *mut Self::Target)
        }
    }
    //
    impl<'a, T: ?Sized + Unpin> PtrLike for Pin<&'a mut T>
    where
        *const T: Pessimize,
    {
        type Target = T;

        fn as_const_ptr(&self) -> *const Self::Target {
            self.deref().as_const_ptr()
        }

        unsafe fn from_const_ptr(ptr: *const Self::Target) -> Self {
            Self::new(<&'a mut T>::from_const_ptr(ptr))
        }
    }
}
