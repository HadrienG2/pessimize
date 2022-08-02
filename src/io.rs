//! Pessimize implementations for std::io

use crate::{assume_accessed, assume_accessed_imut, consume, hide, pessimize_zsts, Pessimize};
use std::io::{Empty, Read, Repeat, Sink};

// Some wrappers use u64 and require u64: Pessimize
#[cfg(target_pointer_width = "64")]
mod u64_is_pessimize {
    use super::*;
    use crate::assume_read;
    use std::io::{Cursor, Take};

    macro_rules! pessimize_wrapper {
        ($name:ident, $get_metadata:ident, $set_metadata:ident, $from_inner_metadata:expr) => {
            #[cfg_attr(
                feature = "nightly",
                doc(cfg(all(feature = "std", target_pointer_width = "64")))
            )]
            unsafe impl<T: Pessimize + Read> Pessimize for $name<T> {
                #[inline(always)]
                fn hide(self) -> Self {
                    let metadata = self.$get_metadata();
                    let (inner, metadata) = hide((self.into_inner(), metadata));
                    $from_inner_metadata(inner, metadata)
                }

                #[inline(always)]
                fn assume_read(&self) {
                    assume_read::<T>(self.get_ref());
                    consume(self.$get_metadata());
                }

                #[inline(always)]
                fn assume_accessed(&mut self) {
                    assume_accessed::<T>(self.get_mut());
                    let mut position = self.$get_metadata();
                    assume_accessed(&mut position);
                    self.$set_metadata(position);
                }

                #[inline(always)]
                fn assume_accessed_imut(&self) {
                    assume_accessed_imut::<T>(self.get_ref());
                    consume(self.$get_metadata());
                }
            }
        };
    }
    //
    pessimize_wrapper!(Cursor, position, set_position, |inner, position| {
        let mut result = Cursor::new(inner);
        result.set_position(position);
        result
    });
    //
    pessimize_wrapper!(Take, limit, set_limit, |inner: T, limit| inner.take(limit));
}

// As zero-sized types, Empty and Sink are trivially pessimizable
pessimize_zsts!(
    doc(cfg(feature = "std"))
    {
        Empty: std::io::empty(),
        Sink: std::io::sink()
    }
);

// Go through the inner byte if possible (need at least &mut self)
#[cfg_attr(feature = "nightly", doc(cfg(feature = "std")))]
unsafe impl Pessimize for Repeat {
    #[inline(always)]
    fn hide(mut self) -> Self {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf[..]).unwrap();
        let byte = hide(buf[0]);
        std::io::repeat(byte)
    }

    #[inline(always)]
    fn assume_read(&self) {
        consume::<&Repeat>(self);
    }

    #[inline(always)]
    fn assume_accessed(&mut self) {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf[..]).unwrap();
        assume_accessed::<u8>(&mut buf[0]);
        *self = std::io::repeat(buf[0])
    }

    #[inline(always)]
    fn assume_accessed_imut(&self) {
        assume_accessed_imut::<&Repeat>(&self);
    }
}

// FIXME: Can't test because the io types don't implement the right traits
