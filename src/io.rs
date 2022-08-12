//! Pessimize implementations for std::io

use crate::{pessimize_once_like, pessimize_zsts};
use std::io::{Empty, Read, Repeat, Sink};

pessimize_zsts!(
    doc(cfg(feature = "std"))
    {
        Empty: std::io::empty(),
        Sink: std::io::sink()
    }
);

pessimize_once_like!(
    doc(cfg(feature = "std"))
    {
        Repeat: (
            u8,
            |self_: &mut Self| {
                let mut buf = [0u8; 1];
                self_.read_exact(&mut buf[..]).unwrap();
                buf[0]
            },
            std::io::repeat
        )
    }
);

// Some wrappers use u64 and require u64: Pessimize
#[cfg(target_pointer_width = "64")]
mod u64_is_pessimize {
    use super::*;
    use crate::assume_read;
    use crate::{assume_accessed, assume_accessed_imut, consume, hide, Pessimize};
    use std::io::{Cursor, Take};

    macro_rules! pessimize_wrapper {
        ($name:ident, $get_metadata:ident, $set_metadata:ident, $from_inner_metadata:expr $(, $extra_trait:path)?) => {
            // NOTE: Can't use PessimizeCast/BorrowPessimize skeleton here
            //       because we need to call assume_read/assume_accessed_imut
            //       for two different types, T and u64.
            #[cfg_attr(
                feature = "nightly",
                doc(cfg(all(feature = "std", target_pointer_width = "64")))
            )]
            unsafe impl<T: Pessimize $(+ $extra_trait)?> Pessimize for $name<T> {
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
                    let mut metadata = self.$get_metadata();
                    assume_accessed(&mut metadata);
                    self.$set_metadata(metadata);
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
    pessimize_wrapper!(
        Take,
        limit,
        set_limit,
        |inner: T, limit| inner.take(limit),
        Read
    );
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::{
        pessimize_newtypes,
        tests::{test_value, test_value_type},
    };

    macro_rules! test_zsts {
        (
            $(
                ($name:ident, $inner:ident)
            ),*
        ) => {
            $(
                #[derive(Clone, Copy, Debug, Default)]
                struct $name($inner);

                impl PartialEq for $name {
                    fn eq(&self, _other: &Self) -> bool {
                        true
                    }
                }

                pessimize_newtypes!( allow(missing_docs) { $name{ $inner } } );

                #[test]
                fn $inner() {
                    test_value($name::default());
                }

                // NOTE: There is no _optim test because Pessimize does not act
                //       as an optimization barrier for stateless ZSTs like
                //       Empty and Sink.
            )*
        };
    }
    //
    test_zsts!((TestableEmpty, Empty), (TestableSink, Sink));

    // NOTE: Can't test Repeat because there is no reliable way to extract the
    //       inner u8 from &self, which would be needed to implement PartialEq
    //       via a newtype.

    #[cfg(target_pointer_width = "64")]
    mod u64_is_pessimize {
        use super::*;
        use crate::tests::{test_unoptimized_value, test_unoptimized_value_type};
        use std::io::{Cursor, Take};

        #[test]
        fn cursor() {
            let mut test1 = Cursor::new((2..8).collect::<Vec<u8>>());
            test1.set_position(3);
            let mut test2 = Cursor::new((37..52).collect::<Vec<u8>>());
            test2.set_position(4);
            test_value_type(test1, test2);
        }

        #[test]
        #[ignore]
        fn cursor_optim() {
            test_unoptimized_value_type::<Cursor<Vec<u8>>>();
        }

        #[derive(Debug)]
        struct TestableTake(Take<Empty>);
        //
        impl TestableTake {
            fn new(limit: u64) -> Self {
                Self(std::io::empty().take(limit))
            }
        }
        //
        impl Clone for TestableTake {
            fn clone(&self) -> Self {
                Self::new(self.0.limit())
            }
        }
        //
        impl PartialEq for TestableTake {
            fn eq(&self, other: &Self) -> bool {
                self.0.limit() == other.0.limit()
            }
        }
        //
        pessimize_newtypes!( allow(missing_docs) { TestableTake{ Take<Empty> } } );
        //
        #[test]
        fn take() {
            test_value(TestableTake::new(42))
        }
        //
        #[test]
        #[ignore]
        fn take_optim() {
            test_unoptimized_value(TestableTake::new(42))
        }
    }
}
