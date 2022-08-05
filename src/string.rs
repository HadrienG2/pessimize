//! Implementations of Pessimize for alloc::string

use crate::pessimize_collections;
use std_alloc::{string::String, vec::Vec};

// String is basically a Vec<u8> with an UTF-8 validity invariant
pessimize_collections!(
    doc(cfg(feature = "alloc"))
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    #[test]
    fn string() {
        test_value_type("a".to_owned(), "Z".to_owned());
    }

    #[test]
    #[ignore]
    fn string_optim() {
        test_unoptimized_value_type::<String>();
    }
}
