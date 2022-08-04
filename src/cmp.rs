//! Pessimize implementations for core::cmp

use crate::{pessimize_newtypes, Pessimize};
use core::cmp::Reverse;

pessimize_newtypes!(
    allow(missing_docs)
    {
        |T: (Pessimize)| Reverse<T>{ T }
    }
);

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value_type, test_value_type};

    #[test]
    fn cell() {
        test_value_type(Reverse(isize::MIN), Reverse(isize::MAX));
    }

    #[test]
    #[ignore]
    fn cell_optim() {
        test_unoptimized_value_type::<Reverse<isize>>();
    }
}
