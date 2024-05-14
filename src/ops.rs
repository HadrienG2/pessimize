//! Implementations of Pessimize for core::ops

use crate::{pessimize_extractible, pessimize_zsts, Pessimize};
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

// Requiring a Clone bound because all useful Range indices are Clone
pessimize_extractible!(
    allow(missing_docs)
    {
        (Idx, Idx): (
            |Idx: (Clone, Pessimize)| Range<Idx> : (
                |Self { start, end }| (start, end),
                |(start, end)| Self { start, end },
                |self_: &Self| (self_.start.clone(), self_.end.clone())
            ),
            |Idx: (Clone, Pessimize)| RangeInclusive<Idx> : (
                |self_: Self| (self_.start().clone(), self_.end().clone()),
                |(start, end)| Self::new(start, end),
                |self_: &Self| (self_.start().clone(), self_.end().clone())
            )
        ),
        Idx: (
            |Idx: (Clone, Pessimize)| RangeFrom<Idx> : (
                |Self { start }| start,
                |start| Self { start },
                |self_: &Self| self_.start.clone()
            ),
            |Idx: (Clone, Pessimize)| RangeTo<Idx> : (
                |Self { end }| end,
                |end| Self { end },
                |self_: &Self| self_.end.clone()
            ),
            |Idx: (Clone, Pessimize)| RangeToInclusive<Idx> : (
                |Self { end }| end,
                |end| Self { end },
                |self_: &Self| self_.end.clone()
            )
        )
    }
);
//
pessimize_zsts!(
    allow(missing_docs)
    {
        RangeFull: RangeFull
    }
);

#[cfg(test)]
mod tests {
    use crate::tests::{test_unoptimized_value, test_value};

    #[test]
    fn range() {
        test_value(isize::MIN..isize::MAX)
    }

    #[test]
    #[ignore]
    fn range_optim() {
        test_unoptimized_value(isize::MIN..isize::MAX)
    }

    #[test]
    fn range_inclusive() {
        test_value(usize::MIN..=usize::MAX)
    }

    #[test]
    #[ignore]
    fn range_inclusive_optim() {
        test_unoptimized_value(usize::MIN..=usize::MAX)
    }

    #[test]
    fn range_from() {
        test_value(i32::MIN..)
    }

    #[test]
    #[ignore]
    fn range_from_optim() {
        test_unoptimized_value(i32::MIN..)
    }

    #[test]
    fn range_to() {
        test_value(..u32::MAX)
    }

    #[test]
    #[ignore]
    fn range_to_optim() {
        test_unoptimized_value(..u32::MAX)
    }

    #[test]
    fn range_to_inclusive() {
        test_value(..=i16::MAX)
    }

    #[test]
    #[ignore]
    fn range_to_inclusive_optim() {
        test_unoptimized_value(..=i16::MAX)
    }

    #[test]
    fn range_full() {
        test_value(..)
    }

    // There is no range_full_optim test because Pessimize does not act as an
    // optimization barrier for stateless ZSTs like RangeFull
}
