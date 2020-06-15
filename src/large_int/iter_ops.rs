use crate::large_int::LargeInt;

use std::iter::{
    Sum,
    Product
};

impl Sum for LargeInt {
    fn sum<I: Iterator<Item = LargeInt>>(iter: I) -> LargeInt {
        let mut sum = LargeInt::default();
        for item in iter {
            sum += item;
        }
        sum
    }
}

impl<'a> Sum<&'a LargeInt> for LargeInt {
    fn sum<I: Iterator<Item = &'a LargeInt>>(iter: I) -> LargeInt {
        let mut sum = LargeInt::default();
        for item in iter {
            sum += item.clone();
        }
        sum
    }
}

impl Product for LargeInt {
    fn product<I: Iterator<Item = LargeInt>>(iter: I) -> LargeInt {
        let mut product = LargeInt::from(1);
        let mut looped = false;
        for item in iter {
            product *= item;
            looped = true;
        }
        if looped {
            product
        } else {
            LargeInt::from(0)
        }
    }
}

impl<'a> Product<&'a LargeInt> for LargeInt {
    fn product<I: Iterator<Item = &'a LargeInt>>(iter: I) -> LargeInt {
        let mut product = LargeInt::from(1);
        let mut looped = false;
        for item in iter {
            product *= item.clone();
            looped = true;
        }
        if looped {
            product
        } else {
            LargeInt::from(0)
        }
    }
}