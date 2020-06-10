use std::u128;
use std::ops::{
    Add,
    AddAssign,
    Sub,
    SubAssign
};

// store a vector of little-endian, 2's compliment figures
// the sign bit is in the most significant figure (more[0])
// littel endian was chosen so vec operations are faster
#[derive(Clone, Debug, PartialEq)]
pub struct LargeInt {
    // use u128 since, if someone needs a LargeInt, it's likely
    // going to end up larger than u128::MAX
    bytes: Vec<u128>
}

const SIGN_BIT: u128 = 1u128 << 127;

fn is_u128_negative(val: u128) -> bool {
    (val & SIGN_BIT) > 1
}

impl LargeInt {
    pub fn new() -> LargeInt {
        LargeInt{bytes: vec!(0)}
    }

    pub fn with_size(size: usize) -> LargeInt {
        LargeInt{bytes: vec!(0; size)}
    }

    pub fn is_negative(&self) -> bool {
        is_u128_negative(self.bytes[self.bytes.len() - 1])
    }

    pub fn is_positive(&self) -> bool {
        !self.is_negative()
    }

    pub fn shrink(&mut self) {
        let (remove_chunk, checker): (u128, Box<dyn Fn(u128) -> bool>) = 
        if self.is_negative() {
            (u128::MAX, Box::new(|chunk| !is_u128_negative(chunk)))
        } else {
            (0u128, Box::new(|chunk| is_u128_negative(chunk)))
        };
        for i in (1..self.bytes.len()).rev() {
            if checker(self.bytes[i - 1]) {
                break;
            } else if self.bytes[i] == remove_chunk {
                self.bytes.pop();
            }
        }
    }

    // represents this int with size u128's if currently
    // represented with less, else leave it as is
    fn expand_to(&mut self, size: usize) {
        let extension = if self.is_negative() {
            u128::MAX
        } else {
            0u128
        };
        while self.bytes.len() < size {
            self.bytes.push(extension);
        }
    }

    fn compliment(&self) -> LargeInt {
        let size = self.bytes.len();
        let mut compliment = LargeInt::with_size(size);
        for i in 0..size {
            compliment.bytes[i] = self.bytes[i] ^ u128::MAX;
        }
        compliment + 1
    }
}

impl Add for LargeInt {
    type Output = LargeInt;

    fn add(mut self, mut other: LargeInt) -> LargeInt {

        // prepare for overflow
        let size = self.bytes.len().max(other.bytes.len()) + 1;
        self.expand_to(size);
        other.expand_to(size);

        // perform addition
        let mut result = LargeInt::with_size(size);
        let mut res;
        let mut o_f = false;
        for i in 0..size {
            let mut add_res = self.bytes[i].overflowing_add(other.bytes[i]);
            let overflowed = add_res.1;

            // check overflow for previous addition
            // this can add at most +1 to this result
            if o_f { 
                add_res = add_res.0.overflowing_add(1u128);
            }
            res = add_res.0;
            o_f = overflowed || add_res.1;
            result.bytes[i] = res;
        }
        result.shrink();
        result
    }
}

impl AddAssign for LargeInt {
    fn add_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() + other).bytes;
    }
}

impl Sub for LargeInt {
    type Output = LargeInt;

    // use the implmentation of addition for subtraction
    fn sub(self, other: LargeInt) -> LargeInt {
        self + other.compliment()
    }
}

impl SubAssign for LargeInt {
    fn sub_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() - other).bytes;
    }
}

macro_rules! ops {
    ( $($t:ident)* ) => {
        $(impl From<$t> for LargeInt {
            fn from(val: $t) -> LargeInt {
                let mut bytes = Vec::new();
                bytes.push(val as u128);
                LargeInt{bytes: bytes}
            }
        })*

        $(impl Add<$t> for LargeInt {
            type Output = LargeInt;

            fn add(self, other: $t) -> LargeInt {
                let oth = LargeInt::from(other);
                self + oth
            }
        })*
    };
}

ops!(i8 i32 i64 i128 isize u8 u32 u64 u128 usize);

#[cfg(test)]
mod tests {
    use crate::large_int::{
        LargeInt,
        SIGN_BIT
    };
    
    use std::u128;

    #[test]
    fn test_from_i64() {
        let li = LargeInt::from(257i64);
        assert_eq!(li.bytes[0], 257u128);
        assert!(li.is_positive());

        let li = LargeInt::from(-1i64);
        assert_eq!(li.bytes[0], u128::MAX); // 2's compliment rep of -1 is all 1s

        assert!(li.is_negative());
    }

    #[test]
    fn test_is_negative() {
        let mut li = LargeInt{bytes: vec!(1, 0)};
        assert!(!li.is_negative());

        li.bytes[0] = u128::MAX;
        assert!(!li.is_negative());

        li.bytes[1] = u128::MAX;
        assert!(li.is_negative());

        li.bytes[1] ^= SIGN_BIT;
        assert!(!li.is_negative());

        li.bytes.push(u128::MAX);
        assert!(li.is_negative());
    }

    #[test]
    fn test_shrink() {
        // shrink because b1111 == b111 == b11 (== d-1) etc
        let mut li = LargeInt{bytes: vec!(u128::MAX; 2)};
        li.shrink();
        assert_eq!(li.bytes.len(), 1);
        assert_eq!(li.bytes[0], u128::MAX);

        // shrink because b0001 == b001 == b01 (== d1) etc
        li = LargeInt{bytes: vec!(1, 0)};
        li.shrink();
        assert_eq!(li.bytes.len(), 1);
        assert_eq!(li.bytes[0], 1);

        // don't shrink because d7 == b0111 != b111 == d-1
        li = LargeInt{bytes: vec!(u128::MAX - 1, 0)};
        li.shrink();
        assert_eq!(li.bytes.len(), 2);
        assert_eq!(li.bytes[0], u128::MAX - 1);
        assert_eq!(li.bytes[1], 0);

        // don't shrink because b0010 != b10
        li = LargeInt{bytes: vec!(1 << 127, 0)};
        li.shrink();
        assert_eq!(li.bytes.len(), 2);
        assert_eq!(li.bytes[0], 1 << 127);
        assert_eq!(li.bytes[1], 0);
    }

    #[test]
    fn test_add() {

        // two negatives
        let li1 = LargeInt{bytes: vec!(u128::MAX; 2)};
        let li2 = LargeInt{bytes: vec!(u128::MAX; 2)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(u128::MAX - 1)});

        // two negatives (and overflow)
        let li1 = LargeInt{bytes: vec!(1 << 127)};
        let li2 = LargeInt{bytes: vec!(1 << 127)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(0, u128::MAX)});

        // two positives
        let li1 = LargeInt{bytes: vec!(23)};
        let li2 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(25)});

        // two positives (and overflow)
        let li1 = LargeInt{bytes: vec!(1 << 126)};
        let li2 = LargeInt{bytes: vec!(1 << 126)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(1 << 127, 0)});

        // positive and negative
        let li1 = LargeInt{bytes: vec!(2)};
        let li2 = LargeInt{bytes: vec!(u128::MAX)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(1)});

        // different sizes positive
        let li1 = LargeInt{bytes: vec!(3, 1, 1 << 126)};
        let li2 = LargeInt{bytes: vec!(4)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(7, 1, 1 << 126)});

        // different sizes positive (and overflow)
        let li1 = LargeInt{bytes: vec!(3, 1, 1 << 126)};
        let li2 = LargeInt{bytes: vec!(u128::MAX, 0)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(2, 2, 1 << 126)});

        // different sizes and signs
        let li1 = LargeInt{bytes: vec!(3, 1, 1 << 126)};
        let li2 = LargeInt{bytes: vec!(u128::MAX)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(2, 1, 1 << 126)});

        // different sizes and signs (and overflow)
        let li1 = LargeInt{bytes: vec!(3, 1, 1 << 126)};
        let li2 = LargeInt{bytes: vec!(u128::MAX ^ (3))}; // represents -4
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(u128::MAX, 0, 1 << 126)});
    }

    // tests for sub are minimal simply because it uses add
    #[test]
    fn test_sub() {
        let li1 = LargeInt{bytes: vec!(4)};
        let li2 = LargeInt{bytes: vec!(1)};
        assert_eq!(li1 - li2, LargeInt{bytes: vec!(3)});

        let li1 = LargeInt{bytes: vec!(4)};
        let li2 = LargeInt{bytes: vec!(u128::MAX)};
        assert_eq!(li1 - li2, LargeInt{bytes: vec!(5)});
    }
}