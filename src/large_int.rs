use std::u128;
use std::str::FromStr;
use std::num::ParseIntError;
use std::ops::{
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    Shr,
    ShrAssign,
    Shl,
    ShlAssign,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign
};

// store a vector of little-endian, 2's compliment figures
// the sign bit is in the most significant figure (more[0])
// littel endian was chosen so vec operations are faster
#[derive(Clone, Debug, PartialEq)]
pub struct LargeInt {
    // use u128 since, if someone needs a LargeInt, it's likely
    // going to end up larger than u128::MAX
    bytes: Vec<u128> // (called "bytes" because it was originally u8)
}

const SIGN_BIT: u128 = 1u128 << 127;

fn is_u128_negative(val: u128) -> bool {
    (val & SIGN_BIT) > 1
}

// fn add_ascii_chars(lhs: u8, rhs: u8) -> (u8, u8) {

// }

// fn add_string_ints(lhs: &str, rhs: &str) -> String {
//     let str_rep = "".to_owned();
//     let mut overflow = 48u8;
//     let mut result;
//     for i in (0..lhs.len().max(rhs.len())).rev() {
//         if i >= lhs.len() {

//         }
//     }
//     str_rep
// }

// fn multiply_string_int(rep: &mut String, by: i64) {

// }

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
                continue;
            }
            break;
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

    fn shr_no_shrink(self, bits: usize) -> LargeInt {
        let mut remaining = bits;
        let mut result = self.clone();
        let size = result.bytes.len();

        // simply shift chunks right while required
        while remaining > 128 {
            for i in 1..size {
                result.bytes[i - 1] = result.bytes[i];
            }
            result.bytes[size - 1] = 0;
            remaining -= 128;
        }

        // shift the remainder
        let mut result_mask = 0;
        let data_mask = (
            1u128.checked_shl(remaining as u32).unwrap_or(0) as i128 - 1
        ) as u128;
        for i in (0..size).rev() {
            let temp_mask = result.bytes[i] & data_mask;
            result.bytes[i] = result.bytes[i].checked_shr(remaining as u32).unwrap_or(0);
            result.bytes[i] |= result_mask;
            result_mask = temp_mask.checked_shl(128 - remaining as u32).unwrap_or(0);
        }
        result
    }

    fn shl_no_shrink(self, bits: usize) -> LargeInt {
        let mut remaining = bits;
        let mut result = self.clone();
        let size = result.bytes.len();

        // shift chunks left while required
        while remaining > 128 {
            for i in (1..size).rev() {
                result.bytes[i] = result.bytes[i - 1];
            }
            result.bytes[0] = 0;
            remaining -= 128;
        }

        // shift the remainder
        let mut result_mask = 0;
        let data_mask = (
            u128::MAX.checked_shl(128 - remaining as u32).unwrap_or(0) as i128
        ) as u128;
        for i in 0..size {
            let temp_mask = result.bytes[i] & data_mask;
            result.bytes[i] = result.bytes[i].checked_shl(remaining as u32).unwrap_or(0);
            result.bytes[i] |= result_mask;
            result_mask = temp_mask.checked_shr(128 - remaining as u32).unwrap_or(0);
        }
        result
    }

    // fn string_rep(&self) -> String {
    //     let result = self.bytes[0].to_string();
    //     let 

    //     result
    // }
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

    // use the implementation of addition for subtraction
    fn sub(self, other: LargeInt) -> LargeInt {
        self + other.compliment()
    }
}

impl SubAssign for LargeInt {
    fn sub_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() - other).bytes;
    }
}

impl Mul for LargeInt {
    type Output = LargeInt;

    // based off information found here:
    // https://en.wikipedia.org/wiki/Two%27s_complement#Multiplication
    fn mul(self, mut other: LargeInt) -> LargeInt {
        let n = self.bytes.len();
        let m = other.bytes.len();
        let size = n.max(m) * 2;
        other.expand_to(size);
        let zero = LargeInt::from(0);
        let mut result = LargeInt::with_size(size);
        let mut mask = LargeInt::from(1);
        mask.expand_to(n);

        for i in 0..(128 * n) {
            if self.clone() & mask.clone() != zero {
                result += other.clone() << i;
            }
            mask = mask.shl_no_shrink(1);
        }
        result
    }
}

impl MulAssign for LargeInt {
    fn mul_assign(&mut self, rhs: LargeInt) {
        self.bytes = (self.clone() * rhs).bytes;
    }
}

// impl FromStr for LargeInt {
//     type Err = ParseIntError;

//     fn from_str(s: &str) -> Result<LargeInt, ParseIntError> {

//         // read the string 38 characters at a time, since this is
//         // the largest an i128 can be

//     }
// }

impl BitAnd for LargeInt {
    type Output = LargeInt;

    fn bitand(mut self, mut rhs: LargeInt) -> LargeInt {
        let size = self.bytes.len().max(rhs.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to(size);
        rhs.expand_to(size);

        for i in 0..size {
            result.bytes[i] = self.bytes[i] & rhs.bytes[i];
        }
        result.shrink();
        result
    }
}

impl BitOr for LargeInt {
    type Output = LargeInt;

    fn bitor(mut self, mut rhs: LargeInt) -> LargeInt {
        let size = self.bytes.len().max(rhs.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to(size);
        rhs.expand_to(size);

        for i in 0..size {
            result.bytes[i] = self.bytes[i] | rhs.bytes[i];
        }
        result.shrink();
        result
    }
}

impl Shr<usize> for LargeInt {
    type Output = LargeInt;

    fn shr(self, bits: usize) -> LargeInt {
        let mut result = self.shr_no_shrink(bits);
        result.shrink();
        result
    }
}

impl ShrAssign<usize> for LargeInt {
    fn shr_assign(&mut self, bits: usize) {
        self.bytes = (self.clone() >> bits).bytes;
    }
}

impl Shl<usize> for LargeInt {
    type Output = LargeInt;

    fn shl(self, bits: usize) -> LargeInt {
        let mut result = self.shl_no_shrink(bits);
        result.shrink();
        result
    }
}

impl ShlAssign<usize> for LargeInt {
    fn shl_assign(&mut self, bits: usize) {
        self.bytes = (self.clone() << bits).bytes;
    }
}

macro_rules! from_unsigned {
    ( $($t:ident)* ) => {
        $(impl From<$t> for LargeInt {
            fn from(val: $t) -> LargeInt {
                let mut bytes = Vec::new();
                bytes.push(val as u128);
                bytes.push(0);
                let mut result = LargeInt{bytes: bytes};
                result.shrink();
                result
            }
        })*

        #[cfg(test)]
        mod from_unsigned_tests {
            use crate::large_int::{
                LargeInt,
                SIGN_BIT
            };
            
            $(use std::$t;)*

            #[test]
            fn test_from_unsigned() {
                let mut tested_u128 = false;
                let mut tested_others = false;

                $(let li = LargeInt::from(127 as $t);
                assert_eq!(li.bytes[0], 127u128);
                assert!(li.is_positive());

                let li = LargeInt::from($t::MAX);
                
                // u128 needs a special case, since all other values will shrink
                if li.bytes[0] == u128::MAX {
                    assert_eq!(li, LargeInt{bytes: vec!(u128::MAX, 0)});
                    tested_u128 = true;
                } else {
                    assert_eq!(li, LargeInt{bytes: vec!($t::MAX as u128)});
                    tested_others = true;
                }

                assert!(!li.is_negative());)*

                assert!(tested_others && tested_u128);
            }
        }
    };
}

macro_rules! from_signed {
    ( $($t:ident)* ) => {
        $(impl From<$t> for LargeInt {
            fn from(val: $t) -> LargeInt {
                let mut bytes = Vec::new();
                bytes.push(val as u128);
                if val < 0 {
                    bytes.push(u128::MAX);
                } else {
                    bytes.push(0);
                }
                let mut result = LargeInt{bytes: bytes};
                result.shrink();
                result
            }
        })*

        #[cfg(test)]
        mod from_signed_tests {
            use crate::large_int::{
                LargeInt,
                SIGN_BIT
            };
            
            use std::u128;

            #[test]
            fn test_from_signed() {
                $(let li = LargeInt::from(127 as $t);
                assert_eq!(li.bytes[0], 127u128);
                assert!(li.is_positive());

                let li = LargeInt::from(-1 as $t);
                assert_eq!(li.bytes[0], u128::MAX); // 2's compliment rep of -1 is all 1s

                assert!(li.is_negative());)*
            }
        }
    };
}

macro_rules! ops {
    ( $($t:ident)* ) => {
        $(impl Add<$t> for LargeInt {
            type Output = LargeInt;

            fn add(self, other: $t) -> LargeInt {
                let oth = LargeInt::from(other);
                self + oth
            }
        })*

        $(impl AddAssign<$t> for LargeInt {
            fn add_assign(&mut self, other: $t) {
                self.bytes = (self.clone() + other).bytes;
            }
        })*

        $(impl Sub<$t> for LargeInt {
            type Output = LargeInt;

            fn sub(self, other: $t) -> LargeInt {
                let oth = LargeInt::from(other);
                self - oth
            }
        })*

        $(impl SubAssign<$t> for LargeInt {
            fn sub_assign(&mut self, other: $t) {
                self.bytes = (self.clone() - other).bytes;
            }
        })*
    };
}

from_signed!(i8 i32 i64 i128 isize);
from_unsigned!(u8 u32 u64 u128 usize);
ops!(i8 i32 i64 i128 isize u8 u32 u64 u128 usize);

#[cfg(test)]
mod tests {
    use crate::large_int::{
        LargeInt,
        SIGN_BIT
    };
    
    use std::u128;

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

        li = LargeInt{bytes: vec!(0, 0, 2)};
        li.shrink();
        assert_eq!(li.bytes.len(), 3);
        assert_eq!(li, LargeInt{bytes: vec!(0, 0, 2)});
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

        let li1 = LargeInt{bytes: vec!(u128::MAX, 0)};
        let li2 = LargeInt{bytes: vec!(u128::MAX, 0)};
        assert_eq!(li1 + li2, LargeInt{bytes: vec!(u128::MAX - 1, 1)});

        // test AddAssign trait too
        let mut li1 = LargeInt{bytes: vec!(2)};
        let li2 = LargeInt{bytes: vec!(5)};
        li1 += li2;
        assert_eq!(li1, LargeInt{bytes: vec!(7)});

        let mut li1 = LargeInt{bytes: vec!(u128::MAX - 3)};
        li1 += 6;
        assert_eq!(li1, LargeInt{bytes: vec!(2)});
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

        let mut li1 = LargeInt{bytes: vec!(10)};
        let li2 = LargeInt{bytes: vec!(u128::MAX)};
        li1 -= li2;
        assert_eq!(li1, LargeInt{bytes: vec!(11)});

        let mut li1 = LargeInt{bytes: vec!(u128::MAX - 4)};
        li1 -= 10;
        assert_eq!(li1, LargeInt{bytes: vec!(u128::MAX - 14)});
    }

    #[test]
    fn test_shr() {

        // test easy case
        let li = LargeInt{bytes: vec!(3)};
        assert_eq!(li >> 1, LargeInt{bytes: vec!(1)});

        let li = LargeInt{bytes: vec!(u128::MAX)};
        assert_eq!(li >> 128, LargeInt{bytes: vec!(0)});

        let li = LargeInt{bytes: vec!(4, 3)};
        assert_eq!(li >> 1, LargeInt{bytes: vec!((1 << 127) + 2, 1)});

        let li = LargeInt{bytes: vec!(4, 3, 4)};
        assert_eq!(li >> 2, LargeInt{bytes: vec!((3 << 126) + 1, 0, 1)});
        let li = LargeInt{bytes: vec!(4, 3, 4)};
        assert_eq!(li >> 3, LargeInt{bytes: vec!((3 << 125), 1 << 127, 0)});

        // test large shifts
        let li = LargeInt{bytes: vec!(4, 3, 4)};
        assert_eq!(li >> 257, LargeInt{bytes: vec!(2)});

        // test shift with 0 as arg
        let li = LargeInt{bytes: vec!(4, 3, 4)};
        assert_eq!(li >> 0, LargeInt{bytes: vec!(4, 3, 4)});
    }

    #[test]
    fn test_shl() {

        // test easy case
        let li = LargeInt{bytes: vec!(3)};
        assert_eq!(li << 1, LargeInt{bytes: vec!(6)});

        let li = LargeInt{bytes: vec!(0, 3)};
        assert_eq!(li << 1, LargeInt{bytes: vec!(0, 6)});

        let li = LargeInt{bytes: vec!(u128::MAX)};
        assert_eq!(li << 2, LargeInt{bytes: vec!(u128::MAX - 3)});

        let li = LargeInt{bytes: vec!(1 << 127)};
        assert_eq!(li << 1, LargeInt{bytes: vec!(0)});

        let li = LargeInt{bytes: vec!(u128::MAX, 1)};
        assert_eq!(li << 1, LargeInt{bytes: vec!(u128::MAX - 1, 3)});

        let li = LargeInt{bytes: vec!(1, 2, 3)};
        assert_eq!(li << 257, LargeInt{bytes: vec!(0, 0, 2)});

        let li = LargeInt{bytes: vec!(1, 2, 3)};
        assert_eq!(li << 0, LargeInt{bytes: vec!(1, 2, 3)});

        let li = LargeInt{bytes: vec!(1, 2, 3)};
        assert_eq!(li << 130, LargeInt{bytes: vec!(0, 4, 8)});
    }

    #[test]
    fn test_mul() {
        let li1 = LargeInt{bytes: vec!(3)};
        let li2 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(6)});

        let li1 = LargeInt{bytes: vec!(3)};
        let li2 = LargeInt{bytes: vec!(u128::MAX)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(u128::MAX - 2)});

        let li1 = LargeInt{bytes: vec!(3)};
        let li2 = LargeInt{bytes: vec!(0)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0)});

        let li1 = LargeInt{bytes: vec!(1u128 << 127)};
        let li2 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0, 1)});

        // check if both orders work the same
        let li2 = LargeInt{bytes: vec!(1u128 << 127, 1)};
        let li1 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0, 3)});
        let li1 = LargeInt{bytes: vec!(1u128 << 127, 1)};
        let li2 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0, 3)});
    }

    #[test]
    fn test_bitand() {
        let li1 = LargeInt{bytes: vec!(3)};
        let li2 = LargeInt{bytes: vec!(1)};
        assert_eq!(li1 & li2, LargeInt{bytes: vec!(1)});

        let li1 = LargeInt{bytes: vec!(3, 4)};
        let li2 = LargeInt{bytes: vec!(1)};
        assert_eq!(li1 & li2, LargeInt{bytes: vec!(1)});

        let li1 = LargeInt{bytes: vec!(3, 4)};
        let li2 = LargeInt{bytes: vec!(1, 2)};
        assert_eq!(li1 & li2, LargeInt{bytes: vec!(1)});

        let li1 = LargeInt{bytes: vec!(3, 5)};
        let li2 = LargeInt{bytes: vec!(1, 4)};
        assert_eq!(li1 & li2, LargeInt{bytes: vec!(1, 4)});
    }

    #[test]
    fn test_bitor() {
        let li1 = LargeInt{bytes: vec!(3)};
        let li2 = LargeInt{bytes: vec!(1)};
        assert_eq!(li1 | li2, LargeInt{bytes: vec!(3)});

        let li1 = LargeInt{bytes: vec!(3, 4)};
        let li2 = LargeInt{bytes: vec!(1)};
        assert_eq!(li1 | li2, LargeInt{bytes: vec!(3, 4)});

        let li1 = LargeInt{bytes: vec!(3, 4)};
        let li2 = LargeInt{bytes: vec!(1, 2)};
        assert_eq!(li1 | li2, LargeInt{bytes: vec!(3, 6)});

        let li1 = LargeInt{bytes: vec!(3, 5)};
        let li2 = LargeInt{bytes: vec!(1, 4)};
        assert_eq!(li1 | li2, LargeInt{bytes: vec!(3, 5)});
    }
}