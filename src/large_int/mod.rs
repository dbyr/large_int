mod ops;
mod utils;
mod iter_ops;

use std::u128;

// store a vector of little-endian, 2's compliment figures
// the sign bit is in the most significant figure (more[0])
// littel endian was chosen so vec operations are faster

/// An unsigned integer that is unbounded in both positive
/// and negative.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LargeInt {
    // use u128 since, if someone needs a LargeInt, it's likely
    // going to end up larger than u128::MAX
    bytes: Vec<u128> // (called "bytes" because it was originally u8)
}

const SIGN_BIT: u128 = 1u128 << 127;
const BITS_PER_CHUNK: u32 = 128;

fn is_u128_negative(val: u128) -> bool {
    (val & SIGN_BIT) > 1
}

fn reorder_by_ones_count(left: LargeInt, right: LargeInt) -> (LargeInt, LargeInt) {
    if right.count_ones() < left.count_ones() {
        (right, left)
    } else {
        (left, right)
    }
}

impl LargeInt {
    /// Returns a default LargeInt (default == 0)
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let zero = LargeInt::new();
    /// ```
    pub fn new() -> LargeInt {
        LargeInt{bytes: vec!(0)}
    }

    fn with_size(size: usize) -> LargeInt {
        LargeInt{bytes: vec!(0; size)}
    }

    /// Checks if this value is negative
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let neg = LargeInt::from(-1);
    /// let pos = LargeInt::from(1);
    /// assert!(neg.is_negative());
    /// assert!(!pos.is_negative());
    /// ```
    /// 
    /// Returns true if this LargeInt is negative, false otherwise
    pub fn is_negative(&self) -> bool {
        is_u128_negative(self.bytes[self.bytes.len() - 1])
    }

    /// Checks if this value is positive
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let neg = LargeInt::from(-1);
    /// let pos = LargeInt::from(1);
    /// assert!(!neg.is_positive());
    /// assert!(pos.is_positive());
    /// ```
    /// 
    /// Returns true if this LargeInt is positive, false otherwise
    pub fn is_positive(&self) -> bool {
        !self.is_negative()
    }

    /// Calculates the absolute value of this LargeInt
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let neg = LargeInt::from(-1).abs();
    /// let pos = LargeInt::from(1).abs();
    /// assert!(neg == pos);
    /// ```
    /// 
    /// Returns the absolute value as a LargeInt
    pub fn abs(&self) -> LargeInt {
        if self.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }

    /// Calculates this LargeInt to the power given
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let val = LargeInt::from(3);
    /// assert!(val.pow(3) == LargeInt::from(27));
    /// ```
    /// 
    /// Returns this LargeInt to the power given as a LargeInt
    pub fn pow(&self, exp: u32) -> LargeInt {
        let mut result = LargeInt::from(1);
        let mut mask = 1u32;
        let mask_max = 1u32 << 31;
        let mask_min = 1u32;
        while mask != mask_max && mask << 1 <= exp {
            mask <<= 1;
        }
        loop {
            result *= result.clone();
            if exp & mask > 0 {
                result *= self.clone();
            }
            if mask == mask_min { break; }
            mask >>= 1;
        }
        result
    }

    fn shrink(&mut self) {
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

    // for use when maintaining sign is undesirable
    // (e.g. for masks)
    fn expand_to_ignore_sign(&mut self, size: usize) {
        while self.bytes.len() < size {
            self.bytes.push(0);
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

    fn count_ones(&self) -> u32 {
        let mut count = 0;
        for byte in self.bytes.iter() {
            count += byte.count_ones();
        }
        count
    }

    fn add_no_shrink(mut self, mut other: LargeInt) -> LargeInt {

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
        result
    }

    // use the implementation of addition for subtraction
    fn sub_no_shrink(self, other: LargeInt) -> LargeInt {
        self.add_no_shrink(other.compliment())
    }

    fn mul_no_shrink(mut self, mut other: LargeInt) -> LargeInt {

        // based off information found here:
        // https://en.wikipedia.org/wiki/Two%27s_complement#Multiplication
        let mut negate = false;
        if self.is_negative() {
            self = self.compliment();
            negate = !negate;
        }
        if other.is_negative() {
            other = other.compliment();
            negate = !negate;
        }
        
        // slightly optimise by multiplying by the value with less 1's
        let (multiplier, mut multiplicand) = reorder_by_ones_count(self, other);
        let n = multiplier.bytes.len();
        let m = multiplicand.bytes.len();
        let size = n.max(m) * 2;
        multiplicand.expand_to(size);
        let zero = LargeInt::from(0);
        let mut result = LargeInt::with_size(size);
        let mut mask = LargeInt::from(1);
        mask.expand_to_ignore_sign(n);

        for i in 0..(128 * n) {
            if multiplier.clone() & mask.clone() != zero {
                result += multiplicand.clone() << i;
            }
            mask <<= 1;
        }
        if negate {
            result.compliment()
        } else {
            result
        }
    }

    /// Divides self by other and returns both the result and remainder
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(10);
    /// let five = LargeInt::from(4);
    /// let (result, remainder) = ten.div_with_remainder(five);
    /// assert!(result == LargeInt::from(2));
    /// assert!(remainder == LargeInt::from(2));
    /// ```
    /// 
    /// Returns a pair of LargeInts that represent the result and remainder
    /// of the division respectively.
    /// 
    /// # Panics
    /// Panics if other is 0
    pub fn div_with_remainder<T: Into<LargeInt>>(self, other: T) -> (LargeInt, LargeInt) {
        let (mut result, mut remainder) = self.div_with_remainder_no_shrink(other.into());
        result.shrink();
        remainder.shrink();
        (result, remainder)
    }

    fn div_with_remainder_no_shrink(mut self, mut other: LargeInt) -> (LargeInt, LargeInt) {
        // adapted from psuedo code here:
        // https://en.wikipedia.org/wiki/Division_algorithm#Long_division
        let zero = LargeInt::from(0);
        if other == zero {
            panic!("Attempted divide by 0");
        }

        // deal with signage
        let mut negative = false;
        if self.is_negative() {
            self = self.compliment();
            negative = !negative;
        }
        if other.is_negative() {
            other = other.compliment();
            negative = !negative;
        }
        
        // perform the division
        let size = self.bytes.len();
        let mut result = LargeInt::with_size(size);
        let mut remainder = LargeInt::with_size(size + 1);
        let mut mask = LargeInt::from(1);
        mask.expand_to_ignore_sign(size);
        mask <<= (size * 128) - 1;
        for _ in 0..(size * 128) {
            remainder <<= 1;
            if self.clone() & mask.clone() != zero {
                remainder += 1;
            }
            if remainder >= other {
                remainder -= other.clone();
                result |= mask.clone();
            }
            remainder.expand_to_ignore_sign(size + 1);
            mask >>= 1;
        }
        if negative {
            result = result.compliment();
        }
        (result, remainder)
    }
}

#[cfg(test)]
mod internal_tests {
    use std::str::FromStr;
    use std::string::ToString;
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
        assert_eq!(li >> 257, LargeInt{bytes: vec!(2, 0, 0)});

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
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0, u128::MAX)});

        // check if both orders work the same
        let li2 = LargeInt{bytes: vec!(1u128 << 127, 1)};
        let li1 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0, 3)});
        let li1 = LargeInt{bytes: vec!(1u128 << 127, 1)};
        let li2 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(0, 3)});

        let li1 = LargeInt{bytes: vec!(1u128 << 126)};
        let li2 = li1.clone();
        assert_eq!(li1 * li2, LargeInt{bytes: vec!(1, 0)} << 252);

        let mut li1 = LargeInt{bytes: vec!(10)};
        li1 *= 10;
        assert_eq!(li1, LargeInt::from(100));
    }

    #[test]
    fn test_div() {
        let li1 = LargeInt{bytes: vec!(6)};
        let li2 = LargeInt{bytes: vec!(3)};
        assert_eq!(li1 / li2, LargeInt{bytes: vec!(2)});

        let li1 = LargeInt{bytes: vec!(6)};
        let li2 = LargeInt{bytes: vec!(4)};
        assert_eq!(li1 / li2, LargeInt{bytes: vec!(1)});

        let li1 = LargeInt{bytes: vec!(0, 1)};
        let li2 = LargeInt{bytes: vec!(2)};
        assert_eq!(li1 / li2, LargeInt{bytes: vec!(1 << 127, 0)});

        let li1 = LargeInt{bytes: vec!(0, 2)};
        let li2 = LargeInt{bytes: vec!(8)};
        assert_eq!(li1 / li2, LargeInt{bytes: vec!(1 << 126)});

        let li1 = LargeInt::from(-10);
        let li2 = LargeInt::from(2);
        assert_eq!(li1 / li2, LargeInt::from(-5));

        let li1 = LargeInt::from(-10);
        let li2 = LargeInt::from(2);
        assert_eq!(li1 / li2, LargeInt::from(-5));
        let li1 = LargeInt::from(-10);
        let li2 = LargeInt::from(-2);
        assert_eq!(li1 / li2, LargeInt::from(5));
        let li1 = LargeInt::from(10);
        let li2 = LargeInt::from(-2);
        assert_eq!(li1 / li2, LargeInt::from(-5));

        let li1 = LargeInt::from(10);
        let li2 = LargeInt::from(20);
        assert_eq!(li1 / li2, LargeInt::from(0));

        let li1 = LargeInt::from(1);
        let li2 = LargeInt::from(10);
        assert_eq!(li1 / li2, LargeInt::from(0));

        let li1 = LargeInt::from(100);
        let li2 = LargeInt::from(10);
        assert_eq!(li1 / li2, LargeInt::from(10));

        let mut li1 = LargeInt::from(100);
        li1 /= 10;
        assert_eq!(li1, LargeInt::from(10));

        let li1 = LargeInt::from_str("340282366920938463463374607431768211465").unwrap();
        let li2 = LargeInt::from_str("100000000000000000000000000000000000000").unwrap();
        assert_eq!(li1 / li2, LargeInt::from(3));

        let li1 = LargeInt::from_str("-340282366920938463463374607431768211465").unwrap();
        let li2 = LargeInt::from_str("-100000000000000000000000000000000000000").unwrap();
        assert_eq!(li1 / li2, LargeInt::from(3));
    }

    #[test]
    #[should_panic(expected = "Attempted divide by 0")]
    fn test_div_by_0() {
        let li1 = LargeInt::from(42);
        let mut val = li1 / 0;
        assert!(false);
        val *= 10; // just to get rid of the "unused" warning...
    }

    #[test]
    fn test_order() {
        let li1 = LargeInt::from(-10);
        let li2 = LargeInt::from(2);
        assert_eq!(li1 < li2, true);

        let li1 = LargeInt::from(-10);
        let li2 = LargeInt::from(2);
        assert_eq!(li1 > li2, false);

        let li1 = LargeInt::from(2);
        let li2 = LargeInt::from(10);
        assert_eq!(li1 <= li2, true);

        let li1 = LargeInt::from(-2);
        let li2 = LargeInt::from(-10);
        assert_eq!(li1 >= li2, true);

        let li1 = LargeInt::from(3);
        let li2 = LargeInt::from(3);
        assert_eq!(li1 <= li2, true);
        let li1 = LargeInt::from(3);
        let li2 = LargeInt::from(3);
        assert_eq!(li1 >= li2, true);
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

    #[test]
    fn test_rem() {
        let li1 = LargeInt{bytes: vec!(6)};
        let li2 = LargeInt{bytes: vec!(3)};
        assert_eq!(li1 % li2, LargeInt{bytes: vec!(0)});

        let li1 = LargeInt{bytes: vec!(6)};
        let li2 = LargeInt{bytes: vec!(4)};
        assert_eq!(li1 % li2, LargeInt{bytes: vec!(2)});

        let li1 = LargeInt{bytes: vec!(6)};
        let li2 = LargeInt{bytes: vec!(7)};
        assert_eq!(li1 % li2, LargeInt{bytes: vec!(6)});
    }

    #[test]
    fn test_neg() {
        let li1 = LargeInt::from(2);
        assert_eq!(-li1, LargeInt::from(-2));

        let li1 = LargeInt::from(-2);
        assert_eq!(-li1, LargeInt::from(2));

        let li1 = LargeInt{bytes: vec!(1, 2)};
        assert_eq!(-li1, LargeInt{bytes: vec!(u128::MAX, u128::MAX - 2)});
    }

    #[test]
    fn test_xor() {
        let li1 = LargeInt::from(2);
        let li2 = LargeInt::from(3);
        assert_eq!(li1 ^ li2, LargeInt::from(1));

        let li1 = LargeInt::from(-2);
        let li2 = LargeInt::from(1);
        assert_eq!(li1 ^ li2, LargeInt::from(-1));

        let li1 = LargeInt{bytes: vec!(2, 4)};
        let li2 = LargeInt{bytes: vec!(1, 6)};
        assert_eq!(li1 ^ li2, LargeInt{bytes: vec!(3, 2)});
    }

    #[test]
    fn test_not() {
        let li1 = LargeInt::from(4);
        assert_eq!(!li1, LargeInt::from(0));
        
        let li1 = LargeInt::from(0);
        assert_eq!(!li1, LargeInt::from(1));

        let li1 = LargeInt::from(-1);
        assert_eq!(!li1, LargeInt::from(0));

        let li1 = LargeInt{bytes: vec!(1, 2, 3)};
        assert_eq!(!li1, LargeInt::from(0));
    }

    #[test]
    fn test_from_string() {
        let li1 = LargeInt::from_str("0").unwrap();
        let li2 = LargeInt::from(0);
        assert_eq!(li1, li2);
        let li1 = LargeInt::from_str("-1").unwrap();
        let li2 = LargeInt::from(-1);
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("5").unwrap();
        let li2 = LargeInt::from(5);
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("-5").unwrap();
        let li2 = LargeInt::from(-5);
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("531").unwrap();
        let li2 = LargeInt::from(531);
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("-531").unwrap();
        let li2 = LargeInt::from(-531);
        assert_eq!(li1, li2);

        // max u128 is 340282366920938463463374607431768211455
        let li1 = LargeInt::from_str("340282366920938463463374607431768211456").unwrap();
        let li2 = LargeInt{bytes: vec!(0, 1)};
        assert_eq!(li1, li2);
        let li1 = LargeInt::from_str("-340282366920938463463374607431768211456").unwrap();
        let li2 = LargeInt{bytes: vec!(0, 1)}.compliment();
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("340282366920938463463374607431768211458").unwrap();
        let li2 = LargeInt{bytes: vec!(2, 1)};
        assert_eq!(li1, li2);
        let li1 = LargeInt::from_str("-340282366920938463463374607431768211458").unwrap();
        let li2 = LargeInt{bytes: vec!(2, 1)}.compliment();
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("340282366920938463463374607431768211465").unwrap();
        let li2 = LargeInt{bytes: vec!(9, 1)};
        assert_eq!(li1, li2);

        let li1 = LargeInt::from_str("123abc");
        match li1 {
            Ok(_) => assert!(false),
            Err(_) => assert!(true)
        }
    }

    #[test]
    fn test_to_string() {
        let li1 = LargeInt::from(0).to_string();
        let li2 = "0";
        assert_eq!(li1, li2);
        let li1 = LargeInt::from(-1).to_string();
        let li2 = "-1";
        assert_eq!(li1, li2);

        let li1 = LargeInt::from(5).to_string();
        let li2 = "5";
        assert_eq!(li1, li2);

        let li1 = LargeInt::from(-5).to_string();
        let li2 = "-5";
        assert_eq!(li1, li2);

        let li1 = LargeInt::from(531).to_string();
        let li2 = "531";
        assert_eq!(li1, li2);

        let li1 = LargeInt::from(-531).to_string();
        let li2 = "-531";
        assert_eq!(li1, li2);

        let li1 = (LargeInt::from(u128::MAX) + 10u8).to_string();
        let li2 = "340282366920938463463374607431768211465";
        assert_eq!(li1, li2);
        let li1 = (LargeInt::from(u128::MAX) + 2u8).compliment().to_string();
        let li2 = "-340282366920938463463374607431768211457";
        assert_eq!(li1, li2);

        let li1 = (LargeInt{bytes: vec!(319435266158123073073250785136463577088, 2)}).to_string();
        let li2 = "1000000000000000000000000000000000000000";
        assert_eq!(li1, li2);
    }
}