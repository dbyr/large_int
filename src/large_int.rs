use std::str::FromStr;
use std::convert::TryFrom;
use std::cmp::Ordering;
use std::num::ParseIntError;
use std::iter::{
    Sum,
    Product
};
use std::fmt::{
    Display,
    Formatter,
    Result as FmtResult,
    LowerExp
};
use std::{
    u128,
    i128,
    u64,
    i64,
    u32,
    i32,
    u16,
    i16,
    u8,
    i8,
    usize,
    isize
};
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
    BitOrAssign,
    Rem,
    RemAssign,
    Neg,
    BitXor,
    BitXorAssign,
    Not
};

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

impl Default for LargeInt {
    /// Returns a default LargeInt (default == 0)
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let zero = LargeInt::new();
    /// let default = LargeInt::default();
    /// assert!(zero == default);
    /// ```
    /// 
    /// Returns the default LargeInt (0)
    fn default() -> LargeInt {
        LargeInt::new()
    }
}

impl Add for LargeInt {
    type Output = LargeInt;

    /// Adds two LargeInts
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// assert!(ten + two == LargeInt::from(12));
    /// ```
    /// 
    /// Returns a LargeInt as the sum of the two LargeInts
    fn add(self, other: LargeInt) -> LargeInt {
        let mut result = self.add_no_shrink(other);
        result.shrink();
        result
    }
}

impl AddAssign for LargeInt {
    /// Adds a LargeInt to this LargeInt
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// ten += two;
    /// assert!(ten == LargeInt::from(12));
    /// ```
    fn add_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() + other).bytes;
    }
}

impl Sub for LargeInt {
    type Output = LargeInt;

    /// Subtracts a LargeInt from another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// assert!(ten - two == LargeInt::from(8));
    /// ```
    /// 
    /// Returns self minus other as a LargeInt
    fn sub(self, other: LargeInt) -> LargeInt {
        let mut result = self.sub_no_shrink(other);
        result.shrink();
        result
    }
}

impl SubAssign for LargeInt {
    /// Subtracts a LargeInt from this LargeInt
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// ten -= two;
    /// assert!(ten == LargeInt::from(8));
    /// ```
    fn sub_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() - other).bytes;
    }
}

impl Mul for LargeInt {
    type Output = LargeInt;

    /// Multiplies two LargeInts
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// assert!(ten * two == LargeInt::from(20));
    /// ```
    /// 
    /// Returns a LargeInt that is the product of the two LargeInts given
    fn mul(self, other: LargeInt) -> LargeInt {
        let mut result = self.mul_no_shrink(other);
        result.shrink();
        result
    }
}

impl MulAssign for LargeInt {
    /// Multiplies this LargeInt by another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// ten *= two;
    /// assert!(ten == LargeInt::from(20));
    /// ```
    fn mul_assign(&mut self, rhs: LargeInt) {
        self.bytes = (self.clone() * rhs).bytes;
    }
}

impl Div for LargeInt {
    type Output = LargeInt;

    /// Divides a LargeInt by another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// assert!(ten / two == LargeInt::from(5));
    /// ```
    /// 
    /// Returns self divided by other as a LargeInt
    /// 
    /// # Panics
    /// Panics if other is 0
    fn div(self, other: LargeInt) -> LargeInt {
        self.div_with_remainder(other).0
    }
}

impl DivAssign for LargeInt {
    /// Divides this LargeInt by another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut ten = LargeInt::from(10);
    /// let two = LargeInt::from(2);
    /// ten /= two;
    /// assert!(ten == LargeInt::from(5));
    /// ```
    /// 
    /// # Panics
    /// Panics if other is 0
    fn div_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() / other).bytes;
    }
}

impl Rem for LargeInt {
    type Output = LargeInt;

    /// Gets the remainder of a division operation
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(11);
    /// let two = LargeInt::from(5);
    /// assert!(ten % two == LargeInt::from(1));
    /// ```
    /// 
    /// Returns the remainder as a LargeInt
    fn rem(self, other: LargeInt) -> LargeInt {
        self.div_with_remainder(other).1
    }
}

impl RemAssign for LargeInt {
    /// Assigns the remainder of this LargeInt divided by another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut eleven = LargeInt::from(11);
    /// let five = LargeInt::from(5);
    /// eleven %= five;
    /// assert!(eleven == LargeInt::from(1));
    /// ```
    fn rem_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() % other).bytes;
    }
}

impl BitAnd for LargeInt {
    type Output = LargeInt;

    /// Performs a bitwise and on two LargeInts
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let three = LargeInt::from(3);
    /// let two = LargeInt::from(2);
    /// assert!(three & two == LargeInt::from(2));
    /// ```
    /// 
    /// Returns the bitwise and as a LargeInt
    fn bitand(mut self, mut rhs: LargeInt) -> LargeInt {
        let size = self.bytes.len().max(rhs.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to_ignore_sign(size);
        rhs.expand_to_ignore_sign(size);

        for i in 0..size {
            result.bytes[i] = self.bytes[i] & rhs.bytes[i];
        }
        result.shrink();
        result
    }
}

impl BitAndAssign for LargeInt {
    /// Bitwise ands this LargeInt with another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut three = LargeInt::from(3);
    /// let two = LargeInt::from(2);
    /// three &= two;
    /// assert!(three == LargeInt::from(2));
    /// ```
    fn bitand_assign(&mut self, rhs: LargeInt) {
        self.bytes = (self.clone() & rhs).bytes;
    }
}

impl BitOr for LargeInt {
    type Output = LargeInt;

    /// Bitwise ors two LargeInts
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let two = LargeInt::from(2);
    /// assert!(five | two == LargeInt::from(7));
    /// ```
    /// 
    /// Returns the bitwise or as a LargeInt
    fn bitor(mut self, mut rhs: LargeInt) -> LargeInt {
        let size = self.bytes.len().max(rhs.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to_ignore_sign(size);
        rhs.expand_to_ignore_sign(size);

        for i in 0..size {
            result.bytes[i] = self.bytes[i] | rhs.bytes[i];
        }
        result.shrink();
        result
    }
}

impl BitOrAssign for LargeInt {
    /// Bitwise ors this LargeInt with another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut five = LargeInt::from(5);
    /// let two = LargeInt::from(2);
    /// five |= two;
    /// assert!(five == LargeInt::from(7));
    /// ```
    fn bitor_assign(&mut self, rhs: LargeInt) {
        self.bytes = (self.clone() | rhs).bytes;
    }
}

impl Shr<usize> for LargeInt {
    type Output = LargeInt;

    /// Shifts the bits right in a LargeInt by bits.
    /// Overflow is lost.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let ten = LargeInt::from(10);
    /// assert!(ten >> 1 == LargeInt::from(5));
    /// ```
    /// 
    /// Returns the shifted bits as a LargeInt
    fn shr(self, bits: usize) -> LargeInt {
        let mut remaining = bits;
        let mut result = self.clone();
        let size = result.bytes.len();

        // simply shift chunks right while required
        let shifts = remaining / BITS_PER_CHUNK as usize;
        remaining = remaining % BITS_PER_CHUNK as usize;
        for i in shifts..size {
            result.bytes[i - shifts] = result.bytes[i];
        }
        for i in size - shifts..size {
            result.bytes[i] = 0;
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
            result_mask = temp_mask.checked_shl(BITS_PER_CHUNK - remaining as u32).unwrap_or(0);
        }
        result
    }
}

impl ShrAssign<usize> for LargeInt {
    /// Shifts the bits right in this LargeInt by bits.
    /// Overflow is lost.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut ten = LargeInt::from(10);
    /// ten >>= 1;
    /// assert!(ten == LargeInt::from(5));
    /// ```
    fn shr_assign(&mut self, bits: usize) {
        self.bytes = (self.clone() >> bits).bytes;
    }
}

impl Shl<usize> for LargeInt {
    type Output = LargeInt;

    /// Shifts the bits left in a LargeInt by bits.
    /// Overflow is lost.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// assert!(five << 1 == LargeInt::from(10));
    /// ```
    /// 
    /// Returns the shifted bits as a LargeInt
    fn shl(self, bits: usize) -> LargeInt {
        let mut remaining = bits;
        let mut result = self.clone();
        let size = result.bytes.len();

        // shift chunks left while required
        let shifts = remaining / BITS_PER_CHUNK as usize;
        remaining = remaining % BITS_PER_CHUNK as usize;
        for i in (shifts..size).rev() {
            result.bytes[i] = result.bytes[i - shifts];
        }
        for i in 0..shifts {
            result.bytes[i] = 0;
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
}

impl ShlAssign<usize> for LargeInt {
    /// Shifts the bits left in this LargeInt by bits.
    /// Overflow is lost.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut five = LargeInt::from(5);
    /// five <<= 1;
    /// assert!(five == LargeInt::from(10));
    /// ```
    fn shl_assign(&mut self, bits: usize) {
        self.bytes = (self.clone() << bits).bytes;
    }
}

impl Neg for LargeInt {
    type Output = LargeInt;

    /// Negates a LargeInt
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// assert!(-five == LargeInt::from(-5));
    /// ```
    /// 
    /// Returns the negated integer as a LargeInt
    fn neg(self) -> LargeInt {
        self.compliment()
    }
}

impl BitXor for LargeInt {
    type Output = LargeInt;

    /// Bitwise xors two LargeInts
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let three = LargeInt::from(3);
    /// assert!(five ^ three == LargeInt::from(6));
    /// ```
    /// 
    /// Returns the bitwise xor as a LargeInt
    fn bitxor(mut self, mut other: LargeInt) -> LargeInt {
        let size = self.bytes.len().max(other.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to_ignore_sign(size);
        other.expand_to_ignore_sign(size);
        for i in 0..size {
            result.bytes[i] = self.bytes[i] ^ other.bytes[i];
        }
        result
    }
}

impl BitXorAssign for LargeInt {
    /// Bitwise ors this LargeInt by another
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let mut five = LargeInt::from(5);
    /// let three = LargeInt::from(3);
    /// five ^= three;
    /// assert!(five == LargeInt::from(6));
    /// ```
    fn bitxor_assign(&mut self, other: LargeInt) {
        self.bytes = (self.clone() ^ other).bytes;
    }
}

impl Not for LargeInt {
    type Output = LargeInt;

    /// Get the C equivalent "not" of a LargeInt
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// assert!(!five == LargeInt::from(0));
    /// ```
    /// 
    /// Returns, as a LargeInt, 0 if self != 0, or 1 otherwise
    fn not(self) -> LargeInt {
        let zero = LargeInt::new();
        if self == zero {
            LargeInt::from(1)
        } else {
            zero
        }
    }
}

impl Ord for LargeInt {
    /// Determines if a LargeInt is larger, equal or greater than another.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let four = LargeInt::from(4);
    /// assert!(five > four);
    /// assert!(four < five);
    /// ```
    /// 
    /// Returns an ordering of the comparison
    fn cmp(&self, other: &LargeInt) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for LargeInt {
    /// Determines if a LargeInt is larger, equal or greater than another.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let four = LargeInt::from(4);
    /// assert!(five > four);
    /// assert!(four < five);
    /// ```
    /// 
    /// Returns an ordering of the comparison
    fn partial_cmp(&self, other: &LargeInt) -> Option<Ordering> {
        let tester = self.clone() - other.clone();
        if tester == LargeInt::new() {
            Some(Ordering::Equal)
        } else if tester.is_negative() {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

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

impl FromStr for LargeInt {
    type Err = ParseIntError;

    /// Creates a LargeInt from a string
    /// 
    /// # Examples
    /// ```
    /// use std::str::FromStr;
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five_str = LargeInt::from_str("5").unwrap();
    /// let five = LargeInt::from(5);
    /// assert!(five == five_str);
    /// ```
    /// 
    /// Returns a Result of either a LargeInt representation 
    /// of the decimal string (Ok), or a ParseIntError (Err)
    fn from_str(s: &str) -> Result<LargeInt, ParseIntError> {
        let mut weight = LargeInt::from(1);
        let mut negative = false;
        let mut result = LargeInt::from(0);
        let inc_fin = if &s[..=0] == "-" {
            negative = true;
            1
        } else {
            0
        };

        // calculate value based off sum of digits * weight
        // (eg, 123 = 1 * 10^2 + 2 * 10^1 + 3 * 10^0)
        for i in (inc_fin..s.len()).rev() {
            let digit = u8::from_str(&s[i..=i])?;
            result += weight.clone() * digit;
            weight *= 10;
        }
        if negative {
            result = result.compliment();
        }
        Ok(result)
    }
}

impl Display for LargeInt {
    /// Converts a LargeInt into a string representation
    /// 
    /// # Examples
    /// ```
    /// use std::string::ToString;
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let four = LargeInt::from(445);
    /// assert!(five.to_string() == "5");
    /// assert!(format!("{:.1}", four) == "4.4e2");
    /// ```
    /// 
    /// Returns a Result of whether or not the conversion was successful
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let mut result = String::new();
        let mut num = self.clone();
        let mut divisor = LargeInt::from(1);
        let zero = LargeInt::from(0);
        if num.is_negative() {
            result.push('-');
            num = num.compliment();
        }
        let keep_going: Box<dyn Fn(usize) -> bool>
        = if let Some(precision) = f.precision() {
            Box::new(move |i| i <= precision)
        } else {
            Box::new(|_| true)
        };

        // find the largest divisor
        let mut power = 0;
        while divisor < num {
            divisor *= 10;
            power += 1;
        }
        divisor /= 10; // re-adjust
        power -= 1;

        // now calculate the digit at each position
        let mut i = 0;
        while divisor != zero && keep_going(i) {
            let digit = num.clone() / divisor.clone();
            result.push_str(&digit.bytes[0].to_string()); // should rep. whole number
            num -= digit * divisor.clone();
            divisor /= 10;
            i += 1;
        }

        // deal with precision and two special cases
        // these cases need to be handled specially because of the
        // two's complement problem
        if let Some(precision) = f.precision() {
            let length = if result == "" {
                power = 0;
                result.push_str("0.0");
                result.len() - 2
            } else if result == "-" {
                power = 0;
                result.push_str("1.0");
                result.len() - 3
            } else if &result[0..=0] == "-" {
                result.insert(2, '.');
                result.len() - 3
            } else {
                result.insert(1, '.');
                result.len() - 2
            };
            if length < precision {
                let remaining = precision - length;
                result.push_str(&"0".repeat(remaining));
            }
            result.push('e');
            result.push_str(&power.to_string());
        } else {
            if result == "" {
                result.push('0');
            } else if result == "-" {
                result.push('1');
            }
        }
        f.write_str(&result)
    }
}

impl LowerExp for LargeInt {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:.1}", self)
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

        // allow attempting to transform into primitives
        $(impl TryFrom<LargeInt> for $t {
            type Error = String;

            fn try_from(val: LargeInt) -> Result<$t, Self::Error> {
                if val >= LargeInt::from($t::MIN) && val <= LargeInt::from($t::MAX) {
                    Ok(val.bytes[0] as $t)
                } else {
                    Err(format!("Value is not in the bound {} <= val <= {}", $t::MIN, $t::MAX))
                }
            }
        })*

        #[cfg(test)]
        mod from_unsigned_tests {
            use crate::large_int::{
                LargeInt
            };
            
            $(use std::$t;)*
            use std::convert::TryFrom;

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

            #[test]
            fn test_try_from_unsigned() {
                $(
                    let li = LargeInt::from($t::MAX);
                    assert_eq!($t::MAX, $t::try_from(li).unwrap());
                    let li = LargeInt::from($t::MIN);
                    assert_eq!($t::MIN, $t::try_from(li).unwrap());
    
                    let li = LargeInt::from(0);
                    assert_eq!(0 as $t, $t::try_from(li).unwrap());
                    let li = LargeInt::from(127);
                    assert_eq!(127 as $t, $t::try_from(li).unwrap());
    
                    let li = LargeInt::from($t::MAX) + 1;
                    assert_eq!(
                        Err(format!("Value is not in the bound {} <= val <= {}", $t::MIN, $t::MAX)),
                        $t::try_from(li)
                    );
                    let li = LargeInt::from($t::MIN) - 1;
                    assert_eq!(
                        Err(format!("Value is not in the bound {} <= val <= {}", $t::MIN, $t::MAX)),
                        $t::try_from(li)
                    );
                    )*
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

        // allow attempting to transform into primitives
        $(impl TryFrom<LargeInt> for $t {
            type Error = String;

            fn try_from(val: LargeInt) -> Result<$t, Self::Error> {
                if val >= LargeInt::from($t::MIN) && val <= LargeInt::from($t::MAX) {
                    Ok(val.bytes[0] as $t)
                } else {
                    Err(format!("Value is not in the bound {} <= val <= {}", $t::MIN, $t::MAX))
                }
            }
        })*

        #[cfg(test)]
        mod from_signed_tests {
            use crate::large_int::{
                LargeInt
            };
            use std::convert::TryFrom;
            
            use std::u128;
            $(use std::$t;)*

            #[test]
            fn test_from_signed() {
                $(
                let li = LargeInt::from(127 as $t);
                assert_eq!(li.bytes[0], 127u128);
                assert!(li.is_positive());

                let li = LargeInt::from(-1 as $t);
                assert_eq!(li.bytes[0], u128::MAX); // 2's compliment rep of -1 is all 1s

                assert!(li.is_negative());

                assert!(LargeInt::from($t::MIN).is_negative());
                )*
            }

            #[test]
            fn test_try_from_signed() {
                $(
                let li = LargeInt::from($t::MAX);
                assert_eq!($t::MAX, $t::try_from(li).unwrap());
                let li = LargeInt::from($t::MIN);
                assert_eq!($t::MIN, $t::try_from(li).unwrap());

                let li = LargeInt::from(0);
                assert_eq!(0 as $t, $t::try_from(li).unwrap());
                let li = LargeInt::from(127);
                assert_eq!(127 as $t, $t::try_from(li).unwrap());

                let li = LargeInt::from($t::MAX) + 1;
                assert_eq!(
                    Err(format!("Value is not in the bound {} <= val <= {}", $t::MIN, $t::MAX)),
                    $t::try_from(li)
                );
                let li = LargeInt::from($t::MIN) - 1;
                assert_eq!(
                    Err(format!("Value is not in the bound {} <= val <= {}", $t::MIN, $t::MAX)),
                    $t::try_from(li)
                );
                )*
            }
        }
    };
}

macro_rules! ex_expr {
    ( $e:expr ) => {
        {$e}
    };
}

// macro for implementing basic int operations for primitives
// so use with LargeInt is natural
macro_rules! ops {
    ( 
        for $($t:ident)* -> impl
        $trait:ident($op_name:ident <-> $op:tt) 
        and $assign_trait:ident($assign_op:ident)
    ) => {
        $(impl $trait<$t> for LargeInt {
            type Output = LargeInt;

            fn $op_name(self, other: $t) -> LargeInt {
                let oth = LargeInt::from(other);
                ex_expr!(self $op oth)
            }
        })*

        $(impl $assign_trait<$t> for LargeInt {
            fn $assign_op(&mut self, other: $t) {
                self.bytes = ex_expr!(self.clone() $op other).bytes;
            }
        })*

        // allow reverse operations too
        $(impl $trait<LargeInt> for $t {
            type Output = LargeInt;

            fn $op_name(self, other: LargeInt) -> LargeInt {
                let lhs = LargeInt::from(self);
                ex_expr!(lhs $op other)
            }
        })*
    };

    ( 
        for $($t:ident)* -> impl
        $trait:ident($op_name:ident <-> $op:tt) 
        and $assign_trait:ident($assign_op:ident), 
        $($remaining:tt)*
    ) => {
        ops!(
            for $($t)* -> impl
            $trait($op_name <-> $op) 
            and $assign_trait($assign_op)
        );
        ops!(for $($t)* -> impl $($remaining)*);
    };
}

from_signed!(i8 i16 i32 i64 i128 isize);
from_unsigned!(u8 u16 u32 u64 u128 usize);
ops!(
    for i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize -> impl
    Add(add <-> +) and AddAssign(add_assign),
    Sub(sub <-> -) and SubAssign(sub_assign),
    Mul(mul <-> *) and MulAssign(mul_assign),
    Div(div <-> /) and DivAssign(div_assign),
    Rem(rem <-> %) and RemAssign(rem_assign)
);

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