use crate::large_int::{
    LargeInt,
    BITS_PER_CHUNK
};

use std::u128;
use std::cmp::Ordering;
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

impl<T: Into<LargeInt>> Add<T> for LargeInt {
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
    fn add(self, other: T) -> LargeInt {
        let mut result = self.add_no_shrink(other.into());
        result.shrink();
        result
    }
}

impl<T: Into<LargeInt>> AddAssign<T> for LargeInt {
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
    fn add_assign(&mut self, other: T) {
        self.bytes = (self.clone() + other).bytes;
    }
}

impl<T: Into<LargeInt>> Sub<T> for LargeInt {
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
    fn sub(self, other: T) -> LargeInt {
        let mut result = self.sub_no_shrink(other.into());
        result.shrink();
        result
    }
}

impl<T: Into<LargeInt>> SubAssign<T> for LargeInt {
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
    fn sub_assign(&mut self, other: T) {
        self.bytes = (self.clone() - other).bytes;
    }
}

impl<T: Into<LargeInt>> Mul<T> for LargeInt {
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
    fn mul(self, other: T) -> LargeInt {
        let mut result = self.mul_no_shrink(other.into());
        result.shrink();
        result
    }
}

impl<T: Into<LargeInt>> MulAssign<T> for LargeInt {
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
    fn mul_assign(&mut self, rhs: T) {
        self.bytes = (self.clone() * rhs).bytes;
    }
}

impl<T: Into<LargeInt>> Div<T> for LargeInt {
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
    fn div(self, other: T) -> LargeInt {
        self.div_with_remainder(other).0
    }
}

impl<T: Into<LargeInt>> DivAssign<T> for LargeInt {
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
    fn div_assign(&mut self, other: T) {
        self.bytes = (self.clone() / other).bytes;
    }
}

impl<T: Into<LargeInt>> Rem<T> for LargeInt {
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
    fn rem(self, other: T) -> LargeInt {
        self.div_with_remainder(other).1
    }
}

impl<T: Into<LargeInt>> RemAssign<T> for LargeInt {
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
    fn rem_assign(&mut self, other: T) {
        self.bytes = (self.clone() % other).bytes;
    }
}

impl<T: Into<LargeInt>> BitAnd<T> for LargeInt {
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
    fn bitand(mut self, rhs: T) -> LargeInt {
        let mut other = rhs.into();
        let size = self.bytes.len().max(other.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to_ignore_sign(size);
        other.expand_to_ignore_sign(size);

        for i in 0..size {
            result.bytes[i] = self.bytes[i] & other.bytes[i];
        }
        result.shrink();
        result
    }
}

impl<T: Into<LargeInt>> BitAndAssign<T> for LargeInt {
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
    fn bitand_assign(&mut self, rhs: T) {
        self.bytes = (self.clone() & rhs).bytes;
    }
}

impl<T: Into<LargeInt>> BitOr<T> for LargeInt {
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
    fn bitor(mut self, rhs: T) -> LargeInt {
        let mut other = rhs.into();
        let size = self.bytes.len().max(other.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to_ignore_sign(size);
        other.expand_to_ignore_sign(size);

        for i in 0..size {
            result.bytes[i] = self.bytes[i] | other.bytes[i];
        }
        result.shrink();
        result
    }
}

impl<T: Into<LargeInt>> BitOrAssign<T> for LargeInt {
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
    fn bitor_assign(&mut self, rhs: T) {
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

impl<T: Into<LargeInt>> BitXor<T> for LargeInt {
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
    fn bitxor(mut self, other: T) -> LargeInt {
        let mut rhs = other.into();
        let size = self.bytes.len().max(rhs.bytes.len());
        let mut result = LargeInt::with_size(size);
        self.expand_to_ignore_sign(size);
        rhs.expand_to_ignore_sign(size);
        for i in 0..size {
            result.bytes[i] = self.bytes[i] ^ rhs.bytes[i];
        }
        result
    }
}

impl<T: Into<LargeInt>> BitXorAssign<T> for LargeInt {
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
    fn bitxor_assign(&mut self, other: T) {
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

impl<T: Into<LargeInt> + Clone> PartialEq<T> for LargeInt {
    /// Determines if a LargeInt is equal to another integer.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let four = LargeInt::from(4);
    /// assert_ne!(five, four);
    /// assert_eq!(four, 4);
    /// ```
    /// 
    /// Returns true if the values are equal, false otherwise.
    fn eq(&self, other: &T) -> bool {
        let oth: LargeInt = other.clone().into();
        self.bytes == oth.bytes
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

impl<T: Into<LargeInt> + Clone> PartialOrd<T> for LargeInt {
    /// Determines if a LargeInt is larger, equal or greater than another integer.
    /// 
    /// # Examples
    /// ```
    /// use large_int::large_int::LargeInt;
    /// 
    /// let five = LargeInt::from(5);
    /// let four = LargeInt::from(4);
    /// assert!(five > four);
    /// assert!(four < 5);
    /// ```
    /// 
    /// Returns an ordering of the comparison
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        let oth: LargeInt = other.clone().into();
        let tester = self.clone() - oth;
        if tester == LargeInt::new() {
            Some(Ordering::Equal)
        } else if tester.is_negative() {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

// macro for implementing basic int operations for primitives
// so use with LargeInt is natural
macro_rules! ops {
    ( 
        for $($t:ident)* -> impl
        $trait:ident($op_name:ident)
    ) => {
        $(impl $trait<LargeInt> for $t {
            type Output = LargeInt;

            fn $op_name(self, other: LargeInt) -> LargeInt {
                let rep = LargeInt::from(self);
                rep.$op_name(other)
            }
        })*
    };

    ( 
        for $($t:ident)* -> impl
        $trait:ident($op_name:ident), 
        $($remaining:tt)*
    ) => {
        ops!(
            for $($t)* -> impl
            $trait($op_name)
        );
        ops!(for $($t)* -> impl $($remaining)*);
    };
}

macro_rules! comparisons {
    ( $($t:ident)* ) => {
        $(impl PartialEq<LargeInt> for $t {
            fn eq(&self, other: &LargeInt) -> bool {
                let lhs: LargeInt = self.clone().into();
                &lhs == other
            }
        })*

        $(impl PartialOrd<LargeInt> for $t {
            fn partial_cmp(&self, other: &LargeInt) -> Option<Ordering> {
                let lhs: LargeInt = self.clone().into();
                lhs.partial_cmp(other)
            }
        })*
    };
}

#[macro_export]
/// Macro to implement reverse operators for foreign types.
/// Requires only that the type(s) being passed to the macro
/// implement From<type> for LargeInt.
macro_rules! reverse_operations {
    ( $($t:ident)* ) => {
        ops!(for $($t)* -> impl
            Add(add),
            Sub(sub),
            Mul(mul),
            Div(div),
            Rem(rem),
            BitOr(bitor),
            BitAnd(bitand),
            BitXor(bitxor)
        );
        comparisons!($($t)*);
    };
}

reverse_operations!(i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize);
