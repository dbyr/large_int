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

ops!(
    for i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize -> impl
    Add(add <-> +) and AddAssign(add_assign),
    Sub(sub <-> -) and SubAssign(sub_assign),
    Mul(mul <-> *) and MulAssign(mul_assign),
    Div(div <-> /) and DivAssign(div_assign),
    Rem(rem <-> %) and RemAssign(rem_assign)
);