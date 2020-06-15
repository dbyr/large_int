use crate::large_int::LargeInt;

use std::str::FromStr;
use std::convert::TryFrom;
use std::num::ParseIntError;

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
use std::fmt::{
    Display,
    Formatter,
    Result as FmtResult,
    LowerExp
};

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

from_signed!(i8 i16 i32 i64 i128 isize);
from_unsigned!(u8 u16 u32 u64 u128 usize);