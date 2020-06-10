use std::u128;
use std::ops::{
    Add,
    Sub
};

// store a vector of little-endian, 2's compliment figures
// the sign bit is in the most significant figure (more[0])
// littel endian was chosen so vec operations are faster
#[derive(Clone, Debug)]
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
        LargeInt{bytes: vec!(0; 1)}
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
        for i in self.bytes.len() - 1..0 {
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
        compliment + LargeInt::from(1)
    }
}

// impl Add<u128> for LargeInt {
//     type Output = LargeInt;

//     fn add(self, other: u128) -> LargeInt {
//         let size
//     }
// }

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
        result
    }
}

impl Sub for LargeInt {
    type Output = LargeInt;

    fn sub(self, other: LargeInt) -> LargeInt {
        LargeInt{bytes: vec!(0;1)}
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

        // $(impl Add<$t> for LargeInt {
        //     type Output = LargeInt;

        //     fn add(self, other: $t) -> LargeInt {

        //     }
        // })*
    };
}

ops!(i8 i32 i64 i128 isize u8 u32 u64 u128 usize);

#[cfg(test)]
mod tests {
    use crate::large_int::LargeInt;
    
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
}