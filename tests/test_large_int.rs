extern crate large_int;

use std::str::FromStr;
use large_int::large_int::LargeInt;

#[test]
fn test_string_conversions() {
    let rep = "777777777777777777777777777777777777777777777777770";
    let int = LargeInt::from_str(rep).unwrap();
    assert_eq!(int.to_string(), rep);

    let rep = "100000000000000000000000000000000000000";
    let int = LargeInt::from_str(rep).unwrap();
    assert_eq!(int, LargeInt::from(100000000000000000000000000000000000000u128));
}

#[test]
fn test_multiplication() {
    let exp = LargeInt::from_str("777777777777777777777777777777777777777777777777770").unwrap();
    let mut init = LargeInt::from_str("11111111111111111111111111111111111111111111111111").unwrap();
    init *= 7;
    init *= 10;
    assert_eq!(exp, init);
}

#[test]
fn test_division() {
    let mut init = LargeInt::from_str("777777777777777777777777777777777777777777777777770").unwrap();
    let exp = LargeInt::from_str("11111111111111111111111111111111111111111111111111").unwrap();
    init /= 7;
    init /= 10;
    assert_eq!(exp, init);

    let lhs = LargeInt::from(265);
    let rhs = LargeInt::from(100);
    assert_eq!(lhs / rhs, LargeInt::from(2));

    let lhs = LargeInt::from_str("340282366920938463463374607431768211465").unwrap();
    let rhs = LargeInt::from_str("100000000000000000000000000000000000000").unwrap();
    assert_eq!(lhs / rhs, LargeInt::from(3));
}

#[test]
fn test_remainder() {
    let lhs = LargeInt::from_str("340282366920938463463374607431768211465").unwrap();
    let rhs = LargeInt::from_str("100000000000000000000000000000000000000").unwrap();
    assert_eq!(lhs % rhs, LargeInt::from_str("40282366920938463463374607431768211465").unwrap());
}
