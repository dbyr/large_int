extern crate large_int;

use std::i128;
use std::str::FromStr;
use large_int::large_int::LargeInt;

#[test]
fn test_string_conversions() {
    let rep = "777777777777777777777777777777777777777777777777770";
    let to_2 = "7.77e50";
    let int = LargeInt::from_str(rep).unwrap();
    assert_eq!(int.to_string(), rep);
    assert_eq!(format!("{:.2}", int), to_2);

    let rep = "100000000000000000000000000000000000000";
    let int = LargeInt::from_str(rep).unwrap();
    assert_eq!(int, LargeInt::from(100000000000000000000000000000000000000u128));

    let int = LargeInt::from(541);
    assert_eq!(int.to_string(), "541");
    assert_eq!(format!("{:.1}", int), "5.4e2");
    assert_eq!(format!("{:.5}", int), "5.41000e2");
    assert_eq!(format!("{:e}", int), "5.4e2");

    let int = LargeInt::from(-3451);
    assert_eq!(format!("{}", int), "-3451");
    assert_eq!(format!("{:.2}", int), "-3.45e3");
    assert_eq!(format!("{:.9}", int), "-3.451000000e3");
    assert_eq!(format!("{:e}", int), "-3.4e3");

    let int = LargeInt::from(0);
    assert_eq!(format!("{:.4}", int), "0.0000e0");
    assert_eq!(format!("{:e}", int), "0.0e0");
    let int = LargeInt::from(-1);
    assert_eq!(format!("{:.2}", int), "-1.00e0");
    assert_eq!(format!("{:e}", int), "-1.0e0");
}

#[test]
fn test_addition() {
    let mut li = LargeInt::from_str("340282366920938463463374607431768211455").unwrap();
    li += 20451;
    assert_eq!(li, LargeInt::from_str("340282366920938463463374607431768231906").unwrap());

    li += LargeInt::from_str("-340282366920938463463374607431768000000").unwrap();
    assert_eq!(li, LargeInt::from(231906));

    let rev = 16 + li;
    assert_eq!(rev, LargeInt::from(231922));
}

#[test]
fn test_subtraction() {
    let mut li = LargeInt::from_str("340282366920938463463374607431769211455").unwrap();
    li -= LargeInt::from_str("340282366920938463463374607431769211455").unwrap() * 2;
    assert_eq!(li, LargeInt::from_str("-340282366920938463463374607431769211455").unwrap());

    let rev = 11 + LargeInt::from(1234);
    assert_eq!(rev, LargeInt::from(1245));
}

#[test]
fn test_multiplication() {
    let exp = LargeInt::from_str("777777777777777777777777777777777777777777777777770").unwrap();
    let mut init = LargeInt::from_str("11111111111111111111111111111111111111111111111111").unwrap();
    init *= 7;
    init *= 10;
    assert_eq!(exp, init);

    let pos = LargeInt::from(i128::MAX) * LargeInt::from(i128::MAX);
    let two_neg = LargeInt::from(i128::MIN + 1) * LargeInt::from(i128::MIN + 1);
    let neg = LargeInt::from(i128::MIN + 1) * LargeInt::from(i128::MAX);
    assert!(i128::MAX == -(i128::MIN + 1));
    assert!(pos == two_neg);
    assert!(pos == -neg);

    let zero = LargeInt::from(0);
    assert_eq!(pos * zero, LargeInt::from(0));

    let rev = 4 * LargeInt::from(9);
    assert_eq!(rev, LargeInt::from(36));
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
                                 
    let lhs = LargeInt::from_str("-8000000000000000000000000000000000000000000000000").unwrap();
    let rhs = LargeInt::from_str("100000000000000000000000000000000000000").unwrap();
    assert_eq!(lhs / rhs, LargeInt::from(-80000000000i128));

    let lhs = LargeInt::from_str("-8000000000000000000000000000000000000000000000000").unwrap();
    let rhs = LargeInt::from_str("-80000").unwrap();
    assert_eq!(lhs / rhs, LargeInt::from_str("100000000000000000000000000000000000000000000").unwrap());

    let lhs = LargeInt::from(43);
    let rhs = LargeInt::from(20);
    let (result, remainder) = lhs.div_with_remainder(rhs);
    assert_eq!(result, LargeInt::from(2));
    assert_eq!(remainder, 3);

    // ensure works with primitives too
    let lhs = LargeInt::from(43);
    let rhs = 20;
    let (result, remainder) = lhs.div_with_remainder(rhs);
    assert_eq!(result, 2);
    assert_eq!(remainder, 3);

    let rev = 12 / remainder;
    assert_eq!(rev, 4);
}

#[test]
fn test_remainder() {
    let lhs = LargeInt::from_str("340282366920938463463374607431768211465").unwrap();
    let rhs = LargeInt::from_str("100000000000000000000000000000000000000").unwrap();
    assert_eq!(lhs % rhs, LargeInt::from_str("40282366920938463463374607431768211465").unwrap());

    let lhs = LargeInt::from_str("-340282366920938463463374607431768211465").unwrap();
    let rhs = LargeInt::from_str("100000000000000000000000000000000000000").unwrap();
    assert_eq!(lhs % rhs, LargeInt::from_str("40282366920938463463374607431768211465").unwrap());
}

#[test]
fn test_default() {
    let def = LargeInt::default();
    let zero = LargeInt::new();
    assert_eq!(def, zero);
}

#[test]
fn test_comparisons() {
    let li = LargeInt::from(42);
    assert_eq!(li, 42);
    assert!(li > 41);
    assert!(li < 43);
    assert!(44 > li);
    assert!(42 == li);
}

#[test]
fn test_abs() {
    let neg = LargeInt::from_str("-8000000000000000000000000000000000000000000000000").unwrap();
    let pos = LargeInt::from_str("8000000000000000000000000000000000000000000000000").unwrap();
    assert_eq!(neg.abs(), pos.abs());
    assert_eq!(pos.abs().to_string(), "8000000000000000000000000000000000000000000000000");
}

#[test]
fn test_pow() {
    let pos = LargeInt::from(3);
    assert_eq!(pos.pow(3), LargeInt::from(27));
    assert_eq!(pos.pow(4), LargeInt::from(81));

    let neg = LargeInt::from(-3);
    assert_eq!(neg.pow(3), LargeInt::from(-27));
    assert_eq!(neg.pow(4), LargeInt::from(81));

    let pos = LargeInt::from(4);
    assert_eq!(pos.pow(13), LargeInt::from(67_108_864));

    let large = LargeInt::from(10);
    assert_eq!(large.pow(40), LargeInt::from_str("10000000000000000000000000000000000000000").unwrap());
}

#[test]
fn test_iter_ops() {
    let vals: Vec<LargeInt> = vec!(
        LargeInt::from(1),
        LargeInt::from(2),
        LargeInt::from(3),
        LargeInt::from(4)
    );
    let sum: LargeInt = vals.iter().sum();
    let product: LargeInt = vals.iter().product();
    assert_eq!(sum, LargeInt::from(10));
    assert_eq!(product, LargeInt::from(24));

    let sum: LargeInt = vals.into_iter().sum();
    assert_eq!(sum, LargeInt::from(10));

    let vals: Vec<LargeInt> = vec!();
    let sum: LargeInt = vals.iter().sum();
    let product: LargeInt = vals.iter().product();
    assert_eq!(sum, LargeInt::from(0));
    assert_eq!(product, LargeInt::from(0));

    let sum: LargeInt = vals.into_iter().sum();
    assert_eq!(sum, LargeInt::from(0));

    let vals: Vec<LargeInt> = vec!(
        LargeInt::from(-1),
        LargeInt::from(2),
        LargeInt::from(3),
        LargeInt::from(4)
    );
    let sum: LargeInt = vals.iter().sum();
    let product: LargeInt = vals.iter().product();
    assert_eq!(sum, LargeInt::from(8));
    assert_eq!(product, LargeInt::from(-24));

    let product: LargeInt = vals.into_iter().product();
    assert_eq!(product, LargeInt::from(-24));
}
