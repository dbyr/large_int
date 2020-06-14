extern crate large_int;

use large_int::large_int::LargeInt;
use std::u128;
use std::i128;
use std::string::ToString;
use std::str::FromStr;

fn main() {

    // can use this data type to perform operations
    // on integers greater than u128::MAX
    let big_num = LargeInt::from(u128::MAX);
    let bigger_num: LargeInt = big_num.clone() + 1;
    println!("{} > u128::MAX (={})!", bigger_num, u128::MAX);
    assert!(bigger_num > big_num);

    println!();
    let huge_num = big_num * u128::MAX;
    println!("Check out this number...");
    println!("{}\nWow, that's definitely over 9000!", huge_num);
    println!("Takes a while to print though...");

    // printing in scientific notation will be faster because there
    // are less digits to calculate. format like floats, where the
    // precision is the number of decmial places
    println!("So you can use scientific notation as well!");
    println!("{:.3}", huge_num);

    // integers that are unbounded in both positive and negative
    println!("");
    let very_tiny = LargeInt::from(i128::MIN) * LargeInt::from(u128::MAX);
    println!("For my next trick:\n\n{}!", very_tiny);

    // you can convert them to strings (as you've seen) or get them from strings
    // NOTE these string conversions take a long time because they have to be
    // converted digit-by-digit, not sure how else to implement it...
    let bigger_than_literal = LargeInt::from_str("234965425784920654070519485679208634056710984651560139845").unwrap();
    let negatives_too = LargeInt::from_str("-123").unwrap();
    assert!(negatives_too.is_negative());
    assert!(bigger_than_literal.is_positive());

    // and all operators have been implemented
    assert!(bigger_than_literal > negatives_too);
    println!("bitwise ops, too! -> {}", negatives_too | bigger_than_literal);

    // and they can interact with primitives with ease
    assert_eq!(500 / LargeInt::from(5), LargeInt::from(100));
}