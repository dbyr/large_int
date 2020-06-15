# LargeInt
An integer datatype that (practically) has no limit on size. This implementation aims to provide a means of performing calculations easily and intuitively with numbers larger than the largest primitive integer type provided in Rust (128 bits). It also aims to be able to integrate easily with existing primitive types, such that they can be added and subtracted easily from LargeInts. LargeInt is always signed, but because it can grow indefinitely there is no downfall to this.

### When to Use LargeInt
LargeInt should ONLY be used when values are expected to go beyond `u128::MAX`. I find the likelihood of this datatype ever really being useful slim (since 128 bits already provides VERY large numbers), however it's a nice-to-have for anyone who may find themselves in need of it. That said, I mostly built it for fun and out of intrigue. Using this type in circumstances outside of this is not recommended because it will be slower than simply using a primitive type.

### How to Use LargeInt
See the example folder for an example on how to use the LargeInt type. Long story short, after initialisation operations can be performed on the numbers using the same operators that can be used with primitive types (i.e. +, -, /, *, |, ^, &). These operations consume the LargeInt (the Copy trait cannot be derived for types containing Vecs) so if you don't want to consume the variable it will require being cloned before performing the operation. 

LargeInt can be initialised from strings (`LargeInt::from_str("123")`) or primitive signed or unsigned integers (`LargeInt::from(123)`).

#### Example
```
use large_int::large_int::LargeInt;

fn main() {
    let mut li = LargeInt::from_str("340282366920938463463374607431768211455").unwrap();
    li += 20451;
    assert_eq!(li, LargeInt::from_str("340282366920938463463374607431768231906").unwrap());

    let smaller = li / 100000000000000000000000000000000_u128;
    assert_eq!(smaller, 3402823_u128.into());

    let multed = smaller * 3;
    assert_eq!(multed, 10208469.into());

    let subbed = multed - 20000000;
    assert_eq!(subbed, -10208459.into());
}
```

### Operations Limitations
Add and Sub are the simplest and least expensive operations and are completed in _O(n)_ time for a number represented with _n_ u128 values (assuming the number being added to it is sized <=_n_).

Mul and Div are more expensive operations. Multiplication is achieved in _O(nm)_ time for _n_ being the largest representation of either value and _m_ being the smallest number of 1's in the binary representation of either number. Division is similar but cannot optimise to the smallest number of 1's due to the operation order being important.

Printing these numbers is by far the most expensive operation as it requires multiple additions and multiplications in order to calculate the decimal representation (as many as the number of digits in the representation). You should only print LargeInts when entirely necessary. In order to ease the requirement of this method they are formattable when printing. This provides a means by which to print the values in scientific notation (e.g. 123 == 1.23*10^2). This format does not round digits.

Shifting operators (i.e. << and >>) do not change the number of u128 values used to represent the number. This means shifting left will not infinitely shift digits left, but will overflow and be lost.

In order to abstract the representation of the numbers the "_op_\_no\_shrink" methods have been kept private.

The Div (/) and Rem (%) operators perform the same operation but return different values. If both values are desired it's better to use the "div\_with\_remainder" method that returns both values as a pair (result, remainder), rather than to call both Div and Rem separately, since this will double the time taken.

## Change Log
### 0.2.1
Ammended the exclusion of `TryFrom` for unsigned integer types.

Additionally the `div_with_remainder` method was made generic such that it now supports both primitives and other `LargeInt`s (and any other type implementing `Into<LargeInt>`).

### 0.2.0
Added extra generic traits `Default`, `Eq`, `Hash`, `Ord`, `LowerExp`. Also added iterator traits `Sum` and `Product`.

New functionality implemented `TryFrom` trait for all primitive int types. Unsigned ints were excluded by mistake.

More new functionality included mirror methods for the primitive integer types, `pow` and `abs`.


### 0.1.0 (First release)
Initial release included most of the standard `std::ops` operators. The base functionality was complete and included in this release.

### 1.0.0 (Yanked initial version)
This version was yanked due to issues with the categorisation on crate.io (after yanking I also realised the version should probably not be 1 yet).