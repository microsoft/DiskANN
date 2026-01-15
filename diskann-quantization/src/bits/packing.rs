/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

const fn mask<const N: usize>() -> u8 {
    const { assert!(N <= 8) };
    0xffu8 >> { 8 - N }
}

/// Unpack `N` bits from `byte` in the bit range `[bitstart, bitstart + N)` and
/// return in the lower bits of the result, zeroing the upper bits.
///
/// # Requires
///
/// Requires `8 - bitstart >= N`. That is - the range to be packed must fit
/// entirely within the byte.
pub(crate) fn unpack_u8<const N: usize>(byte: u8, bitstart: usize) -> u8 {
    debug_assert!(bitstart <= { 8 - N });
    (byte >> bitstart) & mask::<N>()
}

/// Unpack `N` bits from `word` in the bit range `[bitstart, bitstart + N)` and
/// return in the lower bits of the result, zeroing the upper bits.
///
/// This method should only be called if the desired unpacking crosses a byte-boundary,
/// otherwise `unpack_u8` should be used instead.
///
/// # Requires
///
/// Requires `8 - N < bitstart < 8`. That is, the range to pack must cross a byte
/// boundary.
pub(crate) fn unpack_u16<const N: usize>(word: u16, bitstart: usize) -> u8 {
    debug_assert!(bitstart < 8);
    debug_assert!({ 8 - N } < bitstart);

    // Shift, cast, mask.
    ((word >> bitstart) as u8) & mask::<N>()
}

/// Pack the lower `N` of `value` into `before` in the bit positionx
/// `[bitstart, bitstart + N)`, leaving the other bits undisturbed and return the
/// result.
///
/// # Requires
///
/// Requires `8 - bitstart >= N`. That is - the range to be packed must fit
/// entirely within the byte.
pub(crate) fn pack_u8<const N: usize>(before: u8, value: u8, bitstart: usize) -> u8 {
    let mask = mask::<N>();
    debug_assert!(value <= mask);
    debug_assert!(bitstart <= { 8 - N });

    // Move the mask into the correct location and blend the shifted argument.
    let shifted_mask = mask << bitstart;
    (before & !shifted_mask) | ((value << bitstart) & shifted_mask)
}

/// Pack a bit-pattern of `N` into `word` in the position
/// `[bitstart, bitstart + N)`, leaving the other bits undisturbed and return the
/// result.
///
/// This method should only be called if the desired packing crosses a byte-boundary,
/// otherwise `pack_u8` should be used instead.
///
/// # Requires
///
/// Requires `8 - N < bitstart < 8`. That is, the range to pack must cross a byte
/// boundary.
pub(crate) fn pack_u16<const N: usize>(before: u16, value: u8, bitstart: usize) -> u16 {
    let mask = mask::<N>();
    debug_assert!(value <= mask);
    debug_assert!(bitstart < 8);
    debug_assert!({ 8 - N } < bitstart);

    // Move the mask into the correct location and blend the shifted argument.
    let value: u16 = value.into();
    let shifted_mask = <u16 as From<u8>>::from(mask) << bitstart;
    (before & !shifted_mask) | ((value << bitstart) & shifted_mask)
}

// These tests take a long time to run under Miri and do not invoke any unsafe code.
// Exclude when using Miri.
//
// NOTE: We need two separate `cfg` blocks, otherwise `clippy` coes not correctly recognize
// the test module **as** as test module and complains about `unwrap`.
#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use super::*;

    fn upper_bound(nbits: usize) -> usize {
        2usize.pow(nbits.try_into().unwrap()) - 1
    }

    fn test_packing_u8_impl<const NBITS: usize>(mask: u8) {
        let not_mask = !mask;

        // Use all possible values for `before`.
        for before in 0u8..=255u8 {
            // Use all possible values for the encoded value.
            for value in 0u8..=upper_bound(NBITS).try_into().unwrap() {
                // Use all possible bitshifts
                for bitstart in 0..=(8 - NBITS) {
                    let encoded = pack_u8::<NBITS>(before, value, bitstart);

                    // Make sure the masked out portions of the encoded value match what
                    // they were before.
                    assert_eq!(
                        encoded & (not_mask << bitstart),
                        before & (not_mask << bitstart),
                        "failed to preserve unmodified bits for NBITS = {}, before = {}, value = {}, bitstart = {}",
                        NBITS,
                        before,
                        value,
                        bitstart,
                    );

                    // Make sure the encoded portion matches.
                    assert_eq!(
                        (encoded >> bitstart) & mask,
                        value,
                        "failed to propertly encode for NBITS = {}, before = {}, value = {}, bitstart = {}",
                        NBITS,
                        before,
                        value,
                        bitstart,
                    );

                    // Make sure decoding yields the correct value.
                    assert_eq!(unpack_u8::<NBITS>(encoded, bitstart), value);
                }
            }
        }
    }

    fn test_packing_u16_impl<const NBITS: usize>(mask: u16) {
        let not_mask = !mask;

        // Use all possible values for `before`.
        for before in 0u16..=0xffffu16 {
            // Use all possible values for the encoded value.
            for value in 0u8..=upper_bound(NBITS).try_into().unwrap() {
                // Use all possible bitshifts
                for bitstart in ((8 - NBITS) + 1)..=7 {
                    let encoded = pack_u16::<NBITS>(before, value, bitstart);

                    // Make sure the masked out portions of the encoded value match what
                    // they were before.
                    assert_eq!(
                        encoded & (not_mask << bitstart),
                        before & (not_mask << bitstart),
                        "failed to preserve unmodified bits for NBITS = {}, before = {}, value = {}, bitstart = {}",
                        NBITS,
                        before,
                        value,
                        bitstart,
                    );

                    // Make sure the encoded portion matches.
                    let expected: u8 = ((encoded >> bitstart) & mask).try_into().unwrap();
                    assert_eq!(
                        expected,
                        value,
                        "failed to propertly encode for NBITS = {}, before = {}, value = {}, bitstart = {}",
                        NBITS,
                        before,
                        value,
                        bitstart,
                    );

                    // Make sure decoding yields the correct value.
                    assert_eq!(unpack_u16::<NBITS>(encoded, bitstart), value);
                }
            }
        }
    }

    macro_rules! test_packing {
        ($name_u8:ident, $name_u16:ident, $nbits:literal, $mask:literal) => {
            #[test]
            fn $name_u8() {
                test_packing_u8_impl::<$nbits>($mask);
            }

            #[test]
            fn $name_u16() {
                test_packing_u16_impl::<$nbits>($mask);
            }
        };
    }

    test_packing!(test_u8_8bit, test_u16_8bit, 8, 0xff);
    test_packing!(test_u8_7bit, test_u16_7bit, 7, 0x7f);
    test_packing!(test_u8_6bit, test_u16_6bit, 6, 0x3f);
    test_packing!(test_u8_5bit, test_u16_5bit, 5, 0x1f);
    test_packing!(test_u8_4bit, test_u16_4bit, 4, 0x0f);
    test_packing!(test_u8_3bit, test_u16_3bit, 3, 0x07);
    test_packing!(test_u8_2bit, test_u16_2bit, 2, 0x03);
    test_packing!(test_u8_1bit, test_u16_1bit, 1, 0x01);
}
