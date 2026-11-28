//! Reed-Solomon codes over the bytes, which put right the tiles of a frame that are misread.
//!
//! The field is the one of QR codes, GF(256) with the polynomial x^8 + x^4 + x^3 + x^2 + 1 and the element
//! 2 for its powers, and a code of `parity` check bytes has the roots 2^0 to 2^(parity - 1). The check bytes
//! follow the data. The page that `qqcode-show` writes computes the same in JavaScript, so nothing here can
//! change without it.

/// The powers of 2, twice over so that the sum of two logarithms needs no reduction, and the logarithms.
struct Field {
    exp: [u8; 510],
    log: [u8; 256],
}

const FIELD: Field = {
    let mut exp = [0; 510];
    let mut log = [0; 256];
    let mut value: u16 = 1;
    let mut power = 0;
    while power < 255 {
        exp[power] = value as u8;
        exp[power + 255] = value as u8;
        log[value as usize] = power as u8;
        value <<= 1;
        if value & 0x100 != 0 {
            value ^= 0x11d;
        }
        power += 1;
    }
    Field { exp, log }
};

fn multiply(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else {
        FIELD.exp[FIELD.log[a as usize] as usize + FIELD.log[b as usize] as usize]
    }
}

fn divide(a: u8, b: u8) -> u8 {
    debug_assert!(b != 0, "division by zero");
    if a == 0 {
        0
    } else {
        FIELD.exp[FIELD.log[a as usize] as usize + 255 - FIELD.log[b as usize] as usize]
    }
}

/// 2 to the power of `power`, which may be any number.
fn power_of_two(power: usize) -> u8 {
    FIELD.exp[power % 255]
}

/// The value of the polynomial with these coefficients, the highest power first, at `x`.
fn evaluate(coefficients: &[u8], x: u8) -> u8 {
    coefficients
        .iter()
        .fold(0, |value, coefficient| multiply(value, x) ^ coefficient)
}

/// The polynomial with the roots 2^0 to 2^(parity - 1), the highest power first and without its leading 1.
fn generator(parity: usize) -> Vec<u8> {
    let mut generator = vec![1u8];
    for root in 0..parity {
        // times (x + 2^root)
        generator.push(0);
        for index in (1..generator.len()).rev() {
            generator[index] ^= multiply(generator[index - 1], power_of_two(root));
        }
    }
    generator.remove(0);
    generator
}

/// The `parity` check bytes for `data`, which together with it are no more than 255 bytes.
pub fn encode(data: &[u8], parity: usize) -> Vec<u8> {
    assert!(data.len() + parity <= 255, "a code word has 255 bytes at most");
    let generator = generator(parity);
    let mut check = vec![0u8; parity];
    for byte in data {
        let factor = byte ^ check[0];
        check.rotate_left(1);
        check[parity - 1] = 0;
        if factor != 0 {
            for (check, coefficient) in check.iter_mut().zip(&generator) {
                *check ^= multiply(*coefficient, factor);
            }
        }
    }
    check
}

/// Puts right the bytes of a code word, data and then `parity` check bytes, of which up to half as many as
/// there are check bytes may be wrong. Returns how many bytes were wrong, or `None` if they are more than
/// can be put right, in which case the word is left as it was.
pub fn correct(word: &mut [u8], parity: usize) -> Option<usize> {
    let length = word.len();
    assert!(
        parity < length && length <= 255,
        "a code word of data and check bytes"
    );
    let syndromes = (0..parity)
        .map(|root| evaluate(word, power_of_two(root)))
        .collect::<Vec<_>>();
    if syndromes.iter().all(|syndrome| *syndrome == 0) {
        return Some(0);
    }

    // Berlekamp and Massey: the shortest polynomial that generates the syndromes, lowest power first. Its
    // roots are the inverses of the places of the errors.
    let mut locator = vec![1u8];
    let mut previous = vec![1u8];
    let (mut errors, mut shift, mut last) = (0usize, 1usize, 1u8);
    for step in 0..parity {
        let mut discrepancy = syndromes[step];
        for index in 1..=errors.min(locator.len() - 1) {
            discrepancy ^= multiply(locator[index], syndromes[step - index]);
        }
        if discrepancy == 0 {
            shift += 1;
            continue;
        }
        let factor = divide(discrepancy, last);
        let before = locator.clone();
        if locator.len() < previous.len() + shift {
            locator.resize(previous.len() + shift, 0);
        }
        for (index, coefficient) in previous.iter().enumerate() {
            locator[index + shift] ^= multiply(factor, *coefficient);
        }
        if 2 * errors <= step {
            errors = step + 1 - errors;
            previous = before;
            last = discrepancy;
            shift = 1;
        } else {
            shift += 1;
        }
    }
    locator.truncate(errors + 1);
    if errors == 0 || 2 * errors > parity || locator.len() != errors + 1 {
        return None;
    }

    // the places whose inverse is a root, counted from the end of the word
    let highest_first = locator.iter().rev().copied().collect::<Vec<_>>();
    let places = (0..length)
        .filter(|place| evaluate(&highest_first, power_of_two(255 - place % 255)) == 0)
        .collect::<Vec<_>>();
    if places.len() != errors {
        return None;
    }

    // Forney: the error at a place from the evaluator, the syndromes times the locator cut off at the
    // number of check bytes, and the derivative of the locator
    let mut evaluator = vec![0u8; parity];
    for (index, syndrome) in syndromes.iter().enumerate() {
        for (power, coefficient) in locator.iter().enumerate() {
            if index + power < parity {
                evaluator[index + power] ^= multiply(*syndrome, *coefficient);
            }
        }
    }
    let at = |polynomial: &[u8], x: u8| {
        let highest_first = polynomial.iter().rev().copied().collect::<Vec<_>>();
        evaluate(&highest_first, x)
    };
    let derivative = locator
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, coefficient)| if power % 2 == 1 { *coefficient } else { 0 })
        .collect::<Vec<_>>();
    let mut corrected = word.to_vec();
    for place in &places {
        let inverse = power_of_two(255 - place % 255);
        let slope = at(&derivative, inverse);
        if slope == 0 {
            return None;
        }
        let error = multiply(power_of_two(*place), divide(at(&evaluator, inverse), slope));
        corrected[length - 1 - place] ^= error;
    }
    let consistent = (0..parity).all(|root| evaluate(&corrected, power_of_two(root)) == 0);
    if !consistent {
        return None;
    }
    word.copy_from_slice(&corrected);
    Some(errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numbers in no pattern, the same on every run.
    struct Random(u32);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn below(&mut self, limit: usize) -> usize {
            (self.next() >> 8) as usize % limit
        }
    }

    fn word(random: &mut Random, data: usize, parity: usize) -> Vec<u8> {
        let mut word = (0..data).map(|_| random.next() as u8).collect::<Vec<_>>();
        let check = encode(&word, parity);
        word.extend(check);
        word
    }

    /// Makes `count` different bytes of `word` wrong.
    fn damage(random: &mut Random, word: &mut [u8], count: usize) {
        let mut places = Vec::new();
        while places.len() < count {
            let place = random.below(word.len());
            if !places.contains(&place) {
                places.push(place);
                word[place] ^= 1 + random.below(255) as u8;
            }
        }
    }

    #[test]
    fn field_is_that_of_qr_codes() {
        assert_eq!(power_of_two(0), 1);
        assert_eq!(power_of_two(8), 0x1d);
        assert_eq!(power_of_two(255), 1);
        // x^7 times x^2 is x^9, which is x times x^8
        assert_eq!(multiply(0x80, 0x04), 0x3a);
        for value in 1..=255u8 {
            assert_eq!(divide(multiply(value, 0x37), 0x37), value);
            assert_eq!(multiply(value, divide(1, value)), 1);
        }
        assert_eq!((multiply(0, 7), multiply(7, 0), divide(0, 7)), (0, 0, 0));
    }

    #[test]
    fn check_bytes_are_those_of_a_qr_code() {
        // the data and error correction code words of the version 1-M symbol in ISO/IEC 18004, annex I,
        // whose generator has the same roots
        let data = [
            0x10, 0x20, 0x0c, 0x56, 0x61, 0x80, 0xec, 0x11, 0xec, 0x11, 0xec, 0x11, 0xec, 0x11, 0xec, 0x11,
        ];
        assert_eq!(
            encode(&data, 10),
            [0xa5, 0x24, 0xd4, 0xc1, 0xed, 0x36, 0xc7, 0x87, 0x2c, 0x55]
        );
    }

    #[test]
    fn word_without_errors_is_left_alone() {
        let mut random = Random(1);
        let original = word(&mut random, 200, 40);
        let mut received = original.clone();
        assert_eq!(correct(&mut received, 40), Some(0));
        assert_eq!(received, original);
    }

    #[test]
    fn errors_up_to_half_the_check_bytes_are_put_right() {
        let mut random = Random(2);
        for (data, parity) in [(215, 40), (191, 64), (20, 8), (1, 2), (100, 40), (223, 32)] {
            for errors in [1, 2, parity / 4, parity / 2 - 1, parity / 2] {
                if errors == 0 || 2 * errors > parity {
                    continue;
                }
                let original = word(&mut random, data, parity);
                let mut received = original.clone();
                damage(&mut random, &mut received, errors);
                assert_eq!(
                    correct(&mut received, parity),
                    Some(errors),
                    "{errors} errors in {data}+{parity}"
                );
                assert_eq!(received, original);
            }
        }
    }

    #[test]
    fn errors_in_the_check_bytes_and_at_the_ends_are_put_right() {
        let mut random = Random(3);
        let original = word(&mut random, 100, 16);
        for places in [
            &[0usize][..],
            &[115],
            &[0, 115],
            &[100, 101, 102, 103],
            &[0, 1, 2, 99, 100, 114, 115],
        ] {
            let mut received = original.clone();
            for place in places {
                received[*place] ^= 0x5a;
            }
            assert_eq!(correct(&mut received, 16), Some(places.len()), "{places:?}");
            assert_eq!(received, original);
        }
    }

    #[test]
    fn more_errors_than_can_be_put_right_are_noticed() {
        let mut random = Random(4);
        let mut noticed = 0;
        for _ in 0..200 {
            let original = word(&mut random, 215, 40);
            let mut received = original.clone();
            let errors = 21 + random.below(40);
            damage(&mut random, &mut received, errors);
            let damaged = received.clone();
            match correct(&mut received, 40) {
                None => {
                    noticed += 1;
                    assert_eq!(
                        received, damaged,
                        "a word that can't be put right is left as it was"
                    );
                }
                // another code word may be nearer than the one that was sent, though hardly ever
                Some(_) => assert_ne!(received, original),
            }
        }
        assert!(noticed >= 199, "{noticed} of 200");
    }

    #[test]
    fn errors_that_look_like_one_beyond_the_word_are_noticed() {
        // A word of 28 bytes is one of 255 that begins with zeros. Its check bytes are made those of a
        // word with one wrong byte 100 places from its end, a place that this word has not: all eight
        // are wrong, which is more than can be put right, and nothing in the word is the one to put right.
        let mut random = Random(5);
        let original = word(&mut random, 20, 8);
        let mut far = vec![0u8; 93];
        far[0] = 0x5a;
        let mut received = original.clone();
        for (byte, off) in received[20..].iter_mut().zip(encode(&far, 8)) {
            *byte ^= off;
        }
        let damaged = received.clone();
        assert_eq!(correct(&mut received, 8), None);
        assert_eq!(received, damaged);
    }
}
