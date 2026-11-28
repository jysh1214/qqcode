//! The check bits of a frame: a low-density parity-check code, which is put right from how sure the reader
//! is of every bit and not from the bits alone.
//!
//! A code word is the bits of the data and after them one check bit for every check. A check is the sum of
//! some bits of the data, of the check bit before it and of its own check bit, and holds if that sum is
//! even: so the check bits follow from the data one after the other, each the one before it plus the bits
//! of the data that its check has. Every bit of the data is in three checks, and one in ten of them, the
//! first ones, is in thirteen, which is what lets the others be found from few checks to begin with.
//!
//! Which checks a bit is in is drawn from a sequence of numbers that is the same wherever the code is made,
//! so that the code follows from its two lengths alone. The page computes all of this in JavaScript, so
//! nothing here can change without it.
//!
//! The decoder passes on, check after check, how sure each check makes a bit given what is known of its
//! other bits (the least sure of them decides, which is "min-sum", taken somewhat down), and ends as soon
//! as every check holds.

/// How many checks most bits of the data are in, and the first [`MORE`]th of them.
const FEW: usize = 3;
const MANY: usize = 13;
const MORE: usize = 10;

/// Where the sequence of numbers begins that the checks of the bits are drawn from.
const SEED: u32 = 0x2545_f491;

/// What the least sure of the other bits of a check says is taken by this much, as it says too much.
const DAMP: f32 = 0.8;

/// The sequence of numbers that codes are made from.
pub struct Draw(pub u32);

impl Draw {
    pub fn number(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// The numbers from 0 to `count` less one, in no order.
    pub fn order(&mut self, count: usize) -> Vec<u32> {
        let mut order: Vec<u32> = (0..count as u32).collect();
        for last in (1..count).rev() {
            order.swap(last, self.number() as usize % (last + 1));
        }
        order
    }
}

/// A code of words of so many bits, of which so many are data.
pub struct Code {
    bits: usize,
    data: usize,
    /// Where the bits of each check begin in `members`, and where the last one ends
    starts: Vec<u32>,
    /// The bits of the checks, as places in a code word: for each check those of the data, then the check
    /// bit before its own, then its own
    members: Vec<u32>,
    /// The most bits that a check has
    widest: usize,
}

impl Code {
    /// `bits` and `data` are numbers of bits, and at least one bit is a check bit.
    pub fn new(bits: usize, data: usize) -> Self {
        assert!(data > 0 && data < bits, "a code word of data and of check bits");
        let checks = bits - data;
        // The checks of the bits of the data: three rounds through all of the bits and ten more through
        // the first of them, each round in an order of its own, and the checks taken in turn all the way,
        // so that every check gets as many bits as any other. A bit that would come into a check twice
        // takes the next check.
        let mut draw = Draw(SEED);
        let mut of_bit = vec![u32::MAX; data * MANY];
        let mut sizes = vec![0u32; checks];
        let mut turn = 0;
        for round in 0..MANY {
            let count = if round < FEW { data } else { data / MORE };
            for bit in draw.order(count) {
                let own = &mut of_bit[bit as usize * MANY..][..MANY];
                let mut check = (turn % checks) as u32;
                while own[..round].contains(&check) {
                    check = (check + 1) % checks as u32;
                }
                own[round] = check;
                sizes[check as usize] += 1;
                turn += 1;
            }
        }
        let mut starts = Vec::with_capacity(checks + 1);
        let mut total = 0;
        for (check, size) in sizes.iter().enumerate() {
            starts.push(total);
            total += size + if check == 0 { 1 } else { 2 };
        }
        starts.push(total);
        let mut members = vec![0u32; total as usize];
        let mut filled = vec![0u32; checks];
        for bit in 0..data {
            for check in &of_bit[bit * MANY..][..MANY] {
                if *check != u32::MAX {
                    let check = *check as usize;
                    members[(starts[check] + filled[check]) as usize] = bit as u32;
                    filled[check] += 1;
                }
            }
        }
        for check in 0..checks {
            let mut at = (starts[check] + filled[check]) as usize;
            if check > 0 {
                members[at] = (data + check - 1) as u32;
                at += 1;
            }
            members[at] = (data + check) as u32;
        }
        let widest = starts
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .max()
            .unwrap_or(0);
        Self {
            bits,
            data,
            starts,
            members,
            widest,
        }
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn data(&self) -> usize {
        self.data
    }

    fn checks(&self) -> usize {
        self.bits - self.data
    }

    fn members(&self, check: usize) -> &[u32] {
        &self.members[self.starts[check] as usize..self.starts[check + 1] as usize]
    }

    /// The code word of these bits of data, a byte for every bit.
    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        assert_eq!(data.len(), self.data, "as many bits as the code has of data");
        let mut word = Vec::with_capacity(self.bits);
        word.extend(data.iter().map(|bit| bit & 1));
        let mut before = 0;
        for check in 0..self.checks() {
            let members = self.members(check);
            // without the check bits, of which the first check has one and the others two
            let own = &members[..members.len() - if check == 0 { 1 } else { 2 }];
            before ^= own.iter().fold(0, |sum, bit| sum ^ word[*bit as usize]);
            word.push(before);
        }
        word
    }

    /// Whether every check holds for the bits that these values stand for: below zero for a 1.
    fn holds(&self, values: &[f32]) -> bool {
        (0..self.checks()).all(|check| {
            !self
                .members(check)
                .iter()
                .fold(false, |odd, bit| odd ^ (values[*bit as usize] < 0.0))
        })
    }

    /// The code word that was sent, a byte for every bit, from how sure the reader is of each bit: above
    /// zero for a 0 and below it for a 1, the further from zero the surer, and zero for a bit of which
    /// nothing is known. `None` if no code word is found in `rounds` rounds through the checks.
    pub fn decode(&self, sure: &[f32], rounds: usize) -> Option<Vec<u8>> {
        assert_eq!(sure.len(), self.bits, "a value for every bit of the code word");
        let mut values = sure.to_vec();
        // what each check last said of each of its bits
        let mut said = vec![0f32; self.members.len()];
        let mut others = vec![0f32; self.widest];
        for _ in 0..rounds {
            if self.holds(&values) {
                break;
            }
            for check in 0..self.checks() {
                let from = self.starts[check] as usize;
                let members = &self.members[from..self.starts[check + 1] as usize];
                // each bit as the other checks have it, the two least sure of them, and whether the bits
                // that are taken for a 1 are odd in number
                let (mut least, mut second, mut weakest, mut odd) = (f32::MAX, f32::MAX, 0, false);
                for (place, bit) in members.iter().enumerate() {
                    let value = values[*bit as usize] - said[from + place];
                    others[place] = value;
                    let size = value.abs();
                    if size < least {
                        second = least;
                        least = size;
                        weakest = place;
                    } else if size < second {
                        second = size;
                    }
                    odd ^= value < 0.0;
                }
                for (place, bit) in members.iter().enumerate() {
                    let value = others[place];
                    let size = DAMP * if place == weakest { second } else { least };
                    let says = if odd ^ (value < 0.0) { -size } else { size };
                    said[from + place] = says;
                    values[*bit as usize] = value + says;
                }
            }
        }
        self.holds(&values)
            .then(|| values.iter().map(|value| u8::from(*value < 0.0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(count: usize, seed: u32) -> Vec<u8> {
        let mut draw = Draw(seed | 1);
        (0..count).map(|_| (draw.number() >> 7) as u8 & 1).collect()
    }

    /// How sure a reader is of the bits of a code word that it sees without a fault.
    fn certain(word: &[u8]) -> Vec<f32> {
        word.iter()
            .map(|bit| if *bit == 1 { -1.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn code_is_made_as_documented() {
        let code = Code::new(6000, 4000);
        assert_eq!((code.bits(), code.data(), code.checks()), (6000, 4000, 2000));
        // every bit of the data in three checks and the first tenth in thirteen: 16000 places in 2000
        // checks, 8 in each give or take the bits that took the next check, and the check bits
        let mut times = vec![0; 6000];
        assert_eq!(code.members.len(), 16000 + 2 * 2000 - 1);
        for check in 0..2000 {
            let members = code.members(check);
            assert!(
                (8..=12).contains(&members.len()),
                "check {check}: {}",
                members.len()
            );
            let mut sorted = members.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), members.len(), "no bit twice in check {check}");
            for bit in members {
                times[*bit as usize] += 1;
            }
            assert_eq!(*members.last().unwrap(), 4000 + check as u32);
        }
        assert!(times[..400].iter().all(|times| *times == 13));
        assert!(times[400..4000].iter().all(|times| *times == 3));
        assert!(times[4000..5999].iter().all(|times| *times == 2));
        assert_eq!(times[5999], 1);
        // the same code every time
        assert!(Code::new(6000, 4000).members == code.members);
    }

    #[test]
    fn check_bits_follow_from_the_data() {
        let code = Code::new(1200, 900);
        let data = bits(900, 5);
        let word = code.encode(&data);
        assert!(word.len() == 1200 && word[..900] == data[..]);
        assert!(code.holds(&certain(&word)));
        // a bit that is wrong fails the checks it is in
        let mut wrong = word.clone();
        wrong[700] ^= 1;
        assert!(!code.holds(&certain(&wrong)));
        // no data, no check bits
        assert!(code.encode(&[0; 900]) == vec![0; 1200]);
    }

    #[test]
    fn word_comes_back_from_what_a_reader_is_sure_of() {
        let code = Code::new(6000, 4000);
        let word = code.encode(&bits(4000, 9));
        assert!(code.decode(&certain(&word), 50).as_ref() == Some(&word));
        // one bit in twenty-five read wrongly and as surely as the others
        let mut seen = certain(&word);
        for value in seen.iter_mut().step_by(25) {
            *value = -*value;
        }
        assert!(
            code.decode(&seen, 50).as_ref() == Some(&word),
            "bits that are wrong"
        );
        // a quarter of the bits not read at all
        let mut seen = certain(&word);
        for value in seen.iter_mut().skip(1).step_by(4) {
            *value = 0.0;
        }
        assert!(
            code.decode(&seen, 50).as_ref() == Some(&word),
            "bits that are not read"
        );
        // one bit in nine wrong, which the reader was less sure of than of the others: more than could
        // be put right if it was as sure of them
        let mut seen = certain(&word);
        for value in seen.iter_mut().step_by(9) {
            *value *= -0.3;
        }
        assert!(
            code.decode(&seen, 50).as_ref() == Some(&word),
            "bits that the reader doubts"
        );
        for value in seen.iter_mut().step_by(9) {
            *value /= 0.3;
        }
        assert!(code.decode(&seen, 50).is_none(), "bits that it does not doubt");
    }

    #[test]
    fn word_that_is_read_too_badly_is_none() {
        let code = Code::new(6000, 4000);
        let word = code.encode(&bits(4000, 11));
        let mut seen = certain(&word);
        for value in seen.iter_mut().step_by(4) {
            *value = -*value;
        }
        assert!(code.decode(&seen, 30).is_none());
        // a reader that knows nothing takes every bit for a 0, which is a code word
        assert!(code.decode(&vec![0.0; 6000], 30) == Some(vec![0; 6000]));
    }
}
