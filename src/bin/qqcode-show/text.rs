//! Fits what the program has to say into the columns there are.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// The longest beginning of `text` that fits into `width` columns.
pub fn fit(text: &str, width: usize) -> &str {
    let mut used = 0;
    for (index, c) in text.char_indices() {
        used += c.width().unwrap_or(0);
        if used > width {
            return &text[..index];
        }
    }
    text
}

/// `text` as rows of at most `width` columns. Its lines are broken at spaces; a word that is wider than a
/// row on its own is broken wherever the row ends. An empty line is one empty row.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut rows = Vec::new();
    if width == 0 {
        return rows;
    }
    if text.is_empty() {
        rows.push(String::new());
    }
    for line in text.lines() {
        let before = rows.len();
        let mut row = String::new();
        for mut word in line.split(' ').filter(|word| !word.is_empty()) {
            let gap = usize::from(!row.is_empty());
            if row.width() + gap + word.width() <= width {
                if gap == 1 {
                    row.push(' ');
                }
            } else {
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                }
                while word.width() > width {
                    let mut head = fit(word, width);
                    if head.is_empty() {
                        // a character wider than the whole row still has to go somewhere
                        head = &word[..word.chars().next().map_or(0, char::len_utf8)];
                    }
                    rows.push(head.to_owned());
                    word = &word[head.len()..];
                }
            }
            row.push_str(word);
        }
        if !row.is_empty() || rows.len() == before {
            rows.push(row);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_lines_are_broken_between_words() {
        assert_eq!(
            wrap("the quick brown fox jumps", 10),
            ["the quick", "brown fox", "jumps"]
        );
        // exactly as wide as the row
        assert_eq!(wrap("abcde fghij", 5), ["abcde", "fghij"]);
        assert_eq!(wrap("one     two", 5), ["one", "two"]);
    }

    #[test]
    fn lines_stay_lines_and_empty_ones_stay_empty() {
        assert_eq!(wrap("one two\n\nthree", 20), ["one two", "", "three"]);
        assert_eq!(wrap("", 20), [""]);
        assert!(wrap("text", 0).is_empty());
    }

    #[test]
    fn a_word_wider_than_the_row_is_broken_inside() {
        assert_eq!(wrap("a 0123456789abc b", 5), ["a", "01234", "56789", "abc b"]);
        // two columns per character
        assert_eq!(wrap("漢字漢字漢", 4), ["漢字", "漢字", "漢"]);
        assert_eq!(wrap("漢字", 1), ["漢", "字"]);
    }

    #[test]
    fn text_is_cut_to_the_columns_there_are() {
        assert_eq!(fit("0123456789", 4), "0123");
        assert_eq!(fit("ab", 4), "ab");
        assert_eq!(fit("漢字", 3), "漢");
    }
}
