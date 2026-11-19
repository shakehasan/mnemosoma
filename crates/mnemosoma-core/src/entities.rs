//! Finding the names a memory mentions.
//!
//! Graph expansion follows names shared between memories. A caller can list
//! them when writing; this module finds them when the caller does not. It
//! reads capitalisation only: no model and no dictionary of names. That
//! makes it cheap and predictable on the write path, and limited to
//! languages that capitalise names and little else.

use crate::text::is_stopword;
use std::collections::{HashMap, HashSet};

/// Day and month names. They are capitalised like names, but two memories
/// that both say "Monday" have nothing else in common. Keep the list
/// sorted: it is searched by bisection.
const CALENDAR_WORDS: &[&str] = &[
    "april",
    "august",
    "december",
    "february",
    "friday",
    "january",
    "july",
    "june",
    "march",
    "may",
    "monday",
    "november",
    "october",
    "saturday",
    "september",
    "sunday",
    "thursday",
    "tuesday",
    "wednesday",
];

/// Punctuation after which a capital letter says nothing, because the next
/// word starts a sentence, a line, a quotation or a speaker's turn.
const BREAKS: &[char] = &[
    '.', '!', '?', ':', ';', '\n', '"', '(', '[', '\u{201c}', '\u{2022}',
];

/// The only separator allowed between the words of one name.
const NAME_SEPARATOR: &str = " ";

/// A single-word name must have at least this many characters, so a stray
/// capital letter is not taken for one.
const MIN_NAME_CHARS: usize = 2;

/// The words of a name or a question, lower-cased, with punctuation
/// dropped.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The form under which a name is indexed: its words, lower-cased and
/// joined by single spaces, so "New-York" and "new york" are one name.
#[must_use]
pub fn key(name: &str) -> String {
    words(name).join(NAME_SEPARATOR)
}

struct Word<'a> {
    text: &'a str,
    /// Lower-cased, for a capitalised word. Left empty for any other word,
    /// which can never be part of a name.
    lower: String,
    /// Starts with a capital letter and is not a function word.
    name_like: bool,
    /// First word of the text, or follows one of [`BREAKS`].
    initial: bool,
    /// Follows the previous word after exactly one space.
    joined: bool,
}

fn scan(text: &str) -> Vec<Word<'_>> {
    let mut out: Vec<Word<'_>> = Vec::new();
    let mut gap_start = 0;
    let mut word_start: Option<usize> = None;
    let ends = text
        .char_indices()
        .map(|(at, c)| (at, c.is_alphanumeric()))
        .chain(std::iter::once((text.len(), false)));
    for (at, in_word) in ends {
        match (in_word, word_start) {
            (true, None) => word_start = Some(at),
            (false, Some(start)) => {
                let word = &text[start..at];
                let gap = &text[gap_start..start];
                let capital = word.chars().next().is_some_and(char::is_uppercase);
                let lower = if capital {
                    word.to_lowercase()
                } else {
                    String::new()
                };
                let name_like = capital && !is_stopword(&lower);
                out.push(Word {
                    text: word,
                    lower,
                    name_like,
                    initial: out.is_empty() || gap.contains(BREAKS),
                    joined: gap == NAME_SEPARATOR,
                });
                gap_start = at;
                word_start = None;
            }
            _ => {}
        }
    }
    out
}

struct Found {
    display: String,
    count: usize,
    first: usize,
}

/// Count one occurrence of the name made of `run`, then empty `run`.
fn record(run: &mut Vec<&Word<'_>>, found: &mut HashMap<String, Found>) {
    if run.is_empty() {
        return;
    }
    let name: Vec<&str> = run.iter().map(|w| w.lower.as_str()).collect();
    let alone = name.len() == 1;
    let too_short = alone && name[0].chars().count() < MIN_NAME_CHARS;
    let calendar = alone && CALENDAR_WORDS.binary_search(&name[0]).is_ok();
    if !too_short && !calendar {
        let first = found.len();
        let entry = found
            .entry(name.join(NAME_SEPARATOR))
            .or_insert_with(|| Found {
                display: run
                    .iter()
                    .map(|w| w.text)
                    .collect::<Vec<_>>()
                    .join(NAME_SEPARATOR),
                count: 0,
                first,
            });
        entry.count += 1;
    }
    run.clear();
}

/// The words of the name that would start at `scanned[at]`: that word and
/// the name-like words joined to it.
fn run_from<'a, 'b>(scanned: &'b [Word<'a>], at: usize) -> Vec<&'b str> {
    let more = scanned[at + 1..]
        .iter()
        .take_while(|w| w.name_like && w.joined && !w.initial);
    std::iter::once(&scanned[at])
        .chain(more)
        .map(|w| w.lower.as_str())
        .collect()
}

/// Names mentioned in `text`, most frequent first, at most `limit`.
///
/// A name is a run of capitalised words separated by single spaces, such
/// as "Dana", "New York" or "Project Apollo". Function words, and day and
/// month names on their own, are never names.
///
/// A capital that starts a sentence proves nothing on its own. A single
/// capitalised word there counts in two cases only: the same word is
/// capitalised mid-sentence elsewhere in the text, or it is already `known`
/// (given as a [`key`]) as a name from earlier memories. Two or more
/// capitalised words in a row count as a name wherever they stand.
#[must_use]
pub fn extract(text: &str, limit: usize, known: impl Fn(&str) -> bool) -> Vec<String> {
    let scanned = scan(text);
    let mid_sentence: HashSet<&str> = scanned
        .iter()
        .filter(|w| w.name_like && !w.initial)
        .map(|w| w.lower.as_str())
        .collect();

    let mut found: HashMap<String, Found> = HashMap::new();
    let mut run: Vec<&Word<'_>> = Vec::new();
    for (at, word) in scanned.iter().enumerate() {
        let capital_by_position = word.name_like
            && word.initial
            && !mid_sentence.contains(word.lower.as_str())
            && match run_from(&scanned, at).as_slice() {
                [alone] => !known(alone),
                _ => false,
            };
        if !word.name_like || capital_by_position {
            record(&mut run, &mut found);
            continue;
        }
        if !word.joined || word.initial {
            record(&mut run, &mut found);
        }
        run.push(word);
    }
    record(&mut run, &mut found);

    let mut ranked: Vec<Found> = found.into_values().collect();
    ranked.sort_by(|a, b| b.count.cmp(&a.count).then(a.first.cmp(&b.first)));
    ranked.truncate(limit);
    ranked.into_iter().map(|f| f.display).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extraction with no names known beforehand.
    fn extract(text: &str, limit: usize) -> Vec<String> {
        super::extract(text, limit, |_| false)
    }

    #[test]
    fn calendar_list_is_sorted_and_unique() {
        assert!(CALENDAR_WORDS.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn key_ignores_case_and_punctuation() {
        assert_eq!(key("  New-York "), "new york");
        assert_eq!(key("GPT-4"), "gpt 4");
        assert_eq!(key("!!"), "");
    }

    #[test]
    fn finds_single_and_multi_word_names() {
        assert_eq!(
            extract("We moved Project Apollo to Dana in New York.", 8),
            ["Project Apollo", "Dana", "New York"]
        );
    }

    #[test]
    fn a_capital_that_only_starts_a_sentence_is_not_a_name() {
        assert_eq!(
            extract("Yesterday the launch slipped. Nobody told Dana.", 8),
            ["Dana"]
        );
        // The same word mid-sentence elsewhere makes it a name everywhere.
        assert_eq!(extract("Dana called. I rang Dana back.", 8), ["Dana"]);
    }

    #[test]
    fn a_known_name_is_recognised_at_the_start_of_a_sentence() {
        let known = |name: &str| name == "apollo";
        let text = "Apollo ships soon. Tomorrow it rains.";
        assert_eq!(super::extract(text, 8, known), ["Apollo"]);
        assert!(extract(text, 8).is_empty());
    }

    #[test]
    fn two_capitalised_words_in_a_row_are_a_name_wherever_they_stand() {
        assert_eq!(
            extract("Dana Reyes sits on floor four. New York is far.", 8),
            ["Dana Reyes", "New York"]
        );
    }

    #[test]
    fn speaker_labels_and_quotations_start_a_sentence() {
        assert_eq!(
            extract("user: Hello there\nassistant: Sure, ask Priya", 8),
            ["Priya"]
        );
        assert!(extract("She said \"Maybe later\" and left", 8).is_empty());
    }

    #[test]
    fn function_words_and_calendar_words_are_not_names() {
        assert!(extract("so I said The plan starts on Monday in March", 8).is_empty());
        assert_eq!(extract("we shop on Black Friday", 8), ["Black Friday"]);
    }

    #[test]
    fn a_function_word_splits_a_run_and_a_possessive_ends_it() {
        assert_eq!(
            extract("we saw Dana And Priya at the Priya's desk", 8),
            ["Priya", "Dana"]
        );
    }

    #[test]
    fn most_frequent_names_are_kept_when_over_the_limit() {
        let text = "ask Ana, then Ben, then Cy, then Ben again, then Cy, then Ben";
        assert_eq!(extract(text, 2), ["Ben", "Cy"]);
        assert!(extract(text, 0).is_empty());
    }

    #[test]
    fn text_without_capitals_or_letters_has_no_names() {
        assert!(extract("nothing to see here", 8).is_empty());
        assert!(extract("", 8).is_empty());
        assert!(extract("12:30 to 14:00", 8).is_empty());
    }
}
