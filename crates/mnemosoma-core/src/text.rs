//! Turning text into index terms.
//!
//! The same tokenizer runs when a memory is written and when a question is
//! asked, so both sides agree on what a word is. It is built once from
//! [`Tuning`](crate::config::Tuning) when the store is created.

use rust_stemmers::{Algorithm, Stemmer};

/// Languages the stemmer supports, by the name used in configuration.
const LANGUAGES: &[(&str, Algorithm)] = &[
    ("arabic", Algorithm::Arabic),
    ("danish", Algorithm::Danish),
    ("dutch", Algorithm::Dutch),
    ("english", Algorithm::English),
    ("finnish", Algorithm::Finnish),
    ("french", Algorithm::French),
    ("german", Algorithm::German),
    ("greek", Algorithm::Greek),
    ("hungarian", Algorithm::Hungarian),
    ("italian", Algorithm::Italian),
    ("norwegian", Algorithm::Norwegian),
    ("portuguese", Algorithm::Portuguese),
    ("romanian", Algorithm::Romanian),
    ("russian", Algorithm::Russian),
    ("spanish", Algorithm::Spanish),
    ("swedish", Algorithm::Swedish),
    ("tamil", Algorithm::Tamil),
    ("turkish", Algorithm::Turkish),
];

/// Common English function words. They appear in almost every memory, so
/// they say little about which one is relevant. Keep the list sorted: it is
/// searched by bisection.
const ENGLISH_STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "my", "no", "not", "of", "on", "or", "our", "out", "s", "she", "so", "some", "t", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "too", "up",
    "us", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with", "would",
    "you", "your",
];

/// Look up a stemming language by name.
///
/// # Errors
///
/// Returns the list of supported names when `name` is not one of them.
pub fn stemming_algorithm(name: &str) -> Result<Algorithm, String> {
    let wanted = name.to_lowercase();
    LANGUAGES
        .iter()
        .find(|(known, _)| *known == wanted)
        .map(|(_, algorithm)| *algorithm)
        .ok_or_else(|| {
            let known: Vec<&str> = LANGUAGES.iter().map(|(n, _)| *n).collect();
            format!(
                "unknown stemming language '{name}'; choose one of: {}",
                known.join(", ")
            )
        })
}

pub struct Tokenizer {
    stemmer: Option<Stemmer>,
    drop_stopwords: bool,
}

impl Tokenizer {
    /// A tokenizer that lower-cases and splits on anything that is not a
    /// letter or digit, with no stemming and no stopword removal.
    #[must_use]
    pub fn plain() -> Self {
        Self {
            stemmer: None,
            drop_stopwords: false,
        }
    }

    /// Build a tokenizer.
    ///
    /// # Errors
    ///
    /// Fails if `stem_language` names a language the stemmer does not know.
    pub fn new(stem_language: Option<&str>, drop_stopwords: bool) -> Result<Self, String> {
        let stemmer = stem_language
            .map(stemming_algorithm)
            .transpose()?
            .map(Stemmer::create);
        Ok(Self {
            stemmer,
            drop_stopwords,
        })
    }

    /// Split text into index terms.
    #[must_use]
    pub fn tokens(&self, text: &str) -> Vec<String> {
        let mut words: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect();
        // A text made only of stopwords keeps them, so it is still findable.
        if self.drop_stopwords && words.iter().any(|w| !is_stopword(w)) {
            words.retain(|w| !is_stopword(w));
        }
        if let Some(stemmer) = &self.stemmer {
            for word in &mut words {
                let stem = stemmer.stem(word).into_owned();
                *word = stem;
            }
        }
        words
    }
}

/// The stopword list is kept sorted so this is a binary search.
fn is_stopword(word: &str) -> bool {
    ENGLISH_STOPWORDS.binary_search(&word).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopword_list_is_sorted_and_unique() {
        assert!(ENGLISH_STOPWORDS.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn plain_tokenizer_lowercases_and_splits() {
        assert_eq!(
            Tokenizer::plain().tokens("Ana's flight-number is QF12!"),
            ["ana", "s", "flight", "number", "is", "qf12"]
        );
    }

    #[test]
    fn stemming_joins_word_forms() {
        let t = Tokenizer::new(Some("english"), false).unwrap();
        assert_eq!(t.tokens("booked booking books"), ["book", "book", "book"]);
        assert_eq!(t.tokens("Running"), t.tokens("runs"));
    }

    #[test]
    fn stopwords_are_dropped_unless_nothing_else_is_left() {
        let t = Tokenizer::new(None, true).unwrap();
        assert_eq!(
            t.tokens("What is the venue for the dinner"),
            ["venue", "dinner"]
        );
        assert_eq!(t.tokens("who is it"), ["who", "is", "it"]);
    }

    #[test]
    fn unknown_language_is_reported_with_the_choices() {
        let error = Tokenizer::new(Some("klingon"), false).err().unwrap();
        assert!(error.contains("klingon") && error.contains("english"));
    }
}
