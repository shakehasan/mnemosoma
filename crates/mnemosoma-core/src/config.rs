//! Every number that shapes ranking and indexing, in one place.
//!
//! The defaults are what the tests, the eval suite and the published
//! measurements use. Each field says what it trades off, so a deployment can
//! change one knowingly. Unknown fields are rejected when loading from JSON;
//! missing ones take their default.

use crate::ann::HnswParams;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Tuning {
    // ---- Text -------------------------------------------------------
    /// Reduce words to their stems, so "booked" matches "booking". Names a
    /// language such as `"english"`; `None` leaves words as written.
    pub stem_language: Option<String>,
    /// Leave common English function words ("the", "is", "of") out of the
    /// index and out of questions.
    pub drop_stopwords: bool,
    /// Weight of each character trigram, relative to 1 for a whole word, in
    /// the built-in embedding. Higher values make it more tolerant of
    /// spelling differences and less exact.
    pub trigram_weight: f32,
    /// Vector matches at or below this cosine similarity are ignored. The
    /// built-in embedding gives unrelated texts a similarity near zero that
    /// is sometimes slightly positive by chance; this keeps those from
    /// counting as matches. It is off (zero) by default: on LongMemEval,
    /// weak matches still carried signal, and a floor of 0.05 or 0.1 lowered
    /// recall at ten by 0.4 and 1.7 points.
    pub min_vector_similarity: f32,

    // ---- Names ------------------------------------------------------
    /// Find the names a memory mentions when it is written, from
    /// capitalisation alone, and add them to the ones the caller
    /// supplied. Graph expansion and the name ranker then work without
    /// the caller listing names. Suits languages that capitalise names
    /// and little else.
    pub extract_entities: bool,
    /// Most names kept per memory when extracting. The most frequent in
    /// the text are kept.
    pub max_extracted_entities: usize,

    // ---- Rank fusion -------------------------------------------------
    /// Reciprocal rank fusion constant. Larger values flatten the gap
    /// between first and later ranks. 60 is the value from the original
    /// paper and the common default.
    pub rrf_k: f32,
    /// Each ranker returns `k * pool_factor` candidates (at least
    /// `min_pool`) before fusion. Wider pools find more, slower.
    pub pool_factor: usize,
    pub min_pool: usize,

    // ---- Keyword ranking (Okapi BM25) --------------------------------
    /// Term-frequency saturation.
    pub bm25_k1: f32,
    /// Length normalisation, from 0 (none) to 1 (full).
    pub bm25_b: f32,

    // ---- Graph expansion ---------------------------------------------
    /// How many of each ranker's best results start the expansion.
    pub graph_seeds_per_ranker: usize,
    /// Share of a memory's score passed along each entity link.
    pub graph_carry: f32,
    /// Most memories carried forward from one hop to the next.
    pub graph_frontier: usize,

    // ---- Tie-breaking priors -----------------------------------------
    // Rank fusion packs scores within a few percent of each other, so these
    // stay close to 1: they settle near-ties and never let a recently used
    // or more trusted memory outrank a clearly better match.
    pub weight_trusted: f32,
    pub weight_standard: f32,
    /// Applied when untrusted content is explicitly requested.
    pub weight_untrusted: f32,
    /// Applied to an agent's inference that nobody has confirmed yet.
    pub weight_unconfirmed: f32,
    /// Largest share of the score that salience can move.
    pub salience_weight: f32,

    // ---- Approximate vector search -----------------------------------
    /// Below this many eligible rows in a scope, vector scoring is exact.
    pub ann_min_rows: usize,
    /// Below this share of a scope being readable, vector scoring is exact:
    /// the index would mostly walk past rows the caller must skip.
    pub ann_min_share: f32,
    /// Smallest and largest candidate list for an approximate search.
    pub ann_min_ef: usize,
    pub ann_max_ef: usize,
    /// Graph shape for the approximate index.
    pub hnsw: HnswParams,

    // ---- Token budget ------------------------------------------------
    /// Characters per token when estimating how much of a budget a memory
    /// uses. Four is a common rough figure for English text.
    pub chars_per_token: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            stem_language: Some("english".into()),
            drop_stopwords: true,
            trigram_weight: 0.5,
            min_vector_similarity: 0.0,
            extract_entities: false,
            max_extracted_entities: 16,
            rrf_k: 60.0,
            pool_factor: 4,
            min_pool: 32,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            graph_seeds_per_ranker: 5,
            graph_carry: 0.7,
            graph_frontier: 16,
            weight_trusted: 1.0,
            weight_standard: 0.99,
            weight_untrusted: 0.9,
            weight_unconfirmed: 0.99,
            salience_weight: 0.01,
            ann_min_rows: 4_096,
            ann_min_share: 0.05,
            ann_min_ef: 64,
            ann_max_ef: 4_096,
            hnsw: HnswParams::default(),
            chars_per_token: 4,
        }
    }
}

impl Tuning {
    /// Check that every value is in a range the engine can use.
    ///
    /// # Errors
    ///
    /// Returns a description of the first value out of range.
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("rrf_k", self.rrf_k),
            ("trigram_weight", self.trigram_weight),
            ("bm25_k1", self.bm25_k1),
            ("weight_trusted", self.weight_trusted),
            ("weight_standard", self.weight_standard),
            ("weight_untrusted", self.weight_untrusted),
            ("weight_unconfirmed", self.weight_unconfirmed),
        ];
        for (name, value) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{name} must be a positive number, got {value}"));
            }
        }
        let unit = [
            ("bm25_b", self.bm25_b),
            ("graph_carry", self.graph_carry),
            ("salience_weight", self.salience_weight),
            ("min_vector_similarity", self.min_vector_similarity),
            ("ann_min_share", self.ann_min_share),
        ];
        for (name, value) in unit {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be between 0 and 1, got {value}"));
            }
        }
        let counts = [
            ("pool_factor", self.pool_factor),
            ("min_pool", self.min_pool),
            ("max_extracted_entities", self.max_extracted_entities),
            ("graph_frontier", self.graph_frontier),
            ("ann_min_ef", self.ann_min_ef),
            ("chars_per_token", self.chars_per_token),
            ("hnsw.m", self.hnsw.m),
            ("hnsw.ef_construction", self.hnsw.ef_construction),
        ];
        for (name, value) in counts {
            if value == 0 {
                return Err(format!("{name} must be at least 1"));
            }
        }
        if let Some(language) = &self.stem_language {
            crate::text::stemming_algorithm(language)?;
        }
        if self.ann_max_ef < self.ann_min_ef {
            return Err("ann_max_ef must be at least ann_min_ef".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_round_trip_through_json() {
        let tuning = Tuning::default();
        assert_eq!(tuning.validate(), Ok(()));
        let json = serde_json::to_string(&tuning).unwrap();
        assert_eq!(serde_json::from_str::<Tuning>(&json).unwrap(), tuning);
    }

    #[test]
    fn partial_json_fills_in_defaults_and_unknown_fields_are_rejected() {
        let partial: Tuning = serde_json::from_str(r#"{"rrf_k": 30.0}"#).unwrap();
        assert!((partial.rrf_k - 30.0).abs() < f32::EPSILON);
        assert_eq!(partial.bm25_b, Tuning::default().bm25_b);
        assert!(serde_json::from_str::<Tuning>(r#"{"rrf_kk": 30.0}"#).is_err());
    }

    #[test]
    fn out_of_range_values_are_reported() {
        let bad = Tuning {
            bm25_b: 1.5,
            ..Tuning::default()
        };
        assert!(bad.validate().unwrap_err().contains("bm25_b"));
        let bad = Tuning {
            ann_max_ef: 1,
            ..Tuning::default()
        };
        assert!(bad.validate().is_err());
    }
}
