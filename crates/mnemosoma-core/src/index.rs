//! Ranking primitives: a dependency-free hashing embedder,
//! similarity functions, top-n selection and reciprocal rank fusion.

use std::collections::HashMap;

/// Dimensions of the built-in embedding.
pub const HASH_DIM: usize = 256;
/// Characters per n-gram in the built-in embedding.
const TRIGRAM: usize = 3;
/// Prefix that keeps a trigram's hash apart from a whole word spelled the same.
const TRIGRAM_MARK: char = '#';
/// FNV-1a 64-bit offset basis and prime.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// A feature-hashing embedding over words and their character trigrams. It
/// needs no model and no network, so the store works out of the box. Callers
/// who want semantic recall pass their own embeddings instead.
///
/// Each word counts 1 and each of its trigrams counts `trigram_weight`, so
/// words that are spelled alike land near each other. The result has unit
/// length.
#[must_use]
pub fn hash_embed(tokens: &[String], trigram_weight: f32) -> Vec<f32> {
    let mut v = vec![0f32; HASH_DIM];
    {
        let mut add = |feature: &[u8], weight: f32| {
            let h = fnv1a(feature);
            let i = (h % HASH_DIM as u64) as usize;
            let sign = if h >> 63 == 1 { -1.0 } else { 1.0 };
            v[i] += sign * weight;
        };
        for tok in tokens {
            add(tok.as_bytes(), 1.0);
            let chars: Vec<char> = tok.chars().collect();
            for w in chars.windows(TRIGRAM) {
                let gram: String = std::iter::once(TRIGRAM_MARK)
                    .chain(w.iter().copied())
                    .collect();
                add(gram.as_bytes(), trigram_weight);
            }
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Lanes summed independently in [`dot`], so the compiler can vectorise it.
const DOT_LANES: usize = 8;

/// Dot product of two equal-length vectors.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a_chunks, a_tail) = a.as_chunks::<DOT_LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<DOT_LANES>();
    let mut lanes = [0f32; DOT_LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for ((lane, p), q) in lanes.iter_mut().zip(x).zip(y) {
            *lane += p * q;
        }
    }
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(p, q)| p * q).sum();
    lanes.iter().sum::<f32>() + tail
}

/// The ids of the `n` best-scoring entries, best first.
pub fn top_ids(mut scored: Vec<(u64, f32)>, n: usize) -> Vec<u64> {
    let order = |a: &(u64, f32), b: &(u64, f32)| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0));
    if n == 0 {
        return vec![];
    }
    if scored.len() > n {
        scored.select_nth_unstable_by(n - 1, order);
        scored.truncate(n);
    }
    scored.sort_by(order);
    scored.into_iter().map(|entry| entry.0).collect()
}

/// Add one ranker's reciprocal-rank credit to `fused`, for a ranker whose
/// scores often tie. `ranking` is best first. Entries with the same score
/// share the mean of the ranks they occupy, so the order among them, which
/// carries no information, does not decide who gets more credit.
pub fn rrf_tied(fused: &mut HashMap<u64, f32>, ranking: &[(u64, f32)], k: f32) {
    let mut start = 0;
    while start < ranking.len() {
        let score = ranking[start].1;
        let end = start
            + ranking[start..]
                .iter()
                .take_while(|entry| entry.1 == score)
                .count();
        // Ranks are counted from 1; the group holds ranks start+1 to end.
        let mean_rank = (start + 1 + end) as f32 / 2.0;
        for (id, _) in &ranking[start..end] {
            *fused.entry(*id).or_insert(0.0) += 1.0 / (k + mean_rank);
        }
        start = end;
    }
}

pub fn sort_desc(v: &mut [(u64, f32)]) {
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
}

/// Reciprocal rank fusion. Each ranking is a best-first list of ids; `k`
/// controls how quickly credit falls off with rank.
#[must_use]
pub fn rrf(rankings: &[Vec<u64>], k: f32) -> HashMap<u64, f32> {
    let mut fused = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *fused.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_entries_share_the_mean_of_their_ranks() {
        let mut fused = HashMap::new();
        rrf_tied(&mut fused, &[(1, 3.0), (2, 1.0), (3, 1.0), (4, 0.5)], 60.0);
        assert!((fused[&1] - 1.0 / 61.0).abs() < 1e-7);
        // Ranks 2 and 3 are tied: both get the credit of rank 2.5.
        assert!((fused[&2] - 1.0 / 62.5).abs() < 1e-7);
        assert_eq!(fused[&2], fused[&3]);
        assert!((fused[&4] - 1.0 / 64.0).abs() < 1e-7);
    }

    #[test]
    fn without_ties_it_matches_plain_rank_fusion() {
        let mut tied = HashMap::new();
        rrf_tied(&mut tied, &[(7, 0.9), (8, 0.5), (9, 0.1)], 60.0);
        let plain = rrf(&[vec![7, 8, 9]], 60.0);
        for id in [7, 8, 9] {
            assert!((tied[&id] - plain[&id]).abs() < 1e-7);
        }
    }
}
