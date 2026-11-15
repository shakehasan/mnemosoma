//! Ranking primitives: tokenizer, a dependency-free hashing embedder,
//! similarity functions, top-n selection and reciprocal rank fusion.

use std::collections::HashMap;

pub const HASH_DIM: usize = 256;

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A feature-hashing embedding over words and character trigrams. It needs no
/// model and no network, so the store works out of the box. Callers who want
/// semantic recall pass their own embeddings instead.
pub fn hash_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; HASH_DIM];
    {
        let mut add = |feature: &[u8], weight: f32| {
            let h = fnv1a(feature);
            let i = (h % HASH_DIM as u64) as usize;
            let sign = if h >> 63 == 1 { -1.0 } else { 1.0 };
            v[i] += sign * weight;
        };
        for tok in tokenize(text) {
            add(tok.as_bytes(), 1.0);
            let chars: Vec<char> = tok.chars().collect();
            for w in chars.windows(3) {
                let gram: String = std::iter::once('#').chain(w.iter().copied()).collect();
                add(gram.as_bytes(), 0.5);
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

/// Dot product, written so the compiler can vectorise it.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (a8, b8) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a8.remainder().iter().zip(b8.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in a8.zip(b8) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
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

pub fn sort_desc(v: &mut [(u64, f32)]) {
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
}

/// Reciprocal rank fusion. Each ranking is a best-first list of ids.
pub fn rrf(rankings: &[Vec<u64>]) -> HashMap<u64, f32> {
    const K: f32 = 60.0;
    let mut fused = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *fused.entry(*id).or_insert(0.0) += 1.0 / (K + rank as f32 + 1.0);
        }
    }
    fused
}
