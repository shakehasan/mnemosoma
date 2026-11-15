//! Ranking primitives: tokenizer, BM25, a dependency-free hashing embedder,
//! cosine similarity and reciprocal rank fusion.

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

/// Okapi BM25 over the given documents. Returns matches only, best first.
pub fn bm25(query: &[String], docs: &[(u64, &[String])]) -> Vec<(u64, f32)> {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;
    if docs.is_empty() || query.is_empty() {
        return vec![];
    }
    let mut terms: Vec<&String> = query.iter().collect();
    terms.sort();
    terms.dedup();

    let n = docs.len() as f32;
    let avg_len = docs.iter().map(|d| d.1.len()).sum::<usize>() as f32 / n;
    let mut df = vec![0u32; terms.len()];
    let mut tfs: Vec<Vec<u32>> = Vec::with_capacity(docs.len());
    for (_, toks) in docs {
        let mut tf = vec![0u32; terms.len()];
        for t in toks.iter() {
            if let Ok(i) = terms.binary_search(&t) {
                tf[i] += 1;
            }
        }
        for (i, c) in tf.iter().enumerate() {
            if *c > 0 {
                df[i] += 1;
            }
        }
        tfs.push(tf);
    }
    let idf: Vec<f32> = df
        .iter()
        .map(|d| (1.0 + (n - *d as f32 + 0.5) / (*d as f32 + 0.5)).ln())
        .collect();

    let mut out: Vec<(u64, f32)> = docs
        .iter()
        .zip(&tfs)
        .filter_map(|((id, toks), tf)| {
            let len_norm = K1 * (1.0 - B + B * toks.len() as f32 / avg_len.max(1.0));
            let score: f32 = tf
                .iter()
                .zip(&idf)
                .filter(|(c, _)| **c > 0)
                .map(|(c, w)| w * (*c as f32 * (K1 + 1.0)) / (*c as f32 + len_norm))
                .sum();
            (score > 0.0).then_some((*id, score))
        })
        .collect();
    sort_desc(&mut out);
    out
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
