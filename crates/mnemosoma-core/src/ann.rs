//! Approximate nearest-neighbour search: a hierarchical navigable small
//! world (HNSW) graph over unit-length vectors.
//!
//! The graph stores only links. Vectors live in the caller's flat array
//! (`dim` floats per node) and are passed to every call, so the index adds
//! no copy of the data. Similarity is the dot product, which equals cosine
//! similarity for unit-length vectors.
//!
//! Search takes an `accept` filter. Every node can be walked through, but
//! only accepted nodes are returned, so permission and time filters applied
//! before ranking still hold: a memory the caller may not read is never
//! returned, it is only stepped over.

use crate::index::dot;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Highest layer a node can be placed on. With the default link count this
/// is never reached below billions of nodes; it only bounds memory use.
const MAX_LEVEL: usize = 16;
/// Seed for the level generator, so the same inserts build the same graph.
const LEVEL_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Shape of the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HnswParams {
    /// Links per node on upper layers; the bottom layer allows twice as
    /// many. More links give better recall and a larger, slower-built graph.
    pub m: usize,
    /// Candidate list size while inserting. Larger builds a better graph,
    /// more slowly.
    pub ef_construction: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Near {
    sim: f32,
    node: u32,
}
impl Eq for Near {}
impl Ord for Near {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim
            .total_cmp(&other.sim)
            .then(other.node.cmp(&self.node))
    }
}
impl PartialOrd for Near {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Wraps `Near` so a `BinaryHeap` pops the least similar first.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Far(Near);
impl Ord for Far {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}
impl PartialOrd for Far {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Marks which nodes a search has seen. Each search bumps `epoch` instead of
/// clearing the array, so starting a search costs nothing however large the
/// graph is.
#[derive(Default)]
struct Visited {
    stamp: Vec<u32>,
    epoch: u32,
}

impl Visited {
    fn start(&mut self, nodes: usize) {
        if self.stamp.len() < nodes {
            self.stamp.resize(nodes, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
    }

    /// True the first time a node is seen in this search.
    fn first_visit(&mut self, node: u32) -> bool {
        let slot = &mut self.stamp[node as usize];
        let fresh = *slot != self.epoch;
        *slot = self.epoch;
        fresh
    }
}

#[derive(Default)]
pub struct Hnsw {
    params: HnswParams,
    /// `links[node][level]` holds the node's neighbours on that level.
    links: Vec<Vec<Vec<u32>>>,
    entry: Option<u32>,
    top_level: usize,
    seed: u64,
    visited: RefCell<Visited>,
}

fn vector(data: &[f32], dim: usize, node: u32) -> &[f32] {
    let start = node as usize * dim;
    &data[start..start + dim]
}

impl Hnsw {
    #[must_use]
    pub fn new(params: HnswParams) -> Self {
        Self {
            params,
            ..Self::default()
        }
    }

    /// Number of nodes in the graph. Nodes are numbered from zero in the
    /// order they were inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    fn random_level(&mut self) -> usize {
        // xorshift64*, seeded deterministically so builds are repeatable.
        if self.seed == 0 {
            self.seed = LEVEL_SEED;
        }
        self.seed ^= self.seed >> 12;
        self.seed ^= self.seed << 25;
        self.seed ^= self.seed >> 27;
        let bits = self.seed.wrapping_mul(LEVEL_SEED);
        let uniform = ((bits >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let m = self.params.m.max(2) as f64;
        let level = (-uniform.ln() / m.ln()).floor() as usize;
        level.min(MAX_LEVEL)
    }

    /// Add the next node. `data` must already hold its vector at position
    /// `self.len()`.
    pub fn insert(&mut self, data: &[f32], dim: usize) {
        let node = self.links.len() as u32;
        let level = self.random_level();
        self.links.push(vec![Vec::new(); level + 1]);
        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.top_level = level;
            return;
        };
        let query = vector(data, dim, node);
        let mut nearest = Near {
            sim: dot(query, vector(data, dim, entry)),
            node: entry,
        };
        for layer in (level + 1..=self.top_level).rev() {
            nearest = self.greedy(query, data, dim, nearest, layer);
        }
        let mut entries = vec![nearest];
        for layer in (0..=level.min(self.top_level)).rev() {
            let found = self.search_layer(
                query,
                data,
                dim,
                &entries,
                self.params.ef_construction,
                layer,
                &|_| true,
            );
            let limit = if layer == 0 {
                2 * self.params.m
            } else {
                self.params.m
            };
            let chosen = select(&found, data, dim, self.params.m);
            self.links[node as usize][layer] = chosen.iter().map(|n| n.node).collect();
            for neighbour in &chosen {
                let list = &mut self.links[neighbour.node as usize][layer];
                list.push(node);
                if list.len() > limit {
                    let base = vector(data, dim, neighbour.node);
                    let mut candidates: Vec<Near> = list
                        .iter()
                        .map(|n| Near {
                            sim: dot(base, vector(data, dim, *n)),
                            node: *n,
                        })
                        .collect();
                    // Keep the closest. Re-running the diversity heuristic here
                    // costs a full pairwise comparison per overflow, which made
                    // building large indexes several times slower.
                    candidates.sort_by(|a, b| b.cmp(a));
                    candidates.truncate(limit);
                    self.links[neighbour.node as usize][layer] =
                        candidates.iter().map(|n| n.node).collect();
                }
            }
            entries = found;
        }
        if level > self.top_level {
            self.top_level = level;
            self.entry = Some(node);
        }
    }

    fn greedy(
        &self,
        query: &[f32],
        data: &[f32],
        dim: usize,
        mut best: Near,
        layer: usize,
    ) -> Near {
        loop {
            let mut moved = false;
            for n in self.links[best.node as usize]
                .get(layer)
                .into_iter()
                .flatten()
            {
                let sim = dot(query, vector(data, dim, *n));
                if sim > best.sim {
                    best = Near { sim, node: *n };
                    moved = true;
                }
            }
            if !moved {
                return best;
            }
        }
    }

    /// Best-first search of one layer. Returns up to `ef` accepted nodes,
    /// most similar first. Rejected nodes are still walked through.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        &self,
        query: &[f32],
        data: &[f32],
        dim: usize,
        entries: &[Near],
        ef: usize,
        layer: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<Near> {
        let mut visited = self.visited.borrow_mut();
        visited.start(self.links.len());
        let mut frontier: BinaryHeap<Near> = BinaryHeap::new();
        let mut walked: BinaryHeap<Far> = BinaryHeap::new();
        let mut found: BinaryHeap<Far> = BinaryHeap::new();
        for e in entries {
            if visited.first_visit(e.node) {
                frontier.push(*e);
                walked.push(Far(*e));
                if accept(e.node) {
                    found.push(Far(*e));
                }
            }
        }
        // `walked` bounds the search like a plain HNSW result set; `found`
        // keeps the accepted nodes among everything walked.
        let width = ef.max(1);
        while walked.len() > width {
            walked.pop();
        }
        while let Some(current) = frontier.pop() {
            if walked.len() >= width && walked.peek().is_some_and(|worst| current.sim < worst.0.sim)
            {
                break;
            }
            for n in self.links[current.node as usize]
                .get(layer)
                .into_iter()
                .flatten()
            {
                if !visited.first_visit(*n) {
                    continue;
                }
                let near = Near {
                    sim: dot(query, vector(data, dim, *n)),
                    node: *n,
                };
                if walked.len() < width || walked.peek().is_some_and(|worst| near.sim > worst.0.sim)
                {
                    frontier.push(near);
                    walked.push(Far(near));
                    if walked.len() > width {
                        walked.pop();
                    }
                }
                if accept(*n) {
                    found.push(Far(near));
                    if found.len() > width {
                        found.pop();
                    }
                }
            }
        }
        let mut out: Vec<Near> = found.into_iter().map(|f| f.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    }

    /// The `k` accepted nodes most similar to `query`, best first, as
    /// `(node, similarity)`. A larger `ef` trades speed for accuracy.
    pub fn search(
        &self,
        query: &[f32],
        data: &[f32],
        dim: usize,
        k: usize,
        ef: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        let Some(entry) = self.entry else {
            return vec![];
        };
        let mut nearest = Near {
            sim: dot(query, vector(data, dim, entry)),
            node: entry,
        };
        for layer in (1..=self.top_level).rev() {
            nearest = self.greedy(query, data, dim, nearest, layer);
        }
        let found = self.search_layer(query, data, dim, &[nearest], ef.max(k), 0, accept);
        found.into_iter().take(k).map(|n| (n.node, n.sim)).collect()
    }
}

/// Neighbour selection heuristic from the HNSW paper: keep a candidate only
/// if it is closer to the base node than to any neighbour already kept. This
/// keeps links pointing in different directions, which matters for
/// clustered data.
fn select(candidates: &[Near], data: &[f32], dim: usize, limit: usize) -> Vec<Near> {
    let mut kept: Vec<Near> = Vec::with_capacity(limit);
    for c in candidates {
        if kept.len() >= limit {
            break;
        }
        let v = vector(data, dim, c.node);
        if kept
            .iter()
            .all(|k| dot(v, vector(data, dim, k.node)) < c.sim)
        {
            kept.push(*c);
        }
    }
    // Fill any remaining room with the closest leftovers.
    for c in candidates {
        if kept.len() >= limit {
            break;
        }
        if !kept.iter().any(|k| k.node == c.node) {
            kept.push(*c);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2001) as f32 / 1000.0 - 1.0
        };
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            data.extend(v.iter().map(|x| x / norm));
        }
        data
    }

    fn exact(
        query: &[f32],
        data: &[f32],
        dim: usize,
        k: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<u32> {
        let mut all: Vec<(u32, f32)> = (0..(data.len() / dim) as u32)
            .filter(|n| accept(*n))
            .map(|n| (n, dot(query, vector(data, dim, n))))
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        all.into_iter().take(k).map(|x| x.0).collect()
    }

    fn recall_at_10(filter: &dyn Fn(u32) -> bool, ef: usize) -> f32 {
        let (n, dim) = (5_000, 32);
        let data = random_unit_vectors(n, dim, 42);
        let mut index = Hnsw::default();
        for _ in 0..n {
            index.insert(&data, dim);
        }
        let queries = random_unit_vectors(100, dim, 7);
        let mut hit = 0;
        for q in queries.chunks(dim) {
            let truth = exact(q, &data, dim, 10, filter);
            let got: Vec<u32> = index
                .search(q, &data, dim, 10, ef, filter)
                .into_iter()
                .map(|x| x.0)
                .collect();
            assert!(
                got.iter().all(|n| filter(*n)),
                "a rejected node was returned"
            );
            hit += truth.iter().filter(|t| got.contains(t)).count();
        }
        hit as f32 / 1000.0
    }

    #[test]
    fn finds_nearly_all_true_neighbours() {
        let recall = recall_at_10(&|_| true, 64);
        assert!(recall >= 0.95, "recall@10 was {recall}");
    }

    #[test]
    fn filtered_search_returns_only_accepted_nodes() {
        let recall = recall_at_10(&|n| n % 4 == 0, 256);
        assert!(recall >= 0.9, "filtered recall@10 was {recall}");
    }

    #[test]
    fn empty_index_returns_nothing() {
        assert!(Hnsw::default()
            .search(&[1.0, 0.0], &[], 2, 5, 10, &|_| true)
            .is_empty());
    }
}
