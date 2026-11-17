//! The memory store: an append-only event log, the indexes rebuilt from it,
//! and the read and write paths.

use crate::ann::Hnsw;
use crate::config::Tuning;
use crate::identity::{default_roles, Grant, Identity, RolePolicy, Scope, ScopeKind};
use crate::index::{cosine, dot, hash_embed, rrf, sort_desc, top_ids, HASH_DIM};
use crate::memory::{
    Assertion, Memory, MemoryKind, NewMemory, Trust, APPROVED_PREFIX,
    DEFAULT_REQUIRED_CORROBORATIONS,
};
use crate::provenance;
use crate::text::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("permission denied: {0}")]
    Denied(String),
    #[error("memory {0} not found")]
    NotFound(u64),
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("storage error: {0}")]
    Io(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// Default for [`StoreConfig::half_life_ms`]: thirty days.
pub const DEFAULT_HALF_LIFE_MS: i64 = 30 * 24 * 60 * 60 * 1000;
/// Default for [`StoreConfig::archive_below`].
pub const DEFAULT_ARCHIVE_BELOW: f32 = 0.05;

#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Key for write-time HMAC signatures. Keep it outside the log file.
    pub signing_key: Vec<u8>,
    /// Salience halves after this long without the memory being recalled.
    pub half_life_ms: i64,
    /// Episodes whose decayed salience falls below this are archived by `sweep`.
    pub archive_below: f32,
    /// Corroborations from other agents that establish an inference.
    pub required_corroborations: usize,
    /// Ask the operating system to put every write on disk before the call
    /// returns. Off by default: a write then survives the process crashing
    /// but can be lost if the machine loses power within moments of it.
    /// Turning it on trades write speed for that guarantee.
    pub sync_writes: bool,
    pub policies: HashMap<String, RolePolicy>,
    /// Ranking and indexing parameters. See [`Tuning`].
    pub tuning: Tuning,
}

impl StoreConfig {
    pub fn new(signing_key: impl Into<Vec<u8>>) -> Self {
        Self {
            signing_key: signing_key.into(),
            half_life_ms: DEFAULT_HALF_LIFE_MS,
            archive_below: DEFAULT_ARCHIVE_BELOW,
            required_corroborations: DEFAULT_REQUIRED_CORROBORATIONS,
            sync_writes: false,
            policies: default_roles(),
            tuning: Tuning::default(),
        }
    }

    /// Check every setting is usable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] naming the first setting out of range.
    pub fn validate(&self) -> Result<(), Error> {
        if self.signing_key.is_empty() {
            return Err(Error::Invalid("signing_key must not be empty".into()));
        }
        if self.half_life_ms <= 0 {
            return Err(Error::Invalid("half_life_ms must be positive".into()));
        }
        if !(0.0..=1.0).contains(&self.archive_below) {
            return Err(Error::Invalid(
                "archive_below must be between 0 and 1".into(),
            ));
        }
        if self.required_corroborations == 0 {
            return Err(Error::Invalid(
                "required_corroborations must be at least 1".into(),
            ));
        }
        self.tuning.validate().map_err(Error::Invalid)
    }
}

fn default_k() -> usize {
    8
}
fn default_min_trust() -> Trust {
    Trust::Standard
}
fn default_graph_hops() -> u8 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    #[serde(default)]
    pub text: String,
    /// When set, vector ranking uses this against caller-supplied memory
    /// embeddings of the same dimension. Otherwise the built-in hashing
    /// embedder is used on both sides.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default = "default_k")]
    pub k: usize,
    /// Approximate token budget for the returned memories (4 chars per token).
    #[serde(default)]
    pub token_budget: Option<usize>,
    /// Empty means all kinds.
    #[serde(default)]
    pub kinds: Vec<MemoryKind>,
    /// Empty means every scope the caller may read.
    #[serde(default)]
    pub scopes: Vec<ScopeKind>,
    /// Answer as of this moment in valid time. Defaults to now.
    #[serde(default)]
    pub as_of: Option<i64>,
    /// Untrusted content is excluded unless the caller asks for it.
    #[serde(default = "default_min_trust")]
    pub min_trust: Trust,
    #[serde(default)]
    pub include_archived: bool,
    /// How many steps to follow shared entities out from the best matches.
    /// `0` turns graph expansion off.
    #[serde(default = "default_graph_hops")]
    pub graph_hops: u8,
    /// Score every eligible vector instead of using the approximate index.
    #[serde(default)]
    pub exact: bool,
}

impl Query {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            embedding: None,
            k: default_k(),
            token_budget: None,
            kinds: vec![],
            scopes: vec![],
            as_of: None,
            min_trust: default_min_trust(),
            include_archived: false,
            graph_hops: default_graph_hops(),
            exact: false,
        }
    }
}

/// Why a memory was returned: its rank in each ranker (1 is best, `None`
/// means that ranker did not pick it) and whose grant made it visible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Why {
    pub keyword_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub entity_rank: Option<usize>,
    /// Reached by following shared entities from a better match.
    pub graph_rank: Option<usize>,
    /// Set when the memory belongs to another user who shared it.
    pub shared_by: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub memory: Memory,
    pub score: f32,
    /// False for an agent's inference that nobody has corroborated yet.
    pub established: bool,
    pub why: Why,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SweepReport {
    pub expired: usize,
    pub archived: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Stats {
    pub memories: usize,
    pub tenants: usize,
    pub grants: usize,
    /// Log entries rejected on load because their signature did not verify.
    pub quarantined: usize,
    /// Bytes dropped from the end of the log on load: a write that was cut
    /// off by a crash before it finished.
    pub recovered_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Event {
    Meta {
        next_id: u64,
        next_grant_id: u64,
    },
    Write {
        memory: Box<Memory>,
    },
    Supersede {
        old: u64,
        new: u64,
        at: i64,
    },
    Grant {
        grant: Grant,
    },
    Revoke {
        grant_id: u64,
        at: i64,
    },
    Forget {
        id: u64,
    },
    Corroborate {
        id: u64,
        by: String,
    },
    Archive {
        id: u64,
    },
    /// Several events that belong to one operation, written as one line so
    /// a crash leaves either all of them or none.
    Batch {
        events: Vec<Event>,
    },
}

type ChainKey = (String, Scope, String);

/// Stands for "no end" in the time columns, so filters are plain comparisons.
const OPEN: i64 = i64::MAX;

/// The fields recall filters on, kept together so a scan stays in cache.
struct Row {
    id: u64,
    valid_from: i64,
    valid_to: i64,
    expires_at: i64,
    recorded_at: i64,
    doc_len: u32,
    trust: Trust,
    kind: MemoryKind,
    live: bool,
    archived: bool,
    /// Not derived from anything, so scope access alone decides readability.
    plain: bool,
}

/// All memories of one (tenant, scope), stored by column. Recall opens only
/// the segments the caller can read, so its cost follows what the caller may
/// see and not the size of the store.
#[derive(Default)]
struct Segment {
    rows: Vec<Row>,
    /// One built-in embedding per row, `HASH_DIM` floats each.
    vecs: Vec<f32>,
    /// Term to (row, term frequency), rows ascending.
    postings: HashMap<String, Vec<(u32, u32)>>,
    with_entities: Vec<u32>,
    with_embedding: Vec<u32>,
    /// Lower-cased entity name to the rows that mention it: the edges of the
    /// entity graph.
    entity_rows: HashMap<String, Vec<u32>>,
    /// Approximate index over `vecs`; node n is row n.
    hash_ann: Hnsw,
    emb: Option<Embeddings>,
}

fn entity_names(m: &Memory) -> BTreeSet<String> {
    m.entities
        .iter()
        .filter(|e| !e.trim().is_empty())
        .map(|e| e.to_lowercase())
        .collect()
}

/// Caller-supplied embeddings for one segment, normalised and stored flat,
/// with an approximate index over them. All must share the first one's
/// dimension; others are still scored exactly.
struct Embeddings {
    dim: usize,
    data: Vec<f32>,
    node_row: Vec<u32>,
    row_node: HashMap<u32, u32>,
    ann: Hnsw,
}

impl Embeddings {
    fn new(dim: usize, params: crate::ann::HnswParams) -> Self {
        Self {
            dim,
            data: Vec::new(),
            node_row: Vec::new(),
            row_node: HashMap::new(),
            ann: Hnsw::new(params),
        }
    }

    fn catch_up(&mut self) -> usize {
        let mut added = 0;
        while self.ann.len() < self.node_row.len() {
            self.ann.insert(&self.data, self.dim);
            added += 1;
        }
        added
    }
}

fn normalised(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm > 0.0 && norm.is_finite()).then(|| v.iter().map(|x| x / norm).collect())
}

impl Segment {
    /// Add any rows written since the last call to the approximate indexes.
    /// Writes stay fast because indexing happens here, on first use or in
    /// `MemoryStore::build_index`, not on the write path.
    fn catch_up(&mut self) -> usize {
        let mut added = 0;
        while self.hash_ann.len() < self.rows.len() {
            self.hash_ann.insert(&self.vecs, HASH_DIM);
            added += 1;
        }
        if let Some(emb) = self.emb.as_mut() {
            added += emb.catch_up();
        }
        added
    }
}

/// One segment as a particular caller sees it for a particular query.
struct View {
    segment: usize,
    mask: Vec<bool>,
    rows: Vec<u32>,
}

fn index_text(m: &Memory) -> String {
    let mut text = m.content.clone();
    for e in &m.entities {
        text.push(' ');
        text.push_str(e);
    }
    text
}

fn term_counts(mut tokens: Vec<String>) -> Vec<(String, u32)> {
    tokens.sort_unstable();
    let mut counts: Vec<(String, u32)> = Vec::new();
    for token in tokens {
        match counts.last_mut() {
            Some((last, n)) if *last == token => *n += 1,
            _ => counts.push((token, 1)),
        }
    }
    counts
}

pub struct MemoryStore {
    cfg: StoreConfig,
    tokenizer: Tokenizer,
    memories: BTreeMap<u64, Memory>,
    by_tenant: HashMap<String, Vec<u64>>,
    segments: Vec<Segment>,
    segment_of: HashMap<(String, Scope), usize>,
    /// Memory id to (segment, row).
    location: HashMap<u64, (usize, u32)>,
    /// Every memory sharing a (tenant, scope, subject), for supersession.
    chains: HashMap<ChainKey, Vec<u64>>,
    grants: Vec<Grant>,
    next_id: u64,
    next_grant_id: u64,
    quarantined: usize,
    recovered_bytes: u64,
    path: Option<PathBuf>,
    log: Option<File>,
}

impl MemoryStore {
    /// An in-memory store with no persistence.
    ///
    /// # Panics
    ///
    /// Panics if `cfg` names a stemming language that does not exist. Call
    /// [`StoreConfig::validate`] first to get that as an error instead.
    #[must_use]
    pub fn in_memory(cfg: StoreConfig) -> Self {
        let tokenizer = Tokenizer::new(
            cfg.tuning.stem_language.as_deref(),
            cfg.tuning.drop_stopwords,
        )
        .expect("the stemming language is checked by StoreConfig::validate");
        Self {
            cfg,
            tokenizer,
            memories: BTreeMap::new(),
            by_tenant: HashMap::new(),
            segments: vec![],
            segment_of: HashMap::new(),
            location: HashMap::new(),
            chains: HashMap::new(),
            grants: vec![],
            next_id: 1,
            next_grant_id: 1,
            quarantined: 0,
            recovered_bytes: 0,
            path: None,
            log: None,
        }
    }

    /// Open a store backed by an append-only log file, replaying it if it
    /// exists. Entries whose signature does not verify are quarantined.
    ///
    /// # Errors
    ///
    /// Fails if the configuration is invalid, or the log cannot be read,
    /// parsed or opened for appending.
    pub fn open(path: impl AsRef<Path>, cfg: StoreConfig) -> Result<Self, Error> {
        cfg.validate()?;
        let path = path.as_ref().to_path_buf();
        let mut store = Self::in_memory(cfg);
        let mut ends_with_newline = true;
        if path.exists() {
            let mut reader = BufReader::new(File::open(&path)?);
            let mut line = Vec::new();
            let (mut offset, mut number) = (0u64, 0usize);
            loop {
                line.clear();
                let read = reader.read_until(b'\n', &mut line)? as u64;
                if read == 0 {
                    break;
                }
                number += 1;
                let complete = line.last() == Some(&b'\n');
                let parsed = std::str::from_utf8(&line)
                    .map_err(|e| e.to_string())
                    .and_then(|text| {
                        if text.trim().is_empty() {
                            Ok(None)
                        } else {
                            serde_json::from_str::<Event>(text)
                                .map(Some)
                                .map_err(|e| e.to_string())
                        }
                    });
                match parsed {
                    Ok(Some(event)) => store.replay(event),
                    Ok(None) => {}
                    // The last line has no newline and does not parse: a write
                    // was cut off part-way. Everything before it is intact.
                    Err(_) if !complete => {
                        store.recovered_bytes = read;
                        drop(reader);
                        OpenOptions::new()
                            .write(true)
                            .open(&path)?
                            .set_len(offset)?;
                        break;
                    }
                    Err(reason) => {
                        return Err(Error::Io(format!(
                            "log is corrupt at line {number}: {reason}"
                        )));
                    }
                }
                ends_with_newline = complete;
                offset += read;
            }
        }
        let mut log = OpenOptions::new().create(true).append(true).open(&path)?;
        if !ends_with_newline {
            // A complete record that lost only its newline: finish the line.
            log.write_all(b"\n")?;
        }
        store.log = Some(log);
        store.path = Some(path);
        Ok(store)
    }

    // ------------------------------------------------------------------
    // Event log
    // ------------------------------------------------------------------

    /// Apply an event read from the log, setting aside any memory whose
    /// signature does not verify.
    fn replay(&mut self, event: Event) {
        match event {
            Event::Batch { events } => events.into_iter().for_each(|e| self.replay(e)),
            Event::Write { memory } => {
                self.next_id = self.next_id.max(memory.id + 1);
                if provenance::verify(&self.cfg.signing_key, &memory) {
                    self.apply(Event::Write { memory });
                } else {
                    self.quarantined += 1;
                }
            }
            other => self.apply(other),
        }
    }

    /// Append one event to the log, then apply it.
    fn emit(&mut self, event: Event) -> Result<(), Error> {
        if let Some(log) = self.log.as_mut() {
            let mut line = serde_json::to_string(&event)?;
            line.push('\n');
            log.write_all(line.as_bytes())?;
            if self.cfg.sync_writes {
                log.sync_data()?;
            }
        }
        self.apply(event);
        Ok(())
    }

    /// Append the events of one operation as a single line, then apply them.
    fn emit_all(&mut self, mut events: Vec<Event>) -> Result<(), Error> {
        match events.len() {
            0 => Ok(()),
            1 => self.emit(events.remove(0)),
            _ => self.emit(Event::Batch { events }),
        }
    }

    /// Ask the operating system to put everything written so far on disk.
    ///
    /// # Errors
    ///
    /// Fails if the operating system reports an error.
    pub fn sync(&mut self) -> Result<(), Error> {
        if let Some(log) = self.log.as_mut() {
            log.sync_data()?;
        }
        Ok(())
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Meta {
                next_id,
                next_grant_id,
            } => {
                self.next_id = self.next_id.max(next_id);
                self.next_grant_id = self.next_grant_id.max(next_grant_id);
            }
            Event::Write { memory } => {
                let m = *memory;
                let id = m.id;
                self.next_id = self.next_id.max(id + 1);
                let text = index_text(&m);
                let tokens = self.tokenizer.tokens(&text);
                let key = (m.tenant.clone(), m.scope.clone());
                let hnsw_params = self.cfg.tuning.hnsw;
                let at = match self.segment_of.get(&key) {
                    Some(at) => *at,
                    None => {
                        self.segments.push(Segment {
                            hash_ann: Hnsw::new(hnsw_params),
                            ..Segment::default()
                        });
                        self.segment_of.insert(key, self.segments.len() - 1);
                        self.segments.len() - 1
                    }
                };
                let segment = &mut self.segments[at];
                let row = segment.rows.len() as u32;
                segment.rows.push(Row {
                    id,
                    valid_from: m.valid_from,
                    valid_to: m.valid_to.unwrap_or(OPEN),
                    expires_at: m.expires_at.unwrap_or(OPEN),
                    recorded_at: m.recorded_at,
                    doc_len: tokens.len() as u32,
                    trust: m.trust,
                    kind: m.kind,
                    live: true,
                    archived: m.archived,
                    plain: m.derived_from.is_empty(),
                });
                segment
                    .vecs
                    .extend_from_slice(&hash_embed(&tokens, self.cfg.tuning.trigram_weight));
                for (term, count) in term_counts(tokens) {
                    segment.postings.entry(term).or_default().push((row, count));
                }
                if !m.entities.is_empty() {
                    segment.with_entities.push(row);
                }
                for name in entity_names(&m) {
                    segment.entity_rows.entry(name).or_default().push(row);
                }
                if let Some(embedding) = &m.embedding {
                    segment.with_embedding.push(row);
                    if let Some(unit) = normalised(embedding) {
                        let emb = segment
                            .emb
                            .get_or_insert_with(|| Embeddings::new(unit.len(), hnsw_params));
                        if emb.dim == unit.len() {
                            emb.row_node.insert(row, emb.node_row.len() as u32);
                            emb.node_row.push(row);
                            emb.data.extend_from_slice(&unit);
                        }
                    }
                }
                self.location.insert(id, (at, row));
                self.by_tenant.entry(m.tenant.clone()).or_default().push(id);
                if let Some(subject) = &m.subject {
                    self.chains
                        .entry((m.tenant.clone(), m.scope.clone(), subject.clone()))
                        .or_default()
                        .push(id);
                }
                self.memories.insert(id, m);
            }
            Event::Supersede { old, new, at } => {
                if let Some(m) = self.memories.get_mut(&old) {
                    m.valid_to = Some(at);
                    m.superseded_by = Some(new);
                }
                if let Some(row) = self.row_mut(old) {
                    row.valid_to = at;
                }
            }
            Event::Grant { grant } => {
                self.next_grant_id = self.next_grant_id.max(grant.id + 1);
                self.grants.push(grant);
            }
            Event::Revoke { grant_id, at } => {
                if let Some(g) = self.grants.iter_mut().find(|g| g.id == grant_id) {
                    g.revoked_at = Some(at);
                }
            }
            Event::Forget { id } => {
                if let Some(m) = self.memories.remove(&id) {
                    // Remove every trace from the indexes, not only the record:
                    // the words it contained and its embedding go as well.
                    if let Some((at, row)) = self.location.remove(&id) {
                        let segment = &mut self.segments[at];
                        segment.rows[row as usize].live = false;
                        for (term, _) in term_counts(self.tokenizer.tokens(&index_text(&m))) {
                            if let Some(list) = segment.postings.get_mut(&term) {
                                if let Ok(i) = list.binary_search_by_key(&row, |entry| entry.0) {
                                    list.remove(i);
                                }
                                if list.is_empty() {
                                    segment.postings.remove(&term);
                                }
                            }
                        }
                        for name in entity_names(&m) {
                            if let Some(list) = segment.entity_rows.get_mut(&name) {
                                list.retain(|r| *r != row);
                                if list.is_empty() {
                                    segment.entity_rows.remove(&name);
                                }
                            }
                        }
                        let start = row as usize * HASH_DIM;
                        segment.vecs[start..start + HASH_DIM].fill(0.0);
                        if let Some(emb) = segment.emb.as_mut() {
                            if let Some(node) = emb.row_node.get(&row) {
                                let start = *node as usize * emb.dim;
                                emb.data[start..start + emb.dim].fill(0.0);
                            }
                        }
                    }
                    if let Some(ids) = self.by_tenant.get_mut(&m.tenant) {
                        ids.retain(|x| *x != id);
                    }
                    if let Some(subject) = m.subject {
                        let key = (m.tenant, m.scope, subject);
                        if let Some(chain) = self.chains.get_mut(&key) {
                            chain.retain(|x| *x != id);
                            if chain.is_empty() {
                                self.chains.remove(&key);
                            }
                        }
                    }
                }
            }
            Event::Corroborate { id, by } => {
                if let Some(m) = self.memories.get_mut(&id) {
                    if !m.corroborated_by.contains(&by) {
                        m.corroborated_by.push(by);
                    }
                }
            }
            Event::Batch { events } => events.into_iter().for_each(|e| self.apply(e)),
            Event::Archive { id } => {
                if let Some(m) = self.memories.get_mut(&id) {
                    m.archived = true;
                }
                if let Some(row) = self.row_mut(id) {
                    row.archived = true;
                }
            }
        }
    }

    fn row_mut(&mut self, id: u64) -> Option<&mut Row> {
        let (at, row) = *self.location.get(&id)?;
        self.segments.get_mut(at)?.rows.get_mut(row as usize)
    }

    /// Rewrite the log from current state, so that forgotten memories are
    /// gone from disk and not only from the indexes.
    fn compact(&mut self) -> Result<(), Error> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        let tmp = path.with_extension("compacting");
        {
            let mut out = BufWriter::new(File::create(&tmp)?);
            let mut put = |event: &Event| -> Result<(), Error> {
                serde_json::to_writer(&mut out, event)?;
                out.write_all(b"\n")?;
                Ok(())
            };
            put(&Event::Meta {
                next_id: self.next_id,
                next_grant_id: self.next_grant_id,
            })?;
            for g in &self.grants {
                put(&Event::Grant { grant: g.clone() })?;
            }
            for m in self.memories.values() {
                put(&Event::Write {
                    memory: Box::new(m.clone()),
                })?;
            }
            out.flush()?;
            out.get_ref().sync_all()?;
        }
        self.log = None; // Windows cannot replace a file that is still open.
        std::fs::rename(&tmp, &path)?;
        self.log = Some(OpenOptions::new().append(true).open(&path)?);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Access control
    // ------------------------------------------------------------------

    fn policy(&self, who: &Identity) -> Option<&RolePolicy> {
        self.cfg.policies.get(&who.role)
    }

    fn grant_covers(
        &self,
        tenant: &str,
        owner: &str,
        reader: &str,
        tags: &[String],
        now: i64,
    ) -> bool {
        self.grants.iter().any(|g| {
            g.tenant == tenant
                && g.grantor == owner
                && g.grantee == reader
                && g.active(now)
                && g.tag.as_ref().is_none_or(|t| tags.contains(t))
        })
    }

    fn can_read(&self, who: &Identity, m: &Memory, now: i64) -> bool {
        if who.tenant != m.tenant {
            return false;
        }
        let Some(policy) = self.policy(who) else {
            return false;
        };
        if !policy.read.contains(&m.scope.kind()) {
            return false;
        }
        match &m.scope {
            Scope::AgentPrivate(agent) => *agent == who.agent,
            Scope::Role(role) => *role == who.role,
            Scope::Team(team) => who.team.as_ref() == Some(team),
            Scope::UserPrivate(user) => {
                *user == who.user || self.grant_covers(&m.tenant, user, &who.user, &m.tags, now)
            }
            Scope::Tenant => true,
        }
    }

    /// Readable only if every memory it was derived from is readable too.
    /// This is what makes a revoked grant also hide the facts built on it.
    fn can_read_deep(
        &self,
        who: &Identity,
        id: u64,
        now: i64,
        memo: &mut HashMap<u64, bool>,
    ) -> bool {
        if let Some(known) = memo.get(&id) {
            return *known;
        }
        let ok = match self.memories.get(&id) {
            None => false,
            Some(m) => {
                self.can_read(who, m, now)
                    && m.derived_from
                        .iter()
                        .all(|src| self.can_read_deep(who, *src, now, memo))
            }
        };
        memo.insert(id, ok);
        ok
    }

    /// The scopes a caller could possibly read: its own agent, role, team and
    /// user scopes, the tenant scope, and the user scope of anyone who has an
    /// active grant to this user. `can_read_deep` still decides each memory.
    fn readable_scopes(&self, who: &Identity, now: i64) -> Vec<Scope> {
        let mut scopes = vec![
            Scope::AgentPrivate(who.agent.clone()),
            Scope::Role(who.role.clone()),
            Scope::UserPrivate(who.user.clone()),
            Scope::Tenant,
        ];
        if let Some(team) = &who.team {
            scopes.push(Scope::Team(team.clone()));
        }
        for g in &self.grants {
            if g.tenant == who.tenant && g.grantee == who.user && g.active(now) {
                let shared = Scope::UserPrivate(g.grantor.clone());
                if !scopes.contains(&shared) {
                    scopes.push(shared);
                }
            }
        }
        scopes
    }

    /// How a caller may read a scope: `Some(true)` for all of it,
    /// `Some(false)` when each memory must be checked (a tagged grant), and
    /// `None` for no access.
    fn scope_access(&self, who: &Identity, scope: &Scope, now: i64) -> Option<bool> {
        let policy = self.policy(who)?;
        if !policy.read.contains(&scope.kind()) {
            return None;
        }
        match scope {
            Scope::AgentPrivate(agent) => (*agent == who.agent).then_some(true),
            Scope::Role(role) => (*role == who.role).then_some(true),
            Scope::Team(team) => (who.team.as_ref() == Some(team)).then_some(true),
            Scope::Tenant => Some(true),
            Scope::UserPrivate(user) if *user == who.user => Some(true),
            Scope::UserPrivate(user) => {
                let mut tagged = false;
                for g in &self.grants {
                    if g.tenant == who.tenant
                        && g.grantor == *user
                        && g.grantee == who.user
                        && g.active(now)
                    {
                        if g.tag.is_none() {
                            return Some(true);
                        }
                        tagged = true;
                    }
                }
                tagged.then_some(false)
            }
        }
    }

    fn can_write(&self, who: &Identity, scope: &Scope) -> bool {
        let Some(policy) = self.policy(who) else {
            return false;
        };
        if !policy.write.contains(&scope.kind()) {
            return false;
        }
        match scope {
            Scope::AgentPrivate(agent) => *agent == who.agent,
            Scope::Role(role) => *role == who.role,
            Scope::Team(team) => who.team.as_ref() == Some(team),
            Scope::UserPrivate(user) => *user == who.user,
            Scope::Tenant => true,
        }
    }

    // ------------------------------------------------------------------
    // Write path
    // ------------------------------------------------------------------

    /// Append a memory. No model is called here: the write path only checks
    /// permission, signs the record, appends it and indexes it.
    pub fn write(&mut self, who: &Identity, new: NewMemory, now: i64) -> Result<u64, Error> {
        if new.content.trim().is_empty() {
            return Err(Error::Invalid("content is empty".into()));
        }
        if !self.can_write(who, &new.scope) {
            return Err(Error::Denied(format!(
                "role '{}' may not write to {:?}",
                who.role, new.scope
            )));
        }
        // An agent's inference can never outrank what it was told or observed,
        // and a derived memory is never more trusted than its sources.
        let mut trust = new.trust;
        if new.assertion == Assertion::Inferred {
            trust = trust.min(Trust::Standard);
        }
        let mut memo = HashMap::new();
        for src in &new.derived_from {
            let source = self.memories.get(src).ok_or(Error::NotFound(*src))?;
            if !self.can_read_deep(who, *src, now, &mut memo) {
                return Err(Error::Denied(format!("cannot derive from memory {src}")));
            }
            trust = trust.min(source.trust);
        }

        let id = self.next_id;
        let valid_from = new.valid_from.unwrap_or(now);
        let mut memory = Memory {
            id,
            tenant: who.tenant.clone(),
            scope: new.scope,
            kind: new.kind,
            content: new.content,
            subject: new.subject,
            entities: new.entities,
            tags: new.tags,
            embedding: new.embedding,
            author_user: who.user.clone(),
            author_agent: who.agent.clone(),
            author_role: who.role.clone(),
            assertion: new.assertion,
            source: new.source,
            trust,
            derived_from: new.derived_from,
            corroborated_by: vec![],
            valid_from,
            valid_to: None,
            recorded_at: now,
            superseded_by: None,
            expires_at: new.ttl_ms.map(|ttl| now + ttl),
            salience: new.salience.unwrap_or(1.0).clamp(0.0, 1.0),
            access_count: 0,
            last_access: now,
            archived: false,
            signature: String::new(),
        };
        memory.signature = provenance::sign(&self.cfg.signing_key, &memory);

        // Find this fact's neighbours in valid time before it joins the chain.
        // Two agents writing the same subject land in one ordered chain, and a
        // fact that arrives late is slotted into the past where it belongs.
        // Each is (valid_from, id) of the nearest version on that side.
        let mut before: Option<(i64, u64)> = None;
        let mut after: Option<(i64, u64)> = None;
        if let Some(subject) = &memory.subject {
            let key = (memory.tenant.clone(), memory.scope.clone(), subject.clone());
            for other in self.chains.get(&key).into_iter().flatten() {
                let Some(o) = self.memories.get(other) else {
                    continue;
                };
                let entry = (o.valid_from, o.id);
                if o.valid_from <= valid_from {
                    if before.is_none_or(|b| entry > b) {
                        before = Some(entry);
                    }
                } else if after.is_none_or(|a| entry < a) {
                    after = Some(entry);
                }
            }
        }

        let mut events = vec![Event::Write {
            memory: Box::new(memory),
        }];
        if let Some((_, prev)) = before {
            events.push(Event::Supersede {
                old: prev,
                new: id,
                at: valid_from,
            });
        }
        if let Some((next_from, next)) = after {
            events.push(Event::Supersede {
                old: id,
                new: next,
                at: next_from,
            });
        }
        self.emit_all(events)?;
        Ok(id)
    }

    // ------------------------------------------------------------------
    // Read path
    // ------------------------------------------------------------------

    fn decay(&self, m: &Memory, now: i64) -> f32 {
        let idle = (now - m.last_access).max(0) as f32;
        0.5f32.powf(idle / self.cfg.half_life_ms.max(1) as f32)
    }

    /// The rows of each readable segment that pass every filter for this
    /// caller and query, with their count and total length in tokens.
    fn eligible(&self, who: &Identity, q: &Query, now: i64) -> (Vec<View>, usize, u64) {
        let at = q.as_of.unwrap_or(now);
        let mut memo = HashMap::new();
        let (mut views, mut count, mut length) = (Vec::new(), 0usize, 0u64);
        for scope in self.readable_scopes(who, now) {
            if !q.scopes.is_empty() && !q.scopes.contains(&scope.kind()) {
                continue;
            }
            let Some(whole_scope) = self.scope_access(who, &scope, now) else {
                continue;
            };
            let Some(&segment) = self.segment_of.get(&(who.tenant.clone(), scope)) else {
                continue;
            };
            let seg = &self.segments[segment];
            let mut mask = vec![false; seg.rows.len()];
            let mut rows = Vec::new();
            for (i, r) in seg.rows.iter().enumerate() {
                if !r.live
                    || (r.archived && !q.include_archived)
                    || r.expires_at <= now
                    || r.valid_from > at
                    || r.valid_to <= at
                    || r.trust < q.min_trust
                    || (!q.kinds.is_empty() && !q.kinds.contains(&r.kind))
                {
                    continue;
                }
                // Derived memories and tagged grants need the full check.
                if !(whole_scope && r.plain) && !self.can_read_deep(who, r.id, now, &mut memo) {
                    continue;
                }
                mask[i] = true;
                rows.push(i as u32);
                length += u64::from(r.doc_len);
            }
            count += rows.len();
            if !rows.is_empty() {
                views.push(View {
                    segment,
                    mask,
                    rows,
                });
            }
        }
        (views, count, length)
    }

    /// Okapi BM25 through the posting lists. Document counts and lengths come
    /// from the caller's eligible rows only, so scores reveal nothing about
    /// memories the caller cannot read.
    fn rank_keyword(
        &self,
        views: &[View],
        query: &[String],
        count: usize,
        length: u64,
        pool: usize,
    ) -> Vec<u64> {
        let (k1, b) = (self.cfg.tuning.bm25_k1, self.cfg.tuning.bm25_b);
        let mut terms: Vec<&String> = query.iter().collect();
        terms.sort();
        terms.dedup();
        let n = count as f32;
        let avg_len = (length as f32 / n).max(1.0);
        let mut scores: Vec<Vec<f32>> = views.iter().map(|v| vec![0.0; v.mask.len()]).collect();
        let mut touched: Vec<(usize, u32)> = Vec::new();
        for term in terms {
            let lists: Vec<(usize, &Vec<(u32, u32)>)> = views
                .iter()
                .enumerate()
                .filter_map(|(v, view)| {
                    Some((v, self.segments[view.segment].postings.get(term.as_str())?))
                })
                .collect();
            let df: usize = lists
                .iter()
                .map(|(v, list)| {
                    list.iter()
                        .filter(|entry| views[*v].mask[entry.0 as usize])
                        .count()
                })
                .sum();
            if df == 0 {
                continue;
            }
            let idf = (1.0 + (n - df as f32 + 0.5) / (df as f32 + 0.5)).ln();
            for (v, list) in lists {
                let rows = &self.segments[views[v].segment].rows;
                for (row, tf) in list {
                    let r = *row as usize;
                    if !views[v].mask[r] {
                        continue;
                    }
                    let tf = *tf as f32;
                    let len_norm = k1 * (1.0 - b + b * rows[r].doc_len as f32 / avg_len);
                    if scores[v][r] == 0.0 {
                        touched.push((v, *row));
                    }
                    scores[v][r] += idf * (tf * (k1 + 1.0)) / (tf + len_norm);
                }
            }
        }
        let scored = touched
            .into_iter()
            .map(|(v, row)| {
                (
                    self.segments[views[v].segment].rows[row as usize].id,
                    scores[v][row as usize],
                )
            })
            .collect();
        top_ids(scored, pool)
    }

    fn established(&self, m: &Memory) -> bool {
        m.established_with(self.cfg.required_corroborations)
    }

    /// Whether vector scoring for this view should use the approximate index.
    fn use_ann(&self, view: &View, q: &Query) -> bool {
        let total = self.segments[view.segment].rows.len().max(1);
        let tuning = &self.cfg.tuning;
        !q.exact
            && view.rows.len() >= tuning.ann_min_rows
            && view.rows.len() as f32 / total as f32 >= tuning.ann_min_share
    }

    /// Candidate list size for an approximate search: wider when the caller
    /// can read only part of the segment, since skipped rows still take room.
    fn ef_for(&self, view: &View, pool: usize) -> usize {
        let share = view.rows.len() as f32 / self.segments[view.segment].rows.len().max(1) as f32;
        let tuning = &self.cfg.tuning;
        ((pool.max(tuning.ann_min_ef) as f32 / share).ceil() as usize)
            .clamp(pool, tuning.ann_max_ef.max(pool))
    }

    fn rank_vector(&self, views: &[View], q: &Query, pool: usize) -> Vec<u64> {
        let mut scored: Vec<(u64, f32)> = Vec::new();
        match &q.embedding {
            // Caller-supplied embeddings.
            Some(query) => {
                let unit = normalised(query);
                for view in views {
                    let seg = &self.segments[view.segment];
                    let indexed = seg
                        .emb
                        .as_ref()
                        .filter(|e| unit.as_ref().is_some_and(|u| u.len() == e.dim));
                    if let (Some(emb), Some(unit), true) =
                        (indexed, unit.as_ref(), self.use_ann(view, q))
                    {
                        let accept = |node: u32| view.mask[emb.node_row[node as usize] as usize];
                        let ef = self.ef_for(view, pool);
                        for (node, sim) in
                            emb.ann.search(unit, &emb.data, emb.dim, pool, ef, &accept)
                        {
                            scored.push((seg.rows[emb.node_row[node as usize] as usize].id, sim));
                        }
                        continue;
                    }
                    for row in seg
                        .with_embedding
                        .iter()
                        .filter(|r| view.mask[**r as usize])
                    {
                        let id = seg.rows[*row as usize].id;
                        if let Some(e) = self.memories.get(&id).and_then(|m| m.embedding.as_ref()) {
                            scored.push((id, cosine(query, e)));
                        }
                    }
                }
            }
            // Built-in embeddings are unit length, so the dot product is the cosine.
            None => {
                let query = hash_embed(
                    &self.tokenizer.tokens(&q.text),
                    self.cfg.tuning.trigram_weight,
                );
                for view in views {
                    let seg = &self.segments[view.segment];
                    if self.use_ann(view, q) {
                        let accept = |row: u32| view.mask[row as usize];
                        let ef = self.ef_for(view, pool);
                        for (row, sim) in seg
                            .hash_ann
                            .search(&query, &seg.vecs, HASH_DIM, pool, ef, &accept)
                        {
                            scored.push((seg.rows[row as usize].id, sim));
                        }
                        continue;
                    }
                    for row in &view.rows {
                        let start = *row as usize * HASH_DIM;
                        scored.push((
                            seg.rows[*row as usize].id,
                            dot(&query, &seg.vecs[start..start + HASH_DIM]),
                        ));
                    }
                }
            }
        }
        let floor = self.cfg.tuning.min_vector_similarity;
        scored.retain(|entry| entry.1 > floor);
        top_ids(scored, pool)
    }

    fn rank_entity(&self, views: &[View], text: &str, pool: usize) -> Vec<u64> {
        let lowered = text.to_lowercase();
        let mut scored: Vec<(u64, f32)> = Vec::new();
        for view in views {
            let seg = &self.segments[view.segment];
            for row in seg.with_entities.iter().filter(|r| view.mask[**r as usize]) {
                let id = seg.rows[*row as usize].id;
                let Some(m) = self.memories.get(&id) else {
                    continue;
                };
                let named = m
                    .entities
                    .iter()
                    .filter(|e| !e.is_empty() && lowered.contains(&e.to_lowercase()))
                    .count();
                if named > 0 {
                    scored.push((id, named as f32));
                }
            }
        }
        top_ids(scored, pool)
    }

    /// Graph expansion by spreading activation. Each starting memory passes a
    /// share of its score along every entity it mentions to the other
    /// memories that mention it; a further hop passes on a share of that.
    /// An entity mentioned by many memories passes on less to each, so a
    /// common name does not flood the results. Only eligible rows are ever
    /// visited, and a memory never earns credit from its own entities.
    fn spread(&self, views: &[View], seeds: &[(u64, f32)], hops: u8) -> Vec<(u64, f32)> {
        let carry = self.cfg.tuning.graph_carry;
        let mut frontier: Vec<(u64, f32)> = seeds.to_vec();
        let mut reached: HashMap<u64, f32> = HashMap::new();
        for _ in 0..hops {
            // Entity name to the frontier memories that mention it.
            let mut mentions: HashMap<String, Vec<(u64, f32)>> = HashMap::new();
            for (id, activation) in &frontier {
                for name in self.memories.get(id).map(entity_names).unwrap_or_default() {
                    mentions.entry(name).or_default().push((*id, *activation));
                }
            }
            let mut found: HashMap<u64, f32> = HashMap::new();
            for (name, sources) in &mentions {
                let mut mentioned_by: Vec<u64> = Vec::new();
                for view in views {
                    let seg = &self.segments[view.segment];
                    for row in seg.entity_rows.get(name).into_iter().flatten() {
                        if view.mask[*row as usize] {
                            mentioned_by.push(seg.rows[*row as usize].id);
                        }
                    }
                }
                let spread_over = (mentioned_by.len().saturating_sub(1).max(1) as f32).sqrt();
                for id in mentioned_by {
                    if reached.contains_key(&id) {
                        continue;
                    }
                    let strongest = sources
                        .iter()
                        .filter(|(source, _)| *source != id)
                        .map(|(_, activation)| *activation)
                        .fold(0.0, f32::max);
                    let passed_on = carry * strongest / spread_over;
                    if passed_on > 0.0 {
                        *found.entry(id).or_insert(0.0) += passed_on;
                    }
                }
            }
            if found.is_empty() {
                break;
            }
            frontier = found
                .iter()
                .map(|(id, activation)| (*id, *activation))
                .collect();
            sort_desc(&mut frontier);
            frontier.truncate(self.cfg.tuning.graph_frontier);
            reached.extend(found);
        }
        let mut out: Vec<(u64, f32)> = reached.into_iter().collect();
        sort_desc(&mut out);
        out
    }

    /// Recall memories for a caller. Filters run before ranking, so a memory
    /// the caller may not read never reaches a ranker.
    pub fn recall(&mut self, who: &Identity, q: &Query, now: i64) -> Vec<Hit> {
        if q.k == 0 {
            return vec![];
        }
        let (views, count, length) = self.eligible(who, q, now);
        if count == 0 {
            return vec![];
        }
        let tuning = &self.cfg.tuning;
        let pool = (q.k * tuning.pool_factor).max(tuning.min_pool);
        // Bring the approximate index up to date for the segments that will use it.
        for view in &views {
            if self.use_ann(view, q) {
                self.segments[view.segment].catch_up();
            }
        }

        let mut rankings: [Vec<u64>; 4] = Default::default();
        let mut scored: Vec<(u64, f32)> = if q.text.trim().is_empty() && q.embedding.is_none() {
            // No query: newest first. Used to read a blackboard or a handoff.
            let mut newest: Vec<(i64, u64)> = Vec::with_capacity(count);
            for view in &views {
                let rows = &self.segments[view.segment].rows;
                newest.extend(
                    view.rows
                        .iter()
                        .map(|r| (rows[*r as usize].recorded_at, rows[*r as usize].id)),
                );
            }
            newest.sort_unstable_by(|a, b| b.cmp(a));
            newest.truncate(pool);
            newest
                .into_iter()
                .enumerate()
                .map(|(rank, (_, id))| (id, 1.0 / (rank as f32 + 1.0)))
                .collect()
        } else {
            let keyword =
                self.rank_keyword(&views, &self.tokenizer.tokens(&q.text), count, length, pool);
            let vector = self.rank_vector(&views, q, pool);
            let entity = self.rank_entity(&views, &q.text, pool);
            let mut fused = rrf(
                &[keyword.clone(), vector.clone(), entity.clone()],
                self.cfg.tuning.rrf_k,
            );

            // The best few matches from each ranker are where the graph starts.
            let mut seeds: Vec<(u64, f32)> = Vec::new();
            for id in [&keyword, &vector, &entity]
                .into_iter()
                .flat_map(|list| list.iter().take(self.cfg.tuning.graph_seeds_per_ranker))
            {
                if !seeds.iter().any(|(seen, _)| seen == id) {
                    seeds.push((*id, fused[id]));
                }
            }
            let reached = if q.graph_hops == 0 {
                vec![]
            } else {
                self.spread(&views, &seeds, q.graph_hops)
            };
            for (id, activation) in &reached {
                *fused.entry(*id).or_insert(0.0) += activation;
            }
            rankings = [
                keyword,
                vector,
                entity,
                reached.into_iter().map(|r| r.0).collect(),
            ];

            // Relevance decides the order. Rank fusion packs scores close
            // together: the best match in two rankers leads the next by under
            // two percent. So confirmation, trust and salience are kept to
            // about one percent each, enough to settle near-ties and never
            // enough to put a recently used memory above a clearly better
            // match. Untrusted content, when asked for, is set clearly lower.
            fused
                .into_iter()
                .map(|(id, relevance)| {
                    let m = &self.memories[&id];
                    let tuning = &self.cfg.tuning;
                    let trust = match m.trust {
                        Trust::Trusted => tuning.weight_trusted,
                        Trust::Standard => tuning.weight_standard,
                        Trust::Untrusted => tuning.weight_untrusted,
                    };
                    let established = if self.established(m) {
                        1.0
                    } else {
                        tuning.weight_unconfirmed
                    };
                    let salience = (1.0 - tuning.salience_weight)
                        + tuning.salience_weight * m.salience * self.decay(m, now);
                    (id, relevance * trust * established * salience)
                })
                .collect()
        };
        sort_desc(&mut scored);

        // Pack the best memories into the token budget.
        let mut used = 0usize;
        let mut hits = Vec::new();
        for (id, score) in scored {
            if hits.len() >= q.k {
                break;
            }
            let m = &self.memories[&id];
            let cost = m.content.len() / self.cfg.tuning.chars_per_token + 1;
            if q.token_budget.is_some_and(|budget| used + cost > budget) {
                continue;
            }
            used += cost;
            let mut memory = m.clone();
            memory.embedding = None;
            let rank_in = |list: &Vec<u64>| list.iter().position(|x| *x == id).map(|at| at + 1);
            let why = Why {
                keyword_rank: rank_in(&rankings[0]),
                vector_rank: rank_in(&rankings[1]),
                entity_rank: rank_in(&rankings[2]),
                graph_rank: rank_in(&rankings[3]),
                shared_by: match &memory.scope {
                    Scope::UserPrivate(owner) if *owner != who.user => Some(owner.clone()),
                    _ => None,
                },
            };
            hits.push(Hit {
                established: self.established(&memory),
                memory,
                score,
                why,
            });
        }
        // Being recalled is what keeps a memory from decaying.
        for hit in &hits {
            if let Some(m) = self.memories.get_mut(&hit.memory.id) {
                m.access_count = m.access_count.saturating_add(1);
                m.last_access = now;
            }
        }
        hits
    }

    /// Fetch one memory, if the caller may read it.
    pub fn get(&self, who: &Identity, id: u64, now: i64) -> Result<Memory, Error> {
        let mut memo = HashMap::new();
        if !self.can_read_deep(who, id, now, &mut memo) {
            // Same answer whether it is missing or hidden.
            return Err(Error::NotFound(id));
        }
        Ok(self.memories[&id].clone())
    }

    /// Every version of a fact the caller may read, oldest first.
    pub fn history(&self, who: &Identity, scope: &Scope, subject: &str, now: i64) -> Vec<Memory> {
        let key = (who.tenant.clone(), scope.clone(), subject.to_string());
        let mut memo = HashMap::new();
        let mut versions: Vec<Memory> = self
            .chains
            .get(&key)
            .into_iter()
            .flatten()
            .filter(|id| self.can_read_deep(who, **id, now, &mut memo))
            .map(|id| self.memories[id].clone())
            .collect();
        versions.sort_by_key(|m| (m.valid_from, m.id));
        versions
    }

    // ------------------------------------------------------------------
    // Sharing
    // ------------------------------------------------------------------

    /// Let another user read the caller's user-private memories, optionally
    /// only those carrying `tag`, optionally until `expires_at`.
    pub fn grant(
        &mut self,
        who: &Identity,
        grantee: &str,
        tag: Option<String>,
        expires_at: Option<i64>,
    ) -> Result<u64, Error> {
        let allowed = self
            .policy(who)
            .is_some_and(|p| p.read.contains(&ScopeKind::UserPrivate));
        if !allowed {
            return Err(Error::Denied(format!(
                "role '{}' may not share user memory",
                who.role
            )));
        }
        if grantee == who.user {
            return Err(Error::Invalid("cannot grant to yourself".into()));
        }
        let id = self.next_grant_id;
        self.emit(Event::Grant {
            grant: Grant {
                id,
                tenant: who.tenant.clone(),
                grantor: who.user.clone(),
                grantee: grantee.to_string(),
                tag,
                expires_at,
                revoked_at: None,
            },
        })?;
        Ok(id)
    }

    /// Revoke a grant. Only the user who made it can.
    pub fn revoke(&mut self, who: &Identity, grant_id: u64, now: i64) -> Result<(), Error> {
        let owns = self
            .grants
            .iter()
            .any(|g| g.id == grant_id && g.tenant == who.tenant && g.grantor == who.user);
        if !owns {
            return Err(Error::Denied(format!(
                "grant {grant_id} is not yours to revoke"
            )));
        }
        self.emit(Event::Revoke { grant_id, at: now })
    }

    // ------------------------------------------------------------------
    // Corroboration, forgetting, maintenance
    // ------------------------------------------------------------------

    /// Vouch for a memory. A promoting role approves it outright; any other
    /// agent adds one corroboration, and the author cannot vouch for itself.
    pub fn corroborate(&mut self, who: &Identity, id: u64, now: i64) -> Result<(), Error> {
        let memory = self.get(who, id, now)?;
        let by = if self.policy(who).is_some_and(|p| p.can_promote) {
            format!("{APPROVED_PREFIX}{}", who.agent)
        } else if who.agent == memory.author_agent {
            return Err(Error::Invalid(
                "an agent cannot corroborate its own memory".into(),
            ));
        } else {
            who.agent.clone()
        };
        self.emit(Event::Corroborate { id, by })
    }

    /// Hard-delete a memory and everything derived from it, then rewrite the
    /// log so the content is gone from disk. Returns how many were removed.
    pub fn forget(&mut self, who: &Identity, id: u64, now: i64) -> Result<usize, Error> {
        let memory = self.get(who, id, now)?;
        let may = memory.author_user == who.user || self.policy(who).is_some_and(|p| p.can_promote);
        if !may {
            return Err(Error::Denied(format!("memory {id} is not yours to forget")));
        }
        // Ids only grow and a memory can only derive from existing ones, so
        // one ascending pass finds every descendant.
        let mut doomed = BTreeSet::from([id]);
        for other in self.by_tenant.get(&who.tenant).into_iter().flatten() {
            if let Some(m) = self.memories.get(other) {
                if m.derived_from.iter().any(|src| doomed.contains(src)) {
                    doomed.insert(*other);
                }
            }
        }
        self.emit_all(
            doomed
                .iter()
                .map(|gone| Event::Forget { id: *gone })
                .collect(),
        )?;
        self.compact()?;
        Ok(doomed.len())
    }

    /// Drop expired memories and archive episodes whose salience has decayed.
    pub fn sweep(&mut self, now: i64) -> Result<SweepReport, Error> {
        let mut expired = vec![];
        let mut fading = vec![];
        for m in self.memories.values() {
            if m.expires_at.is_some_and(|e| e <= now) {
                expired.push(m.id);
            } else if m.kind == MemoryKind::Episodic
                && !m.archived
                && m.salience * self.decay(m, now) < self.cfg.archive_below
            {
                fading.push(m.id);
            }
        }
        let mut events: Vec<Event> = expired.iter().map(|id| Event::Forget { id: *id }).collect();
        events.extend(fading.iter().map(|id| Event::Archive { id: *id }));
        self.emit_all(events)?;
        Ok(SweepReport {
            expired: expired.len(),
            archived: fading.len(),
        })
    }

    /// Add every memory not yet in the approximate vector indexes and return
    /// how many were added. Recall does this on its own when it needs to; call
    /// it after a bulk load so the first queries do not pay for it.
    ///
    /// Segments are independent, so they are indexed in parallel, one thread
    /// per core.
    pub fn build_index(&mut self) -> usize {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        // Largest segments first, dealt out in turn, so threads finish together.
        let mut work: Vec<&mut Segment> = self.segments.iter_mut().collect();
        work.sort_by_key(|seg| std::cmp::Reverse(seg.rows.len()));
        let mut shares: Vec<Vec<&mut Segment>> = (0..threads).map(|_| Vec::new()).collect();
        for (i, seg) in work.into_iter().enumerate() {
            shares[i % threads].push(seg);
        }
        std::thread::scope(|scope| {
            let handles: Vec<_> = shares
                .into_iter()
                .map(|share| {
                    scope.spawn(move || share.into_iter().map(Segment::catch_up).sum::<usize>())
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
        })
    }

    pub fn stats(&self) -> Stats {
        Stats {
            memories: self.memories.len(),
            tenants: self
                .by_tenant
                .values()
                .filter(|ids| !ids.is_empty())
                .count(),
            grants: self
                .grants
                .iter()
                .filter(|g| g.revoked_at.is_none())
                .count(),
            quarantined: self.quarantined,
            recovered_bytes: self.recovered_bytes,
        }
    }
}
