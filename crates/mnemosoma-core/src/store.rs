//! The memory store: an append-only event log, the indexes rebuilt from it,
//! and the read and write paths.

use crate::identity::{default_roles, Grant, Identity, RolePolicy, Scope, ScopeKind};
use crate::index::{bm25, cosine, hash_embed, rrf, sort_desc, tokenize};
use crate::memory::{Assertion, Memory, MemoryKind, NewMemory, Trust};
use crate::provenance;
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

#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Key for write-time HMAC signatures. Keep it outside the log file.
    pub signing_key: Vec<u8>,
    /// Salience halves after this long without the memory being recalled.
    pub half_life_ms: i64,
    /// Episodes whose decayed salience falls below this are archived by `sweep`.
    pub archive_below: f32,
    pub policies: HashMap<String, RolePolicy>,
}

impl StoreConfig {
    pub fn new(signing_key: impl Into<Vec<u8>>) -> Self {
        Self {
            signing_key: signing_key.into(),
            half_life_ms: 30 * 24 * 60 * 60 * 1000,
            archive_below: 0.05,
            policies: default_roles(),
        }
    }
}

fn default_k() -> usize {
    8
}
fn default_min_trust() -> Trust {
    Trust::Standard
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
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub memory: Memory,
    pub score: f32,
    /// False for an agent's inference that nobody has corroborated yet.
    pub established: bool,
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
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Event {
    Meta { next_id: u64, next_grant_id: u64 },
    Write { memory: Box<Memory> },
    Supersede { old: u64, new: u64, at: i64 },
    Grant { grant: Grant },
    Revoke { grant_id: u64, at: i64 },
    Forget { id: u64 },
    Corroborate { id: u64, by: String },
    Archive { id: u64 },
}

type ChainKey = (String, Scope, String);

pub struct MemoryStore {
    cfg: StoreConfig,
    memories: BTreeMap<u64, Memory>,
    tokens: HashMap<u64, Vec<String>>,
    hash_vecs: HashMap<u64, Vec<f32>>,
    by_tenant: HashMap<String, Vec<u64>>,
    /// Memories grouped by (tenant, scope). Recall walks only the groups the
    /// caller can read, so its cost follows what the caller may see and not
    /// the size of the tenant.
    by_scope: HashMap<(String, Scope), Vec<u64>>,
    /// Every memory sharing a (tenant, scope, subject), for supersession.
    chains: HashMap<ChainKey, Vec<u64>>,
    grants: Vec<Grant>,
    next_id: u64,
    next_grant_id: u64,
    quarantined: usize,
    path: Option<PathBuf>,
    log: Option<File>,
}

impl MemoryStore {
    /// An in-memory store with no persistence.
    pub fn in_memory(cfg: StoreConfig) -> Self {
        Self {
            cfg,
            memories: BTreeMap::new(),
            tokens: HashMap::new(),
            hash_vecs: HashMap::new(),
            by_tenant: HashMap::new(),
            by_scope: HashMap::new(),
            chains: HashMap::new(),
            grants: vec![],
            next_id: 1,
            next_grant_id: 1,
            quarantined: 0,
            path: None,
            log: None,
        }
    }

    /// Open a store backed by an append-only log file, replaying it if it
    /// exists. Entries whose signature does not verify are quarantined.
    pub fn open(path: impl AsRef<Path>, cfg: StoreConfig) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let mut store = Self::in_memory(cfg);
        if path.exists() {
            for line in BufReader::new(File::open(&path)?).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let event: Event = serde_json::from_str(&line)?;
                if let Event::Write { memory } = &event {
                    store.next_id = store.next_id.max(memory.id + 1);
                    if !provenance::verify(&store.cfg.signing_key, memory) {
                        store.quarantined += 1;
                        continue;
                    }
                }
                store.apply(event);
            }
        }
        store.log = Some(OpenOptions::new().create(true).append(true).open(&path)?);
        store.path = Some(path);
        Ok(store)
    }

    // ------------------------------------------------------------------
    // Event log
    // ------------------------------------------------------------------

    fn emit(&mut self, event: Event) -> Result<(), Error> {
        if let Some(log) = self.log.as_mut() {
            let mut line = serde_json::to_string(&event)?;
            line.push('\n');
            log.write_all(line.as_bytes())?;
        }
        self.apply(event);
        Ok(())
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::Meta { next_id, next_grant_id } => {
                self.next_id = self.next_id.max(next_id);
                self.next_grant_id = self.next_grant_id.max(next_grant_id);
            }
            Event::Write { memory } => {
                let m = *memory;
                let id = m.id;
                self.next_id = self.next_id.max(id + 1);
                let mut text = m.content.clone();
                for e in &m.entities {
                    text.push(' ');
                    text.push_str(e);
                }
                self.tokens.insert(id, tokenize(&text));
                self.hash_vecs.insert(id, hash_embed(&text));
                self.by_tenant.entry(m.tenant.clone()).or_default().push(id);
                self.by_scope.entry((m.tenant.clone(), m.scope.clone())).or_default().push(id);
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
                    self.tokens.remove(&id);
                    self.hash_vecs.remove(&id);
                    if let Some(ids) = self.by_tenant.get_mut(&m.tenant) {
                        ids.retain(|x| *x != id);
                    }
                    let group = (m.tenant.clone(), m.scope.clone());
                    if let Some(ids) = self.by_scope.get_mut(&group) {
                        ids.retain(|x| *x != id);
                        if ids.is_empty() {
                            self.by_scope.remove(&group);
                        }
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
            Event::Archive { id } => {
                if let Some(m) = self.memories.get_mut(&id) {
                    m.archived = true;
                }
            }
        }
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
            put(&Event::Meta { next_id: self.next_id, next_grant_id: self.next_grant_id })?;
            for g in &self.grants {
                put(&Event::Grant { grant: g.clone() })?;
            }
            for m in self.memories.values() {
                put(&Event::Write { memory: Box::new(m.clone()) })?;
            }
            out.flush()?;
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

    fn grant_covers(&self, tenant: &str, owner: &str, reader: &str, tags: &[String], now: i64) -> bool {
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
    fn can_read_deep(&self, who: &Identity, id: u64, now: i64, memo: &mut HashMap<u64, bool>) -> bool {
        if let Some(known) = memo.get(&id) {
            return *known;
        }
        let ok = match self.memories.get(&id) {
            None => false,
            Some(m) => {
                self.can_read(who, m, now)
                    && m.derived_from.iter().all(|src| self.can_read_deep(who, *src, now, memo))
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
        let (mut before, mut after): (Option<(i64, u64)>, Option<(i64, u64)>) = (None, None);
        if let Some(subject) = &memory.subject {
            let key = (memory.tenant.clone(), memory.scope.clone(), subject.clone());
            for other in self.chains.get(&key).into_iter().flatten() {
                let Some(o) = self.memories.get(other) else { continue };
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

        self.emit(Event::Write { memory: Box::new(memory) })?;
        if let Some((_, prev)) = before {
            self.emit(Event::Supersede { old: prev, new: id, at: valid_from })?;
        }
        if let Some((next_from, next)) = after {
            self.emit(Event::Supersede { old: id, new: next, at: next_from })?;
        }
        Ok(id)
    }

    // ------------------------------------------------------------------
    // Read path
    // ------------------------------------------------------------------

    fn decay(&self, m: &Memory, now: i64) -> f32 {
        let idle = (now - m.last_access).max(0) as f32;
        0.5f32.powf(idle / self.cfg.half_life_ms.max(1) as f32)
    }

    /// Recall memories for a caller. Filters run before ranking, so a memory
    /// the caller may not read never reaches a ranker.
    pub fn recall(&mut self, who: &Identity, q: &Query, now: i64) -> Vec<Hit> {
        let at = q.as_of.unwrap_or(now);
        let mut memo = HashMap::new();
        let mut candidates: Vec<u64> = Vec::new();
        let readable = self.readable_scopes(who, now);
        let groups = readable.into_iter().filter_map(|scope| self.by_scope.get(&(who.tenant.clone(), scope)));
        for id in groups.flatten() {
            let Some(m) = self.memories.get(id) else { continue };
            if (m.archived && !q.include_archived)
                || m.expires_at.is_some_and(|e| e <= now)
                || m.valid_from > at
                || m.valid_to.is_some_and(|end| end <= at)
                || (!q.kinds.is_empty() && !q.kinds.contains(&m.kind))
                || (!q.scopes.is_empty() && !q.scopes.contains(&m.scope.kind()))
                || m.trust < q.min_trust
            {
                continue;
            }
            if self.can_read_deep(who, *id, now, &mut memo) {
                candidates.push(*id);
            }
        }
        if candidates.is_empty() || q.k == 0 {
            return vec![];
        }

        let mut scored: Vec<(u64, f32)> = if q.text.trim().is_empty() && q.embedding.is_none() {
            // No query: newest first. Used to read a blackboard or a handoff.
            let mut newest: Vec<(i64, u64)> =
                candidates.iter().map(|id| (self.memories[id].recorded_at, *id)).collect();
            newest.sort_unstable_by(|a, b| b.cmp(a));
            newest
                .into_iter()
                .enumerate()
                .map(|(rank, (_, id))| (id, 1.0 / (rank as f32 + 1.0)))
                .collect()
        } else {
            let pool = (q.k * 4).max(32);
            let ids = |ranked: Vec<(u64, f32)>| ranked.into_iter().take(pool).map(|r| r.0).collect();

            let query_tokens = tokenize(&q.text);
            let docs: Vec<(u64, &[String])> = candidates
                .iter()
                .map(|id| (*id, self.tokens[id].as_slice()))
                .collect();
            let keyword: Vec<u64> = ids(bm25(&query_tokens, &docs));

            let mut by_vector: Vec<(u64, f32)> = match &q.embedding {
                Some(qe) => candidates
                    .iter()
                    .filter_map(|id| {
                        let e = self.memories[id].embedding.as_ref()?;
                        Some((*id, cosine(qe, e)))
                    })
                    .collect(),
                None => {
                    let qe = hash_embed(&q.text);
                    candidates.iter().map(|id| (*id, cosine(&qe, &self.hash_vecs[id]))).collect()
                }
            };
            by_vector.retain(|r| r.1 > 0.0);
            sort_desc(&mut by_vector);
            let vector: Vec<u64> = ids(by_vector);

            let lowered = q.text.to_lowercase();
            let mut by_entity: Vec<(u64, f32)> = candidates
                .iter()
                .filter_map(|id| {
                    let n = self.memories[id]
                        .entities
                        .iter()
                        .filter(|e| !e.is_empty() && lowered.contains(&e.to_lowercase()))
                        .count();
                    (n > 0).then_some((*id, n as f32))
                })
                .collect();
            sort_desc(&mut by_entity);
            let entity: Vec<u64> = ids(by_entity);

            rrf(&[keyword, vector, entity])
                .into_iter()
                .map(|(id, fused)| {
                    let m = &self.memories[&id];
                    let trust = match m.trust {
                        Trust::Trusted => 1.0,
                        Trust::Standard => 0.9,
                        Trust::Untrusted => 0.6,
                    };
                    let established = if m.established() { 1.0 } else { 0.85 };
                    let salience = 0.5 + 0.5 * m.salience * self.decay(m, now);
                    (id, fused * trust * established * salience)
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
            let cost = m.content.len() / 4 + 1;
            if q.token_budget.is_some_and(|budget| used + cost > budget) {
                continue;
            }
            used += cost;
            let mut memory = m.clone();
            memory.embedding = None;
            hits.push(Hit { established: memory.established(), memory, score });
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
        let allowed = self.policy(who).is_some_and(|p| p.read.contains(&ScopeKind::UserPrivate));
        if !allowed {
            return Err(Error::Denied(format!("role '{}' may not share user memory", who.role)));
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
            return Err(Error::Denied(format!("grant {grant_id} is not yours to revoke")));
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
            format!("approved:{}", who.agent)
        } else if who.agent == memory.author_agent {
            return Err(Error::Invalid("an agent cannot corroborate its own memory".into()));
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
        for gone in &doomed {
            self.emit(Event::Forget { id: *gone })?;
        }
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
        for id in &expired {
            self.emit(Event::Forget { id: *id })?;
        }
        for id in &fading {
            self.emit(Event::Archive { id: *id })?;
        }
        Ok(SweepReport { expired: expired.len(), archived: fading.len() })
    }

    pub fn stats(&self) -> Stats {
        Stats {
            memories: self.memories.len(),
            tenants: self.by_tenant.values().filter(|ids| !ids.is_empty()).count(),
            grants: self.grants.iter().filter(|g| g.revoked_at.is_none()).count(),
            quarantined: self.quarantined,
        }
    }
}
