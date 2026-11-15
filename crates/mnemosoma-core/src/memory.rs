//! The memory record.

use crate::identity::Scope;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Short-term scratch state. Normally written with a TTL.
    Working,
    /// A raw record of something that happened.
    Episodic,
    /// A fact. Facts with a `subject` form supersession chains.
    Semantic,
    /// A learned skill or playbook.
    Procedural,
    /// What one agent passes to the next.
    Handoff,
}

/// How the author came to hold this memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Assertion {
    Observed,
    ToldByUser,
    /// An agent's own conclusion. Stays marked until corroborated or approved.
    Inferred,
}

/// Ordered: `Untrusted < Standard < Trusted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    /// Content from outside the system: web pages, tool output, uploads.
    Untrusted,
    Standard,
    Trusted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: u64,
    pub tenant: String,
    pub scope: Scope,
    pub kind: MemoryKind,
    pub content: String,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    // Attribution
    pub author_user: String,
    pub author_agent: String,
    pub author_role: String,
    pub assertion: Assertion,
    pub source: String,
    pub trust: Trust,
    #[serde(default)]
    pub derived_from: Vec<u64>,
    #[serde(default)]
    pub corroborated_by: Vec<String>,

    // Valid time: when the fact was true in the world.
    pub valid_from: i64,
    #[serde(default)]
    pub valid_to: Option<i64>,
    // Transaction time: when the store learned it.
    pub recorded_at: i64,
    #[serde(default)]
    pub superseded_by: Option<u64>,
    #[serde(default)]
    pub expires_at: Option<i64>,

    // Salience
    pub salience: f32,
    #[serde(default)]
    pub access_count: u32,
    pub last_access: i64,
    #[serde(default)]
    pub archived: bool,

    /// HMAC-SHA256 over the immutable fields, hex encoded.
    pub signature: String,
}

impl Memory {
    /// An inferred memory counts as established once two other agents have
    /// corroborated it, or a promoting role has approved it.
    pub fn established(&self) -> bool {
        self.assertion != Assertion::Inferred
            || self.corroborated_by.iter().any(|c| c.starts_with("approved:"))
            || self.corroborated_by.len() >= 2
    }
}

fn default_assertion() -> Assertion {
    Assertion::Observed
}
fn default_trust() -> Trust {
    Trust::Standard
}
fn default_source() -> String {
    "agent".into()
}

/// What a caller supplies to write a memory. The store fills in the rest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewMemory {
    pub scope: Scope,
    pub kind: MemoryKind,
    pub content: String,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default = "default_assertion")]
    pub assertion: Assertion,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default = "default_trust")]
    pub trust: Trust,
    #[serde(default)]
    pub derived_from: Vec<u64>,
    #[serde(default)]
    pub valid_from: Option<i64>,
    #[serde(default)]
    pub ttl_ms: Option<i64>,
    #[serde(default)]
    pub salience: Option<f32>,
}

impl NewMemory {
    pub fn new(scope: Scope, kind: MemoryKind, content: impl Into<String>) -> Self {
        Self {
            scope,
            kind,
            content: content.into(),
            subject: None,
            entities: vec![],
            tags: vec![],
            embedding: None,
            assertion: default_assertion(),
            source: default_source(),
            trust: default_trust(),
            derived_from: vec![],
            valid_from: None,
            ttl_ms: None,
            salience: None,
        }
    }
}
