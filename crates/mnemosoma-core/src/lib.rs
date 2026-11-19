//! Mnemosoma core: a permission-aware, bi-temporal memory engine for
//! multi-agent, multi-user systems.
//!
//! The engine has one rule that everything else follows from: a memory the
//! caller may not read is never scored. Permission, time and trust filters
//! run before ranking, inside [`MemoryStore::recall`].

pub mod ann;
pub mod config;
pub mod entities;
pub mod identity;
pub mod index;
pub mod memory;
pub mod provenance;
pub mod store;
pub mod text;

pub use ann::HnswParams;
pub use config::Tuning;
pub use identity::{default_roles, Grant, Identity, RolePolicy, Scope, ScopeKind};
pub use memory::{
    Assertion, Memory, MemoryKind, NewMemory, Trust, APPROVED_PREFIX,
    DEFAULT_REQUIRED_CORROBORATIONS,
};
pub use store::{now_ms, Error, Hit, MemoryStore, Query, Stats, StoreConfig, SweepReport, Why};

/// Compiles the README example as a test.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
