//! Mnemosoma core: a permission-aware, bi-temporal memory engine for
//! multi-agent, multi-user systems.
//!
//! The engine has one rule that everything else follows from: a memory the
//! caller may not read is never scored. Permission, time and trust filters
//! run before ranking, inside [`MemoryStore::recall`].

pub mod ann;
pub mod identity;
pub mod index;
pub mod memory;
pub mod provenance;
pub mod store;

pub use identity::{default_roles, Grant, Identity, RolePolicy, Scope, ScopeKind};
pub use memory::{Assertion, Memory, MemoryKind, NewMemory, Trust};
pub use store::{now_ms, Error, Hit, MemoryStore, Query, Stats, StoreConfig, SweepReport, Why};

/// Compiles the README example as a test.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
