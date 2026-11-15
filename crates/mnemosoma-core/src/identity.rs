//! Who is asking, where a memory lives, and what each role may do.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// The caller of every read and write. Built by the host application from
/// its own authentication; never parsed out of model output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub tenant: String,
    pub user: String,
    pub agent: String,
    pub role: String,
    #[serde(default)]
    pub team: Option<String>,
    #[serde(default)]
    pub session: Option<String>,
}

/// Where a memory lives. Each memory belongs to exactly one scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum Scope {
    /// Visible to one agent.
    AgentPrivate(String),
    /// Visible to every agent holding the role.
    Role(String),
    /// Visible to the agents working one task: the team blackboard.
    Team(String),
    /// Visible to agents acting for the user, and to users holding a grant.
    UserPrivate(String),
    /// Visible to everyone in the tenant.
    Tenant,
}

impl Scope {
    pub fn kind(&self) -> ScopeKind {
        match self {
            Scope::AgentPrivate(_) => ScopeKind::AgentPrivate,
            Scope::Role(_) => ScopeKind::Role,
            Scope::Team(_) => ScopeKind::Team,
            Scope::UserPrivate(_) => ScopeKind::UserPrivate,
            Scope::Tenant => ScopeKind::Tenant,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeKind {
    AgentPrivate,
    Role,
    Team,
    UserPrivate,
    Tenant,
}

/// What a role may read and write. A role with no policy can do nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolePolicy {
    pub read: BTreeSet<ScopeKind>,
    pub write: BTreeSet<ScopeKind>,
    /// May approve inferred memories and delete memories written by others.
    #[serde(default)]
    pub can_promote: bool,
}

/// The starting role set. Hosts can replace it in `StoreConfig::policies`.
pub fn default_roles() -> HashMap<String, RolePolicy> {
    use ScopeKind::*;
    let all = [AgentPrivate, Role, Team, UserPrivate, Tenant];
    let policy = |read: &[ScopeKind], write: &[ScopeKind], can_promote: bool| RolePolicy {
        read: read.iter().copied().collect(),
        write: write.iter().copied().collect(),
        can_promote,
    };
    HashMap::from([
        ("planner".into(), policy(&all, &[AgentPrivate, Team], false)),
        (
            "researcher".into(),
            policy(&all, &[AgentPrivate, Team, UserPrivate], false),
        ),
        (
            "executor".into(),
            policy(&[AgentPrivate, Role, Team, UserPrivate], &[AgentPrivate, Team], false),
        ),
        ("reviewer".into(), policy(&all, &[AgentPrivate, Team, Role], false)),
        ("curator".into(), policy(&all, &all, true)),
    ])
}

/// One user letting another read their user-private memories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub id: u64,
    pub tenant: String,
    pub grantor: String,
    pub grantee: String,
    /// When set, only memories carrying this tag are shared.
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub expires_at: Option<i64>,
    #[serde(default)]
    pub revoked_at: Option<i64>,
}

impl Grant {
    pub fn active(&self, now: i64) -> bool {
        self.revoked_at.is_none() && self.expires_at.is_none_or(|e| e > now)
    }
}
