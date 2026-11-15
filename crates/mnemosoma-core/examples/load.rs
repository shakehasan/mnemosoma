//! Load benchmark: many agents in several roles writing and recalling for
//! many users, with a leak check on every recall.
//!
//!     cargo run --release -p mnemosoma-core --example load -- 100000
//!
//! Arguments: total memories (default 100000), tenants (default 10).

use mnemosoma_core::*;
use std::time::Instant;

const WORDS: &[&str] = &[
    "venue", "contract", "invoice", "launch", "budget", "deadline", "customer", "refund",
    "shipment", "meeting", "roadmap", "pricing", "renewal", "outage", "report", "design",
    "hiring", "travel", "catering", "review", "migration", "release", "audit", "forecast",
    "onboarding", "feedback", "warranty", "delivery", "campaign", "inventory", "policy", "training",
];
const ROLES: &[&str] = &["planner", "researcher", "executor", "reviewer"];

/// Small deterministic generator, so runs are repeatable without a dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn sentence(&mut self, words: usize) -> String {
        (0..words).map(|_| WORDS[self.below(WORDS.len())]).collect::<Vec<_>>().join(" ")
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let total: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let tenants: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10);
    let users_per_tenant = 100;
    let agents_per_tenant = 20;
    let teams_per_tenant = 10;
    let now = 1_800_000_000_000i64;

    let identity = |rng: &mut Rng, tenant: usize| {
        let agent = rng.below(agents_per_tenant);
        Identity {
            tenant: format!("tenant-{tenant}"),
            user: format!("user-{}", rng.below(users_per_tenant)),
            agent: format!("agent-{agent}"),
            role: ROLES[agent % ROLES.len()].to_string(),
            team: Some(format!("team-{}", rng.below(teams_per_tenant))),
            session: None,
        }
    };

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut store = MemoryStore::in_memory(StoreConfig::new("bench-key"));

    // Writes: a mix of scopes the role is allowed to write.
    let mut write_ms = Vec::with_capacity(total);
    let started = Instant::now();
    let mut written = 0usize;
    while written < total {
        let who = identity(&mut rng, written % tenants);
        let scope = match rng.below(3) {
            0 => Scope::AgentPrivate(who.agent.clone()),
            1 => Scope::Team(who.team.clone().unwrap()),
            _ if who.role == "researcher" => Scope::UserPrivate(who.user.clone()),
            _ => Scope::Team(who.team.clone().unwrap()),
        };
        let words = 8 + rng.below(12);
        let memory = NewMemory::new(scope, MemoryKind::Episodic, rng.sentence(words));
        let t = Instant::now();
        store.write(&who, memory, now + written as i64).expect("write allowed");
        write_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        written += 1;
    }
    let write_secs = started.elapsed().as_secs_f64();

    // Recalls: every hit is checked against the scope rules independently.
    let queries = 2_000;
    let mut recall_ms = Vec::with_capacity(queries);
    let mut returned = 0usize;
    let mut leaks = 0usize;
    for i in 0..queries {
        let who = identity(&mut rng, i % tenants);
        let q = Query::text(rng.sentence(3));
        let t = Instant::now();
        let hits = store.recall(&who, &q, now + total as i64);
        recall_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        for hit in &hits {
            let m = &hit.memory;
            let allowed = m.tenant == who.tenant
                && match &m.scope {
                    Scope::AgentPrivate(a) => *a == who.agent,
                    Scope::Team(t) => Some(t) == who.team.as_ref(),
                    Scope::UserPrivate(u) => *u == who.user && who.role != "__none__",
                    Scope::Role(r) => *r == who.role,
                    Scope::Tenant => true,
                };
            if !allowed {
                leaks += 1;
            }
        }
        returned += hits.len();
    }
    write_ms.sort_by(f64::total_cmp);
    recall_ms.sort_by(f64::total_cmp);

    println!("memories            {total}");
    println!("tenants             {tenants}  ({} per tenant)", total / tenants);
    println!("users / agents      {} / {} per tenant", users_per_tenant, agents_per_tenant);
    println!("write throughput    {:.0} per second", total as f64 / write_secs);
    println!("write p50 / p95     {:.4} / {:.4} ms", percentile(&write_ms, 0.5), percentile(&write_ms, 0.95));
    println!("recall p50 / p95    {:.3} / {:.3} ms", percentile(&recall_ms, 0.5), percentile(&recall_ms, 0.95));
    println!("recalls / hits      {queries} / {returned}");
    println!("permission leaks    {leaks}");
    assert_eq!(leaks, 0, "a recall returned a memory outside the caller's scopes");
}
