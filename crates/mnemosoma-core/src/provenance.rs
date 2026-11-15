//! Write-time signatures. Each memory carries an HMAC-SHA256 over the fields
//! that must never change: who wrote it, where, what it says and what it was
//! derived from. A log entry edited on disk fails verification on load.

use crate::memory::Memory;
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

fn mac_for(key: &[u8], m: &Memory) -> HmacSha256 {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    // Length-prefix every field so that no field can bleed into the next.
    let mut field = |bytes: &[u8]| {
        mac.update(&(bytes.len() as u64).to_le_bytes());
        mac.update(bytes);
    };
    field(&m.id.to_le_bytes());
    field(m.tenant.as_bytes());
    field(serde_json::to_string(&m.scope).unwrap_or_default().as_bytes());
    field(serde_json::to_string(&m.kind).unwrap_or_default().as_bytes());
    field(m.content.as_bytes());
    field(m.subject.as_deref().unwrap_or("").as_bytes());
    field(m.author_user.as_bytes());
    field(m.author_agent.as_bytes());
    field(m.author_role.as_bytes());
    field(serde_json::to_string(&m.assertion).unwrap_or_default().as_bytes());
    field(m.source.as_bytes());
    field(serde_json::to_string(&m.trust).unwrap_or_default().as_bytes());
    field(&m.recorded_at.to_le_bytes());
    field(&m.valid_from.to_le_bytes());
    for d in &m.derived_from {
        field(&d.to_le_bytes());
    }
    mac
}

pub fn sign(key: &[u8], m: &Memory) -> String {
    let bytes = mac_for(key, m).finalize().into_bytes();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Constant-time check of a memory's signature.
pub fn verify(key: &[u8], m: &Memory) -> bool {
    let Some(sig) = decode_hex(&m.signature) else {
        return false;
    };
    mac_for(key, m).verify_slice(&sig).is_ok()
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| s.get(i..i + 2).and_then(|p| u8::from_str_radix(p, 16).ok()))
        .collect()
}
