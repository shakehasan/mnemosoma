//! Python bindings. Structured values cross the boundary as JSON, which the
//! Python package wraps in typed dataclasses. The interpreter lock is released
//! for every call into the engine, so agents on different threads do not
//! queue behind each other in Python.

use mnemosoma_core::{
    index, now_ms, Error, Identity, MemoryStore, NewMemory, Query, RolePolicy, Scope, StoreConfig,
};
use pyo3::exceptions::{PyKeyError, PyOSError, PyPermissionError, PyValueError};
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;

fn to_py(e: Error) -> PyErr {
    let msg = e.to_string();
    match e {
        Error::Denied(_) => PyPermissionError::new_err(msg),
        Error::NotFound(_) => PyKeyError::new_err(msg),
        Error::Invalid(_) => PyValueError::new_err(msg),
        Error::Io(_) => PyOSError::new_err(msg),
    }
}

fn parse<T: DeserializeOwned>(json: &str) -> PyResult<T> {
    serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn dump<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(module = "mnemosoma._core")]
struct Store {
    inner: Mutex<MemoryStore>,
}

impl Store {
    /// Run `f` on the engine with the interpreter lock released.
    fn run<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut MemoryStore) -> Result<T, Error> + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let mut store = self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            f(&mut store)
        })
        .map_err(to_py)
    }
}

#[pymethods]
impl Store {
    #[new]
    #[pyo3(signature = (signing_key, path=None, half_life_days=30.0, archive_below=0.05, policies=None))]
    fn new(
        signing_key: Vec<u8>,
        path: Option<String>,
        half_life_days: f64,
        archive_below: f32,
        policies: Option<&str>,
    ) -> PyResult<Self> {
        if signing_key.is_empty() {
            return Err(PyValueError::new_err("signing_key must not be empty"));
        }
        let mut cfg = StoreConfig::new(signing_key);
        cfg.half_life_ms = (half_life_days * 86_400_000.0) as i64;
        cfg.archive_below = archive_below;
        if let Some(json) = policies {
            cfg.policies = parse::<HashMap<String, RolePolicy>>(json)?;
        }
        let store = match path {
            Some(p) => MemoryStore::open(p, cfg).map_err(to_py)?,
            None => MemoryStore::in_memory(cfg),
        };
        Ok(Self { inner: Mutex::new(store) })
    }

    #[pyo3(signature = (identity, memory, now=None))]
    fn write(&self, py: Python<'_>, identity: &str, memory: &str, now: Option<i64>) -> PyResult<u64> {
        let who: Identity = parse(identity)?;
        let new: NewMemory = parse(memory)?;
        self.run(py, move |s| s.write(&who, new, now.unwrap_or_else(now_ms)))
    }

    #[pyo3(signature = (identity, query, now=None))]
    fn recall(&self, py: Python<'_>, identity: &str, query: &str, now: Option<i64>) -> PyResult<String> {
        let who: Identity = parse(identity)?;
        let q: Query = parse(query)?;
        let hits = self.run(py, move |s| Ok(s.recall(&who, &q, now.unwrap_or_else(now_ms))))?;
        dump(&hits)
    }

    #[pyo3(signature = (identity, id, now=None))]
    fn get(&self, py: Python<'_>, identity: &str, id: u64, now: Option<i64>) -> PyResult<String> {
        let who: Identity = parse(identity)?;
        let memory = self.run(py, move |s| s.get(&who, id, now.unwrap_or_else(now_ms)))?;
        dump(&memory)
    }

    #[pyo3(signature = (identity, scope, subject, now=None))]
    fn history(
        &self,
        py: Python<'_>,
        identity: &str,
        scope: &str,
        subject: String,
        now: Option<i64>,
    ) -> PyResult<String> {
        let who: Identity = parse(identity)?;
        let scope: Scope = parse(scope)?;
        let versions =
            self.run(py, move |s| Ok(s.history(&who, &scope, &subject, now.unwrap_or_else(now_ms))))?;
        dump(&versions)
    }

    #[pyo3(signature = (identity, grantee, tag=None, expires_at=None))]
    fn grant(
        &self,
        py: Python<'_>,
        identity: &str,
        grantee: String,
        tag: Option<String>,
        expires_at: Option<i64>,
    ) -> PyResult<u64> {
        let who: Identity = parse(identity)?;
        self.run(py, move |s| s.grant(&who, &grantee, tag, expires_at))
    }

    #[pyo3(signature = (identity, grant_id, now=None))]
    fn revoke(&self, py: Python<'_>, identity: &str, grant_id: u64, now: Option<i64>) -> PyResult<()> {
        let who: Identity = parse(identity)?;
        self.run(py, move |s| s.revoke(&who, grant_id, now.unwrap_or_else(now_ms)))
    }

    #[pyo3(signature = (identity, id, now=None))]
    fn corroborate(&self, py: Python<'_>, identity: &str, id: u64, now: Option<i64>) -> PyResult<()> {
        let who: Identity = parse(identity)?;
        self.run(py, move |s| s.corroborate(&who, id, now.unwrap_or_else(now_ms)))
    }

    #[pyo3(signature = (identity, id, now=None))]
    fn forget(&self, py: Python<'_>, identity: &str, id: u64, now: Option<i64>) -> PyResult<usize> {
        let who: Identity = parse(identity)?;
        self.run(py, move |s| s.forget(&who, id, now.unwrap_or_else(now_ms)))
    }

    #[pyo3(signature = (now=None))]
    fn sweep(&self, py: Python<'_>, now: Option<i64>) -> PyResult<String> {
        let report = self.run(py, move |s| s.sweep(now.unwrap_or_else(now_ms)))?;
        dump(&report)
    }

    fn stats(&self, py: Python<'_>) -> PyResult<String> {
        let stats = self.run(py, |s| Ok(s.stats()))?;
        dump(&stats)
    }
}

/// The built-in hashing embedder, exposed for inspection and tests.
#[pyfunction]
fn hash_embed(text: &str) -> Vec<f32> {
    index::hash_embed(text)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Store>()?;
    m.add_function(wrap_pyfunction!(hash_embed, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
