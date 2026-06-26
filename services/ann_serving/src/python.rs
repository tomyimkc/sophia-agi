// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//
//! In-process PyO3 binding — the dense recall core callable from Python without a subprocess.
//!
//! The default Python bridge (`agent/ann_client.py`) drives the `serve` binary over a text
//! protocol: robust and build-decoupled, but it pays a process boundary + float-text
//! serialization per query. This binding removes both — Python holds a `ShardedHnsw` directly
//! and calls `search` in-process. Built only with `--features python` (off by default, so the
//! core stays dependency-free); package as an importable extension with maturin.
//!
//!   maturin develop --features python      # or: cargo build --release --features python
//!   >>> import sophia_ann
//!   >>> idx = sophia_ann.ShardedHnsw(num_shards=4, dim=384, m=16, ef_construction=200)
//!   >>> idx.add(0, vec); idx.search(query, k=10, ef=128); idx.save("x.idx")
//!   >>> idx2 = sophia_ann.ShardedHnsw.load("x.idx")

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use crate::ShardedHnsw;

/// Python-facing handle to a sharded HNSW index (1:1 with the Rust `ShardedHnsw`).
#[pyclass(name = "ShardedHnsw")]
pub struct PyShardedHnsw {
    inner: ShardedHnsw,
}

#[pymethods]
impl PyShardedHnsw {
    #[new]
    #[pyo3(signature = (num_shards, dim, m = 16, ef_construction = 200))]
    fn new(num_shards: usize, dim: usize, m: usize, ef_construction: usize) -> Self {
        Self { inner: ShardedHnsw::new(num_shards, dim, m, ef_construction) }
    }

    /// Insert a vector under `id` (routed to its shard). Raises on a dimensionality mismatch.
    fn add(&mut self, id: u32, vec: Vec<f32>) -> PyResult<()> {
        if vec.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected dim {}, got {}", self.inner.dim(), vec.len()
            )));
        }
        self.inner.add(id, &vec);
        Ok(())
    }

    /// Global approximate top-`k` as a list of `(id, similarity)`, best first.
    #[pyo3(signature = (query, k = 10, ef = 64))]
    fn search(&self, query: Vec<f32>, k: usize, ef: usize) -> PyResult<Vec<(u32, f32)>> {
        if query.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected dim {}, got {}", self.inner.dim(), query.len()
            )));
        }
        Ok(self.inner.search(&query, k, ef))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn num_shards(&self) -> usize {
        self.inner.num_shards()
    }

    fn shard_sizes(&self) -> Vec<usize> {
        self.inner.shard_sizes()
    }

    /// Persist the built graph to a portable `.idx` file (build once, load fast).
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.inner.to_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load a persisted index. Raises on a missing file or a corrupt/incompatible blob.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        ShardedHnsw::from_bytes(&bytes)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err("corrupt or incompatible .idx"))
    }
}

/// The `sophia_ann` Python module.
#[pymodule]
fn sophia_ann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyShardedHnsw>()?;
    m.add("__doc__", "In-process sharded HNSW nearest-neighbour index (Rust core).")?;
    Ok(())
}
