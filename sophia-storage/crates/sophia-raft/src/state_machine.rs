// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! The replicated state machine: what committed log entries *do*.
//!
//! Raft guarantees every node applies the same commands in the same order; the
//! state machine turns that ordered command stream into state. For Sophia this
//! is the decision log / task queue — and because `queue.py` keys every task
//! with an idempotency key, applying the same committed entry twice is safe,
//! which is exactly what makes leader-change retries correct.

use std::collections::HashMap;

/// Apply committed commands in log order. Implementations must be deterministic:
/// the same command sequence yields the same state on every node.
pub trait StateMachine {
    fn apply(&mut self, command: &[u8]);
}

/// A simple key/value state machine for tests and as the reference shape for a
/// durable backend (e.g. an `sophia-lsm` engine). Commands are `key=value` or
/// `del key` byte strings.
#[derive(Debug, Default, Clone)]
pub struct KvStateMachine {
    pub map: HashMap<String, String>,
    /// Every command applied, in order — lets tests assert cross-node agreement.
    pub applied: Vec<Vec<u8>>,
}

impl KvStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.map.get(key)
    }
}

impl StateMachine for KvStateMachine {
    fn apply(&mut self, command: &[u8]) {
        self.applied.push(command.to_vec());
        let text = String::from_utf8_lossy(command);
        if let Some(rest) = text.strip_prefix("del ") {
            self.map.remove(rest.trim());
        } else if let Some((k, v)) = text.split_once('=') {
            self.map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
}
