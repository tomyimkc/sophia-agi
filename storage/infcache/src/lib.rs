// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! `infcache` — a prefix-keyed, tiered KV-block cache for LLM inference.
//!
//! Phase 4 of the Sophia distributed-storage roadmap, and the part that maps
//! most directly onto the role's #1 responsibility ("支撑大模型推理的高性能
//! KVCache 存储系统"). It composes the earlier phases:
//!
//! - **prefix keying** ([`prefix`]) turns a token stream into block keys so a
//!   shared prompt prefix is a cache hit (context caching);
//! - the **RAM tier** is the in-process [`kvcache::ShardedCache`] (Phase 1);
//! - the **SSD tier** is [`diskstore::Bitcask`] (Phase 2), durable and larger.
//!
//! A lookup hits RAM first, falls back to SSD (promoting the block back into
//! RAM), and a prefill *plan* reports how much of a prompt can be reused vs.
//! must be recomputed — the metric that drives inference cost down.
//!
//! Scope: this is the storage/reuse layer, not an attention kernel. Block
//! payloads are opaque bytes (the serialized K/V for a block); wiring it to a
//! real engine (vLLM/SGLang-style) is the integration step.

pub mod prefix;
pub mod tier;

pub use prefix::block_keys;
pub use tier::{PrefillPlan, TierMetrics, TieredKvCache};
