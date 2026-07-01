---
id: event:7e2cb6f2eb3e
node_type: event
sources: ["bridge/STATUS.json"]
links: []
verifier: null
verdict: fail
moral_standard: null
title: bench-a-04 stalled — heartbeat frozen, GPU idle, no result
---

The cloud-dispatched kappa re-run (seeds {1,2,10}) wedged: STATUS.json heartbeat froze at 05:25:21Z, GPU read 0% idle, and no bridge/results/...bench-a-04.json was ever written — 27+ minutes of silence while the poller still claimed `running`.
