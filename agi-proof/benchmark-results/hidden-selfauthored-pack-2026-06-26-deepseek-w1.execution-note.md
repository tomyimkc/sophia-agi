# W1 Hidden Eval Attempt — v1 artifact retention failure

Run ID: `hidden-selfauthored-pack-2026-06-26-deepseek-w1`  
Pack ID: `selfauthored-fugu-w1-2026-06-26-v1`  
Pack SHA-256: `d9fba6493f9453ebabe552c21ccba4c3e09f84c1b10630649693b6a8b177ea7c`  
Backend: `deepseek` (`deepseek-v4-pro` observed).

The first execution used a fresh/self-authored pack and printed an aggregate of 8/8 nonempty answers, 0 backend failures, auto score 27.34/40 (68.35%), and strict-ready 0/8/manual pending 8/8. The requested raw response/private/public/manual artifacts were not retained for checksum and commit after a shell-wrapper capture mistake, so those console numbers are **not artifact-backed evidence**.

A diagnostic rerun on the same now-spent pack printed 31.21/40 (78.03%) with 8/8 nonempty and 0 backend failures, but it is invalid as fresh hidden evidence because the pack had already been spent.

This file exists so the failed attempt is not hidden. The valid artifact-backed W1 execution-health run is the separate v2 record. `canClaimAGI` remains false.
