"""Sophia ↔ AIpp bridge.

A thin, authenticated HTTP surface that lets the AIpp iOS boss-cockpit consume
Sophia's epistemic gate, RAG grounding, and conscience kernel. The raw RAG API
(``services/rag_api``) is intentionally unauthenticated and shape-rich; this
bridge adds Bearer-token auth and normalizes Sophia's internal gate/conscience
dicts into the compact verdict contract the app maps onto its governance states.
"""
