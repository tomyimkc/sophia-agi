// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Deterministic, single-DFA "no-overclaim" scanner.
//!
//! `tools/lint_claims.py` enforces the no-overclaim gate with ~18 Python
//! regexes. Regexes can backtrack, are scattered, and their word-boundary
//! semantics are implicit. This module instead tokenizes each (lowercased) line
//! into Unicode WORD tokens with a `logos`-generated DFA, then matches the
//! forbidden phrases over the *token stream*. Word boundaries are then exact by
//! construction (a token IS a word), the scan is linear-time with no
//! backtracking, and the forbidden list is one auditable table.
//!
//! PARITY NOTE (honest scope): this AGREES with the Python oracle on the
//! committed corpus and on the shared test-vector suite
//! (`tools/sophia-lex/fixtures/overclaim_vectors.jsonl`). It is NOT proven
//! byte-identical for every adversarial input — the fuzzy `first .{0,12} agi`
//! gap is reproduced as a byte-distance window, an approximation of the regex.
//! The Python tool stays the reference oracle; this is an optional accelerator.

use logos::Logos;

/// One forbidden phrase reported at a line/column, carrying the SAME `why`
/// string the Python linter uses, so verdict sets compare directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub line: usize,   // 1-based, like the Python linter
    pub col: usize,    // 1-based byte column of the match start
    pub why: String,
}

/// Line opt-out marker, identical to the Python linter.
const ALLOW_MARKER: &str = "claim-ok";

/// Token alphabet for prose scanning. We only care about word runs and the `%`
/// sign (for the "100% on all" rule); everything else is whitespace/punctuation
/// and is skipped. `\p{Alphabetic}` keeps "gödel" a single token.
#[derive(Logos, Debug, PartialEq)]
enum Tok {
    #[regex(r"[\p{Alphabetic}\p{Nd}_]+")]
    Word,

    #[token("%")]
    Percent,
}

/// A materialized token: its kind-agnostic lowercased text plus byte span on the
/// line. (The line is lowercased before lexing, so text is already lowercase.)
struct Span<'a> {
    text: &'a str,
    is_percent: bool,
    start: usize,
    end: usize,
}

/// Allowed separator BEFORE a phrase word, so the token scan reproduces the
/// regex's literal spaces / `\b` boundaries instead of silently bridging
/// punctuation (e.g. markdown emphasis `is **agi**` or `proven. [agi`).
#[derive(Clone, Copy, PartialEq)]
enum Sep {
    /// First word of a phrase — no preceding separator to check.
    Start,
    /// One or more whitespace chars (mirrors a literal space / `\s+`).
    Ws,
    /// Exactly a hyphen (e.g. `self-improvement`).
    Hyphen,
    /// Exactly an apostrophe (e.g. `world's`).
    Apostrophe,
    /// A hyphen or whitespace (mirrors `ai[- ]authored`).
    HyphenOrWs,
}

fn lex_line(line: &str) -> Vec<Span<'_>> {
    let mut out = Vec::new();
    let mut lx = Tok::lexer(line);
    while let Some(res) = lx.next() {
        let span = lx.span();
        match res {
            Ok(Tok::Word) => out.push(Span {
                text: &line[span.start..span.end],
                is_percent: false,
                start: span.start,
                end: span.end,
            }),
            Ok(Tok::Percent) => out.push(Span {
                text: "%",
                is_percent: true,
                start: span.start,
                end: span.end,
            }),
            Err(_) => {} // whitespace / punctuation — skipped
        }
    }
    out
}

/// Is the byte slice between two tokens an acceptable separator of kind `sep`?
fn sep_ok(low: &str, prev_end: usize, cur_start: usize, sep: Sep) -> bool {
    let gap = &low[prev_end..cur_start];
    match sep {
        Sep::Start => true,
        Sep::Ws => !gap.is_empty() && gap.chars().all(char::is_whitespace),
        Sep::Hyphen => gap == "-",
        Sep::Apostrophe => gap == "'",
        Sep::HyphenOrWs => gap == "-" || (!gap.is_empty() && gap.chars().all(char::is_whitespace)),
    }
}

/// Do the word tokens at `i..` match `phrase` (text + required preceding
/// separator)? Percent tokens never match a word literal. Reproduces the
/// regexes' literal-space / `\b` semantics so punctuation between words (markdown
/// emphasis, sentence boundaries) is NOT silently bridged.
fn phrase_at(low: &str, toks: &[Span], i: usize, phrase: &[(&str, Sep)]) -> bool {
    if i + phrase.len() > toks.len() {
        return false;
    }
    for (j, (w, sep)) in phrase.iter().enumerate() {
        let t = &toks[i + j];
        if t.is_percent || t.text != *w {
            return false;
        }
        if j > 0 && !sep_ok(low, toks[i + j - 1].end, t.start, *sep) {
            return false;
        }
    }
    true
}

/// Convenience: a phrase whose words are all whitespace-separated.
macro_rules! ws_phrase {
    ($first:literal $(, $rest:literal)* $(,)?) => {
        &[($first, Sep::Start) $(, ($rest, Sep::Ws))*]
    };
}

/// The forbidden phrase table, expressed as token-sequence alternatives.
/// Each entry is `(why, &[&[phrase-alternative]])`; a line violates the entry if
/// ANY alternative occurs as consecutive word tokens. The `why` strings are
/// copied verbatim from `tools/lint_claims.py` so the two linters' verdict sets
/// are directly comparable.
struct Rule {
    why: &'static str,
    alts: &'static [&'static [(&'static str, Sep)]],
}

const RULES: &[Rule] = &[
    Rule { why: "implies a guarantee; the gate is a filter (23.6% residual)",
           alts: &[ws_phrase!("safe", "to", "ship")] },
    Rule { why: "implies no oversight needed",
           alts: &[ws_phrase!("trust", "in", "production", "without")] },
    Rule { why: "implies autonomy the evidence does not support",
           alts: &[ws_phrase!("without", "constant", "oversight")] },
    Rule { why: "unqualified safety claim",
           alts: &[ws_phrase!("makes", "ai", "safe")] },
    Rule { why: "AGI capability claim",
           alts: &[ws_phrase!("proven", "agi"), ws_phrase!("is", "agi")] },
    Rule { why: "AGI primacy / hype",
           alts: &[ws_phrase!("birth", "the", "first"), ws_phrase!("birthing", "the", "first")] },
    Rule { why: "unfalsifiable superiority claim",
           alts: &[ws_phrase!("the", "only", "open", "project")] },
    Rule { why: "primacy claim",
           alts: &[
               // world's first  (world + ' + s + space + first)
               &[("world", Sep::Start), ("s", Sep::Apostrophe), ("first", Sep::Ws)],
               // worlds first
               ws_phrase!("worlds", "first"),
           ] },
    Rule { why: "hype term without a cited result",
           alts: &[&[("breakthrough", Sep::Start)]] },
    Rule { why: "unformalizable alignment overclaim",
           alts: &[ws_phrase!("proves", "alignment")] },
    Rule { why: "unqualified safe-self-improvement claim",
           // proves safe self-improvement  (hyphen required before "improvement")
           alts: &[&[("proves", Sep::Start), ("safe", Sep::Ws), ("self", Sep::Ws),
                     ("improvement", Sep::Hyphen)]] },
    Rule { why: "misleading Gödel-machine framing",
           alts: &[ws_phrase!("gödel", "machine"), ws_phrase!("godel", "machine")] },
    Rule { why: "unqualified trustworthiness proof claim",
           alts: &[ws_phrase!("proves", "it", "is", "trustworthy")] },
    Rule { why: "Leiden: AI is a tool, not an author",
           // ai[- ]authored
           alts: &[&[("ai", Sep::Start), ("authored", Sep::HyphenOrWs)]] },
];

fn scan_line(line_no: usize, line: &str, out: &mut Vec<Violation>) {
    if line.contains(ALLOW_MARKER) {
        return;
    }
    let low = line.to_lowercase();
    let toks = lex_line(&low);

    // The Python oracle (lint_claims FORBIDDEN) matches with `re.search`, so it
    // records at most ONE hit per pattern per line. This token scan visits every
    // token start, so a phrase that recurs on a line — or two alternatives of the
    // same rule both matching — would otherwise push the same `why` more than once.
    // Dedupe per (line, why) so the emitted verdict set matches the Python linter
    // exactly (each `why` maps 1:1 to a Python FORBIDDEN pattern).
    let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    let mut push = |why: &'static str, col_byte: usize| {
        if seen.insert(why) {
            out.push(Violation { line: line_no, col: col_byte + 1, why: why.to_string() });
        }
    };

    // Whitespace-separated word run starting at i matches `words`?
    let ws_run = |i: usize, words: &[&str]| -> bool {
        if words.is_empty() {
            return true;
        }
        let mut phrase: Vec<(&str, Sep)> = Vec::with_capacity(words.len());
        for (k, w) in words.iter().enumerate() {
            phrase.push((w, if k == 0 { Sep::Start } else { Sep::Ws }));
        }
        phrase_at(&low, &toks, i, &phrase)
    };

    for i in 0..toks.len() {
        // Straight phrase-table rules.
        for rule in RULES {
            for alt in rule.alts {
                if phrase_at(&low, &toks, i, alt) {
                    push(rule.why, toks[i].start);
                }
            }
        }

        // "authored by (claude|gpt|copilot|glm|an? ai|the model|the llm)"
        if ws_run(i, &["authored", "by"]) {
            let hit = toks.get(i + 2).map_or(false, |t| {
                !t.is_percent
                    && matches!(t.text, "claude" | "gpt" | "copilot" | "glm")
                    && sep_ok(&low, toks[i + 1].end, t.start, Sep::Ws)
            }) || ws_run(i + 2, &["a", "ai"])
                || ws_run(i + 2, &["an", "ai"])
                || ws_run(i + 2, &["the", "model"])
                || ws_run(i + 2, &["the", "llm"]);
            if hit {
                push("Leiden: results are authored by humans, not by an automated system",
                     toks[i].start);
            }
        }

        // "(claude|gpt|the model|the llm) (is|was) (the|a|an )?(author|inventor|discoverer)"
        let subj = if matches!(toks[i].text, "claude" | "gpt") && !toks[i].is_percent {
            Some(i + 1)
        } else if ws_run(i, &["the", "model"]) || ws_run(i, &["the", "llm"]) {
            Some(i + 2)
        } else {
            None
        };
        if let Some(mut j) = subj {
            let ws_before = |toks: &[Span], j: usize| -> bool {
                j > 0 && j < toks.len() && sep_ok(&low, toks[j - 1].end, toks[j].start, Sep::Ws)
            };
            if ws_before(&toks, j)
                && toks.get(j).map_or(false, |t| matches!(t.text, "is" | "was"))
            {
                j += 1;
                if ws_before(&toks, j)
                    && toks.get(j).map_or(false, |t| matches!(t.text, "the" | "a" | "an"))
                {
                    j += 1;
                }
                if ws_before(&toks, j)
                    && toks.get(j).map_or(false, |t| {
                        matches!(t.text, "author" | "inventor" | "discoverer")
                    })
                {
                    push("Leiden: credit for results belongs to humans, not automated systems",
                         toks[i].start);
                }
            }
        }

        // "100% on all" — Word("100") immediately-followed-by Percent, then "on all".
        if !toks[i].is_percent && toks[i].text == "100" {
            if let Some(pct) = toks.get(i + 1) {
                if pct.is_percent
                    && pct.start == toks[i].end // no gap: "100%"
                    && ws_run(i + 2, &["on", "all"])
                    && toks
                        .get(i + 2)
                        .map_or(false, |t| sep_ok(&low, pct.end, t.start, Sep::Ws))
                {
                    push("first-party benchmark stated as universal result", toks[i].start);
                }
            }
        }

        // "(the )?first .{0,12} agi" — reproduced as a byte-distance window:
        // an "agi" word starting within 13 bytes after a "first" word.
        if !toks[i].is_percent && toks[i].text == "first" {
            for t in &toks[i + 1..] {
                if t.start > toks[i].end + 13 {
                    break;
                }
                if !t.is_percent && t.text == "agi" && t.start >= toks[i].end {
                    push("AGI primacy claim", toks[i].start);
                    break;
                }
            }
        }
    }
}

/// Scan a whole file body, returning violations in (line, then discovery) order.
pub fn scan(body: &str) -> Vec<Violation> {
    let mut out = Vec::new();
    for (idx, line) in body.split('\n').enumerate() {
        // strip a trailing '\r' so CRLF files match LF byte columns
        let line = line.strip_suffix('\r').unwrap_or(line);
        scan_line(idx + 1, line, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whys(body: &str) -> Vec<String> {
        scan(body).into_iter().map(|v| v.why).collect()
    }

    #[test]
    fn flags_basic_overclaims() {
        assert!(whys("This is safe to ship today.").iter().any(|w| w.contains("filter")));
        assert!(whys("a real breakthrough").iter().any(|w| w.contains("hype")));
        assert!(whys("the world's first gate").iter().any(|w| w.contains("primacy")));
        assert!(whys("a gödel machine").iter().any(|w| w.contains("Gödel")));
        assert!(whys("100% on all benchmarks").iter().any(|w| w.contains("universal")));
        assert!(whys("the first true agi").iter().any(|w| w.contains("primacy")));
    }

    #[test]
    fn respects_allow_marker() {
        assert!(whys("a breakthrough <!-- claim-ok -->").is_empty());
    }

    #[test]
    fn clean_prose_is_clean() {
        assert!(whys("A provenance-aware gate that abstains instead of fabricating.").is_empty());
    }

    #[test]
    fn word_boundary_no_false_positive() {
        // "agile" must not trip the "is agi" rule; "firstly" is not "first".
        assert!(whys("this is agile software").is_empty());
    }

    #[test]
    fn leiden_author_rules() {
        assert!(whys("authored by claude").iter().any(|w| w.contains("automated system")));
        assert!(whys("authored by an ai").iter().any(|w| w.contains("automated system")));
        assert!(whys("gpt is the author here").iter().any(|w| w.contains("belongs to humans")));
        assert!(whys("ai-authored result").iter().any(|w| w.contains("tool")));
    }
}
