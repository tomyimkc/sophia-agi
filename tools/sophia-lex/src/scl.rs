// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Sophia Claim Language (SCL) — a tiny, deterministic surface syntax for
//! provenance assertions, lexed by a `logos` DFA and parsed by a hand-written
//! recursive descent parser.
//!
//! Motivation: today a claim like "the Analects is attributed to the Confucian
//! school, not personally to Confucius" reaches the belief graph (`okf/`) via an
//! LLM/regex extractor — non-deterministic and hard to audit. SCL gives an
//! OPTIONAL, machine-checkable canonical form that compiles to the same triple
//! deterministically, so a corpus author (or a verifier) can pin ground truth
//! that never depends on a model:
//!
//! ```text
//! attribute("Analects" school:"Confucian" confidence:0.6)
//!   not_to("Confucius")
//!   source:"wikidata:Q17592"
//! ```
//!
//! This is net-new (it duplicates nothing): it is a fail-closed *interface* with
//! a real, tested reference implementation, in the repo's idiom. It does NOT
//! claim to replace the extractor — it is a deterministic anchor alongside it.

use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\r\n]+")]
enum Tok {
    #[token("attribute")]
    Attribute,
    #[token("not_to")]
    NotTo,
    #[token("school")]
    School,
    #[token("confidence")]
    Confidence,
    #[token("source")]
    Source,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(":")]
    Colon,
    // double-quoted string with no escapes (claim subjects are plain text)
    #[regex(r#""[^"]*""#, |lex| {
        let s = lex.slice();
        s[1..s.len() - 1].to_string()
    })]
    Str(String),
    #[regex(r"[0-9]+\.[0-9]+|[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Num(f64),
}

/// A parsed claim. Serialized to JSON by `to_json` for hand-off to `okf/`.
#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    pub subject: String,
    pub school: Option<String>,
    pub confidence: Option<f64>,
    pub not_to: Vec<String>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SCL parse error: {}", self.0)
    }
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), ParseError> {
        match self.next() {
            Some(ref t) if t == want => Ok(()),
            other => Err(ParseError(format!("expected {what}, found {other:?}"))),
        }
    }
    fn expect_str(&mut self, what: &str) -> Result<String, ParseError> {
        match self.next() {
            Some(Tok::Str(s)) => Ok(s),
            other => Err(ParseError(format!("expected string for {what}, found {other:?}"))),
        }
    }
}

/// Parse a single SCL claim. Grammar:
///   claim   := "attribute" "(" Str (kv)* ")" tail*
///   kv      := ("school" ":" Str) | ("confidence" ":" Num)
///   tail    := ("not_to" "(" Str ")") | ("source" ":" Str)
pub fn parse(src: &str) -> Result<Claim, ParseError> {
    let toks: Vec<Tok> = Tok::lexer(src)
        .map(|r| r.map_err(|_| ParseError("unrecognized token".into())))
        .collect::<Result<_, _>>()?;
    let mut p = Parser { toks, pos: 0 };

    p.expect(&Tok::Attribute, "`attribute`")?;
    p.expect(&Tok::LParen, "`(`")?;
    let subject = p.expect_str("attribute subject")?;

    let mut claim = Claim {
        subject,
        school: None,
        confidence: None,
        not_to: Vec::new(),
        source: None,
    };

    // inner key/value pairs until ')'
    loop {
        match p.peek() {
            Some(Tok::RParen) => {
                p.next();
                break;
            }
            Some(Tok::School) => {
                p.next();
                p.expect(&Tok::Colon, "`:`")?;
                claim.school = Some(p.expect_str("school")?);
            }
            Some(Tok::Confidence) => {
                p.next();
                p.expect(&Tok::Colon, "`:`")?;
                match p.next() {
                    Some(Tok::Num(n)) => claim.confidence = Some(n),
                    other => {
                        return Err(ParseError(format!("expected number for confidence, found {other:?}")))
                    }
                }
            }
            other => {
                return Err(ParseError(format!("unexpected token inside attribute(...): {other:?}")))
            }
        }
    }

    // optional tail clauses
    while let Some(t) = p.peek() {
        match t {
            Tok::NotTo => {
                p.next();
                p.expect(&Tok::LParen, "`(`")?;
                let who = p.expect_str("not_to target")?;
                p.expect(&Tok::RParen, "`)`")?;
                claim.not_to.push(who);
            }
            Tok::Source => {
                p.next();
                p.expect(&Tok::Colon, "`:`")?;
                claim.source = Some(p.expect_str("source")?);
            }
            other => return Err(ParseError(format!("unexpected trailing token: {other:?}"))),
        }
    }

    if let Some(c) = claim.confidence {
        if !(0.0..=1.0).contains(&c) {
            return Err(ParseError(format!("confidence {c} out of range [0,1]")));
        }
    }
    Ok(claim)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

/// Deterministic JSON serialization (stable key order) for hand-off to `okf/`.
pub fn to_json(c: &Claim) -> String {
    let mut s = String::new();
    s.push_str(&format!("{{\"subject\":\"{}\"", json_escape(&c.subject)));
    match &c.school {
        Some(v) => s.push_str(&format!(",\"school\":\"{}\"", json_escape(v))),
        None => s.push_str(",\"school\":null"),
    }
    match c.confidence {
        Some(v) => s.push_str(&format!(",\"confidence\":{v}")),
        None => s.push_str(",\"confidence\":null"),
    }
    s.push_str(",\"not_to\":[");
    for (i, n) in c.not_to.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{}\"", json_escape(n)));
    }
    s.push(']');
    match &c.source {
        Some(v) => s.push_str(&format!(",\"source\":\"{}\"", json_escape(v))),
        None => s.push_str(",\"source\":null"),
    }
    s.push('}');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_claim() {
        let c = parse(
            r#"attribute("Analects" school:"Confucian" confidence:0.6) not_to("Confucius") source:"wikidata:Q17592""#,
        )
        .unwrap();
        assert_eq!(c.subject, "Analects");
        assert_eq!(c.school.as_deref(), Some("Confucian"));
        assert_eq!(c.confidence, Some(0.6));
        assert_eq!(c.not_to, vec!["Confucius".to_string()]);
        assert_eq!(c.source.as_deref(), Some("wikidata:Q17592"));
    }

    #[test]
    fn minimal_claim() {
        let c = parse(r#"attribute("Tao Te Ching")"#).unwrap();
        assert_eq!(c.subject, "Tao Te Ching");
        assert!(c.school.is_none() && c.confidence.is_none() && c.not_to.is_empty());
    }

    #[test]
    fn multiple_not_to() {
        let c = parse(r#"attribute("X") not_to("A") not_to("B")"#).unwrap();
        assert_eq!(c.not_to, vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn rejects_bad_confidence() {
        assert!(parse(r#"attribute("X" confidence:1.5)"#).is_err());
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("attribute Analects").is_err());
        assert!(parse(r#"attribute("X" school:0.5)"#).is_err());
    }

    #[test]
    fn json_is_stable() {
        let c = parse(r#"attribute("A" confidence:0.5) not_to("B")"#).unwrap();
        assert_eq!(
            to_json(&c),
            r#"{"subject":"A","school":null,"confidence":0.5,"not_to":["B"],"source":null}"#
        );
    }
}
