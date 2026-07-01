// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! `sophia-lex` CLI — the optional accelerator entrypoint the Python gate tools
//! shell out to. All output is line-oriented JSON so Python parses it with the
//! stdlib `json` module; no serde dependency is needed on either side.
//!
//! Subcommands:
//!   overclaim <file>...      Scan files; emit one JSON object per violation.
//!   decontam                 Read a length-prefixed stream of (k, jaccard,
//!                            train[], eval[]) on stdin; emit flagged train
//!                            prompts (full eval coverage, no cap).
//!   scl                      Read one SCL claim on stdin; emit canonical JSON.

use std::io::{self, Read, Write};

use sophia_lex::normalize::normalize;
use sophia_lex::{overclaim, scl, shingle};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let code = match cmd {
        "overclaim" => cmd_overclaim(&args[2..]),
        "decontam" => cmd_decontam(),
        "scl" => cmd_scl(),
        other => {
            eprintln!("sophia-lex: unknown subcommand {other:?} (overclaim|decontam|scl)");
            2
        }
    };
    std::process::exit(code);
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One JSON object per violation: {"file":..,"line":..,"col":..,"why":..}.
/// Exit 1 if any violation found, 0 if clean (matches lint_claims semantics).
fn cmd_overclaim(files: &[String]) -> i32 {
    let mut found = false;
    let stdout = io::stdout();
    let mut w = stdout.lock();
    for f in files {
        let body = match std::fs::read_to_string(f) {
            Ok(b) => b,
            Err(_) => continue, // mirror the Python linter's silent skip on read error
        };
        for v in overclaim::scan(&body) {
            found = true;
            let _ = writeln!(
                w,
                "{{\"file\":{},\"line\":{},\"col\":{},\"why\":{}}}",
                json_str(f),
                v.line,
                v.col,
                json_str(&v.why)
            );
        }
    }
    if found {
        1
    } else {
        0
    }
}

/// Length-prefixed stdin protocol (dependency-free, content-safe):
///   line: k
///   line: jaccard (float)
///   line: n_train
///   then n_train records, each: "<byte_len>\n<bytes>"
///   line: n_eval
///   then n_eval records, each: "<byte_len>\n<bytes>"
/// Output: for each train prompt with a near-duplicate eval prompt, one JSON
/// object {"j":..,"train":..,"eval":..} (train/eval truncated to 80 chars, as
/// the Python report does). Full eval coverage — no cap.
fn cmd_decontam() -> i32 {
    let mut buf = Vec::new();
    if io::stdin().read_to_end(&mut buf).is_err() {
        eprintln!("sophia-lex decontam: failed to read stdin");
        return 2;
    }
    let mut cur = 0usize;

    fn read_line(buf: &[u8], cur: &mut usize) -> Option<String> {
        let start = *cur;
        while *cur < buf.len() && buf[*cur] != b'\n' {
            *cur += 1;
        }
        if *cur >= buf.len() {
            return None;
        }
        let s = String::from_utf8_lossy(&buf[start..*cur]).into_owned();
        *cur += 1; // skip newline
        Some(s)
    }
    fn read_record(buf: &[u8], cur: &mut usize) -> Option<String> {
        let len: usize = read_line(buf, cur)?.trim().parse().ok()?;
        if *cur + len > buf.len() {
            return None;
        }
        let s = String::from_utf8_lossy(&buf[*cur..*cur + len]).into_owned();
        *cur += len;
        if *cur < buf.len() && buf[*cur] == b'\n' {
            *cur += 1;
        }
        Some(s)
    }

    let parse_err = || -> i32 {
        eprintln!("sophia-lex decontam: malformed input stream");
        2
    };

    let k: usize = match read_line(&buf, &mut cur).and_then(|s| s.trim().parse().ok()) {
        Some(v) => v,
        None => return parse_err(),
    };
    let jaccard_thr: f64 = match read_line(&buf, &mut cur).and_then(|s| s.trim().parse().ok()) {
        Some(v) => v,
        None => return parse_err(),
    };
    let n_train: usize = match read_line(&buf, &mut cur).and_then(|s| s.trim().parse().ok()) {
        Some(v) => v,
        None => return parse_err(),
    };
    let mut train = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        match read_record(&buf, &mut cur) {
            Some(s) => train.push(s),
            None => return parse_err(),
        }
    }
    let n_eval: usize = match read_line(&buf, &mut cur).and_then(|s| s.trim().parse().ok()) {
        Some(v) => v,
        None => return parse_err(),
    };
    let mut eval = Vec::with_capacity(n_eval);
    for _ in 0..n_eval {
        match read_record(&buf, &mut cur) {
            Some(s) => eval.push(s),
            None => return parse_err(),
        }
    }

    // Precompute eval shingles (full coverage).
    let eval_sh: Vec<(String, _)> = eval
        .iter()
        .map(|e| (normalize(e), shingle::shingles(e, k)))
        .collect();

    let stdout = io::stdout();
    let mut w = stdout.lock();
    let mut seen = std::collections::HashSet::new();
    for pr in &train {
        let npr = normalize(pr);
        if !seen.insert(npr.clone()) {
            continue;
        }
        let tsh = shingle::shingles(pr, k);
        if tsh.is_empty() {
            continue;
        }
        for (e_norm, esh) in &eval_sh {
            let j = shingle::jaccard(&tsh, esh);
            if j >= jaccard_thr && &npr != e_norm {
                let tprefix: String = pr.chars().take(80).collect();
                let eprefix: String = e_norm.chars().take(80).collect();
                let _ = writeln!(
                    w,
                    "{{\"j\":{:.3},\"train\":{},\"eval\":{}}}",
                    j,
                    json_str(&tprefix),
                    json_str(&eprefix)
                );
                break; // first eval match, like the Python loop
            }
        }
    }
    0
}

/// Parse one SCL claim from stdin; emit canonical JSON on success (exit 0) or a
/// JSON error object on failure (exit 1).
fn cmd_scl() -> i32 {
    let mut src = String::new();
    if io::stdin().read_to_string(&mut src).is_err() {
        eprintln!("sophia-lex scl: failed to read stdin");
        return 2;
    }
    match scl::parse(&src) {
        Ok(c) => {
            println!("{}", scl::to_json(&c));
            0
        }
        Err(e) => {
            println!("{{\"error\":{}}}", json_str(&e.0));
            1
        }
    }
}
