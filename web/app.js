const MODES = ["advisor", "repo", "life"];
let activeMode = "advisor";
let manifest = null;

async function loadManifest() {
  try {
    const res = await fetch("data/manifest.json");
    manifest = await res.json();
    renderStats();
    renderLeaderboards();
    renderComparisons();
    renderProofPackage();
    renderFooter();
  } catch {
    document.getElementById("stats").innerHTML =
      "<p class='agent-hint'>Run <code>python tools/build_web_data.py</code>, then serve via <code>python tools/serve_web.py</code> to populate live stats.</p>";
  }
}

function renderStats() {
  if (!manifest) return;
  const ragChunks = manifest.rag?.indexChunks ?? 0;
  const loraScore = manifest.localModel?.benchmark;
  const loraPct = loraScore?.scorePct != null ? `${loraScore.scorePct}%` : "—";
  const domains = (manifest.domains || []).length || 4;
  document.getElementById("stats").innerHTML = `
    <div class="stat-card"><div class="stat-value">v${manifest.version}</div><div class="stat-label">Release</div></div>
    <div class="stat-card"><div class="stat-value">${manifest.trainingExamples}</div><div class="stat-label">Bilingual examples</div></div>
    <div class="stat-card"><div class="stat-value">${ragChunks || "—"}</div><div class="stat-label">RAG index chunks</div></div>
    <div class="stat-card"><div class="stat-value">${loraPct}</div><div class="stat-label">Local model</div></div>
    <div class="stat-card"><div class="stat-value">${domains}</div><div class="stat-label">Active domains</div></div>`;
}

function barClass(pct) {
  if (pct >= 100) return "full";
  if (pct < 50) return "low";
  return "";
}

function renderLeaderboards() {
  const root = document.getElementById("leaderboards");
  if (!manifest?.leaderboards || !root) return;
  root.innerHTML = "";
  for (const [domain, board] of Object.entries(manifest.leaderboards)) {
    const section = document.createElement("div");
    section.innerHTML = `<h3>${domain.charAt(0).toUpperCase() + domain.slice(1)} <span style="font-weight:400;color:var(--muted)">(${board.cases} cases)</span></h3>`;
    const table = document.createElement("table");
    table.innerHTML = "<tr><th>Model</th><th>Score</th><th>Passed</th></tr>";
    for (const entry of board.entries || []) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${entry.model}</td>
        <td>
          <div>${entry.score_pct}%</div>
          <div class="score-bar"><div class="score-fill ${barClass(entry.score_pct)}" style="width:${entry.score_pct}%"></div></div>
        </td>
        <td>${entry.passed}/${entry.total}</td>`;
      table.appendChild(tr);
    }
    section.appendChild(table);
    root.appendChild(section);
  }
}

// ---- Benchmark comparison charts (dependency-free, accessible) -------------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderBar(bar, max, unit, lowerIsBetter) {
  const pct = Math.max(0, Math.min(100, (bar.value / max) * 100));
  const tone = bar.highlight ? "is-sophia" : lowerIsBetter ? "is-rival-low" : "is-rival";
  const ci = Array.isArray(bar.ci)
    ? `<span class="bar-ci">95% CI [${bar.ci[0]}, ${bar.ci[1]}]</span>`
    : "";
  return `
    <div class="cbar-row">
      <div class="cbar-label">${escapeHtml(bar.label)}</div>
      <div class="cbar-track" role="img" aria-label="${escapeHtml(bar.label)}: ${bar.value}${unit}">
        <div class="cbar-fill ${tone}" style="width:${pct}%"></div>
        <span class="cbar-value">${bar.value}${unit}${ci}</span>
      </div>
    </div>`;
}

function renderComparisons() {
  const root = document.getElementById("comparison-charts");
  const comp = manifest?.comparisons;
  if (!root || !comp) return;
  root.innerHTML = "";

  for (const chart of comp.charts || []) {
    const card = document.createElement("article");
    card.className = "cchart";
    const unit = chart.unit || "";
    const max = chart.max || 100;
    const better = `${chart.lowerIsBetter ? "Lower" : "Higher"} is better`;
    const vClass =
      chart.verdict === "win" ? "v-win" : chart.verdict === "tradeoff" ? "v-mixed" : "v-loss";

    let bars = "";
    if (chart.groups) {
      bars = chart.groups
        .map(
          (g) =>
            `<div class="cgroup"><div class="cgroup-label">${escapeHtml(g.label)}</div>${(g.bars || [])
              .map((b) => renderBar(b, max, unit, chart.lowerIsBetter))
              .join("")}</div>`
        )
        .join("");
    } else {
      bars = (chart.bars || []).map((b) => renderBar(b, max, unit, chart.lowerIsBetter)).join("");
    }

    card.innerHTML = `
      <div class="cchart-head">
        <h3>${escapeHtml(chart.title)}</h3>
        ${chart.verdictLabel ? `<span class="cverdict ${vClass}">${escapeHtml(chart.verdictLabel)}</span>` : ""}
      </div>
      <p class="cchart-sub">${escapeHtml(chart.subtitle || "")}</p>
      <p class="cchart-metric">${escapeHtml(chart.metric || "")} · <span>${better}</span></p>
      <div class="cchart-bars">${bars}</div>
      ${chart.note ? `<p class="cchart-note">${escapeHtml(chart.note)}</p>` : ""}`;
    root.appendChild(card);
  }

  const honesty = document.getElementById("comparison-honesty");
  if (honesty) {
    honesty.innerHTML = "";
    for (const item of comp.honesty || []) {
      const li = document.createElement("li");
      li.textContent = item;
      honesty.appendChild(li);
    }
  }
}

function statusClass(status) {
  const s = (status || "").toLowerCase();
  if (s.includes("implement") || s === "done") return "is-done";
  if (s.includes("not_run") || s.includes("not run")) return "is-open";
  return "is-partial"; // protocol-ready, awaiting-live-run, etc.
}

function renderProofPackage() {
  const proof = manifest?.agiProof;
  if (!proof) return;
  const boundary = document.getElementById("proof-boundary");
  if (boundary && proof.claimBoundary) boundary.textContent = proof.claimBoundary;

  const ladder = document.getElementById("proof-ladder");
  if (ladder) {
    ladder.innerHTML = "";
    for (const item of proof.proofLadder || []) {
      const card = document.createElement("article");
      card.className = "proof-card";
      card.innerHTML = `
        <div class="proof-level">Level ${item.level}</div>
        <h3>${item.name}</h3>
        <span class="proof-status ${statusClass(item.status)}">${item.status}</span>`;
      ladder.appendChild(card);
    }
  }

  const required = document.getElementById("proof-required");
  if (required) {
    required.innerHTML = "";
    for (const item of proof.requiredProofData || []) {
      const li = document.createElement("li");
      li.textContent = item;
      required.appendChild(li);
    }
  }

  const external = document.getElementById("proof-external");
  if (external) {
    const table = document.createElement("table");
    table.innerHTML = "<tr><th>Benchmark</th><th>Status</th><th>Purpose</th></tr>";
    for (const item of proof.externalBenchmarks || []) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${item.name}</td><td><span class="proof-status ${statusClass(item.status)}">${item.status}</span></td><td>${item.purpose}</td>`;
      table.appendChild(tr);
    }
    external.replaceChildren(table);
  }
}

function renderFooter() {
  const footer = document.getElementById("site-footer");
  if (footer && manifest?.version) {
    footer.textContent = `Sophia · the Wisdom Gate · v${manifest.version} · Apache 2.0 · UI decided by council panel · Wisdom before intelligence.`;
  }
}

function setupTabs() {
  const tabs = document.querySelectorAll(".agent-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      activeMode = tab.dataset.mode;
      tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === activeMode));
      document.getElementById("agent-hint").textContent =
        activeMode === "repo"
          ? "Repo mode may suggest tools; server-side execution requires API + approval on the CLI."
          : activeMode === "life"
            ? "Not a substitute for licensed medical, legal, or financial advice."
            : "Project, corpus, benchmark, and growth decisions.";
    });
  });
}

async function askAgent() {
  const q = document.getElementById("agent-question").value.trim();
  const out = document.getElementById("agent-output");
  const btn = document.getElementById("agent-submit");
  if (!q) return;
  btn.disabled = true;
  out.classList.add("visible");
  out.textContent = "Council deliberation in progress…";

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: activeMode, question: q }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || res.statusText);
    }
    const data = await res.json();
    let text = data.answer || "";
    if (data.gate) {
      const gateLine = `[Gate ${data.gate.passed ? "PASS" : "FAIL"}]`;
      const extras = [...(data.gate.warnings || []), ...(data.gate.violations || [])];
      if (extras.length) {
        text += `\n\n${gateLine} ${extras.join("; ")}`;
      } else if (data.gate.checks?.length) {
        text += `\n\n${gateLine} ${data.gate.checks.length} attribution check(s) run`;
      }
    }
    out.textContent = text;
  } catch (e) {
    out.textContent =
      `Live agent unavailable (${e.message}).\n\nRun locally:\npython tools/serve_web.py\n\nOr via CLI:\npython tools/sophia_agent.py ${activeMode} "${q.replace(/"/g, '\\"')}"`;
  }
  btn.disabled = false;
}

function setupNav() {
  const links = document.querySelectorAll("nav.toc a");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`));
        }
      });
    },
    { rootMargin: "-20% 0px -70% 0px" }
  );
  document.querySelectorAll(".chapter").forEach((ch) => observer.observe(ch));
}

document.getElementById("agent-submit")?.addEventListener("click", askAgent);
setupTabs();
setupNav();
loadManifest();
