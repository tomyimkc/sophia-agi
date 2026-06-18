const MODES = ["advisor", "repo", "life"];
let activeMode = "advisor";
let manifest = null;

async function loadManifest() {
  try {
    const res = await fetch("data/manifest.json");
    manifest = await res.json();
    renderStats();
    renderLeaderboards();
  } catch {
    document.getElementById("stats").innerHTML =
      "<p class='agent-hint'>Run <code>python tools/build_web_data.py</code> and serve via <code>python tools/serve_web.py</code>.</p>";
  }
}

function renderStats() {
  if (!manifest) return;
  document.getElementById("stats").innerHTML = `
    <div class="stat-card"><div class="stat-value">v${manifest.version}</div><div class="stat-label">Release</div></div>
    <div class="stat-card"><div class="stat-value">${manifest.trainingExamples}</div><div class="stat-label">Training examples</div></div>
    <div class="stat-card"><div class="stat-value">4</div><div class="stat-label">Active domains</div></div>`;
}

function renderLeaderboards() {
  const root = document.getElementById("leaderboards");
  if (!manifest?.leaderboards) return;
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
          <div class="score-bar"><div class="score-fill" style="width:${entry.score_pct}%"></div></div>
        </td>
        <td>${entry.passed}/${entry.total}</td>`;
      table.appendChild(tr);
    }
    section.appendChild(table);
    root.appendChild(section);
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
          ? "Repo mode may suggest tools; server-side execution requires API + approval on CLI."
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
      `Live agent unavailable (${e.message}).\n\nRun locally:\npython tools/serve_web.py\n\nOr CLI:\npython tools/sophia_agent.py ${activeMode} "${q.replace(/"/g, '\\"')}"`;
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