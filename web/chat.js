// The analyst chat (Phase 12). Chat is per browser: questions go to POST /api/chat (not the shared
// WebSocket), and the conversation lives here and is sent back with each question (last turns
// only), so follow-ups ("and at 4:00?") work.
//
// Rendering: the model's text is ESCAPED first, then a tiny markdown subset is applied (paragraphs,
// "- " bullets, **bold**) and citations [n] become buttons that highlight that source. Nothing the
// model writes is ever inserted as raw HTML.
//
// Scenario proposals are shown as a card (current -> proposed, with the model's assumptions) and
// only applied when the user presses Apply, through the same validated WebSocket commands the
// sliders send.

const HISTORY_TURNS = 3; // exchanges sent back for context (matches analyst.history_turns)
const KIND_LABEL = { run_window: "run, minute", rl_decisions: "AI decisions", run_summary: "run summary",
  doc_page: "document", live_status: "live dashboard" };
const MIX_ORDER = ["car", "bike", "auto", "bus"];

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Escaped text -> paragraphs, bullet lists, bold, citation buttons. */
function renderMarkdown(text, nSources) {
  const lines = esc(text.trim()).split(/\n/);
  let html = "", list = false, para = [];
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+(?:\s*[,;–-]\s*\d+)*)\]/g, (m, inner) => {
      const ns = inner.split(/\s*[,;]\s*/).flatMap((p) => {
        const r = p.split(/\s*[–-]\s*/).map(Number);
        return r.length === 2 && r[1] >= r[0] && r[1] - r[0] < 10 ? Array.from({ length: r[1] - r[0] + 1 }, (_, i) => r[0] + i) : [r[0]];
      });
      return ns.map((n) => (n >= 1 && n <= nSources
        ? `<button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">[${n}]</button>`
        : `[${n}]`)).join("");
    });
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  for (const raw of lines) {
    const line = raw.trim();
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    if (bullet) {
      flushPara();
      if (!list) { html += "<ul>"; list = true; }
      html += `<li>${inline(bullet[1])}</li>`;
    } else {
      if (list) { html += "</ul>"; list = false; }
      if (!line) flushPara(); else para.push(line);
    }
  }
  flushPara();
  if (list) html += "</ul>";
  return html;
}

function fmtValue(key, v) {
  if (v == null) return "–";
  if (key === "vehicle_mix") return MIX_ORDER.map((t) => `${t} ${Math.round(v[t] ?? 0)}%`).join(", ");
  if (key === "controller") return v === "rl" ? "AI" : "fixed timer";
  if (key === "aggressiveness") return Number(v).toFixed(2);
  return `${Math.round(Number(v))}`;
}

export class Chat {
  constructor(root, { send }) {
    this.root = root;
    this.send = send;
    this.log = root.querySelector("#chat-log");
    this.form = root.querySelector("#chat-form");
    this.input = root.querySelector("#chat-input");
    this.button = root.querySelector("#chat-send");
    this.suggest = root.querySelector("#chat-suggest");
    this.history = [];
    this.busy = false;
    this.status = null;
    this.form.addEventListener("submit", (e) => { e.preventDefault(); this.ask(this.input.value); });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.ask(this.input.value); }
    });
    for (const chip of this.suggest.querySelectorAll(".chip")) chip.addEventListener("click", () => this.ask(chip.textContent));
  }

  setStatus(st) {
    this.status = st;
    const note = this.root.querySelector("#analyst-note");
    const ok = !!st?.available;
    this.root.querySelector("#analyst-model").textContent = st?.model ? st.model.replace(/^openai\//, "") : "";
    if (!ok) {
      note.textContent = `The analyst is unavailable: ${st?.reason ?? "the server did not report its status."}`;
      note.classList.add("unavailable");
    }
    this.input.disabled = this.button.disabled = !ok;
    for (const c of this.suggest.querySelectorAll(".chip")) c.disabled = !ok;
  }

  focus() { this.input.focus(); }

  #add(cls, html) {
    const li = document.createElement("li");
    li.className = `msg ${cls}`;
    li.innerHTML = html;
    this.log.append(li);
    this.log.scrollTop = this.log.scrollHeight;
    return li;
  }

  async ask(text) {
    const q = text.trim();
    if (!q || this.busy || !this.status?.available) return;
    this.busy = true;
    this.button.disabled = true;
    this.input.value = "";
    this.suggest.hidden = true;
    this.#add("user", `<div class="bubble">${esc(q)}</div>`);
    const first = !this.status.loaded;
    const pending = this.#add("bot pending", `<div class="bubble">${first
      ? "Loading the search models (the first question takes about 10 s)" : "Searching the run log and documents"}</div>`);
    const history = this.history.slice(-2 * HISTORY_TURNS);
    let res;
    try {
      const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, history }) });
      res = r.ok ? await r.json() : { type: "error", message: `The server rejected the question (${r.status}).` };
    } catch {
      res = { type: "error", message: "Could not reach the server." };
    }
    this.status.loaded = true;
    pending.remove();
    if (res.type === "answer") this.#answer(res);
    else if (res.type === "scenario") this.#scenario(res);
    else this.#add("bot error", `<div class="bubble">${esc(res.message ?? "Something went wrong.")}</div>`);
    this.history.push({ role: "user", content: q });
    this.history.push({ role: "assistant", content: res.answer ?? (res.type === "scenario"
      ? `Proposed scenario: ${res.proposal?.title ?? ""}` : res.message ?? "") });
    this.busy = false;
    this.button.disabled = false;
    this.input.focus();
  }

  #answer(res) {
    const n = res.sources?.length ?? 0;
    const g = res.grounding ?? {};
    let meta = "";
    if (n) {
      meta += g.unsupported?.length
        ? `<span class="ground-warn">Not found in the sources: ${esc(g.unsupported.join(", "))}</span>`
        : `<span class="ground-ok" title="Every number in the answer appears in the sources">Numbers checked against the sources</span>`;
    }
    if (res.timing?.total_s != null) meta += `<span>${res.timing.total_s.toFixed(1)} s</span>`;
    const sources = n ? `<details class="sources"><summary>${n} source${n > 1 ? "s" : ""}</summary><ol>${
      res.sources.map((s) => this.#source(s)).join("")}</ol></details>` : "";
    const li = this.#add("bot", `<div class="bubble">${renderMarkdown(res.answer ?? "", n)}</div>
      ${meta ? `<div class="msg-meta">${meta}</div>` : ""}${sources}`);
    li.querySelectorAll(".cite").forEach((b) => b.addEventListener("click", () => {
      const det = li.querySelector(".sources");
      det.open = true;
      const row = li.querySelector(`.src[data-n="${b.dataset.n}"]`);
      row?.scrollIntoView({ block: "nearest" });
      row?.classList.add("flash");
      setTimeout(() => row?.classList.remove("flash"), 1400);
    }));
  }

  #source(s) {
    const title = s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.citation)}</a>` : esc(s.citation);
    const extra = s.section ? ` · ${esc(s.section)}` : "";
    const snippet = esc((s.text ?? "").split("\n").slice(s.source_type === "doc" ? 1 : 0).join(" ").slice(0, 220));
    return `<li class="src" data-n="${s.n}"><span class="src-n">[${s.n}]</span><span class="src-title">${title}${extra}
      <span class="src-kind">${esc(KIND_LABEL[s.kind] ?? s.kind ?? "")}</span></span><span class="src-snip">${snippet}…</span></li>`;
  }

  #scenario(res) {
    const p = res.proposal, changes = res.changes ?? [];
    if (!changes.length) {
      this.#add("bot", `<div class="bubble"><p>That matches the current settings already: nothing to change.</p></div>`);
      return;
    }
    const cell = (c) => {
      if (c.key === "vehicle_mix") { // one line per type that changes: "bike 45 → 60%"
        return MIX_ORDER.filter((t) => Math.round(c.current?.[t] ?? 0) !== Math.round(c.proposed[t] ?? 0))
          .map((t) => `${t} <span class="old">${Math.round(c.current?.[t] ?? 0)}</span>${Math.round(c.proposed[t] ?? 0)}%`)
          .join("<br>");
      }
      return `<span class="old">${esc(fmtValue(c.key, c.current))}</span>${esc(fmtValue(c.key, c.proposed))}`;
    };
    const rows = changes.map((c) => `<tr><th scope="row">${esc(c.label)}</th><td>${cell(c)}</td></tr>`).join("");
    const assumptions = (p.assumptions ?? []).map((a) => `<li>${esc(a)}</li>`).join("");
    const li = this.#add("bot", `<div class="scenario" role="group" aria-label="Proposed scenario">
      <h3>${esc(p.title ?? "Proposed scenario")}</h3>
      <table><tbody>${rows}</tbody></table>
      ${assumptions ? `<ul aria-label="Assumptions">${assumptions}</ul>` : ""}
      <div class="row"><button type="button" class="btn primary" data-act="apply">Apply</button>
        <button type="button" class="btn" data-act="cancel">Cancel</button></div></div>`);
    const card = li.querySelector(".scenario");
    const done = (msg) => {
      card.querySelector(".row").outerHTML = `<p class="scenario-done" role="status">${esc(msg)}</p>`;
    };
    card.querySelector('[data-act="apply"]').addEventListener("click", () => {
      const settings = {};
      for (const c of changes) if (c.key !== "controller") settings[c.key] = c.proposed;
      if (Object.keys(settings).length) this.send({ cmd: "settings", ...settings });
      const ctl = changes.find((c) => c.key === "controller");
      if (ctl) this.send({ cmd: "controller", value: ctl.proposed });
      done("Applied to the simulation. The sliders on the Dashboard tab show the new values.");
    });
    card.querySelector('[data-act="cancel"]').addEventListener("click", () => done("Cancelled: nothing was changed."));
  }
}
