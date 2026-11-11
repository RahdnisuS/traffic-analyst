// Dashboard controller: WebSocket connection, message dispatch, controls, render loop.
// The browser never simulates anything. It sends commands and displays server state.

import { Renderer2D } from "./render2d.js";
import { FrameBuffer } from "./interp.js";
import { Charts } from "./charts.js";
import { Chat } from "./chat.js";

const $ = (id) => document.getElementById(id);
const frames = new FrameBuffer();
// Comparison mode: the AI twin streams its own frames and gets its own pair of renderers, created
// on first use. Both panes share ONE camera (see syncCameras) so they can be compared spot for spot.
const twinFrames = new FrameBuffer();
const twinViews = { "2d": null, "3d": null };
let Renderer3DClass = null;
let camLeader = "main"; // the pane the user last panned/zoomed: the other one copies its camera
// Two interchangeable views fed by the same interpolated frames. Only the visible one draws,
// so the hidden view costs nothing. The 3D view loads lazily (see load3D).
const views = { "2d": new Renderer2D($("view2d")), "3d": null };
let mode = "2d";
let layoutArgs = null; // last init's layout, replayed into the 3D view if it loads later
const active = () => views[mode];
const charts = new Charts($("ch-queue"), $("ch-delay"));
// This tab's session id. On the public demo each tab gets its own simulation (the server keys it
// by this id); sessionStorage survives a reload, so a reload reconnects to the same sim, while a
// new tab starts a new one. Locally the server ignores it (one shared sim).
const SESSION = (() => {
  const fresh = () => (crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`);
  try {
    let id = sessionStorage.getItem("ta-session");
    if (!id) sessionStorage.setItem("ta-session", (id = fresh()));
    return id;
  } catch {
    return fresh();
  }
})();
const chat = new Chat($("tab-analyst"), { send, session: SESSION });
const state = { limits: null, playing: false, speed: 1, ws: null, retry: 0, types: [],
  controller: "fixed", aiAvailable: false, comparing: false, seed: null };
const CONTROLLER_NAMES = { fixed: "fixed timer", rl: "AI" };
const MAX_EVENTS = 80;
const TYPE_NAMES = { car: "Cars", bike: "Bikes", auto: "Autos", bus: "Buses" };

// ------------------------------------------------------------------ connection
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/sim?session=${encodeURIComponent(SESSION)}`);
  state.ws = ws;
  state.busy = false;
  ws.onopen = () => { state.retry = 0; setConn("Live", "ok"); };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
  ws.onclose = () => {
    if (state.busy) {
      setConn("Demo busy, retrying…", "bad");
      setTimeout(connect, 20000); // every slot taken: someone else's sim must close first
      return;
    }
    setConn("Disconnected, retrying…", "bad");
    // Exponential backoff (0.5 s, 1 s, 2 s … capped at 8 s) so a restarting server isn't hammered.
    const wait = Math.min(8000, 500 * 2 ** state.retry++);
    setTimeout(connect, wait);
  };
}

function send(cmd) {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(cmd));
}

function setConn(text, cls) {
  const el = $("conn");
  el.textContent = text;
  el.className = `conn ${cls}`;
}

// ------------------------------------------------------------------ messages
function handle(msg) {
  switch (msg.type) {
    case "init":
      state.limits = msg.limits;
      chat.setStatus(msg.analyst);
      layoutArgs = [msg.layout, msg.vehicle_types.map((t) => msg.vehicle_dims[t]), msg.vehicle_types, msg.ped_radius];
      for (const v of [...Object.values(views), ...Object.values(twinViews)]) v?.setLayout(...layoutArgs);
      buildMix(msg.vehicle_types);
      frames.clear();
      twinFrames.clear();
      state.predictor = msg.predictor;
      charts.setForecast(msg.predictor?.horizon_s);
      showForecastNote();
      charts.reset(msg.history, msg.twin_history, msg.forecast_history);
      $("events").innerHTML = "";
      addEvents(msg.events, msg.twin_events ? "Fixed" : null);
      if (msg.twin_events) addEvents(msg.twin_events, "AI");
      buildSpeeds(msg.limits.speeds);
      showBenchmark(msg.benchmark);
      configureSliders(msg.limits);
      applyStatus(msg);
      if (msg.history.length) updateKpis(msg.history.at(-1));
      break;
    case "frame":
      frames.push(msg, performance.now());
      if (msg.twin) {
        twinFrames.push(msg.twin, performance.now());
        $("twin-phase").textContent = phaseText(msg.twin);
      }
      updateStressProgress(msg.t);
      $("hud-clock").textContent = clock(msg.t);
      $("hud-phase").textContent = phaseText(msg);
      break;
    case "metrics":
      charts.addRows(msg.rows, msg.twin_rows, msg.forecast);
      if (msg.rows.length) updateKpis(msg.rows.at(-1));
      if (msg.compare) showCompare(msg.compare);
      break;
    case "events":
      addEvents(msg.events.filter((e) => e.type !== "phase_change"), msg.twin_events ? "Fixed" : null);
      if (msg.twin_events) addEvents(msg.twin_events.filter((e) => e.type !== "phase_change"), "AI");
      break;
    case "status":
      applyStatus(msg);
      break;
    case "error":
      setSwitchBusy(false);
      toast(msg.message);
      break;
    case "busy":
      state.busy = true;
      toast(msg.message);
      break;
  }
}

function phaseText(f) {
  return f.phase === "PED"
    ? `Pedestrian phase · ${{ walk: "walk", ped_clear: "finish crossing", all_red: "all-red" }[f.stage] ?? f.stage}`
    : `Phase ${f.phase} · ${f.stage.replace("_", "-")}`;
}

function applyStatus(s) {
  state.playing = s.playing;
  state.speed = s.speed;
  const play = $("btn-play");
  play.textContent = s.playing ? "Pause" : "Play";
  play.setAttribute("aria-pressed", String(s.playing));
  if (document.activeElement !== $("seed")) $("seed").value = s.seed;
  for (const b of $("speeds").children) b.setAttribute("aria-pressed", String(Number(b.dataset.v) === s.speed));
  const st = s.settings;
  setSlider("sl-volume", st.volume_vph);
  setSlider("sl-aggr", st.aggressiveness);
  if (st.green_s != null) setSlider("sl-green", st.green_s);
  if (st.ped_rate_ph != null) setSlider("sl-peds", st.ped_rate_ph);
  if (st.vehicle_mix) for (const [t, w] of Object.entries(st.vehicle_mix)) {
    const el = $(`mix-${t}`);
    if (el && document.activeElement !== el) el.value = w;
  }
  updateMixLabels();
  applyStress(s.stress);
  applyAi(s);
  if (s.error) toast(s.error);
}

// ------------------------------------------------------------------ AI control & comparison
// The switch keeps one accessible name ("AI signal control"); loading is said in the note below it.
function setSwitchBusy(busy) {
  $("ai-switch").setAttribute("aria-busy", String(busy));
  if (busy) $("ai-note").textContent = "Loading the AI model… (a few seconds, first time only)";
}

function applyAi(s) {
  state.controller = s.controller ?? "fixed";
  state.aiAvailable = !!s.ai_available;
  state.seed = s.seed;
  setSwitchBusy(false);
  const comparing = !!s.compare;
  const stressing = !!s.stress?.active;
  const sw = $("ai-switch");
  sw.setAttribute("aria-checked", String(state.controller === "rl"));
  sw.disabled = !state.aiAvailable || comparing;
  // One place to stop a comparison (the Keep buttons in its panel); here it just says it's running.
  const cmp = $("btn-compare");
  cmp.disabled = !state.aiAvailable || stressing || comparing;
  cmp.textContent = comparing ? "Comparing…" : "Compare side by side";
  // The comparison and the stress test exclude each other (the server refuses both ways).
  $("btn-stress").disabled = comparing;
  const tag = $("hud-ctrl");
  const aiDrives = !comparing && state.controller === "rl";
  tag.textContent = aiDrives ? "AI" : "Fixed timer";
  tag.classList.toggle("ai", aiDrives);

  const note = $("ai-note");
  if (!state.aiAvailable) note.innerHTML = "No trained AI yet. Run <code>python -m rl.train</code>, then restart.";
  else if (comparing) note.textContent = "Both controllers are running on the same traffic.";
  else if (aiDrives) note.textContent = "The AI decides when each green ends, checking every 5 s. The green-time slider applies to the fixed timer.";
  else note.textContent = "";
  setComparing(comparing);
  if (comparing) showCompare(s.compare);
}

/** The queue chart's note on the dashed lines: what they are and how far to trust them (the
 *  model's error on held-out runs next to the naive "same as now" forecast). */
function showForecastNote() {
  const p = state.predictor, el = $("ch-queue-note");
  el.hidden = !p || state.comparing;
  if (!p) return;
  const mins = Math.round(p.horizon_s / 60);
  el.textContent = `Dashed: forecast ${mins} min ahead (LightGBM). On average off by ${p.mae.toFixed(1)} vehicles, ` +
    `vs ${p.persistence_mae.toFixed(1)} for "same as now".`;
  $("ch-queue").setAttribute("aria-label", `Line chart of queued vehicles per approach over the last 5 minutes, ` +
    `with dashed forecasts ${mins} minutes ahead`);
}

function setComparing(on) {
  if (on === state.comparing) return;
  state.comparing = on;
  if (on) ensureTwinViews();
  $("stage").classList.toggle("comparing", on);
  $("view-twin").hidden = !on;
  $("compare").hidden = !on;
  $("kpis").hidden = on; // the comparison panel carries the numbers while it runs
  $("ch-queue-title").textContent = on ? "Queued vehicles, all approaches" : "Queued vehicles by approach";
  showForecastNote();
  if (!on) {
    twinFrames.clear();
    state.verdict = null;
    // The Keep button that was just pressed has disappeared with its panel: don't strand focus.
    if (state.refocus) $("btn-compare").focus();
  }
  state.refocus = false;
  camLeader = "main";
}

function ensureTwinViews() {
  if (!twinViews["2d"]) {
    twinViews["2d"] = new Renderer2D($("twin2d"));
    bindLeader($("twin2d"), "twin");
  }
  if (Renderer3DClass && !twinViews["3d"]) {
    twinViews["3d"] = new Renderer3DClass($("twin3d"));
    bindLeader($("twin3d"), "twin");
  }
  for (const v of Object.values(twinViews)) if (v && layoutArgs && !v.layout) v.setLayout(...layoutArgs);
  $("twin3d").hidden = mode !== "3d";
  $("twin2d").hidden = mode !== "2d";
}

/** Whichever pane the user touches leads; the other copies its camera every frame. */
function bindLeader(canvas, who) {
  for (const ev of ["pointerdown", "wheel"]) {
    canvas.addEventListener(ev, () => { camLeader = who; }, { capture: true, passive: true });
  }
}

function syncCameras(main, twin) {
  const [src, dst] = camLeader === "main" ? [main, twin] : [twin, main];
  dst.zoom = src.zoom;
  if (src.pan) dst.pan = { ...src.pan };                       // 2D
  if (src.target) {                                            // 3D
    dst.target = { ...src.target };
    dst.azimuth = src.azimuth;
    dst.spin = src.spin ? { ...src.spin } : null;
  }
}

const WARMUP_S = 60;
const pct = (v) => `${Math.abs(v).toFixed(0)}%`;

/** Lower is better for every compared measure. Within 1 % counts as "about the same". */
function verdict(change) {
  if (change == null) return null;
  if (Math.abs(change) < 1) return "same";
  return change < 0 ? "better" : "worse";
}

/** "5% less" / "3% more" / "about the same": the change said in words, never a bare signed number. */
function inWords(change, more = "more", less = "less") {
  const v = verdict(change);
  return v === "same" ? "about the same" : `${pct(change)} ${v === "better" ? less : more}`;
}

// Every pair reads in pane order: fixed timer first, then the AI.
function showCompare(c) {
  if (!c) return;
  $("cmp-seed").textContent = state.seed ?? "–";
  const f = c.fixed, a = c.rl, ch = c.change_pct;
  const ready = c.t >= WARMUP_S && ch.waiting != null; // the first minute is mostly empty road: noise
  const vmin = (s) => `${Math.round(s / 60).toLocaleString()} veh·min`;
  const hero = $("cmp-hero");
  const v = ready ? verdict(ch.waiting) : null;
  hero.dataset.verdict = v ?? "";
  if (ready) {
    hero.textContent = v === "same" ? "About the same waiting" : `${inWords(ch.waiting)} waiting`;
    $("cmp-hero-label").textContent = "with the AI than with the fixed timer, counting every vehicle-second of waiting";
    $("cmp-hero-sub").textContent = `Fixed timer ${vmin(f.waiting_veh_s)} → AI ${vmin(a.waiting_veh_s)} since 0:00`;
  } else {
    hero.textContent = "Measuring…";
    $("cmp-hero-label").textContent = "Total time vehicles spend waiting, fixed timer vs AI";
    $("cmp-hero-sub").textContent = `Results appear after ${clock(WARMUP_S)} of simulated time.`;
  }
  // Screen readers hear the verdict once when it appears and again only when it changes,
  // not every second (the numbers update at 1 Hz).
  if (v && v !== state.verdict) {
    $("cmp-announce").textContent = `Comparison: ${hero.textContent} with the AI than with the fixed timer.`;
  }
  state.verdict = v;
  const tile = (id, fixed, ai, change) => {
    $(id).textContent = fixed == null || ai == null ? "–" : `${fixed.toFixed(1)} → ${ai.toFixed(1)} s`;
    $(`${id}-sub`).textContent = fixed == null || ai == null ? "fixed timer → AI"
      : ready && change != null ? `fixed timer → AI · ${inWords(change, "longer", "shorter")}` : "fixed timer → AI";
  };
  tile("cmp-delay", f.mean_delay_s, a.mean_delay_s, ch.delay);
  tile("cmp-ped", f.mean_ped_wait_s, a.mean_ped_wait_s, ch.ped_wait);
  $("cmp-now").textContent = `${f.waiting_now} → ${a.waiting_now}`;
}

// ------------------------------------------------------------------ capacity / stress test
const fmt = (v) => Math.round(v).toLocaleString();

/** The official score: the latest saved `python -m bench` result (sent in `init`). */
function showBenchmark(b) {
  const el = $("bench-official");
  if (!b?.score) {
    el.innerHTML = "No benchmark saved yet. Run <code>python -m bench</code>.";
    return;
  }
  const sc = b.score;
  el.textContent = "";
  const strong = document.createElement("strong");
  strong.textContent = `${sc.censored ? "≥ " : ""}${fmt(sc.mean_vph)} ± ${fmt(sc.std_vph)} veh/h`;
  el.append(strong, document.createElement("br"),
    `Benchmark over ${sc.n} seeds (${new Date(b.created_at).toLocaleDateString()}). ` +
    `Peak while overloaded: ${fmt(sc.peak_served_mean_vph)} veh/h.`);
  el.title = b.definition;
}

function applyStress(st) {
  state.stress = st;
  const running = !!st?.active;
  const btn = $("btn-stress");
  btn.textContent = running ? "Stop stress test" : "Run stress test";
  btn.classList.toggle("primary", running);
  // The ramp owns the demand while it runs: settings are locked (the server enforces it too).
  document.querySelector(".controls").classList.toggle("locked", running);
  for (const el of document.querySelectorAll(".controls .slider input")) el.disabled = running;

  const list = $("stress-steps");
  list.hidden = !st?.steps?.length;
  list.innerHTML = "";
  for (const s of st?.steps ?? []) {
    const li = document.createElement("li");
    const left = document.createElement("span");
    left.textContent = `${fmt(s.volume_vph)} veh/h per approach: served ${fmt(s.served_vph)} of ${fmt(s.offered_vph)}`;
    const right = document.createElement("span");
    right.className = s.passed ? "ok" : "bad";
    // Words as well as colour, never colour alone.
    right.textContent = s.passed ? "kept up"
      : `overloaded${s.failing_approaches.length ? ` (${s.failing_approaches.join(", ")})` : ""}`;
    li.append(left, right);
    list.append(li);
  }
  const live = $("stress-live");
  if (!running && st?.result) {
    const r = st.result;
    live.hidden = false;
    live.textContent = r.capacity_vph == null
      ? `Stress test ${r.reason === "stopped" ? "stopped" : "ended"}: no step kept up.`
      : `Kept up to ${fmt(r.capacity_level_vph)} veh/h per approach` +
        (r.overloaded_level_vph != null ? `, clearly overloaded at ${fmt(r.overloaded_level_vph)}` : "") +
        (r.reason === "stopped" ? " (stopped early)" : r.censored ? " (never overloaded)" : "") +
        `. Volume restored to ${fmt(r.restored_vph)}. ` +
        "Short live steps only catch clear overload, so this reads higher than the benchmark score.";
  } else live.hidden = !running;
}

/** Progress within the current ramp step, from the sim clock in each frame. */
function updateStressProgress(t) {
  const st = state.stress;
  if (!st?.active) return;
  const into = Math.max(0, t - st.step_start_t);
  const live = $("stress-live");
  live.textContent = `Step ${st.step}: ${fmt(st.level_vph)} veh/h per approach · ` +
    (into >= st.settle_s ? `measuring ${clock(into - st.settle_s)} of ${clock(st.step_s - st.settle_s)}`
                         : `settling ${clock(into)}`);
  const bar = document.createElement("span");
  bar.className = "bar";
  const fill = document.createElement("i");
  fill.style.width = `${Math.min(100, (100 * into) / st.step_s)}%`;
  bar.append(fill);
  live.append(bar);
}

// ------------------------------------------------------------------ KPIs
const LOS_STATUS = { A: ["--good", "free flow"], B: ["--good", "stable"], C: ["--warning", "stable"],
  D: ["--warning", "approaching unstable"], E: ["--serious", "unstable"], F: ["--critical", "breakdown"] };

function updateKpis(r) {
  if (!r) return;
  $("kpi-clock").textContent = r.clock;
  $("kpi-veh").textContent = r.in_network;
  $("kpi-backlog").textContent = r.backlog ? `+${r.backlog} waiting to enter` : "on the road";
  $("kpi-thru").textContent = Math.round(r.throughput_vph).toLocaleString();
  const hasDelay = r.mean_delay_window !== "" && r.mean_delay_window != null;
  $("kpi-delay").textContent = hasDelay ? `${Number(r.mean_delay_window).toFixed(1)} s` : "–";
  const badge = $("kpi-los");
  if (r.los) {
    const [color, words] = LOS_STATUS[r.los];
    badge.hidden = false;
    badge.textContent = `LOS ${r.los} · ${words}`; // letter + words: never colour alone
    badge.style.setProperty("--los-color", `var(${color})`);
  } else badge.hidden = true;
  $("kpi-peds").textContent = r.peds_in_network ?? 0;
  $("kpi-peds-sub").textContent = r.peds_crossing ? `${r.peds_crossing} crossing, ${r.peds_waiting} waiting`
    : r.peds_waiting ? `${r.peds_waiting} waiting (longest ${Math.round(r.ped_longest_wait)} s)` : "none waiting";
  const pw = r.ped_wait_window;
  $("kpi-pwait").textContent = pw !== "" && pw != null ? `${Number(pw).toFixed(0)} s` : "–";
}

// ------------------------------------------------------------------ events list
function describe(e) {
  const on = e.status === "start";
  switch (e.type) {
    case "queue_spike":
      return on ? `Queue spike on ${e.approach}: ${e.value} queued` +
                  (e.approach_light === "R" && e.seconds_since_approach_green != null ? `, red for ${Math.round(e.seconds_since_approach_green)} s` : "")
                : `${e.approach} queue cleared (peak ${e.peak} at ${e.peak_clock}, ${Math.round(e.duration_s)} s)`;
    case "spillback":
      return on ? `Spillback on ${e.lane}: queue ${e.queue_m} m long` : `Spillback on ${e.lane} ended after ${Math.round(e.duration_s)} s`;
    case "gridlock":
      return on ? `GRIDLOCK: ${e.in_network} vehicles, none leaving` : `Gridlock cleared after ${Math.round(e.duration_s)} s`;
    case "ped_phase":
      return `Pedestrian phase: ${e.waiting} crossing` + (e.waiting ? `, longest wait ${Math.round(e.max_wait)} s` : "");
    case "stress_start":
      return `Stress test started: ${fmt(e.start_vph)} veh/h per approach, +${fmt(e.step_vph)} every ${clock(e.step_s)}`;
    case "stress_step":
      return `Stress step ${e.step} (${fmt(e.volume_vph)} veh/h per approach): served ${fmt(e.served_vph)} of ` +
             `${fmt(e.offered_vph)} veh/h, ${e.passed ? "kept up" : "overloaded"}`;
    case "stress_result":
      return e.capacity_vph == null ? `Stress test ${e.reason}: no step kept up`
        : `Stress test ${e.reason === "stopped" ? "stopped" : "finished"}: kept up to ${fmt(e.capacity_level_vph)} ` +
          `veh/h per approach` + (e.overloaded_level_vph != null ? `, overloaded at ${fmt(e.overloaded_level_vph)}` : "");
    case "controller_change":
      return `Signal control: ${CONTROLLER_NAMES[e.from] ?? e.from} → ${CONTROLLER_NAMES[e.to] ?? e.to}`;
    case "los_change":
      return `Level of Service ${e.from ?? "–"} → ${e.to} (delay ${e.mean_delay_window} s)`;
    case "config_change": {
      const c = e.changes, parts = [];
      if (c.per_approach_vph) parts.push(`volume ${Object.entries(c.per_approach_vph).map(([a, v]) => `${a} ${Math.round(v)}`).join(", ")} veh/h`);
      if (c.aggressiveness != null) parts.push(`aggressiveness ${c.aggressiveness}`);
      if (c.green_s != null) parts.push(`green ${c.green_s} s`);
      if (c.ped_rate_ph != null) parts.push(`pedestrians ${Math.round(c.ped_rate_ph)}/h`);
      if (c.vehicle_mix) parts.push(`mix ${Object.entries(c.vehicle_mix).filter(([, w]) => w > 0)
        .map(([t, w]) => `${Math.round(w * 100)}% ${t}`).join(", ")}`);
      return `Settings changed: ${parts.join("; ")}`;
    }
    default:
      return e.type;
  }
}

/** `side` labels events in comparison mode ("Fixed" / "AI"); null outside it. */
function addEvents(list, side = null) {
  const ol = $("events");
  for (const e of list) {
    // The ramp's own volume changes are logged, but the stress-step lines already say it here.
    if (e.type === "config_change" && e.source === "stress_test") continue;
    const li = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = e.clock;
    const span = document.createElement("span");
    span.textContent = describe(e); // textContent, never innerHTML: no injection from data
    li.append(time);
    if (side) {
      const tag = document.createElement("b");
      tag.className = `side-tag${side === "AI" ? " ai" : ""}`;
      tag.textContent = side;
      li.append(tag);
    }
    li.append(span);
    ol.prepend(li);
  }
  while (ol.children.length > MAX_EVENTS) ol.lastChild.remove();
  if (!ol.children.length) ol.innerHTML = '<li class="empty">No notable events yet.</li>';
  else ol.querySelector(".empty")?.remove();
}

// ------------------------------------------------------------------ controls
function buildSpeeds(speeds) {
  const box = $("speeds");
  box.innerHTML = "";
  for (const v of speeds) {
    const b = document.createElement("button");
    b.dataset.v = v;
    b.textContent = `${v}×`;
    b.setAttribute("aria-pressed", "false");
    b.onclick = () => send({ cmd: "speed", value: v });
    box.append(b);
  }
}

// Vehicle mix: one slider per type. Values are weights (the server normalises), shown as shares.
function buildMix(types) {
  if (types.join() === state.types.join()) return;
  state.types = types;
  const box = $("mix-sliders");
  box.innerHTML = "";
  for (const t of types) {
    const wrap = document.createElement("div");
    wrap.className = "slider";
    const label = document.createElement("label");
    label.htmlFor = `mix-${t}`;
    label.append(TYPE_NAMES[t] ?? t, " ");
    const out = document.createElement("output");
    out.id = `mixout-${t}`;
    label.append(out);
    const input = document.createElement("input");
    Object.assign(input, { id: `mix-${t}`, type: "range", min: 0, max: 100, step: 5 });
    input.addEventListener("input", updateMixLabels);
    input.addEventListener("change", () => {
      const mix = Object.fromEntries(state.types.map((k) => [k, Number($(`mix-${k}`).value)]));
      if (Object.values(mix).some((w) => w > 0)) send({ cmd: "settings", vehicle_mix: mix });
      else toast("At least one vehicle type needs a share above 0%.");
    });
    wrap.append(label, input);
    box.append(wrap);
  }
}

function updateMixLabels() {
  const vals = state.types.map((t) => Number($(`mix-${t}`)?.value ?? 0));
  const total = vals.reduce((a, b) => a + b, 0);
  state.types.forEach((t, i) => {
    const out = $(`mixout-${t}`);
    if (out) out.textContent = total ? `${Math.round((100 * vals[i]) / total)}%` : "0%";
  });
}

function configureSliders(limits) {
  $("sl-volume").max = limits.volume_vph[1];
  if (limits.ped_rate_ph) $("sl-peds").max = limits.ped_rate_ph[1];
  $("sl-green").min = limits.green_s[0];
  $("sl-green").max = limits.green_s[1];
}

const labels = {
  "sl-volume": (v) => `${Math.round(v)} veh/h per approach`,
  "sl-aggr": (v) => `${v < 0.34 ? "Timid" : v < 0.67 ? "Normal" : "Aggressive"} (${Number(v).toFixed(2)})`,
  "sl-green": (v) => {
    const L = state.limits;
    if (!L) return `${v} s`;
    const cycle = L.n_phases * (Number(v) + L.amber_s + L.all_red_s);
    // An exclusive pedestrian phase, when called, adds WALK + clearance + all-red once per cycle.
    const ped = L.walk_s + L.ped_clear_s + L.all_red_s;
    return `${v} s · cycle ${cycle} s (+${Math.round(ped)} s with walkers)`;
  },
  "sl-peds": (v) => `${Math.round(v)} per hour`,
};
const outputs = { "sl-volume": "out-volume", "sl-aggr": "out-aggr", "sl-green": "out-green", "sl-peds": "out-peds" };

function setSlider(id, value) {
  const el = $(id);
  if (document.activeElement === el) return; // don't fight the user's thumb
  el.value = value;
  $(outputs[id]).textContent = labels[id](el.value);
}

for (const [id, key] of [["sl-volume", "volume_vph"], ["sl-aggr", "aggressiveness"], ["sl-green", "green_s"],
                         ["sl-peds", "ped_rate_ph"]]) {
  const el = $(id);
  el.addEventListener("input", () => { $(outputs[id]).textContent = labels[id](el.value); });
  // Send on release ("change"), not on every drag pixel: one validated command per adjustment.
  el.addEventListener("change", () => send({ cmd: "settings", [key]: Number(el.value) }));
}

$("btn-play").onclick = () => send({ cmd: state.playing ? "pause" : "play" });
$("ai-switch").onclick = () => {
  const want = state.controller === "rl" ? "fixed" : "rl";
  if (want === "rl") setSwitchBusy(true); // the first switch loads the network (a few seconds)
  send({ cmd: "controller", value: want });
};
$("btn-compare").onclick = () => send({ cmd: "compare", action: "start" });
for (const id of ["btn-keep-fixed", "btn-keep-rl"]) {
  $(id).onclick = (e) => {
    state.refocus = true;
    send({ cmd: "compare", action: "stop", keep: e.currentTarget.dataset.keep });
  };
}
$("btn-stress").onclick = () => send({ cmd: "stress", action: state.stress?.active ? "stop" : "start" });
$("btn-reset").onclick = () => {
  const seed = Number.parseInt($("seed").value, 10);
  send({ cmd: "reset", seed: Number.isFinite(seed) && seed >= 0 ? seed : null });
};
// The view tools act on the left pane, which then leads the shared camera.
const lead = (fn) => () => { camLeader = "main"; fn(); };
$("zoom-in").onclick = lead(() => active().zoomBy(1.4));
$("zoom-out").onclick = lead(() => active().zoomBy(1 / 1.4));
$("zoom-fit").onclick = lead(() => active().fit());
$("rot-l").onclick = lead(() => views["3d"]?.rotate(-Math.PI / 2));
$("rot-r").onclick = lead(() => views["3d"]?.rotate(Math.PI / 2));
$("mode-2d").onclick = () => setMode("2d");
$("mode-3d").onclick = () => setMode("3d");
document.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.code === "Space" && document.activeElement.tagName !== "BUTTON") {
    e.preventDefault();
    send({ cmd: state.playing ? "pause" : "play" });
  } else if (mode === "3d" && (e.code === "KeyQ" || e.code === "KeyE")) {
    camLeader = "main";
    views["3d"].rotate(e.code === "KeyQ" ? -Math.PI / 2 : Math.PI / 2);
  } else if (e.code === "KeyA" && !$("ai-switch").disabled) {
    $("ai-switch").click(); // A: AI on/off
  } else if (e.code === "KeyC" && !state.comparing && !$("btn-compare").disabled) {
    $("btn-compare").click(); // C: start a comparison (stopping means choosing a side: use the buttons)
  } else if (e.key === "/") {
    e.preventDefault(); // /: ask the analyst
    setTab("analyst");
    chat.focus();
  } else if (e.code === "KeyF") {
    $("hud-fps").hidden = !$("hud-fps").hidden; // frame-rate readout, for checking performance
  }
});

// ------------------------------------------------------------------ side-panel tabs
// Dashboard (controls, numbers, charts, events) and Analyst (chat). The junction view stays visible
// with either. Remembered per browser, like the view mode.
const TABS = { dash: ["tab-btn-dash", "tab-dash"], analyst: ["tab-btn-analyst", "tab-analyst"] };
function setTab(name, focus = false) {
  for (const [key, [btn, panel]] of Object.entries(TABS)) {
    const on = key === name;
    $(btn).setAttribute("aria-selected", String(on));
    $(btn).tabIndex = on ? 0 : -1;
    $(panel).hidden = !on;
  }
  if (focus) $(TABS[name][0]).focus();
  try { localStorage.setItem("side-tab", name); } catch { /* not essential */ }
}
$("tab-btn-dash").onclick = () => setTab("dash");
$("tab-btn-analyst").onclick = () => setTab("analyst");
for (const id of ["tab-btn-dash", "tab-btn-analyst"]) {
  $(id).addEventListener("keydown", (e) => { // arrow keys move between tabs (WAI-ARIA tabs pattern)
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      setTab(id === "tab-btn-dash" ? "analyst" : "dash", true);
    }
  });
}
{
  // ?tab=analyst opens the chat directly (a shareable link), else the remembered tab.
  let initial = new URLSearchParams(location.search).get("tab");
  if (!TABS[initial]) { try { initial = localStorage.getItem("side-tab"); } catch { initial = null; } }
  if (initial === "analyst") setTab("analyst");
}

// ------------------------------------------------------------------ view mode
const store = {
  // Remembered view choice. Storage can be blocked (private mode), so failures are ignored.
  get: () => { try { return localStorage.getItem("view-mode"); } catch { return null; } },
  set: (v) => { try { localStorage.setItem("view-mode", v); } catch { /* not essential */ } },
};

function setMode(m) {
  if (m === "3d" && !views["3d"]) return;
  mode = m;
  $("view3d").hidden = m !== "3d";
  $("view2d").hidden = m !== "2d";
  $("twin3d").hidden = m !== "3d";
  $("twin2d").hidden = m !== "2d";
  if (m === "3d" && state.comparing) ensureTwinViews();
  $("legend-3d").hidden = m !== "3d";
  $("legend-2d").hidden = m !== "2d";
  for (const b of document.querySelectorAll(".only-3d")) b.hidden = m !== "3d";
  $("mode-3d").setAttribute("aria-pressed", String(m === "3d"));
  $("mode-2d").setAttribute("aria-pressed", String(m === "2d"));
  store.set(m);
}

/**
 * Load Three.js and the 3D renderer on demand. A dynamic import means a CDN outage or a
 * browser without WebGL only costs the 3D view: the dashboard keeps running in 2D.
 */
async function load3D() {
  try {
    const { Renderer3D } = await import("./render3d.js");
    Renderer3DClass = Renderer3D;
    views["3d"] = new Renderer3D($("view3d"));
    bindLeader($("view3d"), "main");
    if (layoutArgs) views["3d"].setLayout(...layoutArgs);
    $("mode-3d").disabled = false;
    // 3D is the default; 2D is the debug view. ?view=2d|3d in the URL overrides the remembered choice.
    const wanted = new URLSearchParams(location.search).get("view") ?? store.get();
    if (wanted !== "2d") setMode("3d");
  } catch (err) {
    console.warn("3D view unavailable:", err);
    $("mode-3d").title = "3D view unavailable (WebGL or Three.js failed to load)";
    toast("3D view unavailable here, showing the 2D view.");
  }
}

let toastTimer;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 4000);
}

function clock(t) { const s = Math.max(0, Math.floor(t)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

// ------------------------------------------------------------------ render loop
const fps = { last: 0, avg: 60, shown: 0 };
function loop(now) {
  const twin = state.comparing ? twinViews[mode] : null;
  if (twin) syncCameras(active(), twin);
  active().draw(frames.cur, frames.sample(now), frames.samplePeds(now), now);
  if (twin && twinFrames.cur) twin.draw(twinFrames.cur, twinFrames.sample(now), twinFrames.samplePeds(now), now);
  if (fps.last) fps.avg = 0.95 * fps.avg + 0.05 * (1000 / Math.max(now - fps.last, 1));
  fps.last = now;
  if (!$("hud-fps").hidden && now - fps.shown > 500) {
    fps.shown = now;
    $("hud-fps").textContent = `${Math.round(fps.avg)} fps · ${frames.cur?.vehicles.length ?? 0} veh`;
  }
  requestAnimationFrame(loop);
}

bindLeader($("view2d"), "main");
addEvents([]);
connect();
load3D();
requestAnimationFrame(loop);
