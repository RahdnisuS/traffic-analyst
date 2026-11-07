// Dashboard controller: WebSocket connection, message dispatch, controls, render loop.
// The browser never simulates anything. It sends commands and displays server state.

import { Renderer2D } from "./render2d.js";
import { FrameBuffer } from "./interp.js";
import { Charts } from "./charts.js";

const $ = (id) => document.getElementById(id);
const frames = new FrameBuffer();
// Two interchangeable views fed by the same interpolated frames. Only the visible one draws,
// so the hidden view costs nothing. The 3D view loads lazily (see load3D).
const views = { "2d": new Renderer2D($("view2d")), "3d": null };
let mode = "2d";
let layoutArgs = null; // last init's layout, replayed into the 3D view if it loads later
const active = () => views[mode];
const charts = new Charts($("ch-queue"), $("ch-delay"));
const state = { limits: null, playing: false, speed: 1, ws: null, retry: 0, types: [] };
const MAX_EVENTS = 80;
const TYPE_NAMES = { car: "Cars", bike: "Bikes", auto: "Autos", bus: "Buses" };

// ------------------------------------------------------------------ connection
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/sim`);
  state.ws = ws;
  ws.onopen = () => { state.retry = 0; setConn("Live", "ok"); };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
  ws.onclose = () => {
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
      layoutArgs = [msg.layout, msg.vehicle_types.map((t) => msg.vehicle_dims[t]), msg.vehicle_types, msg.ped_radius];
      for (const v of Object.values(views)) v?.setLayout(...layoutArgs);
      buildMix(msg.vehicle_types);
      frames.clear();
      charts.reset(msg.history);
      $("events").innerHTML = "";
      addEvents(msg.events);
      buildSpeeds(msg.limits.speeds);
      configureSliders(msg.limits);
      applyStatus(msg);
      if (msg.history.length) updateKpis(msg.history.at(-1));
      break;
    case "frame":
      frames.push(msg, performance.now());
      $("hud-clock").textContent = clock(msg.t);
      $("hud-phase").textContent = msg.phase === "PED"
        ? `Pedestrian phase · ${{ walk: "walk", ped_clear: "finish crossing", all_red: "all-red" }[msg.stage] ?? msg.stage}`
        : `Phase ${msg.phase} · ${msg.stage.replace("_", "-")}`;
      break;
    case "metrics":
      charts.addRows(msg.rows);
      updateKpis(msg.rows.at(-1));
      break;
    case "events":
      addEvents(msg.events.filter((e) => e.type !== "phase_change"));
      break;
    case "status":
      applyStatus(msg);
      break;
    case "error":
      toast(msg.message);
      break;
  }
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
  if (s.error) toast(s.error);
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

function addEvents(list) {
  const ol = $("events");
  for (const e of list) {
    const li = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = e.clock;
    const span = document.createElement("span");
    span.textContent = describe(e); // textContent, never innerHTML: no injection from data
    li.append(time, span);
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
$("btn-reset").onclick = () => {
  const seed = Number.parseInt($("seed").value, 10);
  send({ cmd: "reset", seed: Number.isFinite(seed) && seed >= 0 ? seed : null });
};
$("zoom-in").onclick = () => active().zoomBy(1.4);
$("zoom-out").onclick = () => active().zoomBy(1 / 1.4);
$("zoom-fit").onclick = () => active().fit();
$("rot-l").onclick = () => views["3d"]?.rotate(-Math.PI / 2);
$("rot-r").onclick = () => views["3d"]?.rotate(Math.PI / 2);
$("mode-2d").onclick = () => setMode("2d");
$("mode-3d").onclick = () => setMode("3d");
document.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.code === "Space" && document.activeElement.tagName !== "BUTTON") {
    e.preventDefault();
    send({ cmd: state.playing ? "pause" : "play" });
  } else if (mode === "3d" && (e.code === "KeyQ" || e.code === "KeyE")) {
    views["3d"].rotate(e.code === "KeyQ" ? -Math.PI / 2 : Math.PI / 2);
  } else if (e.code === "KeyF") {
    $("hud-fps").hidden = !$("hud-fps").hidden; // frame-rate readout, for checking performance
  }
});

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
    views["3d"] = new Renderer3D($("view3d"));
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
  active().draw(frames.cur, frames.sample(now), frames.samplePeds(now), now);
  if (fps.last) fps.avg = 0.95 * fps.avg + 0.05 * (1000 / Math.max(now - fps.last, 1));
  fps.last = now;
  if (!$("hud-fps").hidden && now - fps.shown > 500) {
    fps.shown = now;
    $("hud-fps").textContent = `${Math.round(fps.avg)} fps · ${frames.cur?.vehicles.length ?? 0} veh`;
  }
  requestAnimationFrame(loop);
}

addEvents([]);
connect();
load3D();
requestAnimationFrame(loop);
