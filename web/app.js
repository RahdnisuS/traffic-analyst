// Dashboard controller: WebSocket connection, message dispatch, controls, render loop.
// The browser never simulates anything. It sends commands and displays server state.

import { Renderer2D } from "./render2d.js";
import { FrameBuffer } from "./interp.js";
import { Charts } from "./charts.js";

const $ = (id) => document.getElementById(id);
const frames = new FrameBuffer();
const renderer = new Renderer2D($("view2d"));
const charts = new Charts($("ch-queue"), $("ch-delay"));
const state = { limits: null, playing: false, speed: 1, ws: null, retry: 0 };
const MAX_EVENTS = 80;

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
      renderer.setLayout(msg.layout, msg.vehicle_types.map((t) => msg.vehicle_dims[t]));
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
      $("hud-phase").textContent = `Phase ${msg.phase} · ${msg.stage.replace("_", "-")}`;
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
    case "los_change":
      return `Level of Service ${e.from ?? "–"} → ${e.to} (delay ${e.mean_delay_window} s)`;
    case "config_change": {
      const c = e.changes, parts = [];
      if (c.per_approach_vph) parts.push(`volume ${Object.entries(c.per_approach_vph).map(([a, v]) => `${a} ${Math.round(v)}`).join(", ")} veh/h`);
      if (c.aggressiveness != null) parts.push(`aggressiveness ${c.aggressiveness}`);
      if (c.green_s != null) parts.push(`green ${c.green_s} s`);
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

function configureSliders(limits) {
  $("sl-volume").max = limits.volume_vph[1];
  $("sl-green").min = limits.green_s[0];
  $("sl-green").max = limits.green_s[1];
}

const labels = {
  "sl-volume": (v) => `${Math.round(v)} veh/h per approach`,
  "sl-aggr": (v) => `${v < 0.34 ? "Timid" : v < 0.67 ? "Normal" : "Aggressive"} (${Number(v).toFixed(2)})`,
  "sl-green": (v) => {
    const L = state.limits;
    return L ? `${v} s · cycle ${L.n_phases * (Number(v) + L.amber_s + L.all_red_s)} s` : `${v} s`;
  },
};
const outputs = { "sl-volume": "out-volume", "sl-aggr": "out-aggr", "sl-green": "out-green" };

function setSlider(id, value) {
  const el = $(id);
  if (document.activeElement === el) return; // don't fight the user's thumb
  el.value = value;
  $(outputs[id]).textContent = labels[id](el.value);
}

for (const [id, key] of [["sl-volume", "volume_vph"], ["sl-aggr", "aggressiveness"], ["sl-green", "green_s"]]) {
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
$("zoom-in").onclick = () => renderer.zoomBy(1.4);
$("zoom-out").onclick = () => renderer.zoomBy(1 / 1.4);
$("zoom-fit").onclick = () => renderer.fit();
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !["INPUT", "BUTTON"].includes(document.activeElement.tagName)) {
    e.preventDefault();
    send({ cmd: state.playing ? "pause" : "play" });
  }
});

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
function loop(now) {
  renderer.draw(frames.cur, frames.sample(now));
  requestAnimationFrame(loop);
}

addEvents([]);
connect();
requestAnimationFrame(loop);
