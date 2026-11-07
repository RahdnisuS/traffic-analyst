// 2D top-down debug renderer (HTML canvas).
//
// It draws only what the server sends: the static layout once, then vehicle poses and signal
// states. World units are metres with y pointing north; one canvas transform maps world to
// screen, so all drawing code below works in metres.

const QUEUE_SPEED = 2.0;  // matches metrics.queue_speed
const STOP_SPEED = 0.1;   // matches the engine's WAIT_SPEED
const PED_MIN_PX = 2.4;   // pedestrians are 0.5 m wide: at "fit" zoom that is under a pixel, so
                          // they get a minimum on-screen radius (a debug view should show them)
const PED_WAIT = 1, PED_CROSS = 2;

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class Renderer2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.layout = null;
    this.dims = [];
    this.zoom = 1;
    this.pan = { x: 0, y: 0 };
    this.readTheme();
    new ResizeObserver(() => this.resize()).observe(canvas);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.readTheme());
    this.resize();
    this.#bindPointer();
  }

  readTheme() {
    this.c = {
      ground: css("--ground"), asphalt: css("--asphalt"), marking: css("--marking"), median: css("--median"),
      moving: css("--veh-moving"), queued: css("--veh-queued"), stopped: css("--veh-stopped"),
      sidewalk: css("--sidewalk"), ped: css("--ped"), pedHalo: css("--ped-halo"),
    };
  }

  setLayout(layout, vehicleDims, vehicleTypes = [], pedRadius = 0.25) {
    this.layout = layout;
    this.dims = vehicleDims;
    this.types = vehicleTypes;
    this.pedRadius = pedRadius;
    let ext = 0;
    for (const e of Object.values(layout.edges)) for (const [x, y] of e.points) ext = Math.max(ext, Math.abs(x), Math.abs(y));
    this.extent = ext;
    this.arms = this.#armGeometry(layout);
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, r.width);
    this.h = Math.max(1, r.height);
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
  }

  fit() { this.zoom = 1; this.pan = { x: 0, y: 0 }; }
  zoomBy(f) { this.zoom = Math.min(Math.max(this.zoom * f, 0.5), 12); }

  #scale() { return (Math.min(this.w, this.h) / (2 * (this.extent || 100) * 1.02)) * this.zoom; }

  #bindPointer() {
    // Drag to pan (mouse or one finger), wheel to zoom. Pointer events cover touch too.
    let drag = null;
    this.canvas.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, pan: { ...this.pan } };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const s = this.#scale();
      this.pan.x = drag.pan.x - (e.clientX - drag.x) / s;
      this.pan.y = drag.pan.y + (e.clientY - drag.y) / s;
    });
    const end = () => { drag = null; };
    this.canvas.addEventListener("pointerup", end);
    this.canvas.addEventListener("pointercancel", end);
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });
  }

  /** Per-arm road extents, derived from lane polylines (so any JSON layout renders). */
  #armGeometry(layout) {
    const arms = {};
    const w = layout.lane_width;
    for (const e of Object.values(layout.edges)) {
      if (e.kind === "connector") continue;
      const a = (arms[e.arm] ??= { lat: [Infinity, -Infinity], inbound: [] });
      const vertical = e.arm === "N" || e.arm === "S";
      for (const [x, y] of e.points) {
        const lat = vertical ? x : y;
        a.lat[0] = Math.min(a.lat[0], lat - w / 2);
        a.lat[1] = Math.max(a.lat[1], lat + w / 2);
      }
      if (e.kind === "inbound") a.inbound.push(e);
    }
    return arms;
  }

  draw(frame, vehicles, peds = [], now = 0) {
    const { ctx } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = this.c.ground;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.layout) return;

    const s = this.#scale();
    // World (metres, y up) -> device pixels.
    ctx.setTransform(this.dpr * s, 0, 0, -this.dpr * s,
                     this.dpr * (this.w / 2 - this.pan.x * s), this.dpr * (this.h / 2 + this.pan.y * s));
    this.#drawSidewalks();
    this.#drawRoads();
    this.#drawSignals(frame?.signals);
    this.#drawPedSignals(frame?.ped_signal, now);
    this.#drawVehicles(vehicles);
    this.#drawPeds(peds, s);
  }

  /** World point from a crosswalk's own axes: along the arm (u) and across it (n). */
  #pt(cw, along, across) { return [cw.u[0] * along + cw.n[0] * across, cw.u[1] * along + cw.n[1] * across]; }

  #quad(cw, a0, a1, b0, b1) {
    const { ctx } = this;
    const pts = [this.#pt(cw, a0, b0), this.#pt(cw, a1, b0), this.#pt(cw, a1, b1), this.#pt(cw, a0, b1)];
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    for (const p of pts.slice(1)) ctx.lineTo(...p);
    ctx.closePath();
    ctx.fill();
  }

  #drawSidewalks() {
    // A sidewalk strip along both kerbs of every arm, from the junction box outwards.
    const L = this.layout, sw = L.sidewalk_width ?? 3;
    this.ctx.fillStyle = this.c.sidewalk;
    for (const cw of Object.values(L.crosswalks ?? {})) {
      this.#quad(cw, cw.u0, this.extent, cw.hi, cw.hi + sw);
      this.#quad(cw, cw.u0, this.extent, cw.lo - sw, cw.lo);
    }
  }

  #drawPedSignals(state, now) {
    // A small pedestrian head at both kerbs of every crosswalk. Flashing DON'T WALK blinks at 1 Hz.
    if (!state) return;
    const { ctx } = this;
    const on = state === "walk" ? "#22c55e" : state === "flash" ? (Math.floor(now / 500) % 2 ? "#f5b301" : "#1b1b1b") : "#e5484d";
    for (const cw of Object.values(this.layout.crosswalks ?? {})) {
      for (const across of [cw.hi + 0.9, cw.lo - 0.9]) {
        const [x, y] = this.#pt(cw, cw.u1 + 0.6, across);
        ctx.fillStyle = "#1b1b1b";
        ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
        ctx.fillStyle = on;
        ctx.fillRect(x - 0.5, y - 0.5, 1.0, 1.0);
      }
    }
  }

  #drawPeds(peds, scale) {
    if (!peds.length) return;
    const { ctx } = this;
    const r = Math.max(this.pedRadius, PED_MIN_PX / scale);
    for (const [, x, y, st] of peds) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = this.c.ped;
      ctx.fill();
      if (st === PED_WAIT || st === PED_CROSS) {
        // Halo: waiting at the kerb or on the road, i.e. the walkers the signal is serving.
        ctx.lineWidth = r * 0.45;
        ctx.strokeStyle = this.c.pedHalo;
        ctx.stroke();
      }
    }
  }

  #drawRoads() {
    const { ctx, layout } = this;
    const L = layout, w = L.lane_width, B = L.box_half, S = L.stop_offset;
    ctx.fillStyle = this.c.asphalt;
    ctx.fillRect(-B, -B, 2 * B, 2 * B);
    for (const [arm, g] of Object.entries(this.arms)) {
      const far = this.extent;
      const [lo, hi] = g.lat;
      ctx.fillStyle = this.c.asphalt; // reset every arm: the crosswalk below changes fillStyle
      if (arm === "N") ctx.fillRect(lo, B, hi - lo, far - B);
      if (arm === "S") ctx.fillRect(lo, -far, hi - lo, far - B);
      if (arm === "E") ctx.fillRect(B, lo, far - B, hi - lo);
      if (arm === "W") ctx.fillRect(-far, lo, far - B, hi - lo);
      this.#drawCrosswalk(arm, lo, hi, B, S);
    }
    // Lane dividers (dashed) between adjacent same-direction lanes, and the centre line.
    ctx.lineWidth = 0.15;
    ctx.strokeStyle = this.c.marking;
    ctx.setLineDash([3, 3]);
    const groups = {};
    for (const [id, e] of Object.entries(L.edges)) {
      // Filter lanes (flt_*, oflt_*) run along the dividers themselves: no extra markings.
      if (e.kind === "connector" || e.kind === "filter" || id.startsWith("oflt_")) continue;
      (groups[`${e.kind}_${e.arm}`] ??= []).push([Number(id.split("_").pop()), e]);
    }
    for (const lanes of Object.values(groups)) {
      lanes.sort((a, b) => a[0] - b[0]);
      for (let i = 0; i + 1 < lanes.length; i++) {
        const p = lanes[i][1].points, q = lanes[i + 1][1].points;
        ctx.beginPath();
        ctx.moveTo((p[0][0] + q[0][0]) / 2, (p[0][1] + q[0][1]) / 2);
        ctx.lineTo((p.at(-1)[0] + q.at(-1)[0]) / 2, (p.at(-1)[1] + q.at(-1)[1]) / 2);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = this.c.median;
    ctx.lineWidth = 0.3;
    for (const arm of Object.keys(this.arms)) {
      const u = { N: [0, 1], E: [1, 0], S: [0, -1], W: [-1, 0] }[arm];
      ctx.beginPath();
      ctx.moveTo(u[0] * S, u[1] * S);
      ctx.lineTo(u[0] * this.extent, u[1] * this.extent);
      ctx.stroke();
    }
    // Stop lines across each inbound lane.
    ctx.strokeStyle = this.c.marking;
    ctx.lineWidth = 0.45;
    for (const g of Object.values(this.arms)) {
      for (const e of g.inbound) {
        const p = e.points, a = p.at(-2), b = p.at(-1);
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len;
        ctx.beginPath();
        ctx.moveTo(b[0] - nx * w / 2, b[1] - ny * w / 2);
        ctx.lineTo(b[0] + nx * w / 2, b[1] + ny * w / 2);
        ctx.stroke();
      }
    }
  }

  #drawCrosswalk(arm, lo, hi, B, S) {
    // Zebra stripes parallel to traffic flow, in the strip between box edge and stop line.
    const { ctx } = this;
    ctx.fillStyle = this.c.marking;
    const stripe = 0.5, gap = 0.6, inset = 0.4;
    for (let p = lo + 0.3; p + stripe <= hi - 0.3; p += stripe + gap) {
      if (arm === "N") ctx.fillRect(p, B + inset, stripe, S - B - 2 * inset);
      if (arm === "S") ctx.fillRect(p, -S + inset, stripe, S - B - 2 * inset);
      if (arm === "E") ctx.fillRect(B + inset, p, S - B - 2 * inset, stripe);
      if (arm === "W") ctx.fillRect(-S + inset, p, S - B - 2 * inset, stripe);
    }
  }

  #drawSignals(signals) {
    if (!signals) return;
    const { ctx } = this;
    const w = this.layout.lane_width;
    const colors = { G: "#22c55e", Y: "#f5b301", R: "#e5484d" };
    for (const [arm, g] of Object.entries(this.arms)) {
      const kerb = g.inbound.find((e) => e.points) && g.inbound[0];
      if (!kerb) continue;
      // Signal head on the kerb side (driver's left in left-hand traffic), at the stop line.
      const p = kerb.points, a = p.at(-2), b = p.at(-1);
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const lx = -(b[1] - a[1]) / len, ly = (b[0] - a[0]) / len;
      const x = b[0] + lx * (w / 2 + 1.6), y = b[1] + ly * (w / 2 + 1.6);
      ctx.fillStyle = "#1b1b1b";
      ctx.beginPath(); ctx.arc(x, y, 1.6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = colors[signals[arm]] || "#777";
      ctx.beginPath(); ctx.arc(x, y, 1.15, 0, 2 * Math.PI); ctx.fill();
    }
  }

  #drawVehicles(vehicles) {
    // Colour shows speed state; size and a simple mark show the type, so both read at once.
    const { ctx } = this;
    for (const [, x, y, h, v, ti] of vehicles) {
      const [len, wid] = this.dims[ti] || [4.5, 1.8];
      const type = this.types[ti];
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(h);
      // Position is the FRONT bumper, so the body extends backwards along -heading.
      ctx.fillStyle = v < STOP_SPEED ? this.c.stopped : v < QUEUE_SPEED ? this.c.queued : this.c.moving;
      ctx.fillRect(-len, -wid / 2, len, wid);
      if (type === "bike") {
        ctx.fillStyle = "rgba(20,20,20,0.75)"; // rider's helmet
        ctx.beginPath(); ctx.arc(-len * 0.55, 0, wid * 0.42, 0, 2 * Math.PI); ctx.fill();
      } else if (type === "auto") {
        ctx.fillStyle = "rgba(20,20,20,0.55)"; // canopy over the rear two-thirds
        ctx.fillRect(-len * 0.95, -wid / 2 + 0.15, len * 0.6, wid - 0.3);
      } else if (type === "bus") {
        ctx.fillStyle = "rgba(255,255,255,0.55)"; // window strip along each side
        ctx.fillRect(-len + 0.8, -wid / 2 + 0.2, len - 1.8, 0.35);
        ctx.fillRect(-len + 0.8, wid / 2 - 0.55, len - 1.8, 0.35);
      } else {
        ctx.fillStyle = "rgba(255,255,255,0.55)"; // windscreen marks the front
        ctx.fillRect(-1.3, -wid / 2 + 0.25, 0.5, wid - 0.5);
      }
      ctx.restore();
    }
  }
}
