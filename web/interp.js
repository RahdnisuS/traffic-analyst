// Smooth motion between server frames.
//
// The server sends positions at 20 Hz; the screen refreshes at ~60 Hz. Drawing the raw frames
// would make vehicles jump every 50 ms. Instead we render ONE FRAME IN THE PAST and
// interpolate between the two most recent frames. This adds 50 ms of latency (invisible) and
// gives smooth motion. Multiplayer games use the same trick ("entity interpolation").
// Shared by the 2D and (Phase 5) 3D renderers. It never simulates anything itself.

const TELEPORT_M = 25; // if a vehicle moved further than this between frames, don't lerp it

function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

export class FrameBuffer {
  constructor() { this.clear(); }

  clear() {
    this.prev = null;
    this.cur = null;
    this.interval = 50;
  }

  push(frame, now) {
    if (this.cur) {
      // Smoothed arrival interval, so network jitter doesn't make motion stutter.
      const dt = Math.min(Math.max(now - this.cur.recv, 16), 250);
      this.interval = 0.8 * this.interval + 0.2 * dt;
    }
    const map = new Map();
    for (const v of frame.vehicles) map.set(v[0], v);
    this.prev = this.cur;
    this.cur = { ...frame, recv: now, map };
  }

  /** Interpolated vehicle list: [id, x, y, heading, speed, typeIdx]. */
  sample(now) {
    if (!this.cur) return [];
    if (!this.prev) return this.cur.vehicles;
    const t = Math.min(Math.max((now - this.cur.recv) / this.interval, 0), 1);
    const out = new Array(this.cur.vehicles.length);
    let i = 0;
    for (const v of this.cur.vehicles) {
      const p = this.prev.map.get(v[0]);
      if (!p || Math.hypot(v[1] - p[1], v[2] - p[2]) > TELEPORT_M) { out[i++] = v; continue; }
      out[i++] = [v[0], p[1] + (v[1] - p[1]) * t, p[2] + (v[2] - p[2]) * t,
                  lerpAngle(p[3], v[3], t), p[4] + (v[4] - p[4]) * t, v[5]];
    }
    return out;
  }
}
