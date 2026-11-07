// Layout helpers shared by the 2D and 3D renderers.
//
// Both views derive road surfaces from the same lane polylines the server sends, so they can
// never disagree about where a road is, and any junction JSON renders without renderer changes.

/** Largest |x| or |y| of any lane point: how far the drawn world extends from the centre. */
export function layoutExtent(layout) {
  let ext = 0;
  for (const e of Object.values(layout.edges)) for (const [x, y] of e.points) ext = Math.max(ext, Math.abs(x), Math.abs(y));
  return ext;
}

/** Per-arm road extents: lateral range [lo, hi] covered by its lanes, plus its inbound lanes. */
export function armGeometry(layout) {
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

/**
 * Where an arm's signal head stands: beside the kerb lane at the stop line, on the driver's
 * left (left-hand traffic). Returns the point and the unit direction traffic travels in.
 */
export function signalSpot(layout, arm, offset = 1.6) {
  const kerb = arm.inbound[0];
  if (!kerb) return null;
  const p = kerb.points, a = p.at(-2), b = p.at(-1);
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const dx = (b[0] - a[0]) / len, dy = (b[1] - a[1]) / len;
  const lx = -dy, ly = dx; // left-hand normal
  const d = layout.lane_width / 2 + offset;
  return { x: b[0] + lx * d, y: b[1] + ly * d, dx, dy };
}

/** Pairs of adjacent same-direction lanes, for drawing the dashed dividers between them. */
export function dividerPairs(layout) {
  const groups = {};
  for (const [id, e] of Object.entries(layout.edges)) {
    // Filter lanes (flt_*, oflt_*) run along the dividers themselves: no extra markings.
    if (e.kind === "connector" || e.kind === "filter" || id.startsWith("oflt_")) continue;
    (groups[`${e.kind}_${e.arm}`] ??= []).push([Number(id.split("_").pop()), e]);
  }
  const pairs = [];
  for (const lanes of Object.values(groups)) {
    lanes.sort((a, b) => a[0] - b[0]);
    for (let i = 0; i + 1 < lanes.length; i++) {
      const p = lanes[i][1].points, q = lanes[i + 1][1].points;
      pairs.push([[(p[0][0] + q[0][0]) / 2, (p[0][1] + q[0][1]) / 2],
                  [(p.at(-1)[0] + q.at(-1)[0]) / 2, (p.at(-1)[1] + q.at(-1)[1]) / 2]]);
    }
  }
  return pairs;
}

export const ARM_DIR = { N: [0, 1], E: [1, 0], S: [0, -1], W: [-1, 0] };

/** Small deterministic PRNG (mulberry32), so scenery is the same on every page load. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Integer hash of an agent id, for stable per-agent colours. */
export function hashId(id) {
  let h = Math.imul(Number(id) ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}
