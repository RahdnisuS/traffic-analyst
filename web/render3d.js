// 2.5D renderer (Three.js, orthographic camera with an isometric-style tilt).
//
// Like the 2D view it is a pure consumer of server state: the static layout once (`init`),
// then interpolated vehicle/pedestrian poses and signal states every screen refresh. There is
// no simulation logic here, so the 2D and 3D views always show the same run.
//
// Coordinates: the sim uses metres with x east and y north. Three.js is y-up, so a sim point
// (x, y) maps to (x, height, -y). A rotation of `heading` about Three's y axis then points a
// mesh's local +x along the sim heading (cos h, sin h), so vehicle headings need no conversion.
//
// Performance (target: 60 FPS with 200+ agents on a laptop):
//  - One InstancedMesh per vehicle part per type (body, cabin, …): every car's body is drawn
//    in a single draw call, so ~15 draw calls cover all traffic however many vehicles there are.
//  - Part geometries are pre-offset so their origin is the vehicle's FRONT BUMPER at ground
//    level (the point the server sends). One matrix per vehicle (rotate, then translate) is
//    then shared by all its parts: no per-part maths.
//  - Static scenery is built once per layout; signals only swap colours.
//  - Lambert (diffuse-only) materials and a single shadow-casting light keep fragments cheap.

import * as THREE from "three";
import { ARM_DIR, armGeometry, dividerPairs, hashId, layoutExtent, rng, signalSpot } from "./geometry.js";

const QUEUE_SPEED = 2.0;     // matches metrics.queue_speed: below this a vehicle counts as queued
const DEG = Math.PI / 180;
const ELEVATION = 55 * DEG;  // camera tilt above the ground: a "slight" isometric tilt (true
                             // isometric is 35°); steeper keeps queues readable behind buildings
const HOME_AZIMUTH = -45 * DEG; // camera sits to the south-east, looking north-west
const HOME_ZOOM = 1.6;
const KERB_H = 0.15;         // sidewalks are raised: a visible kerb reads as "road edge" at a glance
const MARK_Y = 0.02;         // markings float just above asphalt to avoid z-fighting
const PED_CROSS = 2;

// Body colours. Cars draw from common real-world paint colours; autos and buses use the
// typical Indian liveries, so type is readable from colour as well as shape.
const PAINT = {
  car: ["#f2f2ef", "#c9ccd1", "#8d9198", "#2f3237", "#b8322d", "#2f5d9e", "#d9d2c3", "#6b2430"],
  bike: ["#1f2328", "#b8322d", "#2f5d9e", "#3a3d42", "#e0e0dc"],
  auto: ["#2f8f3e", "#3a9a46"],
  bus: ["#c8432f", "#2f6fb0", "#d98c1f"],
};
const SHIRTS = ["#3b6fb6", "#c0504d", "#e0b040", "#5b8c4a", "#8a5ca8", "#e8e6df", "#33363b", "#d07a3c"];
const SIGNAL_RGB = { R: "#ff4d4f", Y: "#ffb300", G: "#2ee66b" };

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** A box geometry whose origin is offset so the vehicle's front bumper sits at (0, 0, 0). */
function partBox(lx, ly, lz, cx, cy) {
  return new THREE.BoxGeometry(lx, ly, lz).translate(cx, cy, 0);
}

/**
 * Low-poly part list per vehicle type, from its [length, width] (metres, from config.yaml).
 * `tint` says how a part is coloured: "paint"/"shirt" = per-vehicle instance colour, otherwise a
 * shared material. `tail` parts are unlit brake-light bars (instance colour = on/off).
 */
function vehicleParts(type, len, wid) {
  const back = (f) => -len * f; // x position as a fraction of the length behind the bumper
  switch (type) {
    case "bike":
      return [
        { g: partBox(len, 0.45, wid * 0.35, back(0.5), 0.5), tint: "paint" },
        { g: partBox(len * 0.3, 0.65, wid * 0.75, back(0.6), 1.05), tint: "shirt" },   // rider
        { g: new THREE.SphereGeometry(wid * 0.3, 8, 6).translate(back(0.55), 1.55, 0), tint: "helmet" },
        { g: partBox(0.06, 0.12, wid * 0.3, -len - 0.02, 0.6), tint: "tail" },
      ];
    case "auto":
      return [
        { g: partBox(len * 0.92, 0.2, wid * 0.85, back(0.5), 0.15), tint: "chassis" },
        { g: partBox(len, 0.6, wid, back(0.5), 0.55), tint: "paint" },
        { g: partBox(len * 0.7, 0.75, wid * 0.94, back(0.6), 1.22), tint: "canopy" }, // black hood
        { g: partBox(0.06, 0.14, wid * 0.8, -len - 0.02, 0.6), tint: "tail" },
      ];
    case "bus":
      return [
        { g: partBox(len * 0.94, 0.3, wid * 0.9, back(0.5), 0.2), tint: "chassis" },
        { g: partBox(len, 2.6, wid, back(0.5), 1.65), tint: "paint" },
        { g: partBox(len * 0.94, 0.8, wid + 0.04, back(0.5), 2.1), tint: "glass" },     // window band
        { g: partBox(0.06, 0.25, wid * 0.85, -len - 0.02, 0.8), tint: "tail" },
      ];
    default: // car: body box with a cabin on top, set slightly back from the bonnet
      return [
        { g: partBox(len * 0.92, 0.22, wid * 0.9, back(0.5), 0.14), tint: "chassis" },
        { g: partBox(len, 0.62, wid, back(0.5), 0.56), tint: "paint" },
        { g: partBox(len * 0.52, 0.5, wid * 0.86, back(0.56), 1.12), tint: "glass" },
        { g: partBox(0.06, 0.14, wid * 0.86, -len - 0.02, 0.72), tint: "tail" },
      ];
  }
}

/** Soft round glow texture (drawn once on a canvas): additive sprites make lamps "glow". */
function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.25, "rgba(255,255,255,0.55)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * An InstancedMesh that grows on demand. Instance buffers have a fixed size, so when traffic
 * exceeds the capacity we replace the mesh with one twice as large (amortised O(1), and it only
 * happens a handful of times per session).
 */
class InstancePool {
  constructor(parent, geometry, material, { colored = false, shadow = true } = {}) {
    Object.assign(this, { parent, geometry, material, colored, shadow, mesh: null });
    this.#alloc(64);
  }

  #alloc(capacity) {
    if (this.mesh) this.parent.remove(this.mesh);
    const m = new THREE.InstancedMesh(this.geometry, this.material, capacity);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    if (this.colored) m.setColorAt(0, new THREE.Color()); // creates the instanceColor buffer
    m.castShadow = this.shadow;
    // Instances move all over the map, but the bounding sphere is computed from the base
    // geometry at the origin: culling would wrongly hide them. 1 draw call: just draw it.
    m.frustumCulled = false;
    m.count = 0;
    this.parent.add(m);
    this.mesh = m;
  }

  ensure(n) { if (n > this.mesh.instanceMatrix.count) this.#alloc(2 ** Math.ceil(Math.log2(n))); }

  finish(n) {
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.colored && this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  dispose() {
    this.parent.remove(this.mesh);
    this.mesh.dispose();
    this.geometry.dispose();
  }
}

export class Renderer3D {
  constructor(canvas) {
    this.canvas = canvas;
    // Throws if WebGL is unavailable; the caller catches that and stays on the 2D view.
    this.gl = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.gl.shadowMap.enabled = true;
    this.gl.shadowMap.type = THREE.PCFSoftShadowMap; // soft shadow edges, one light: affordable
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 3000);

    this.target = { x: 0, y: 0 };  // sim-frame point the camera looks at
    this.azimuth = HOME_AZIMUTH;
    this.zoom = HOME_ZOOM;
    this.spin = null;              // in-progress camera rotation animation

    this.#lights();
    this.static = new THREE.Group();
    this.dynamic = new THREE.Group();
    this.scene.add(this.static, this.dynamic);
    this.glow = glowTexture();
    this.materials = this.#materials();
    this.readTheme();

    this.pools = {};      // vehicle type -> [{pool, tint}]
    this.pedPools = null;
    this.mat = new THREE.Matrix4();
    this.color = new THREE.Color();

    new ResizeObserver(() => this.resize()).observe(canvas);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.readTheme());
    this.resize();
    this.#bindPointer();
  }

  // ---------------------------------------------------------------- setup
  #lights() {
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x6b6f63, 1.1);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.03;
    this.scene.add(this.hemi, this.sun, this.sun.target);
  }

  #materials() {
    const lam = (c) => new THREE.MeshLambertMaterial({ color: c });
    return {
      ground: lam(0xffffff), asphalt: lam(0xffffff), sidewalk: lam(0xffffff),
      marking: new THREE.MeshLambertMaterial({ color: 0xffffff }),
      building: lam(0xffffff), roof: lam(0xffffff), trunk: lam(0x6b4a2f), crown: lam(0x4f8a3c),
      pole: lam(0x2a2c30), head: lam(0x15171a),
      paint: lam(0xffffff), shirt: lam(0xffffff), skin: lam(0xc68c5a),
      glass: lam(0x24303c), canopy: lam(0x1b1c1e), chassis: lam(0x151618), helmet: lam(0x18191b),
      tail: new THREE.MeshBasicMaterial({ color: 0xffffff }), // unlit: brake lights should glow
    };
  }

  /** Re-read colours from the CSS tokens, so the 3D view follows light/dark like the page. */
  readTheme() {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches
      && document.documentElement.dataset.theme !== "light"
      || document.documentElement.dataset.theme === "dark";
    const m = this.materials;
    m.ground.color.set(css("--ground"));
    m.asphalt.color.set(css("--asphalt"));
    m.sidewalk.color.set(css("--sidewalk"));
    m.marking.color.set(0xffffff); // instance colours carry white vs median yellow
    m.building.color.set(dark ? 0x9a9688 : 0xffffff);
    m.roof.color.set(dark ? 0x6f6c64 : 0xbdb8aa);
    this.markColors = { white: new THREE.Color(css("--marking")), yellow: new THREE.Color(css("--median")) };
    this.scene.background = new THREE.Color(css("--ground"));
    // Dark mode is dusk: dimmer, cooler light, so the signal lamps and tail-lights stand out.
    this.hemi.intensity = dark ? 0.55 : 1.1;
    this.hemi.color.set(dark ? 0x9fb3d6 : 0xffffff);
    this.sun.intensity = dark ? 0.7 : 2.0;
    if (this.layout) this.#buildStatic(); // markings take their colours at build time
  }

  setLayout(layout, vehicleDims, vehicleTypes = [], pedRadius = 0.25) {
    this.layout = layout;
    this.dims = vehicleDims;
    this.types = vehicleTypes;
    this.pedRadius = pedRadius;
    this.extent = layoutExtent(layout);
    this.arms = armGeometry(layout);
    this.#buildStatic();
    this.#buildPools();
    this.#placeSun();
  }

  #placeSun() {
    // Low-ish sun (~42° up) from the west-south-west. Shadows about as long as objects are tall
    // are what sell the "2.5D" depth; with a high sun they shrink under the vehicles. They fall
    // east-north-east, across the screen rather than towards the default camera.
    const e = this.extent;
    this.sun.position.set(-0.9 * e, 0.9 * e, 0.45 * e);
    this.sun.target.position.set(0, 0, 0);
    const cam = this.sun.shadow.camera;
    // The shadow map covers the inner part of the map only (±60% of the extent): spending all
    // 2048² texels on 330 m would give blurry, blocky car shadows near the junction, where
    // people actually look. Far-out queues simply get no shadow.
    const r = 0.6 * e;
    Object.assign(cam, { left: -r, right: r, top: r, bottom: -r, near: 1, far: 4 * e });
    cam.updateProjectionMatrix();
  }

  #clearGroup(group) {
    for (const o of [...group.children]) {
      group.remove(o);
      // Sprites share one internal geometry across the whole library: never dispose it.
      o.traverse?.((c) => { if (c.geometry && !c.isSprite) c.geometry.dispose(); });
    }
  }

  /** Static scenery: ground, asphalt, kerbs, markings, buildings, trees, signal heads. */
  #buildStatic() {
    const L = this.layout, m = this.materials, g = this.static;
    this.#clearGroup(g);
    const E = this.extent, B = L.box_half, sw = L.sidewalk_width ?? 3;

    const flat = (x0, y0, x1, y1, h, mat) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(Math.abs(x1 - x0), Math.abs(y1 - y0)).rotateX(-Math.PI / 2), mat);
      mesh.position.set((x0 + x1) / 2, h, -(y0 + y1) / 2);
      mesh.receiveShadow = true;
      g.add(mesh);
    };
    flat(-E * 1.6, -E * 1.6, E * 1.6, E * 1.6, -0.01, m.ground);
    flat(-B, -B, B, B, 0, m.asphalt);
    for (const [arm, a] of Object.entries(this.arms)) {
      const [lo, hi] = a.lat;
      if (arm === "N") flat(lo, B, hi, E, 0, m.asphalt);
      if (arm === "S") flat(lo, -E, hi, -B, 0, m.asphalt);
      if (arm === "E") flat(B, lo, E, hi, 0, m.asphalt);
      if (arm === "W") flat(-E, lo, -B, hi, 0, m.asphalt);
    }

    // Sidewalks: raised boxes along both kerbs of every arm (same footprint as the 2D view).
    let k = 0;
    for (const cw of Object.values(L.crosswalks ?? {})) {
      for (const [a0, a1] of [[cw.hi, cw.hi + sw], [cw.lo - sw, cw.lo]]) {
        const len = E - cw.u0, across = (a0 + a1) / 2, along = (cw.u0 + E) / 2;
        // Sidewalks of adjacent arms overlap at the corners; a hair of height difference stops
        // their coplanar tops flickering (z-fighting).
        const h = KERB_H + 0.004 * (k++ % 2);
        const box = new THREE.Mesh(new THREE.BoxGeometry(len, h, sw), m.sidewalk);
        const x = cw.u[0] * along + cw.n[0] * across, y = cw.u[1] * along + cw.n[1] * across;
        box.position.set(x, h / 2, -y);
        box.rotation.y = Math.atan2(cw.u[1], cw.u[0]);
        box.receiveShadow = true;
        g.add(box);
      }
    }

    this.#buildMarkings();
    this.#buildScenery();
    this.#buildSignals();
  }

  /** All road paint in ONE instanced mesh of unit quads (scaled/rotated per stripe). */
  #buildMarkings() {
    const L = this.layout, rects = [];
    const W = this.markColors.white, Yl = this.markColors.yellow;
    // rect: centre (x, y), length along `ang`, width across it
    const seg = (p, q, wid, col, dash = 0, gap = 0) => {
      const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy), ang = Math.atan2(dy, dx);
      if (!dash) { rects.push([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, len, wid, ang, col]); return; }
      for (let s = 0; s + dash <= len; s += dash + gap) {
        const t = (s + dash / 2) / len;
        rects.push([p[0] + dx * t, p[1] + dy * t, dash, wid, ang, col]);
      }
    };
    for (const [p, q] of dividerPairs(L)) seg(p, q, 0.15, W, 3, 3);
    for (const arm of Object.keys(this.arms)) {
      const u = ARM_DIR[arm];
      seg([u[0] * L.stop_offset, u[1] * L.stop_offset], [u[0] * this.extent, u[1] * this.extent], 0.3, Yl);
    }
    for (const a of Object.values(this.arms)) {
      for (const e of a.inbound) {           // stop line across each inbound lane
        const p = e.points, s = p.at(-2), b = p.at(-1);
        const len = Math.hypot(b[0] - s[0], b[1] - s[1]);
        const nx = -(b[1] - s[1]) / len, ny = (b[0] - s[0]) / len, hw = L.lane_width / 2;
        seg([b[0] - nx * hw, b[1] - ny * hw], [b[0] + nx * hw, b[1] + ny * hw], 0.45, W);
      }
    }
    // Zebra crossings: stripes parallel to traffic, between the box edge and the stop line.
    const stripe = 0.5, gap = 0.6, inset = 0.4;
    for (const cw of Object.values(L.crosswalks ?? {})) {
      const along = (cw.u0 + cw.u1) / 2, len = cw.u1 - cw.u0 - 2 * inset, ang = Math.atan2(cw.u[1], cw.u[0]);
      for (let p = cw.lo + 0.3; p + stripe <= cw.hi - 0.3; p += stripe + gap) {
        const c = p + stripe / 2;
        rects.push([cw.u[0] * along + cw.n[0] * c, cw.u[1] * along + cw.n[1] * c, len, stripe, ang, W]);
      }
    }
    const quad = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
    const mesh = new THREE.InstancedMesh(quad, this.materials.marking, rects.length);
    const o = new THREE.Object3D();
    rects.forEach(([x, y, len, wid, ang, col], i) => {
      o.position.set(x, MARK_Y, -y);
      o.rotation.set(0, ang, 0);
      o.scale.set(len, 1, wid);
      o.updateMatrix();
      mesh.setMatrixAt(i, o.matrix);
      mesh.setColorAt(i, col);
    });
    mesh.receiveShadow = true;
    this.static.add(mesh);
  }

  /**
   * Procedural corner buildings and street trees. Generated from a FIXED seed, so the city
   * looks the same on every load. It is decoration only: none of it affects the simulation.
   */
  #buildScenery() {
    const L = this.layout, rand = rng(1234), sw = L.sidewalk_width ?? 3;
    const boxes = [], tanks = [], trees = [];
    const reach = Math.min(this.extent * 0.8, 130);
    for (const [sx, sy] of [[1, 1], [-1, 1], [-1, -1], [1, -1]]) {
      // Lot starts behind the sidewalk of the two arms bounding this corner, plus a setback.
      const vert = this.arms[sy > 0 ? "N" : "S"], horz = this.arms[sx > 0 ? "E" : "W"];
      if (!vert || !horz) continue;
      const x0 = (sx > 0 ? vert.lat[1] : -vert.lat[0]) + sw + 4;
      const y0 = (sy > 0 ? horz.lat[1] : -horz.lat[0]) + sw + 4;
      // Rows of blocks with random footprints, filling the corner lot.
      for (let y = y0; y < reach; ) {
        const depth = 10 + rand() * 14;
        for (let x = x0; x < reach; ) {
          const width = 10 + rand() * 14;
          if (rand() > 0.12) { // occasional empty plot breaks up the grid
            const dist = Math.hypot(x + width / 2, y + depth / 2);
            // Low near the junction, taller further out: tall blocks in the near corner
            // would hide the stop lines from the tilted camera.
            const h = 4 + rand() * (4 + Math.min(dist, 110) * 0.2);
            const cx = sx * (x + width / 2), cy = sy * (y + depth / 2);
            boxes.push([cx, cy, width, h, depth, rand()]);
            // Rooftop water tanks: a staple of Indian roofs, and a cheap silhouette detail.
            if (rand() < 0.45) tanks.push([cx + (rand() - 0.5) * width * 0.5, cy + (rand() - 0.5) * depth * 0.5, h]);
          }
          x += width + 3 + rand() * 3;
        }
        y += depth + 3 + rand() * 3;
      }
    }
    // Street trees along the outer edge of each sidewalk, clear of the crossing and signals.
    for (const cw of Object.values(L.crosswalks ?? {})) {
      for (const across of [cw.hi + sw - 0.6, cw.lo - sw + 0.6]) {
        for (let along = cw.u1 + 9 + rand() * 4; along < this.extent - 4; along += 11 + rand() * 6) {
          trees.push([cw.u[0] * along + cw.n[0] * across, cw.u[1] * along + cw.n[1] * across, 0.8 + rand() * 0.5]);
        }
      }
    }

    const tones = ["#e8dcc4", "#d9c9a8", "#cfd3d6", "#e3c7b0", "#c9b79c", "#dfe1d8", "#bfc6b8", "#e6d3b8"];
    const o = new THREE.Object3D(), c = new THREE.Color();
    const unit = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0); // base at y = 0
    const bmesh = new THREE.InstancedMesh(unit, this.materials.building, boxes.length);
    boxes.forEach(([x, y, w, h, d, t], i) => {
      o.position.set(x, KERB_H, -y); o.rotation.set(0, 0, 0); o.scale.set(w, h, d); o.updateMatrix();
      bmesh.setMatrixAt(i, o.matrix);
      bmesh.setColorAt(i, c.set(tones[Math.floor(t * tones.length)]));
    });
    const tmesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.9, 0.9, 1.4, 8).translate(0, 0.7, 0),
                                          this.materials.roof, tanks.length);
    tanks.forEach(([x, y, h], i) => {
      o.position.set(x, KERB_H + h, -y); o.scale.set(1, 1, 1); o.updateMatrix();
      tmesh.setMatrixAt(i, o.matrix);
    });
    const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.15, 0.2, 2, 5).translate(0, 1, 0),
                                          this.materials.trunk, trees.length);
    const crown = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1.6, 0).translate(0, 3.2, 0),
                                          this.materials.crown, trees.length);
    trees.forEach(([x, y, s], i) => {
      o.position.set(x, KERB_H, -y); o.scale.set(s, s, s); o.rotation.set(0, s * 7, 0); o.updateMatrix();
      trunk.setMatrixAt(i, o.matrix);
      crown.setMatrixAt(i, o.matrix);
      crown.setColorAt(i, c.setHSL(0.27 + (s - 1) * 0.1, 0.42, 0.36 + (s - 1) * 0.15));
    });
    for (const mesh of [bmesh, tmesh, trunk, crown]) {
      mesh.castShadow = mesh.receiveShadow = true;
      this.static.add(mesh);
    }
  }

  /** Signal poles. Lamps are meshes whose colour we swap; a glow sprite marks the lit one. */
  #buildSignals() {
    const L = this.layout, m = this.materials;
    const lampGeo = new THREE.SphereGeometry(0.32, 10, 8);
    const sprite = (color, size) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: this.glow, color, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      s.scale.set(size, size, 1);
      return s;
    };
    const pole = (x, y, h) => {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, h, 6), m.pole);
      p.position.set(x, KERB_H + h / 2, -y);
      p.castShadow = true;
      this.static.add(p);
    };

    this.signalHeads = {};
    for (const [arm, a] of Object.entries(this.arms)) {
      const spot = signalSpot(L, a);
      if (!spot) continue;
      const H = 4.2;
      pole(spot.x, spot.y, H);
      const head = new THREE.Group();
      head.position.set(spot.x, KERB_H + H, -spot.y);
      // Face the approaching drivers: they travel along (dx, dy), so the lamps face -(dx, dy).
      head.rotation.y = Math.atan2(-spot.dy, -spot.dx);
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.5, 2.1, 0.8), m.head);
      box.castShadow = true;
      head.add(box);
      const lamps = {};
      ["R", "Y", "G"].forEach((s, i) => {
        const lamp = new THREE.Mesh(lampGeo, new THREE.MeshBasicMaterial({ color: 0x222222 }));
        lamp.position.set(0.22, 0.65 - i * 0.65, 0);
        const glow = sprite(SIGNAL_RGB[s], 3.2);
        glow.position.copy(lamp.position).x += 0.3;
        glow.visible = false;
        head.add(lamp, glow);
        lamps[s] = { lamp, glow, on: new THREE.Color(SIGNAL_RGB[s]), off: new THREE.Color(SIGNAL_RGB[s]).multiplyScalar(0.12) };
      });
      this.static.add(head);
      this.signalHeads[arm] = lamps;
    }

    // Pedestrian signals: one small head at both kerbs of every crosswalk.
    this.pedHeads = [];
    for (const cw of Object.values(L.crosswalks ?? {})) {
      for (const across of [cw.hi + 0.9, cw.lo - 0.9]) {
        const x = cw.u[0] * (cw.u1 + 0.6) + cw.n[0] * across, y = cw.u[1] * (cw.u1 + 0.6) + cw.n[1] * across;
        pole(x, y, 2.6);
        const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.45, 0.45), new THREE.MeshBasicMaterial({ color: 0x222222 }));
        lamp.position.set(x, KERB_H + 2.8, -y);
        const glow = sprite(0xffffff, 1.3);
        glow.position.copy(lamp.position);
        this.static.add(lamp, glow);
        this.pedHeads.push({ lamp, glow });
      }
    }
  }

  #buildPools() {
    for (const parts of Object.values(this.pools)) for (const p of parts) p.pool.dispose();
    this.pools = {};
    const m = this.materials;
    this.types.forEach((type, ti) => {
      const [len, wid] = this.dims[ti] || [4.5, 1.8];
      this.pools[ti] = vehicleParts(type, len, wid).map(({ g, tint }) => {
        const colored = tint === "paint" || tint === "shirt" || tint === "tail";
        const mat = tint === "paint" ? m.paint : tint === "shirt" ? m.shirt : m[tint];
        return { tint, pool: new InstancePool(this.dynamic, g, mat, { colored, shadow: tint !== "tail" }) };
      });
    });
    if (this.pedPools) for (const p of this.pedPools) p.dispose();
    const r = this.pedRadius;
    // Capsule body (legs + torso) and a head: reads as a person even from far away.
    this.pedPools = [
      new InstancePool(this.dynamic, new THREE.CapsuleGeometry(r, 1.0, 3, 8).translate(0, r + 0.5, 0), m.shirt, { colored: true }),
      new InstancePool(this.dynamic, new THREE.SphereGeometry(r * 0.75, 8, 6).translate(0, 1.0 + 2 * r + 0.1, 0), m.skin),
    ];
  }

  // ---------------------------------------------------------------- camera
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, r.width);
    this.h = Math.max(1, r.height);
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); // 3× phones: cap the fill cost
    this.gl.setSize(this.w, this.h, false);
  }

  fit() { this.zoom = HOME_ZOOM; this.target = { x: 0, y: 0 }; this.rotate(HOME_AZIMUTH - this.azimuth); }
  zoomBy(f) { this.zoom = Math.min(Math.max(this.zoom * f, 0.5), 14); }

  /** Rotate the camera around the junction by `delta` radians, animated (or snapped). */
  rotate(delta) {
    if (!delta) return;
    const from = this.spin ? this.spin.to : this.azimuth;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { this.azimuth = from + delta; this.spin = null; return; }
    this.spin = { from: this.azimuth, to: from + delta, t0: performance.now(), dur: 450 };
  }

  #halfHeight() { return (this.extent || 100) * 0.62 / this.zoom; } // world half-height of the view

  #updateCamera(now) {
    if (this.spin) {
      const k = Math.min((now - this.spin.t0) / this.spin.dur, 1);
      const e = 1 - (1 - k) ** 3; // ease-out: starts promptly, settles gently
      this.azimuth = this.spin.from + (this.spin.to - this.spin.from) * e;
      if (k === 1) this.spin = null;
    }
    const hh = this.#halfHeight(), aspect = this.w / this.h, cam = this.camera;
    const D = 4 * (this.extent || 100); // distance only matters for clipping in an ortho view
    Object.assign(cam, { left: -hh * aspect, right: hh * aspect, top: hh, bottom: -hh, near: D * 0.5, far: D * 1.6 });
    cam.updateProjectionMatrix();
    const ce = Math.cos(ELEVATION);
    cam.position.set(this.target.x + D * ce * Math.cos(this.azimuth), D * Math.sin(ELEVATION),
                     -(this.target.y + D * ce * Math.sin(this.azimuth)));
    cam.lookAt(this.target.x, 0, -this.target.y);
  }

  #bindPointer() {
    // Drag pans across the ground plane (mouse or one finger); wheel zooms; Q/E rotate.
    let drag = null;
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, t: { ...this.target } };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const wpp = (2 * this.#halfHeight()) / this.h;          // world metres per screen pixel
      const fx = -Math.cos(this.azimuth), fy = -Math.sin(this.azimuth); // view direction on the ground
      const rx = fy, ry = -fx;                                  // screen-right on the ground
      const dx = (e.clientX - drag.x) * wpp, dy = (e.clientY - drag.y) * wpp / Math.sin(ELEVATION);
      // Content follows the pointer: moving it right moves the target left, and so on.
      this.target = { x: drag.t.x - rx * dx + fx * dy, y: drag.t.y - ry * dx + fy * dy };
    });
    const end = () => { drag = null; };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });
  }

  // ---------------------------------------------------------------- per frame
  draw(frame, vehicles, peds = [], now = 0) {
    if (!this.layout) { this.gl.render(this.scene, this.camera); return; }
    this.#updateCamera(now);
    this.#updateSignals(frame, now);
    this.#updateVehicles(vehicles);
    this.#updatePeds(peds);
    this.gl.render(this.scene, this.camera);
  }

  #updateSignals(frame, now) {
    const sig = frame?.signals || {};
    for (const [arm, lamps] of Object.entries(this.signalHeads || {})) {
      for (const [s, l] of Object.entries(lamps)) {
        const lit = sig[arm] === s;
        l.lamp.material.color.copy(lit ? l.on : l.off);
        l.glow.visible = lit;
      }
    }
    const ps = frame?.ped_signal;
    // Flashing DON'T WALK blinks at 1 Hz, same as the 2D view.
    const col = ps === "walk" ? SIGNAL_RGB.G : ps === "flash" ? (Math.floor(now / 500) % 2 ? SIGNAL_RGB.Y : null)
      : ps ? SIGNAL_RGB.R : null;
    for (const { lamp, glow } of this.pedHeads || []) {
      lamp.material.color.set(col ?? 0x222222);
      glow.visible = !!col;
      if (col) glow.material.color.set(col);
    }
  }

  #updateVehicles(vehicles) {
    // Bucket by type, then write one matrix per vehicle into every part pool of that type.
    const byType = {};
    for (const v of vehicles) (byType[v[5]] ??= []).push(v);
    const { mat, color } = this;
    for (const [ti, parts] of Object.entries(this.pools)) {
      const list = byType[ti] || [];
      const type = this.types[ti];
      const paints = PAINT[type] || PAINT.car;
      for (const { pool } of parts) pool.ensure(list.length);
      list.forEach(([id, x, y, h, speed], i) => {
        mat.makeRotationY(h);
        mat.setPosition(x, 0, -y);
        const hsh = hashId(id);
        for (const { tint, pool } of parts) {
          const mesh = pool.mesh;
          mesh.setMatrixAt(i, mat);
          if (tint === "paint") mesh.setColorAt(i, color.set(paints[hsh % paints.length]));
          else if (tint === "shirt") mesh.setColorAt(i, color.set(SHIRTS[(hsh >>> 8) % SHIRTS.length]));
          // Brake lights on when queued or stopped: the 3D stand-in for the 2D speed colours.
          else if (tint === "tail") mesh.setColorAt(i, color.set(speed < QUEUE_SPEED ? 0xff2a2a : 0x4a1010));
        }
      });
      for (const { pool } of parts) pool.finish(list.length);
    }
  }

  #updatePeds(peds) {
    const [body, head] = this.pedPools || [];
    if (!body) return;
    body.ensure(peds.length);
    head.ensure(peds.length);
    const { mat, color } = this;
    peds.forEach(([id, x, y, st], i) => {
      // Walkers stand on the raised sidewalk except while crossing the carriageway.
      mat.makeTranslation(x, st === PED_CROSS ? 0 : KERB_H, -y);
      body.mesh.setMatrixAt(i, mat);
      head.mesh.setMatrixAt(i, mat);
      body.mesh.setColorAt(i, color.set(SHIRTS[hashId(id) % SHIRTS.length]));
    });
    body.finish(peds.length);
    head.finish(peds.length);
  }
}
