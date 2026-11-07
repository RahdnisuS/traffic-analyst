// Live Chart.js charts. Each chart has a single y-axis (two measures with different units get
// two charts, never a shared dual axis). Series colours follow the entity (N/E/S/W always the
// same colour), and lines are 2 px with a crosshair-style index tooltip.

const WINDOW_S = 300; // visible history
const ARM_SLOTS = { N: "--series-1", E: "--series-2", S: "--series-3", W: "--series-4" };
const ARM_NAMES = { N: "North", E: "East", S: "South", W: "West" };
const LOS_LINES = [[35, "LOS D"], [55, "LOS E"], [80, "LOS F"]];

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function clock(t) { const s = Math.max(0, Math.round(t)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

// Direct labels: series name at the end of each line (text in ink colour, a dot carries the hue).
const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(chart) {
    if (chart.data.datasets.length < 2) return;
    const { ctx } = chart;
    ctx.save();
    ctx.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.textBaseline = "middle";
    const placed = [];
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      const pt = meta.data.at(-1);
      if (!pt || meta.hidden) return;
      let y = pt.y;
      for (const p of placed) if (Math.abs(p - y) < 12) y = p + 12; // nudge to avoid overlap
      placed.push(y);
      ctx.fillStyle = ds.borderColor;
      ctx.beginPath(); ctx.arc(pt.x + 6, y, 3, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = css("--ink-2");
      ctx.fillText(ds.shortLabel, pt.x + 12, y);
    });
    ctx.restore();
  },
};

// Faint reference lines at Level-of-Service boundaries give the delay number some meaning.
const losLines = {
  id: "losLines",
  beforeDatasetsDraw(chart) {
    if (!chart.options.plugins.losLines?.enabled) return;
    const { ctx, chartArea: a, scales: { y } } = chart;
    ctx.save();
    ctx.strokeStyle = css("--axis");
    ctx.fillStyle = css("--muted");
    ctx.setLineDash([4, 4]);
    ctx.font = "10px system-ui, -apple-system, 'Segoe UI', sans-serif";
    for (const [v, label] of LOS_LINES) {
      if (v > y.max) continue;
      const py = y.getPixelForValue(v);
      ctx.beginPath(); ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); ctx.stroke();
      ctx.fillText(label, a.left + 4, py - 4);
    }
    ctx.restore();
  },
};

export class Charts {
  constructor(queueCanvas, delayCanvas) {
    this.ok = typeof window.Chart !== "undefined";
    if (!this.ok) {
      for (const c of [queueCanvas, delayCanvas]) {
        c.parentElement.textContent = "Charts unavailable (Chart.js could not be loaded).";
      }
      return;
    }
    this.arms = ["N", "E", "S", "W"];
    this.queue = new Chart(queueCanvas, this.#options({
      datasets: this.arms.map((a) => ({ label: ARM_NAMES[a], shortLabel: a, data: [], slot: ARM_SLOTS[a] })),
    }, { legend: true, yTitle: "vehicles" }));
    this.delay = new Chart(delayCanvas, this.#options({
      datasets: [{ label: "Mean delay", shortLabel: "", data: [], slot: "--ink-2" }],
    }, { legend: false, yTitle: "seconds", los: true }));
    this.applyTheme();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.applyTheme());
  }

  #options(data, { legend, yTitle, los }) {
    return {
      type: "line",
      data,
      plugins: [endLabels, losLines],
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
        layout: { padding: { right: legend ? 26 : 8 } },
        interaction: { mode: "index", intersect: false, axis: "x" },
        elements: { line: { borderWidth: 2, tension: 0.2 }, point: { radius: 0, hoverRadius: 4, hitRadius: 8 } },
        scales: {
          x: { type: "linear", ticks: { callback: (v) => clock(v), maxTicksLimit: 6 }, grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: yTitle }, ticks: { maxTicksLimit: 5, precision: 0 } },
        },
        plugins: {
          legend: { display: legend, position: "top", align: "start",
                    labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 8, boxHeight: 8 } },
          tooltip: { callbacks: { title: (items) => (items.length ? `t = ${clock(items[0].parsed.x)}` : "") } },
          losLines: { enabled: !!los },
        },
      },
    };
  }

  applyTheme() {
    if (!this.ok) return;
    const ink2 = css("--ink-2"), muted = css("--muted"), grid = css("--grid"), axis = css("--axis");
    for (const ch of [this.queue, this.delay]) {
      ch.data.datasets.forEach((ds) => { ds.borderColor = css(ds.slot); ds.backgroundColor = ds.borderColor; });
      for (const sc of Object.values(ch.options.scales)) {
        sc.ticks.color = muted;
        sc.grid.color = grid;
        sc.border = { color: axis };
        if (sc.title) sc.title.color = muted;
      }
      ch.options.plugins.legend.labels.color = ink2;
      ch.update("none");
    }
  }

  reset(rows) {
    if (!this.ok) return;
    for (const ch of [this.queue, this.delay]) ch.data.datasets.forEach((ds) => { ds.data = []; });
    this.addRows(rows);
  }

  addRows(rows) {
    if (!this.ok || !rows.length) return;
    for (const r of rows) {
      this.arms.forEach((a, i) => this.queue.data.datasets[i].data.push({ x: r.t, y: r[`queue_${a}`] }));
      if (r.mean_delay_window !== "" && r.mean_delay_window != null) {
        this.delay.data.datasets[0].data.push({ x: r.t, y: r.mean_delay_window });
      }
    }
    const tMax = rows.at(-1).t, tMin = Math.max(0, tMax - WINDOW_S);
    for (const ch of [this.queue, this.delay]) {
      ch.data.datasets.forEach((ds) => {
        let k = 0;
        while (k < ds.data.length && ds.data[k].x < tMin) k++;
        if (k) ds.data.splice(0, k);
      });
      ch.options.scales.x.min = tMin;
      ch.options.scales.x.max = Math.max(tMax, tMin + 60);
      ch.update("none");
    }
  }
}
