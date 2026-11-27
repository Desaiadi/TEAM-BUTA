/* Shared chart helpers for the BUTA building reports (Chart.js 4).
 *
 * Every chart: thin bars with rounded data ends, 2px lines, a hover tooltip,
 * a legend for 2+ series, and a "View as table" toggle under it, so no value
 * depends on color alone. Problem types keep the same color everywhere.
 *
 * Colors follow the BUTA palette: teal is always the subject of the chart
 * (this building, this landlord, the model) and grey-green is always the
 * reference it is measured against.
 */
const PALETTE = {
  // Fixed categorical order (validated for color-vision deficiency, adjacent pairs)
  typeColors: {
    "heat/hot water": "#2a78d6", "pests": "#d87844", "leaks/mold": "#178570", "plumbing": "#d9a300",
    "paint/plaster": "#d1719c", "doors/windows": "#1f6f4a", "safety": "#6250d6",
  },
  teal: "#178570",              // the subject: this building / main model
  orange: "#d87844",            // the counterpart, as on the compare page
  navy: "#142e32",
  grey: "#a3b5ae",              // reference: typical building, NYC, baseline
  ordinal: ["#a7ddcb", "#2f9b83", "#16403c"],   // light → dark (A/B/C, Low/Medium/High)
  ink: "#183537", muted: "#647575", grid: "#e7edea",
};

Chart.defaults.font.family = '"DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
Chart.defaults.font.size = 13;
Chart.defaults.color = PALETTE.muted;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.boxHeight = 12;
Chart.defaults.plugins.legend.labels.color = PALETTE.ink;
Chart.defaults.plugins.tooltip.backgroundColor = PALETTE.navy;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.maintainAspectRatio = false;

const fmt = {
  pct: (v) => (v == null ? "–" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`),
  int: (v) => (v == null ? "–" : Math.round(v).toLocaleString()),
  dec: (v) => (v == null ? "–" : v.toFixed(3)),
  one: (v) => (v == null ? "–" : v.toFixed(1)),
};

/** Build a chart in the <canvas id=…> and a table view in the <details> after it.
 *  kind: "bar" | "line"; opts: {stacked, horizontal, format, yTitle, yMin, yMax, legend, tickEvery} */
function butaChart(id, kind, labels, datasets, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return null;
  Chart.getChart(el)?.destroy();                   // redraw after a filter change
  const f = fmt[opts.format || "int"];
  const valueAxis = {
    beginAtZero: opts.yMin == null, min: opts.yMin, max: opts.yMax, stacked: !!opts.stacked,
    grid: { color: PALETTE.grid }, border: { display: false },
    // Whole-number charts get whole-number ticks; otherwise an empty chart labels 0.2, 0.4… as "0, 0, 1, 1".
    ticks: { callback: (v) => f(v), maxTicksLimit: 6, precision: opts.format === "int" ? 0 : undefined },
    title: opts.yTitle ? { display: true, text: opts.yTitle, color: PALETTE.muted } : undefined,
  };
  const catAxis = { stacked: !!opts.stacked, grid: { display: false }, border: { color: PALETTE.grid },
                    ticks: { color: PALETTE.ink, autoSkip: !opts.tickEvery && labels.length > 12,
                             maxRotation: labels.length > 12 ? 0 : 40,
                             // tickEvery: n labels every n-th category, plus the last one. Only set when
                             // asked: an explicit `callback: undefined` would replace Chart.js's labels with indices.
                             ...(opts.tickEvery && { callback: (v, i) => (i % opts.tickEvery === 0 || i === labels.length - 1 ? labels[i] : null) }) } };
  const ds = datasets.map((d) => kind === "bar"
    ? { borderRadius: opts.stacked ? 0 : 4, borderSkipped: "start", borderColor: "#ffffff",
        borderWidth: opts.stacked ? { top: 2 } : 0, maxBarThickness: opts.horizontal ? 18 : 28,
        backgroundColor: d.color, ...d }
    : { borderWidth: 2, pointRadius: d.points ? 4 : 0, pointHoverRadius: 5, tension: 0.25, fill: false,
        borderColor: d.color, backgroundColor: d.color, ...d });
  const chart = new Chart(el, {
    type: kind,
    data: { labels, datasets: ds },
    options: {
      indexAxis: opts.horizontal ? "y" : "x",
      interaction: { mode: kind === "line" ? "index" : "nearest", intersect: kind !== "line", axis: opts.horizontal ? "y" : "x" },
      scales: opts.horizontal ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
      plugins: {
        legend: { display: opts.legend ?? datasets.length > 1, position: "top", align: "start" },
        tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${f(c.parsed[opts.horizontal ? "x" : "y"])}` } },
      },
    },
  });
  addTable(el, labels, datasets, f, opts.labelHeader || "");
  return chart;
}

/** "View as table" under a chart: same numbers, no color needed. */
function addTable(canvas, labels, datasets, f, labelHeader) {
  const box = canvas.closest(".chart");
  if (!box) return;
  const old = box.nextElementSibling;              // redraw: replace the old table
  if (old && old.classList.contains("table-view")) old.remove();
  const rows = labels.map((l, i) => `<tr><td>${l}</td>${datasets.map((d) => `<td>${f(d.data[i])}</td>`).join("")}</tr>`);
  box.insertAdjacentHTML("afterend",
    `<details class="table-view"><summary>View as table</summary><div class="table-scroll"><table>
      <tr><th>${labelHeader}</th>${datasets.map((d) => `<th>${d.label}</th>`).join("")}</tr>${rows.join("")}
    </table></div></details>`);
}
