const fs = require("fs");
const path = require("path");

const PACKAGE_DIR = path.resolve(__dirname, "..");
const RECOMPUTED_ROOT = process.env.AIH_RECOMPUTED_ROOT
  ? path.resolve(process.env.AIH_RECOMPUTED_ROOT)
  : "";
const FIGURE_DIR = RECOMPUTED_ROOT
  ? path.join(RECOMPUTED_ROOT, "figure")
  : path.join(PACKAGE_DIR, "figure");
const RESULTS_DIR = path.join(PACKAGE_DIR, "results");

const FILES = {
  humanMetrics: RECOMPUTED_ROOT
    ? path.join(RECOMPUTED_ROOT, "human/experiment-2-human-accuracy-metrics.json")
    : path.join(RESULTS_DIR, "human/experiment-2-human-accuracy-metrics.json"),
  llmLatestPointer: path.join(PACKAGE_DIR, "outputs/latest_direct_llm_output_dir.txt"),
  structuredLatestPointer: path.join(PACKAGE_DIR, "outputs/latest_structured_llm_output_dir.txt"),
  frozenDirect: path.join(RESULTS_DIR, "direct-llm"),
  frozenStructured: path.join(RESULTS_DIR, "structured-llm"),
};

const METHODS = [
  { id: "human", label: "Human", color: "#b7791f" },
  { id: "llm_subagent", label: "Direct LLM", color: "#2f7d62" },
  { id: "aih_agent_api", label: "Structured LLM", color: "#3b6fb6" },
];

const BLOCKS = [
  { id: "A", label: "Block A" },
  { id: "B", label: "Block B" },
  { id: "C", label: "Block C" },
];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8").trim();
}

function fmtPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function fmtTime(seconds) {
  if (!Number.isFinite(seconds)) return "n/a";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = Math.round(seconds % 60);
  if (hours) return `${hours}h ${minutes}m`;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

function groupStrictRows(detail) {
  const grouped = new Map();
  for (const row of detail) {
    const key = `${row.form_id}::${row.block}::${row.question_id}`;
    if (!grouped.has(key)) {
      grouped.set(key, { block: row.block, correct: 0, total: 0 });
    }
    const item = grouped.get(key);
    item.correct += Number(row.is_correct || 0);
    item.total += 1;
  }
  return Array.from(grouped.values()).map((row) => ({
    block: row.block,
    isCorrect: row.total > 0 && row.correct === row.total ? 1 : 0,
  }));
}

function aggregateStrictByBlock(detail) {
  const rows = groupStrictRows(detail);
  const byBlock = new Map();
  for (const row of rows) {
    if (!byBlock.has(row.block)) byBlock.set(row.block, { correct: 0, total: 0 });
    const item = byBlock.get(row.block);
    item.correct += row.isCorrect;
    item.total += 1;
  }
  return BLOCKS.map((block) => {
    const item = byBlock.get(block.id) || { correct: 0, total: 0 };
    return {
      block: block.id,
      correct: item.correct,
      total: item.total,
      accuracy: item.total ? item.correct / item.total : 0,
    };
  });
}

function aggregateLlmElapsedByBlock(llmOutputDir, totalSeconds) {
  const rawFiles = fs
    .readdirSync(llmOutputDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^run_\d+$/.test(entry.name))
    .map((entry) => path.join(llmOutputDir, entry.name, "raw_responses.json"))
    .filter((filePath) => fs.existsSync(filePath));

  const elapsedByBlock = new Map(BLOCKS.map((block) => [block.id, 0]));
  for (const filePath of rawFiles) {
    const rows = readJson(filePath);
    for (const row of rows) {
      const block = row?.question?.block || row?.row?.block;
      if (!elapsedByBlock.has(block)) continue;
      elapsedByBlock.set(block, elapsedByBlock.get(block) + Number(row.elapsed_seconds || 0));
    }
  }

  const elapsedTotal = Array.from(elapsedByBlock.values()).reduce((sum, value) => sum + value, 0);
  return BLOCKS.map((block) => {
    const rawElapsed = elapsedByBlock.get(block.id) || 0;
    return {
      block: block.id,
      seconds: elapsedTotal > 0 ? (rawElapsed / elapsedTotal) * totalSeconds : totalSeconds / BLOCKS.length,
      rawElapsedSeconds: rawElapsed,
    };
  });
}

function aggregateRawElapsedByBlock(rawRows, totalSeconds) {
  const elapsedByBlock = new Map(BLOCKS.map((block) => [block.id, 0]));
  for (const row of rawRows) {
    const block = row?.question?.block || row?.row?.block;
    if (!elapsedByBlock.has(block)) continue;
    elapsedByBlock.set(block, elapsedByBlock.get(block) + Number(row.elapsed_seconds || 0));
  }
  const total = Array.from(elapsedByBlock.values()).reduce((sum, value) => sum + value, 0);
  return BLOCKS.map((block) => ({
    block: block.id,
    seconds: total > 0 ? (elapsedByBlock.get(block.id) / total) * totalSeconds : totalSeconds / BLOCKS.length,
  }));
}

function byBlockLookup(rows, blockKey = "block") {
  return new Map(rows.map((row) => [row[blockKey], row]));
}

function buildData() {
  const human = readJson(FILES.humanMetrics);
  const llmOutputDir = fs.existsSync(FILES.llmLatestPointer) ? readText(FILES.llmLatestPointer) : FILES.frozenDirect;
  const llmScorePath = path.join(llmOutputDir, "experiment-2-llm-subagents-score.json");
  const llm = readJson(llmScorePath);
  const aihOutputDir = fs.existsSync(FILES.structuredLatestPointer) ? readText(FILES.structuredLatestPointer) : FILES.frozenStructured;
  const aihScorePath = aihOutputDir ? path.join(aihOutputDir, "experiment-2-aih-agent-api-score.json") : "";
  const aihRawPath = aihOutputDir ? path.join(aihOutputDir, "raw-responses.json") : "";
  const aih = aihScorePath && fs.existsSync(aihScorePath) ? readJson(aihScorePath) : null;
  const aihRaw = aihRawPath && fs.existsSync(aihRawPath) ? readJson(aihRawPath) : [];

  const humanStrictByBlock = byBlockLookup(human.strict.byBlock);
  const humanTimingByBlock = byBlockLookup(human.timing.byBlock);
  const llmStrictByBlock = byBlockLookup(aggregateStrictByBlock(llm.detail));
  const llmTimingByBlock = byBlockLookup(aggregateLlmElapsedByBlock(llmOutputDir, llm.total_seconds));
  const aihStrictByBlock = aih ? byBlockLookup(aggregateStrictByBlock(aih.detail)) : new Map();
  const aihTimingByBlock = aih ? byBlockLookup(aggregateRawElapsedByBlock(aihRaw, aih.total_seconds || 0)) : new Map();
  const aihMethodText = !aih
    ? ""
    : aih.evidence_mode === "visible_only"
      ? "Structured LLM rows use human-visible question text only for all three blocks, with block-specific structured prompts and three-run majority vote."
      : "Structured LLM rows use additional upstream sentence/timeblock/cross-document evidence with three-run majority vote.";

  const overall = [
    {
      method: "human",
      label: "Human",
      color: METHODS[0].color,
      strictAccuracy: human.strict.overall.accuracy,
      correct: human.strict.overall.correct,
      total: human.strict.overall.total,
      seconds: human.timing.totalSeconds,
    },
    {
      method: "llm_subagent",
      label: METHODS[1].label,
      color: METHODS[1].color,
      strictAccuracy: llm.row_strict_accuracy,
      correct: llm.row_strict_correct,
      total: llm.row_strict_total,
      seconds: llm.total_seconds,
    },
  ];
  if (aih) {
    overall.push({
      method: "aih_agent_api",
      label: METHODS[2].label,
      color: METHODS[2].color,
      strictAccuracy: aih.row_strict_accuracy,
      correct: aih.row_strict_correct,
      total: aih.row_strict_total,
      seconds: aih.total_seconds || 0,
    });
  }

  const blocks = BLOCKS.map((block) => {
    const humanStrict = humanStrictByBlock.get(block.id);
    const humanTime = humanTimingByBlock.get(block.id);
    const llmStrict = llmStrictByBlock.get(block.id);
    const llmTime = llmTimingByBlock.get(block.id);
    const values = [
      {
        method: "human",
        label: "Human",
        color: METHODS[0].color,
        strictAccuracy: humanStrict.accuracy,
        correct: humanStrict.correct,
        total: humanStrict.total,
        seconds: humanTime.seconds,
      },
      {
        method: "llm_subagent",
        label: METHODS[1].label,
        color: METHODS[1].color,
        strictAccuracy: llmStrict.accuracy,
        correct: llmStrict.correct,
        total: llmStrict.total,
        seconds: llmTime.seconds,
      },
    ];
    if (aih) {
      const aihStrict = aihStrictByBlock.get(block.id) || { accuracy: 0, correct: 0, total: 0 };
      const aihTime = aihTimingByBlock.get(block.id) || { seconds: 0 };
      values.push({
        method: "aih_agent_api",
        label: METHODS[2].label,
        color: METHODS[2].color,
        strictAccuracy: aihStrict.accuracy,
        correct: aihStrict.correct,
        total: aihStrict.total,
        seconds: aihTime.seconds,
      });
    }
    return {
      id: block.id,
      label: block.label,
      values,
    };
  });

  return {
    generatedAt: new Date().toISOString(),
    metric: "row_strict_accuracy",
    overall,
    blocks,
    timeNotes: {
      human: "Human block times are read from participant timing records.",
      llm: "Direct LLM block times allocate total wall-clock runtime by each block's share of per-question API elapsed seconds across the three runs.",
      aihAgent: "Structured LLM block times allocate total wall-clock runtime by each block's share of per-question API elapsed seconds across the three runs.",
    },
    methodNotes: {
      aihAgent: aihMethodText,
    },
    sources: {
      humanMetrics: "experiment-2-human-accuracy-metrics.json",
      llmSource: "latest direct-LLM consensus output",
      structuredLlmSource: "latest structured-LLM consensus output",
    },
  };
}

function chartScript(data) {
  return `
const DATA = ${JSON.stringify(data, null, 2)};

function fmtPct(value) { return (value * 100).toFixed(1) + '%'; }
function fmtTime(seconds) {
  if (!Number.isFinite(seconds)) return 'n/a';
  if (seconds < 60) return seconds.toFixed(0) + 's';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.round(seconds % 60);
  if (h) return h + 'h ' + m + 'm';
  return s ? m + 'm ' + s + 's' : m + 'm';
}

function svgEl(tag, attrs = {}, text = '') {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text) el.textContent = text;
  return el;
}

function drawGroupedChart(svgId, groups) {
  const svg = document.getElementById(svgId);
  const width = 1040, height = 430;
  const margin = { top: 30, right: 24, bottom: 112, left: 72 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
  svg.innerHTML = '';

  const y = (v) => margin.top + plotH - Math.max(0, Math.min(1, v)) * plotH;
  [0, .25, .5, .75, 1].forEach((tick) => {
    const yy = y(tick);
    svg.appendChild(svgEl('line', { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: 'grid' }));
    svg.appendChild(svgEl('text', { x: margin.left - 12, y: yy + 4, 'text-anchor': 'end', class: 'axis' }, Math.round(tick * 100) + '%'));
  });

  const groupW = plotW / groups.length;
  const maxBars = Math.max(...groups.map((group) => group.values.length));
  const barW = Math.min(72, groupW * 0.2);
  const innerGap = 18;

  groups.forEach((group, groupIndex) => {
    const center = margin.left + groupW * groupIndex + groupW / 2;
    const totalBarsW = group.values.length * barW + (group.values.length - 1) * innerGap;
    const startX = center - totalBarsW / 2;
    group.values.forEach((item, itemIndex) => {
      const x = startX + itemIndex * (barW + innerGap);
      const yy = y(item.strictAccuracy);
      const h = margin.top + plotH - yy;
      svg.appendChild(svgEl('rect', { x, y: yy, width: barW, height: h, rx: 6, fill: item.color }));
      svg.appendChild(svgEl('text', { x: x + barW / 2, y: yy - 10, 'text-anchor': 'middle', class: 'bar-label' }, fmtPct(item.strictAccuracy)));
      svg.appendChild(svgEl('text', { x: x + barW / 2, y: margin.top + plotH + 42 + itemIndex * 20, 'text-anchor': 'middle', class: 'time-label' }, item.label + ': ' + item.correct + '/' + item.total + ', ' + fmtTime(item.seconds)));
    });
    svg.appendChild(svgEl('text', { x: center, y: margin.top + plotH + 24, 'text-anchor': 'middle', class: 'category' }, group.label));
  });

  svg.appendChild(svgEl('text', {
    x: 18,
    y: margin.top + plotH / 2,
    'text-anchor': 'middle',
    transform: 'rotate(-90 18 ' + (margin.top + plotH / 2) + ')',
    class: 'axis-title',
  }, 'Strict accuracy'));
}

function fillTables() {
  document.getElementById('generatedAt').textContent = new Date(DATA.generatedAt).toLocaleString();
  const rows = [];
  rows.push(...DATA.overall.map((item) => ({ group: 'Overall', ...item })));
  DATA.blocks.forEach((block) => block.values.forEach((item) => rows.push({ group: block.label, ...item })));
  document.getElementById('metricsBody').innerHTML = rows.map((item) => '<tr>' +
    '<td>' + item.group + '</td>' +
    '<td>' + item.label + '</td>' +
    '<td>' + fmtPct(item.strictAccuracy) + '</td>' +
    '<td>' + item.correct + ' / ' + item.total + '</td>' +
    '<td>' + fmtTime(item.seconds) + '</td>' +
  '</tr>').join('');

  const best = DATA.overall.reduce((a, b) => a.strictAccuracy >= b.strictAccuracy ? a : b);
  const human = DATA.overall.find((item) => item.method === 'human');
  const diff = best.strictAccuracy - human.strictAccuracy;
  const timeRatio = best.seconds > 0 ? human.seconds / best.seconds : Infinity;
  document.getElementById('takeaway').textContent =
    'Best overall strict accuracy: ' + best.label + ' is ' + (diff * 100).toFixed(1) +
    ' percentage points above Human; displayed runtime ratio vs Human is ' + (Number.isFinite(timeRatio) ? timeRatio.toFixed(1) + 'x' : 'n/a') + '.';
}

drawGroupedChart('overallChart', [{ label: 'Overall', values: DATA.overall }]);
drawGroupedChart('blockChart', DATA.blocks);
fillTables();
`;
}

function html(data) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment 2: Strict Accuracy + Time</title>
  <style>
    :root {
      --bg: #f7f5f0;
      --ink: #20231f;
      --muted: #626a62;
      --line: #d9ddd6;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
    }
    header {
      padding: 34px 42px 26px;
      border-bottom: 1px solid var(--line);
      background: #efeee8;
    }
    h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }
    .sub { margin: 0; color: var(--muted); font-size: 16px; line-height: 1.5; max-width: 1100px; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px 24px 54px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px 24px 18px;
      margin-bottom: 22px;
    }
    h2 { margin: 0 0 4px; font-size: 22px; }
    .note { margin: 0 0 16px; color: var(--muted); line-height: 1.45; }
    .legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 14px 0 4px; color: var(--muted); }
    .legend-item { display: inline-flex; align-items: center; gap: 8px; }
    .swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
    svg { width: 100%; height: auto; display: block; }
    .grid { stroke: #e4e7e1; stroke-width: 1; }
    .axis, .time-label { fill: #6d746c; font-size: 12px; }
    .axis-title { fill: #5f675f; font-size: 13px; font-weight: 650; }
    .bar-label { fill: #20231f; font-size: 15px; font-weight: 700; }
    .category { fill: #20231f; font-size: 14px; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); background: #f8faf8; }
    .takeaway {
      padding: 14px 16px;
      border: 1px solid #cfd9d1;
      border-radius: 8px;
      background: #f4faf5;
      color: #245b43;
      font-weight: 650;
      line-height: 1.45;
    }
    .foot {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      margin-top: 12px;
    }
    code { background: #f0f1ed; padding: 1px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>Experiment 2 Strict Accuracy + Time</h1>
    <p class="sub">Comparison of human performance, direct LLM prompting, and structured LLM prompting on Experiment 2. Bar height is strict row accuracy; labels under each bar show correct rows and elapsed time.</p>
    <div class="legend">
      ${METHODS.map((item) => `<span class="legend-item"><span class="swatch" style="background:${item.color}"></span>${item.label}</span>`).join("")}
    </div>
  </header>
  <main>
    <section>
      <h2>Overall</h2>
      <p class="note">Strict accuracy counts a row as correct only when every required field in that question is correct.</p>
      <svg id="overallChart" role="img" aria-label="Experiment 2 overall strict accuracy and time chart"></svg>
    </section>
    <section>
      <h2>By Block</h2>
      <p class="note">Block A requires time information, sink, and interlude to all match. Blocks B and C require the selected choice to match.</p>
      <svg id="blockChart" role="img" aria-label="Experiment 2 strict accuracy and time by block chart"></svg>
    </section>
    <section>
      <h2>Summary</h2>
      <p id="takeaway" class="takeaway"></p>
      <table>
        <thead><tr><th>Scope</th><th>Condition</th><th>Strict accuracy</th><th>Correct / total</th><th>Time</th></tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </section>
    <p class="foot">
      Generated at <span id="generatedAt"></span>. Human rows aggregate the six participant Experiment 2 sheets. Direct LLM rows aggregate the T1, T2, and T3 consensus output from three model runs. ${data.methodNotes.aihAgent} Metric source: <code>row_strict_accuracy</code>. ${data.timeNotes.llm} ${data.timeNotes.aihAgent}
    </p>
  </main>
  <script>${chartScript(data)}</script>
</body>
</html>
`;
}

function main() {
  const data = buildData();
  fs.mkdirSync(FIGURE_DIR, { recursive: true });
  const outHtml = path.join(FIGURE_DIR, "experiment-2-strict-total-comparison.html");
  const outMetrics = path.join(FIGURE_DIR, "experiment-2-strict-total-comparison-metrics.json");
  fs.writeFileSync(outHtml, html(data), "utf8");
  fs.writeFileSync(outMetrics, JSON.stringify(data, null, 2), "utf8");
  console.log(`Wrote ${outHtml}`);
  console.log(`Wrote ${outMetrics}`);
  for (const row of data.overall) {
    console.log(`Overall ${row.label}: ${fmtPct(row.strictAccuracy)} | ${row.correct}/${row.total} | ${fmtTime(row.seconds)}`);
  }
  for (const block of data.blocks) {
    for (const row of block.values) {
      console.log(`${block.label} ${row.label}: ${fmtPct(row.strictAccuracy)} | ${row.correct}/${row.total} | ${fmtTime(row.seconds)}`);
    }
  }
}

main();
