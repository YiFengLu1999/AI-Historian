const fs = require("fs");
const path = require("path");

const DIR = __dirname;
const PACKAGE_DIR = path.resolve(DIR, "../..");
const DEFAULT_VARIANT = path.join(
  PACKAGE_DIR,
  "direct-llm-results/generated_results_direct_llm_20260615_190423/tables/all_cases_direct_llm_prefill.csv",
);

const variantPath = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_VARIANT;
const label = process.argv[3] || path.basename(path.dirname(path.dirname(variantPath)));

const INPUTS = {
  gold: path.join(PACKAGE_DIR, "inputs/annotations/gold-iso-ranges.csv"),
  human: path.join(PACKAGE_DIR, "inputs/annotations/human-iso-ranges.csv"),
  isoMap: path.join(PACKAGE_DIR, "inputs/config/time-string-iso-map.json"),
};
const EPS = 1e-6;

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
      continue;
    }
    if (ch === '"') inQuotes = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") field += ch;
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  const header = (rows.shift() || []).map((key) => key.replace(/^\uFEFF/, ""));
  return rows
    .filter((values) => values.some((value) => value !== ""))
    .map((values) => Object.fromEntries(header.map((key, index) => [key, values[index] || ""])));
}

function readCsv(filePath) {
  return parseCsv(fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
}

function monthPrecisionIso(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  if (text === "-infinity" || text === "+infinity") return text;
  const match = text.match(/^([+-]?\d{4,})-(\d{2})(?:-\d{2})?$/);
  return match ? `${match[1]}-${match[2]}` : text;
}

function normalizeTimeText(value) {
  return String(value || "").trim();
}

function makeIsoLookup() {
  const data = JSON.parse(fs.readFileSync(INPUTS.isoMap, "utf8"));
  const lookup = new Map();
  for (const [key, value] of Object.entries(data.map || {})) {
    lookup.set(normalizeTimeText(key), monthPrecisionIso(value));
  }
  return lookup;
}

function mapTimeText(value, lookup, fallbackIso = "") {
  const text = normalizeTimeText(value);
  const fallback = monthPrecisionIso(fallbackIso);
  if (!text || text === "无") return { iso: fallback, unresolved: fallback ? "" : "" };
  if (text === "-infinity" || text === "+infinity") return { iso: text, unresolved: "" };
  if (lookup.has(text)) return { iso: lookup.get(text), unresolved: "" };
  return { iso: fallback, unresolved: fallback ? "" : text };
}

function monthToIndex(value) {
  const match = String(value || "").match(/^([+-]?\d{4,})-(\d{2})$/);
  return match ? Number(match[1]) * 12 + Number(match[2]) - 1 : null;
}

function rowKey(row) {
  return `${row.case_id}::${row.part_id}::${row.item_no}::${row.sentence_id}`;
}

function isSunk(row) {
  return ["sunk", "sink", "interlude"].includes(row.state) || row.sink === "true" || row.interlude === "true";
}

function isKnownRange(row) {
  return !isSunk(row) && row.state !== "unknown" && Boolean(row.iso_start && row.iso_end && row.iso_range);
}

function buildGoldWindows(goldRows) {
  const windows = {};
  for (const row of goldRows) {
    if (!windows[row.case_id]) windows[row.case_id] = { start: null, end: null };
    for (const boundary of [row.iso_start, row.iso_end]) {
      const idx = monthToIndex(boundary);
      if (idx == null) continue;
      windows[row.case_id].start = windows[row.case_id].start == null ? idx : Math.min(windows[row.case_id].start, idx);
      windows[row.case_id].end = windows[row.case_id].end == null ? idx : Math.max(windows[row.case_id].end, idx);
    }
  }
  return windows;
}

function clippedRange(row, window) {
  if (!window || window.start == null || window.end == null) return null;
  let start = row.iso_start === "-infinity" ? window.start : monthToIndex(row.iso_start);
  let end = row.iso_end === "+infinity" ? window.end : monthToIndex(row.iso_end);
  if (start == null || end == null) return null;
  start = Math.max(start, window.start);
  end = Math.min(end, window.end);
  return { start, end, size: start <= end ? end - start + 1 : 0 };
}

function intersectionSize(a, b) {
  if (!a || !b || a.size === 0 || b.size === 0) return 0;
  const start = Math.max(a.start, b.start);
  const end = Math.min(a.end, b.end);
  return start <= end ? end - start + 1 : 0;
}

function unionSize(a, b) {
  if (!a || a.size === 0) return b ? b.size : 0;
  if (!b || b.size === 0) return a.size;
  return a.size + b.size - intersectionSize(a, b);
}

function scoreRows(predRows, goldByKey, windows) {
  const score = {
    rows: 0,
    goldKnownRows: 0,
    intersectionMonths: 0,
    unionMonths: 0,
    missingRows: 0,
    unresolvedRows: 0,
  };
  for (const pred of predRows) {
    const gold = goldByKey.get(rowKey(pred));
    if (!gold) continue;
    score.rows += 1;
    if (pred.unresolved_time_texts) score.unresolvedRows += 1;
    if (!isKnownRange(gold)) continue;
    score.goldKnownRows += 1;
    const goldRange = clippedRange(gold, windows[gold.case_id]);
    const predRange = isKnownRange(pred) ? clippedRange(pred, windows[gold.case_id]) : { size: 0 };
    if (!isKnownRange(pred)) score.missingRows += 1;
    score.intersectionMonths += intersectionSize(goldRange, predRange);
    score.unionMonths += unionSize(goldRange, predRange);
  }
  return {
    ...score,
    microIoU: score.unionMonths ? score.intersectionMonths / score.unionMonths : null,
  };
}

function aiRowsFromPrefill(rawRows, lookup) {
  return rawRows.map((row) => {
    const startIso = monthPrecisionIso(row.ai_start_ym);
    const endIso = monthPrecisionIso(row.ai_end_ym);
    const startText = normalizeTimeText(startIso || row.ai_timeblock_start_tm);
    const endText = normalizeTimeText(endIso || row.ai_timeblock_end_tm);
    const start = startIso
      ? { iso: startIso, unresolved: "" }
      : mapTimeText(startText, lookup, row.ai_start_ym);
    const end = endIso
      ? { iso: endIso, unresolved: "" }
      : mapTimeText(endText, lookup, row.ai_end_ym);
    const isoStart = start.iso;
    const isoEnd = end.iso;
    return {
      source: "ai_variant",
      case_id: row.case_id,
      part_id: row.part_id,
      item_no: row.item_no,
      sentence_id: row.sentence_id,
      source_text: row.source_text,
      sentence_text: row.sentence,
      state: row.ai_unknown ? "unknown" : row.ai_sink ? "sunk" : row.ai_interlude ? "interlude" : "time_range",
      iso_start: isoStart,
      iso_end: isoEnd,
      iso_range: isoStart && isoEnd ? `${isoStart}to${isoEnd}` : "",
      sink: row.ai_sink ? "true" : "",
      interlude: row.ai_interlude ? "true" : "",
      ai_crossdoc_used: row.ai_crossdoc_used ? "true" : "",
      unresolved_time_texts: [start.unresolved, end.unresolved].filter(Boolean).join("|"),
    };
  });
}

function fmt(value) {
  return value == null ? null : Number(value.toFixed(6));
}

function ceilingAwarePassed(aiMicroIoU, baselineMicroIoU) {
  if (aiMicroIoU == null || baselineMicroIoU == null || !Number.isFinite(baselineMicroIoU)) {
    return false;
  }
  if (baselineMicroIoU >= 1 - EPS) return aiMicroIoU >= baselineMicroIoU - EPS;
  return aiMicroIoU > baselineMicroIoU + EPS;
}

function main() {
  if (!fs.existsSync(variantPath)) {
    throw new Error(`Variant CSV not found: ${variantPath}`);
  }
  const goldRows = readCsv(INPUTS.gold);
  const humanRows = readCsv(INPUTS.human);
  const lookup = makeIsoLookup();
  const aiRows = aiRowsFromPrefill(readCsv(variantPath), lookup);
  const goldByKey = new Map(goldRows.map((row) => [rowKey(row), row]));
  const windows = buildGoldWindows(goldRows);
  const cases = [...new Set(goldRows.map((row) => row.case_id))].sort();

  const rows = cases.map((caseId) => {
    const humanOnly = scoreRows(humanRows.filter((row) => row.case_id === caseId && row.condition === "Human-only"), goldByKey, windows);
    const humanAi = scoreRows(humanRows.filter((row) => row.case_id === caseId && row.condition === "Human+AI"), goldByKey, windows);
    const ai = scoreRows(aiRows.filter((row) => row.case_id === caseId), goldByKey, windows);
    const humanBest = Math.max(humanOnly.microIoU ?? -Infinity, humanAi.microIoU ?? -Infinity);
    const humanBaseline = humanOnly.microIoU;
    const passed = ceilingAwarePassed(ai.microIoU, humanBaseline);
    return {
      caseId,
      aiMicroIoU: fmt(ai.microIoU),
      humanOnlyMicroIoU: fmt(humanOnly.microIoU),
      humanAiMicroIoU: fmt(humanAi.microIoU),
      humanBaselineCondition: "Human-only",
      humanBaselineMicroIoU: fmt(humanBaseline),
      humanBestMicroIoU: fmt(humanBest),
      aiMinusHumanBaseline: fmt(ai.microIoU == null || humanBaseline == null ? null : ai.microIoU - humanBaseline),
      aiMinusHumanOnly: fmt(ai.microIoU == null || humanOnly.microIoU == null ? null : ai.microIoU - humanOnly.microIoU),
      aiMinusHumanBest: fmt((ai.microIoU ?? 0) - humanBest),
      aiRows: ai.rows,
      aiMissingRows: ai.missingRows,
      aiUnresolvedRows: ai.unresolvedRows,
      aiCrossdocRows: aiRows.filter((row) => row.case_id === caseId && row.ai_crossdoc_used === "true").length,
      strictlyPassed: (ai.microIoU ?? -Infinity) > humanBaseline + EPS,
      ceilingAwarePassed: passed,
      passed,
    };
  });

  const summary = {
    label,
    variantPath,
    generatedAt: new Date().toISOString(),
    passedAllCases: rows.every((row) => row.passed),
    failingCases: rows.filter((row) => !row.passed).map((row) => row.caseId),
    rows,
  };
  const outPath = path.join(PACKAGE_DIR, "recomputed", `ai_variant_score_${label.replace(/[^A-Za-z0-9_-]+/g, "_")}.json`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(summary, null, 2));
  console.error(`Wrote ${outPath}`);
}

main();
