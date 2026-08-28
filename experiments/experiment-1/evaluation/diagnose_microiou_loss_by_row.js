const fs = require("fs");
const path = require("path");

const DIR = __dirname;

const variantPath = process.argv[2] ? path.resolve(process.argv[2]) : "";
const label = process.argv[3] || (variantPath ? path.basename(path.dirname(path.dirname(variantPath))) : "variant");
const outPath = process.argv[4]
  ? path.resolve(process.argv[4])
  : path.join(DIR, "../../outputs", `microiou_loss_rows_${label}.csv`);

const INPUTS = {
  gold: path.join(DIR, "../inputs/annotations/gold-iso-ranges.csv"),
  isoMap: path.join(DIR, "../inputs/config/time-string-iso-map.json"),
};

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

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function writeCsv(filePath, rows, fields) {
  const lines = [fields.join(",")];
  for (const row of rows) {
    lines.push(fields.map((field) => csvEscape(row[field])).join(","));
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `\uFEFF${lines.join("\n")}\n`, "utf8");
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

function aiRowsFromPrefill(rawRows, lookup) {
  return rawRows.map((row) => {
    const startText = normalizeTimeText(row.ai_timeblock_start_tm || row.ai_start_ym);
    const endText = normalizeTimeText(row.ai_timeblock_end_tm || row.ai_end_ym);
    const start = mapTimeText(startText, lookup, row.ai_start_ym);
    const end = mapTimeText(endText, lookup, row.ai_end_ym);
    const isoStart = start.iso;
    const isoEnd = end.iso;
    return {
      ...row,
      source: "ai_variant",
      state: row.ai_unknown ? "unknown" : row.ai_sink ? "sunk" : row.ai_interlude ? "interlude" : "time_range",
      iso_start: isoStart,
      iso_end: isoEnd,
      iso_range: isoStart && isoEnd ? `${isoStart}to${isoEnd}` : "",
      sink: row.ai_sink ? "true" : "",
      interlude: row.ai_interlude ? "true" : "",
      unresolved_time_texts: [start.unresolved, end.unresolved].filter(Boolean).join("|"),
    };
  });
}

function rangeLabel(range) {
  if (!range) return "";
  return `${range.start}..${range.end}(${range.size})`;
}

function classifyError(ai, gold, intersection, union, predKnown) {
  if (!isKnownRange(gold)) return "gold_not_known";
  if (!predKnown) return "missing_or_unknown";
  if (ai.unresolved_time_texts) return "unresolved_time_text";
  if (intersection === union && union > 0) return "exact";
  if (intersection === 0) return "disjoint";
  const predStart = monthToIndex(ai.iso_start);
  const predEnd = monthToIndex(ai.iso_end);
  const goldStart = monthToIndex(gold.iso_start);
  const goldEnd = monthToIndex(gold.iso_end);
  if (predStart != null && predEnd != null && predStart <= goldStart && predEnd >= goldEnd) return "too_wide";
  if (predStart != null && predEnd != null && predStart >= goldStart && predEnd <= goldEnd) return "too_narrow";
  return "partial_overlap";
}

function main() {
  if (!variantPath || !fs.existsSync(variantPath)) {
    throw new Error(`Variant CSV not found: ${variantPath}`);
  }
  const goldRows = readCsv(INPUTS.gold);
  const lookup = makeIsoLookup();
  const aiRows = aiRowsFromPrefill(readCsv(variantPath), lookup);
  const goldByKey = new Map(goldRows.map((row) => [rowKey(row), row]));
  const windows = buildGoldWindows(goldRows);
  const outRows = [];

  for (const ai of aiRows) {
    const gold = goldByKey.get(rowKey(ai));
    if (!gold || !isKnownRange(gold)) continue;
    const window = windows[gold.case_id];
    const goldRange = clippedRange(gold, window);
    const predKnown = isKnownRange(ai);
    const predRange = predKnown ? clippedRange(ai, window) : { size: 0 };
    const intersection = intersectionSize(goldRange, predRange);
    const union = unionSize(goldRange, predRange);
    const loss = union - intersection;
    outRows.push({
      label,
      case_id: ai.case_id,
      part_id: ai.part_id,
      item_no: ai.item_no,
      sentence_id: ai.sentence_id,
      sentence: ai.sentence,
      gold_iso_range: gold.iso_range,
      ai_iso_range: ai.iso_range,
      gold_clipped: rangeLabel(goldRange),
      ai_clipped: rangeLabel(predRange),
      intersection_months: intersection,
      union_months: union,
      loss_months: loss,
      row_iou: union ? (intersection / union).toFixed(6) : "",
      error_type: classifyError(ai, gold, intersection, union, predKnown),
      ai_tm: ai.ai_tm,
      ai_timeblock_id: ai.ai_timeblock_id,
      ai_crossdoc_used: ai.ai_crossdoc_used,
      ai_crossdoc_source_timeblock: ai.ai_crossdoc_source_timeblock,
      unresolved_time_texts: ai.unresolved_time_texts,
    });
  }
  outRows.sort((a, b) => Number(b.loss_months) - Number(a.loss_months));
  const fields = [
    "label",
    "case_id",
    "part_id",
    "item_no",
    "sentence_id",
    "sentence",
    "gold_iso_range",
    "ai_iso_range",
    "gold_clipped",
    "ai_clipped",
    "intersection_months",
    "union_months",
    "loss_months",
    "row_iou",
    "error_type",
    "ai_tm",
    "ai_timeblock_id",
    "ai_crossdoc_used",
    "ai_crossdoc_source_timeblock",
    "unresolved_time_texts",
  ];
  writeCsv(outPath, outRows, fields);
  console.log(`Wrote ${outRows.length} loss rows to ${outPath}`);
  console.log(JSON.stringify(outRows.slice(0, 12).map((row) => ({
    sentence_id: row.sentence_id,
    loss_months: row.loss_months,
    error_type: row.error_type,
    gold_iso_range: row.gold_iso_range,
    ai_iso_range: row.ai_iso_range,
    ai_crossdoc_used: row.ai_crossdoc_used,
  })), null, 2));
}

main();
