const fs = require("fs");
const path = require("path");

const baselinePath = process.argv[2] ? path.resolve(process.argv[2]) : "";
const candidatePath = process.argv[3] ? path.resolve(process.argv[3]) : "";
const mode = process.argv[4] || "non-regression";
const requestedCases = process.argv[5]
  ? new Set(process.argv[5].split(",").map((item) => item.trim()).filter(Boolean))
  : null;

if (!baselinePath || !candidatePath) {
  console.error("Usage: node compare_ai_score_to_baseline.js <baseline_score.json> <candidate_score.json> [non-regression|human-baseline] [case_ids_csv]");
  process.exit(2);
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function byCase(scoreJson) {
  return new Map((scoreJson.rows || []).map((row) => [row.caseId, row]));
}

function fmt(value) {
  if (value == null || Number.isNaN(Number(value))) return "";
  return Number(value).toFixed(6);
}

function okForMode(base, cand) {
  if (cand.aiMicroIoU == null) return false;
  if (mode === "human-baseline") {
    return cand.passed === true;
  }
  if (!base || base.aiMicroIoU == null) return false;
  return Number(cand.aiMicroIoU) + 1e-6 >= Number(base.aiMicroIoU);
}

const baseline = readJson(baselinePath);
const candidate = readJson(candidatePath);
const baselineRows = byCase(baseline);
const candidateRows = byCase(candidate);
const cases = [...new Set([...baselineRows.keys(), ...candidateRows.keys()])]
  .filter((caseId) => !requestedCases || requestedCases.has(caseId))
  .sort();

const rows = cases.map((caseId) => {
  const base = baselineRows.get(caseId);
  const cand = candidateRows.get(caseId);
  const delta = base && cand && base.aiMicroIoU != null && cand.aiMicroIoU != null
    ? Number(cand.aiMicroIoU) - Number(base.aiMicroIoU)
    : null;
  return {
    caseId,
    baseline: base ? base.aiMicroIoU : null,
    candidate: cand ? cand.aiMicroIoU : null,
    delta,
    humanOnly: cand ? cand.humanOnlyMicroIoU : base ? base.humanOnlyMicroIoU : null,
    candidatePassedHumanOnly: cand ? cand.passed : false,
    nonRegressionPassed: okForMode(base, cand),
  };
});

const failed = rows.filter((row) => !row.nonRegressionPassed);

console.log(`Baseline:  ${baseline.label || path.basename(baselinePath)}`);
console.log(`Candidate: ${candidate.label || path.basename(candidatePath)}`);
console.log(`Mode:      ${mode}`);
console.log("");
console.log(["case", "baseline", "candidate", "delta", "humanOnly", "pass"].join("\t"));
for (const row of rows) {
  console.log([
    row.caseId,
    fmt(row.baseline),
    fmt(row.candidate),
    fmt(row.delta),
    fmt(row.humanOnly),
    row.nonRegressionPassed ? "yes" : "no",
  ].join("\t"));
}

if (failed.length) {
  console.error(`\nFAILED: ${failed.map((row) => row.caseId).join(", ")}`);
  process.exit(1);
}

console.error("\nPASSED");
