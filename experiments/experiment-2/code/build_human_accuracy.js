const fs = require("fs");
const path = require("path");

const PACKAGE_DIR = path.resolve(__dirname, "..");
const PARTICIPANT_JSON = path.join(
  PACKAGE_DIR,
  "inputs/responses/experiment-2-participant-responses.json"
);
const STANDARD_JSON = path.join(
  PACKAGE_DIR,
  "inputs/scoring/experiment-2-standard-answers.json"
);
const OUT_DIR = process.env.AIH_RECOMPUTED_ROOT
  ? path.join(path.resolve(process.env.AIH_RECOMPUTED_ROOT), "human")
  : path.join(PACKAGE_DIR, "results/human");
const DETAIL_CSV = path.join(OUT_DIR, "experiment-2-human-accuracy-detail.csv");
const SUMMARY_CSV = path.join(OUT_DIR, "experiment-2-human-accuracy-summary.csv");
const METRICS_JSON = path.join(OUT_DIR, "experiment-2-human-accuracy-metrics.json");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function normText(value) {
  return String(value ?? "")
    .trim()
    .replace(/\s+/g, "")
    .replace(/[，,。．.；;：:、]/g, "");
}

function normYesNo(value) {
  const v = normText(value);
  if (["是", "yes", "y", "true", "1"].includes(v.toLowerCase())) return "是";
  if (["否", "no", "n", "false", "0"].includes(v.toLowerCase())) return "否";
  if (["不确定", "不知道", "unsure", "unknown"].includes(v.toLowerCase())) return "不确定";
  return v;
}

function normChoice(value) {
  return String(value ?? "").trim().toUpperCase();
}

function csvEscape(value) {
  const s = String(value ?? "");
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function writeCsv(filePath, rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

function makeGoldMap(standard) {
  const map = new Map();
  for (const formId of ["T1", "T2", "T3"]) {
    const unit = standard.units?.[formId];
    if (!unit) throw new Error(`Missing standard unit: ${formId}`);
    for (const row of unit.rows || []) {
      map.set(`${formId}::${row.questionId}`, row);
    }
  }
  return map;
}

function getParticipantExperiment2Stages(participants) {
  const stages = [];
  for (const [participantId, participant] of Object.entries(participants.participants || {})) {
    for (const [stageId, stage] of Object.entries(participant.stages || {})) {
      if (stage.type === "experiment2") {
        stages.push({ participantId, stageId, stage });
      }
    }
  }
  return stages;
}

function scoreRow({ participantId, stageId, formId, participantRow, goldRow }) {
  const block = participantRow.block || goldRow.block;
  const base = {
    participantId,
    stageId,
    formId,
    block,
    questionId: participantRow.questionId,
    sourceId: participantRow.sourceId || goldRow.sourceId || "",
    doc: participantRow.doc || goldRow.doc || "",
  };

  const fields = [];
  if (block === "A") {
    fields.push({
      field: "time_span",
      answer: participantRow.timeSpan,
      correct: goldRow.correctTimeSpan,
      isCorrect: normText(participantRow.timeSpan) === normText(goldRow.correctTimeSpan),
    });
    fields.push({
      field: "sink",
      answer: participantRow.sinkYesNo,
      correct: goldRow.correctSinkYesNo,
      isCorrect: normYesNo(participantRow.sinkYesNo) === normYesNo(goldRow.correctSinkYesNo),
    });
    fields.push({
      field: "interlude",
      answer: participantRow.interludeYesNo,
      correct: goldRow.correctInterludeYesNo,
      isCorrect: normYesNo(participantRow.interludeYesNo) === normYesNo(goldRow.correctInterludeYesNo),
    });
  } else {
    fields.push({
      field: "choice",
      answer: participantRow.choice,
      correct: goldRow.correctChoice,
      isCorrect: normChoice(participantRow.choice) === normChoice(goldRow.correctChoice),
    });
  }

  return fields.map((field) => ({
    ...base,
    field: field.field,
    answer: field.answer ?? "",
    correct: field.correct ?? "",
    isCorrect: field.isCorrect ? 1 : 0,
  }));
}

function summarize(rows, keyFields) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFields.map((field) => row[field]).join("\u0001");
    if (!groups.has(key)) {
      const out = {};
      for (const field of keyFields) out[field] = row[field];
      out.correct = 0;
      out.total = 0;
      groups.set(key, out);
    }
    const group = groups.get(key);
    group.correct += Number(row.isCorrect) || 0;
    group.total += 1;
  }
  return [...groups.values()].map((group) => ({
    ...group,
    accuracy: group.total ? group.correct / group.total : null,
  }));
}

function summarizeStrict(detailRows) {
  const byQuestion = new Map();
  for (const row of detailRows) {
    const key = [row.participantId, row.formId, row.block, row.questionId].join("\u0001");
    if (!byQuestion.has(key)) {
      byQuestion.set(key, {
        participantId: row.participantId,
        formId: row.formId,
        block: row.block,
        questionId: row.questionId,
        fieldsCorrect: 0,
        fieldsTotal: 0,
      });
    }
    const group = byQuestion.get(key);
    group.fieldsCorrect += Number(row.isCorrect) || 0;
    group.fieldsTotal += 1;
  }
  const strictRows = [...byQuestion.values()].map((row) => ({
    ...row,
    isCorrect: row.fieldsTotal > 0 && row.fieldsCorrect === row.fieldsTotal ? 1 : 0,
  }));
  return {
    strictRows,
    byParticipant: summarize(strictRows, ["participantId", "formId"]),
    byForm: summarize(strictRows, ["formId"]),
    byBlock: summarize(strictRows, ["block"]),
    overall: summarize(strictRows.map((row) => ({ ...row, scope: "overall" })), ["scope"])[0],
  };
}

function getTiming(stages) {
  const rows = [];
  for (const { participantId, stage } of stages) {
    for (const block of ["A", "B", "C"]) {
      const duration = stage.blockDurations?.[block];
      rows.push({
        participantId,
        formId: stage.formId,
        block,
        seconds: Number(duration?.durationSeconds ?? 0) || 0,
      });
    }
  }
  return rows;
}

function summarizeSeconds(rows, keyFields) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFields.map((field) => row[field]).join("\u0001");
    if (!groups.has(key)) {
      const out = {};
      for (const field of keyFields) out[field] = row[field];
      out.seconds = 0;
      out.count = 0;
      groups.set(key, out);
    }
    const group = groups.get(key);
    group.seconds += row.seconds;
    group.count += 1;
  }
  return [...groups.values()].map((group) => ({
    ...group,
    averageSeconds: group.count ? group.seconds / group.count : null,
  }));
}

function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const participants = readJson(PARTICIPANT_JSON);
  const standard = readJson(STANDARD_JSON);
  const goldMap = makeGoldMap(standard);
  const stages = getParticipantExperiment2Stages(participants);
  const detailRows = [];

  for (const { participantId, stageId, stage } of stages) {
    const formId = stage.formId;
    for (const participantRow of stage.rows || []) {
      const goldRow = goldMap.get(`${formId}::${participantRow.questionId}`);
      if (!goldRow) {
        throw new Error(`Missing gold row for ${formId} ${participantRow.questionId}`);
      }
      detailRows.push(...scoreRow({ participantId, stageId, formId, participantRow, goldRow }));
    }
  }

  const component = {
    overall: summarize(detailRows.map((row) => ({ ...row, scope: "overall" })), ["scope"])[0],
    byParticipant: summarize(detailRows, ["participantId", "formId"]),
    byForm: summarize(detailRows, ["formId"]),
    byBlock: summarize(detailRows, ["block"]),
    byFormBlock: summarize(detailRows, ["formId", "block"]),
    byField: summarize(detailRows, ["field"]),
    byBlockField: summarize(detailRows, ["block", "field"]),
  };

  const strict = summarizeStrict(detailRows);
  const timingRows = getTiming(stages);
  const timing = {
    totalSeconds: timingRows.reduce((sum, row) => sum + row.seconds, 0),
    byParticipant: summarizeSeconds(timingRows, ["participantId", "formId"]),
    byForm: summarizeSeconds(timingRows, ["formId"]),
    byBlock: summarizeSeconds(timingRows, ["block"]),
    byFormBlock: summarizeSeconds(timingRows, ["formId", "block"]),
  };

  const summaryRows = [
    { metric: "component_accuracy", scope: "overall", group: "all", correct: component.overall.correct, total: component.overall.total, accuracy: component.overall.accuracy },
    { metric: "row_strict_accuracy", scope: "overall", group: "all", correct: strict.overall.correct, total: strict.overall.total, accuracy: strict.overall.accuracy },
    ...component.byForm.map((row) => ({ metric: "component_accuracy", scope: "form", group: row.formId, ...row })),
    ...strict.byForm.map((row) => ({ metric: "row_strict_accuracy", scope: "form", group: row.formId, ...row })),
    ...component.byBlock.map((row) => ({ metric: "component_accuracy", scope: "block", group: row.block, ...row })),
    ...strict.byBlock.map((row) => ({ metric: "row_strict_accuracy", scope: "block", group: row.block, ...row })),
    ...component.byField.map((row) => ({ metric: "component_accuracy", scope: "field", group: row.field, ...row })),
  ];

  writeCsv(DETAIL_CSV, detailRows, [
    "participantId",
    "stageId",
    "formId",
    "block",
    "questionId",
    "sourceId",
    "doc",
    "field",
    "answer",
    "correct",
    "isCorrect",
  ]);
  writeCsv(SUMMARY_CSV, summaryRows, ["metric", "scope", "group", "correct", "total", "accuracy"]);
  fs.writeFileSync(
    METRICS_JSON,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        sources: {
          participantJson: path.relative(PACKAGE_DIR, PARTICIPANT_JSON),
          standardJson: path.relative(PACKAGE_DIR, STANDARD_JSON),
        },
        participantCount: stages.length,
        component,
        strict,
        timing,
      },
      null,
      2
    ),
    "utf8"
  );

  console.log(`Wrote ${DETAIL_CSV}`);
  console.log(`Wrote ${SUMMARY_CSV}`);
  console.log(`Wrote ${METRICS_JSON}`);
  console.log(
    JSON.stringify(
      {
        componentAccuracy: component.overall.accuracy,
        rowStrictAccuracy: strict.overall.accuracy,
        totalComponentCorrect: component.overall.correct,
        totalComponentFields: component.overall.total,
        totalStrictCorrect: strict.overall.correct,
        totalQuestions: strict.overall.total,
        totalSeconds: timing.totalSeconds,
      },
      null,
      2
    )
  );
}

main();
