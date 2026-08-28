import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const analysisDir = path.dirname(fileURLToPath(import.meta.url));
const experimentRoot = path.resolve(analysisDir, '../../..');
const resultsRoot = path.join(experimentRoot, 'results');
const scoreRoot = path.join(resultsRoot, 'multimodel', 'scores');
const DEEPSEEK_DISPLAY_NAME = 'DeepSeek-V4-Flash（非思考模式）';

const sources = [
  [DEEPSEEK_DISPLAY_NAME, 'agent', path.join(resultsRoot, 'ai-variant-score-experiment-1-ai-only-final-api-consensus.json')],
  [DEEPSEEK_DISPLAY_NAME, 'direct', path.join(experimentRoot, 'direct-llm-results', 'ai-variant-score-direct-llm-baseline-step-11-fixed.json')],
  ['GPT-5.6 SOL', 'agent', path.join(scoreRoot, 'ai-variant-score-multimodel-gpt-5.6-sol-agent.json')],
  ['GPT-5.6 SOL', 'direct', path.join(scoreRoot, 'ai-variant-score-multimodel-gpt-5.6-sol-direct.json')],
  ['Gemini 3.1 Pro', 'agent', path.join(scoreRoot, 'ai-variant-score-multimodel-gemini-3.1-pro-agent.json')],
  ['Gemini 3.1 Pro', 'direct', path.join(scoreRoot, 'ai-variant-score-multimodel-gemini-3.1-pro-direct.json')],
  ['Claude Opus 5', 'agent', path.join(scoreRoot, 'ai-variant-score-multimodel-claude-opus-5-agent.json')],
  ['Claude Opus 5', 'direct', path.join(scoreRoot, 'ai-variant-score-multimodel-claude-opus-5-direct.json')],
  ['Qwen 3.6', 'agent', path.join(scoreRoot, 'ai-variant-score-multimodel-westlake-qwen-agent.json')],
  ['Qwen 3.6', 'direct', path.join(scoreRoot, 'ai-variant-score-multimodel-westlake-qwen-direct.json')],
];

const round = (value, digits = 6) => Number(value.toFixed(digits));
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const loaded = sources.map(([model, mode, file]) => ({ model, mode, file, data: JSON.parse(fs.readFileSync(file, 'utf8')) }));
const deepSeekByMode = Object.fromEntries(loaded.filter((item) => item.model === DEEPSEEK_DISPLAY_NAME).map((item) => [item.mode, item]));

const summaries = loaded.map(({ model, mode, file, data }) => {
  const caseScores = Object.fromEntries(data.rows.map((row) => [row.caseId, row.aiMicroIoU]));
  const baseline = deepSeekByMode[mode];
  const baselineMacro = mean(baseline.data.rows.map((row) => row.aiMicroIoU));
  const macroMicroIoU = mean(data.rows.map((row) => row.aiMicroIoU));
  return {
    model,
    mode,
    macro_micro_iou: round(macroMicroIoU),
    delta_vs_deepseek: round(macroMicroIoU - baselineMacro),
    cases_beating_human_only: data.rows.filter((row) => row.passed).length,
    missing_rows: data.rows.reduce((sum, row) => sum + row.aiMissingRows, 0),
    unresolved_rows: data.rows.reduce((sum, row) => sum + row.aiUnresolvedRows, 0),
    predicted_rows: data.rows.reduce((sum, row) => sum + row.aiRows, 0),
    case_scores: caseScores,
    score_source: path.relative(experimentRoot, file),
  };
});

const modes = ['agent', 'direct'];
const winners = Object.fromEntries(modes.map((mode) => [mode, Object.fromEntries(
  ['H-C1', 'H-C2', 'H-C3', 'H-C4', 'H-C5', 'H-C6'].map((caseId) => {
    const rows = summaries.filter((row) => row.mode === mode);
    const best = Math.max(...rows.map((row) => row.case_scores[caseId]));
    return [caseId, {
      score: best,
      models: rows.filter((row) => row.case_scores[caseId] === best).map((row) => row.model),
    }];
  }),
)]));

const agentByModel = Object.fromEntries(summaries.filter((row) => row.mode === 'agent').map((row) => [row.model, row]));
const directByModel = Object.fromEntries(summaries.filter((row) => row.mode === 'direct').map((row) => [row.model, row]));
const agentLift = Object.keys(agentByModel).map((model) => ({
  model,
  agent_macro_micro_iou: agentByModel[model].macro_micro_iou,
  direct_macro_micro_iou: directByModel[model].macro_micro_iou,
  agent_minus_direct: round(agentByModel[model].macro_micro_iou - directByModel[model].macro_micro_iou),
}));

const humanOnly = loaded[0].data.rows.map((row) => ({ case_id: row.caseId, micro_iou: row.humanOnlyMicroIoU }));
const result = {
  generated_at: new Date().toISOString(),
  metric: 'Macro-average of the six case-level month-precision MicroIoU values',
  expected_rows: 249,
  data_quality: {
    all_score_files_present: sources.every(([, , file]) => fs.existsSync(file)),
    all_have_six_cases: loaded.every((item) => item.data.rows.length === 6),
    note: 'All variants were rescored with the frozen Experiment1 scorer and identical Gold/Human inputs. Macro averages are descriptive summaries; the official scorer is case-level.',
  },
  human_only: humanOnly,
  summaries,
  case_winners: winners,
  agent_lift_over_direct: agentLift,
};

fs.writeFileSync(path.join(analysisDir, 'experiment-1-multimodel-metrics.json'), `${JSON.stringify(result, null, 2)}\n`);

const headers = ['model', 'mode', 'macro_micro_iou', 'delta_vs_deepseek', 'cases_beating_human_only', 'predicted_rows', 'missing_rows', 'unresolved_rows', 'H-C1', 'H-C2', 'H-C3', 'H-C4', 'H-C5', 'H-C6'];
const csvRows = summaries.map((row) => [
  row.model, row.mode, row.macro_micro_iou, row.delta_vs_deepseek, row.cases_beating_human_only,
  row.predicted_rows, row.missing_rows, row.unresolved_rows,
  ...['H-C1', 'H-C2', 'H-C3', 'H-C4', 'H-C5', 'H-C6'].map((caseId) => row.case_scores[caseId]),
]);
const csv = [headers, ...csvRows].map((row) => row.map((value) => {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}).join(',')).join('\n');
fs.writeFileSync(path.join(analysisDir, 'experiment-1-multimodel-metrics.csv'), `${csv}\n`);
console.log(JSON.stringify(result, null, 2));
