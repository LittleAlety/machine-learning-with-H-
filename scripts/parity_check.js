#!/usr/bin/env node
/*
 * parity_check.js — JS 推理引擎与 Python XGBoost 的逐行奇偶校验
 *
 * 用法: node parity_check.js <parity_inputs.json> <assets_dir>
 *   parity_inputs.json: { comps, py_preds, csv_features, feature_names,
 *                         random_comps, random_py_preds }
 *   assets_dir: 含 model.json / elements.json / predict.js 的目录，
 *               parity_report.json 也写入该目录。
 *
 * 交付门槛: 1836 行逐行 max|diff| < 1e-9，且 50 个随机组成 max|diff| < 1e-9。
 */
'use strict';
const fs = require('fs');
const path = require('path');

const THRESHOLD = 1e-9;

function main() {
  const inputsPath = path.resolve(process.argv[2]);
  const assetsDir = path.resolve(process.argv[3]);
  const inputs = JSON.parse(fs.readFileSync(inputsPath, 'utf8'));
  const model = JSON.parse(fs.readFileSync(path.join(assetsDir, 'model.json'), 'utf8'));
  const elements = JSON.parse(fs.readFileSync(path.join(assetsDir, 'elements.json'), 'utf8'));
  const HStarPredict = require(path.join(assetsDir, 'predict.js'));
  const engine = HStarPredict.createEngine(model, elements);

  const report = {
    meta: {
      threshold: THRESHOLD,
      engine: 'predict.js (纯 JS, float32 累加 + float32 分裂比较)',
      reference: 'Python xgboost booster.predict (float32)',
      node: process.version,
      generated_at: new Date().toISOString(),
    },
    dataset: null, features: null, random: null, pass: false,
  };

  // ---- 1) 1836 行全管线 parity（comp → JS 特征 → JS 树遍历）----
  let maxDiff = 0, sumDiff = 0, worstIdx = -1;
  let fMaxDiff = 0, fWorst = null;
  const fnames = inputs.feature_names;
  for (let i = 0; i < inputs.comps.length; i++) {
    const r = engine.predictFromComp(inputs.comps[i]);
    const diff = Math.abs(r.E_ads - inputs.py_preds[i]);
    sumDiff += diff;
    if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
    // 特征级 parity（JS 重算 vs CSV 存储值）
    const csvRow = inputs.csv_features[i];
    for (let j = 0; j < fnames.length; j++) {
      const fd = Math.abs(r.features[fnames[j]] - csvRow[j]);
      if (fd > fMaxDiff) {
        fMaxDiff = fd;
        fWorst = { row: i, comp: inputs.comps[i], feature: fnames[j],
                   js: r.features[fnames[j]], csv: csvRow[j] };
      }
    }
  }
  report.dataset = {
    n_rows: inputs.comps.length,
    max_diff: maxDiff,
    mean_diff: sumDiff / inputs.comps.length,
    worst: worstIdx >= 0 ? { row: worstIdx, comp: inputs.comps[worstIdx],
                             py: inputs.py_preds[worstIdx] } : null,
    pass: maxDiff < THRESHOLD,
  };
  report.features = { n_checks: inputs.comps.length * fnames.length,
                      max_diff: fMaxDiff, worst: fWorst };

  // ---- 2) 50 个随机合法组成双端对比 ----
  let rMax = 0, rSum = 0, rWorst = null;
  for (let i = 0; i < inputs.random_comps.length; i++) {
    const r = engine.predictFromComp(inputs.random_comps[i]);
    const diff = Math.abs(r.E_ads - inputs.random_py_preds[i]);
    rSum += diff;
    if (diff > rMax) {
      rMax = diff;
      rWorst = { comp: inputs.random_comps[i], js: r.E_ads,
                 py: inputs.random_py_preds[i], in_domain: r.in_domain };
    }
  }
  report.random = {
    n: inputs.random_comps.length,
    max_diff: rMax,
    mean_diff: rSum / inputs.random_comps.length,
    worst: rWorst,
    pass: rMax < THRESHOLD,
  };

  report.pass = report.dataset.pass && report.random.pass;
  fs.writeFileSync(path.join(assetsDir, 'parity_report.json'),
                   JSON.stringify(report, null, 1));
  console.log(JSON.stringify({
    dataset: report.dataset, features: report.features,
    random: report.random, pass: report.pass,
  }, null, 1));
  process.exit(report.pass ? 0 : 1);
}

main();
