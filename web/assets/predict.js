/*
 * predict.js — H* 吸附能 XGBoost 模型的纯前端 JS 推理引擎（v3，84 特征）
 *
 * 零依赖，浏览器（window.HStarPredict）与 Node（module.exports）双端可用。
 * 特征规范见 feature_spec.md（唯一事实来源，任何修改两端同步）。
 *
 * 推理与 XGBoost 3.4.1 C++ 预测器位级一致的关键点：
 *   1. 分裂比较用 float32：Math.fround(feature) < Math.fround(split_condition)；
 *   2. 叶值按树序以 float32 逐步累加：acc = fround(acc + fround(leaf))；
 *   3. 初始 acc = fround(base_score)；objective=reg:squarederror 无 link 变换；
 *   4. 缺失特征（null/undefined/NaN）走节点的 missing 分支（本模型特征无缺失）。
 *
 * 用法：
 *   const engine = HStarPredict.createEngine(modelJson, elementsJson);
 *   const r = engine.predictFromComp('CuPt3');
 *   // r = { comp, n_elements, structure, facet, in_domain, warnings,
 *   //       features, E_ads, dG_H }
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.HStarPredict = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* 属性表（feature_spec v3 §2，13 项）：
   * 前 6 项为基线属性，后 7 项为描述符精准化新增属性 */
  var BASE_PROPS = ['en', 'radius', 'group', 'period', 'ie1', 'd_el'];
  var NEW_PROPS = ['melting_point', 'mendeleev_number', 'covalent_radius',
    'n_valence', 'nd_valence', 'n_unfilled', 'nd_unfilled'];
  var ALL_PROPS = BASE_PROPS.concat(NEW_PROPS);
  var COMP_TOKEN = /([A-Z][a-z]?)(\d*)/g;
  var DG_SHIFT = 0.24; // ΔG_H* = E_ads + 0.24 eV

  function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }

  /* 组成解析 + 规范化（元素字典序 + gcd 约简），返回 {counts, els, canon} */
  function parseComp(str) {
    if (typeof str !== 'string' || str.length === 0) {
      throw new Error('组成不能为空');
    }
    COMP_TOKEN.lastIndex = 0;
    var m, rebuilt = '', raw = {};
    while ((m = COMP_TOKEN.exec(str)) !== null) {
      rebuilt += m[1] + m[2];
      if (Object.prototype.hasOwnProperty.call(raw, m[1])) {
        throw new Error('组成含重复元素 ' + m[1] + ': ' + str);
      }
      raw[m[1]] = m[2] ? parseInt(m[2], 10) : 1;
    }
    if (rebuilt !== str) {
      throw new Error('组成解析失败（非法字符或大小写）: ' + str);
    }
    var els = Object.keys(raw).sort();
    var g = 0;
    for (var i = 0; i < els.length; i++) g = gcd(g, raw[els[i]]);
    var counts = {};
    for (i = 0; i < els.length; i++) counts[els[i]] = raw[els[i]] / g;
    var canon = els.map(function (e) {
      return counts[e] > 1 ? e + counts[e] : e;
    }).join('');
    return { counts: counts, els: els, canon: canon };
  }

  /* structure/facet 推断（feature_spec §4，源自 05 的 12 原子超胞计量模式） */
  function inferStructure(counts, els) {
    if (els.length === 1) return { structure: 'A1', facet: '111', known: true };
    if (els.length === 2) {
      var c0 = counts[els[0]], c1 = counts[els[1]];
      var mx = Math.max(c0, c1), mn = Math.min(c0, c1);
      if (mx === mn) return { structure: 'L10', facet: '101', known: true };
      if (mx === 3 * mn) return { structure: 'L12', facet: '111', known: true };
    }
    return { structure: null, facet: null, known: false };
  }

  /* 84 特征计算。opts: {structure, facet} 可显式覆盖推断结果。 */
  /* 5 种加权统计（wmean/max/min/range/wstd），6 基线 + 7 新增属性共用 */
  function propStats(features, prop, v, w, n) {
    // wmean = Σ(v_i*w_i)/Σw_i （np.average 语义，v*w 逐元素相乘后顺序求和）
    var num = 0, den = 0, i2;
    for (i2 = 0; i2 < n; i2++) num += v[i2] * w[i2];
    for (i2 = 0; i2 < n; i2++) den += w[i2];
    var mu = num / den;
    var vmax = v[0], vmin = v[0];
    for (i2 = 1; i2 < n; i2++) {
      if (v[i2] > vmax) vmax = v[i2];
      if (v[i2] < vmin) vmin = v[i2];
    }
    // wstd = sqrt( Σ((v_i-mu)^2 * w_i) / Σw_i )
    var snum = 0, sden = 0;
    for (i2 = 0; i2 < n; i2++) { var d = v[i2] - mu; snum += (d * d) * w[i2]; }
    for (i2 = 0; i2 < n; i2++) sden += w[i2];
    features[prop + '_wmean'] = mu;
    features[prop + '_max'] = vmax;
    features[prop + '_min'] = vmin;
    features[prop + '_range'] = vmax - vmin;
    features[prop + '_wstd'] = Math.sqrt(snum / sden);
  }
  function featurize(engine, compStr, opts) {
    var p = parseComp(compStr);
    var counts = p.counts, els = p.els;
    var warnings = [];
    var inDomain = true;

    for (var i = 0; i < els.length; i++) {
      if (!engine.elements[els[i]]) {
        throw new Error('元素 ' + els[i] + ' 不在 elements.json 属性表中，无法计算特征');
      }
      if (engine.trainingElements.indexOf(els[i]) < 0) {
        inDomain = false;
        warnings.push('元素 ' + els[i] + ' 未出现在训练集（外推）');
      }
    }

    var st;
    if (opts && opts.structure) {
      st = { structure: String(opts.structure), facet: String(opts.facet), known: true };
    } else {
      st = inferStructure(counts, els);
      if (!st.known) {
        inDomain = false;
        warnings.push('计量模式无法推断 structure/facet（仅支持纯金属 / 1:1(L10) / ' +
          '3:1(L12)），one-hot 全 0，预测为外推');
      }
    }

    // 权重：与 06 完全一致 —— c 转浮点、逐步求和、逐元素相除
    var n = els.length;
    var c = new Array(n), wsum = 0, i2;
    for (i2 = 0; i2 < n; i2++) { c[i2] = counts[els[i2]]; wsum += c[i2]; }
    var w = new Array(n);
    for (i2 = 0; i2 < n; i2++) w[i2] = c[i2] / wsum;

    var features = { n_elements: n };
    /* 2–31：6 基线属性 × 5 统计 */
    var STAT_PROPS = BASE_PROPS.concat(NEW_PROPS);
    var vals = {};
    for (var pi = 0; pi < STAT_PROPS.length; pi++) {
      var prop = STAT_PROPS[pi];
      var v = new Array(n);
      for (i2 = 0; i2 < n; i2++) v[i2] = engine.elements[els[i2]][prop];
      vals[prop] = v;
      propStats(features, prop, v, w, n);
    }
    /* 32–36：结构/晶面 one-hot */
    features.structure_A1 = st.structure === 'A1' ? 1 : 0;
    features.structure_L10 = st.structure === 'L10' ? 1 : 0;
    features.structure_L12 = st.structure === 'L12' ? 1 : 0;
    features.facet_101 = st.facet === '101' ? 1 : 0;
    features.facet_111 = st.facet === '111' ? 1 : 0;

    /* 72–84：13 属性的 mode 统计 —— 计量数最大的组元的属性原始值；
     * 计量数并列时取 els（ASCII 字母序）最靠前者（即规范化化学式首元素），
     * els 已按字典序排列，故严格大于才替换即实现平局取首。 */
    var mi = 0;
    for (i2 = 1; i2 < n; i2++) {
      if (c[i2] > c[mi]) mi = i2;
    }
    for (pi = 0; pi < ALL_PROPS.length; pi++) {
      features[ALL_PROPS[pi] + '_mode'] = vals[ALL_PROPS[pi]][mi];
    }

    return {
      comp: p.canon, n_elements: n,
      structure: st.structure, facet: st.facet,
      in_domain: inDomain, warnings: warnings, features: features
    };
  }

  /* 单棵树遍历：float32 比较 + missing 分支 */
  function evalTree(node, features) {
    var n = node;
    while (!Object.prototype.hasOwnProperty.call(n, 'leaf')) {
      var fv = features[n.split];
      var next;
      if (fv === null || fv === undefined || fv !== fv) { // NaN 检查
        next = n.missing;
      } else {
        next = (Math.fround(fv) < Math.fround(n.split_condition)) ? n.yes : n.no;
      }
      var kids = n.children;
      n = kids[0].nodeid === next ? kids[0] : kids[1];
    }
    return Math.fround(n.leaf);
  }

  /* 树集成求和：float32 逐步累加，与 C++ bst_float 累加位级一致 */
  function predictFeatures(engine, features) {
    var acc = Math.fround(engine.model.base_score);
    var trees = engine.model.trees;
    for (var i = 0; i < trees.length; i++) {
      acc = Math.fround(acc + evalTree(trees[i], features));
    }
    return acc;
  }

  function createEngine(modelJson, elementsJson) {
    if (!modelJson || !modelJson.trees || typeof modelJson.base_score !== 'number') {
      throw new Error('model.json 格式不正确（需要 base_score 与 trees）');
    }
    var elements = elementsJson.elements || elementsJson;
    var training = (elementsJson.meta && elementsJson.meta.training_elements) || [];
    var engine = {
      model: modelJson,
      elements: elements,
      trainingElements: training,
      featureNames: modelJson.feature_names,
      parseComp: parseComp,
      featurize: function (compStr, opts) { return featurize(engine, compStr, opts); },
      predictFeatures: function (features) { return predictFeatures(engine, features); },
      /* 主入口：组成字符串 → 吸附能预测 */
      predictFromComp: function (compStr, opts) {
        var f = featurize(engine, compStr, opts);
        var eads = predictFeatures(engine, f.features);
        f.E_ads = eads;
        f.dG_H = eads + DG_SHIFT; // ΔG_H* = E_ads + 0.24 eV
        return f;
      }
    };
    return engine;
  }

  return { createEngine: createEngine, DG_SHIFT: DG_SHIFT };
});
