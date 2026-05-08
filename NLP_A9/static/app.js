/* ============================================================
   前端交互逻辑：与 Flask 后端 API 对接 + 图表绘制
   ============================================================ */

// ---------- Tabs 切换 ----------
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    // 为延迟渲染的图表强制 resize
    window.dispatchEvent(new Event('resize'));
  });
});

// ---------- 工具：示例填充 ----------
document.querySelectorAll('[data-fill]').forEach(b => {
  b.addEventListener('click', () => {
    document.getElementById(b.dataset.fill).value = b.dataset.text;
  });
});

// ---------- API 封装 ----------
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ============================================================
//  模块 1：单文本极性 + Plotly 仪表盘
// ============================================================
function renderGauge(score, label) {
  const colorMap = { Positive: '#2ed47a', Negative: '#ff5e6c', Neutral: '#f5b942' };
  const color = colorMap[label] || '#1ee6ff';

  const data = [{
    type: 'indicator',
    mode: 'gauge+number',
    value: +(score * 100).toFixed(2),
    number: { suffix: '%', font: { size: 36, color: '#fff' } },
    gauge: {
      shape: 'angular',
      axis: { range: [0, 100], tickcolor: '#9bb0d8', tickfont: { color: '#9bb0d8' } },
      bar: { color },
      bgcolor: 'rgba(0,0,0,0)',
      borderwidth: 0,
      steps: [
        { range: [0, 50], color: 'rgba(255,94,108,.18)' },
        { range: [50, 75], color: 'rgba(245,185,66,.20)' },
        { range: [75, 100], color: 'rgba(46,212,122,.22)' },
      ],
      threshold: { line: { color, width: 4 }, thickness: 0.85, value: +(score * 100).toFixed(2) },
    },
  }];
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e6f0ff' },
    margin: { t: 20, r: 20, l: 20, b: 10 },
  };
  Plotly.newPlot('t1Gauge', data, layout, { displayModeBar: false, responsive: true });
}

document.getElementById('t1Btn').addEventListener('click', async () => {
  const text = document.getElementById('t1Text').value.trim();
  if (!text) { alert('请输入评论内容'); return; }
  try {
    const r = await postJSON('/api/analyze', { text });
    const labelEl = document.getElementById('t1Label');
    labelEl.textContent = r.label;
    labelEl.className = 'result-label ' + r.label;
    document.getElementById('t1Conf').textContent = (r.confidence * 100).toFixed(2) + '%';
    document.getElementById('t1Engine').textContent = r.engine;
    renderGauge(r.confidence, r.label);
  } catch (e) { alert('分析失败：' + e.message); }
});

// 初始空仪表盘
renderGauge(0, 'Neutral');

// ============================================================
//  模块 2：显式 vs 隐式
// ============================================================
document.getElementById('t2Demo').addEventListener('click', () => {
  document.getElementById('t2Exp').value = '这屏幕画质太垃圾了！';
  document.getElementById('t2Imp').value = '在太阳底下根本看不清屏幕上的字。';
});

document.getElementById('t2Btn').addEventListener('click', async () => {
  const explicit = document.getElementById('t2Exp').value.trim();
  const implicit = document.getElementById('t2Imp').value.trim();
  if (!explicit && !implicit) { alert('请至少输入一段文本'); return; }
  try {
    const r = await postJSON('/api/compare', { explicit, implicit });
    const setRes = (labelId, confId, data) => {
      const el = document.getElementById(labelId);
      el.textContent = data.label;
      el.className = 'result-label ' + data.label;
      document.getElementById(confId).textContent = (data.confidence * 100).toFixed(2) + '%';
    };
    setRes('t2ExpLabel', 't2ExpConf', r.explicit);
    setRes('t2ImpLabel', 't2ImpConf', r.implicit);
  } catch (e) { alert('分析失败：' + e.message); }
});

// ============================================================
//  模块 3：舆情大屏
// ============================================================
let pieChart, radarChart;

function ensureCharts() {
  if (!pieChart)   pieChart   = echarts.init(document.getElementById('t3Pie'));
  if (!radarChart) radarChart = echarts.init(document.getElementById('t3Radar'));
}

function renderPie(pieData) {
  ensureCharts();
  const colorMap = { Positive: '#2ed47a', Negative: '#ff5e6c', Neutral: '#f5b942' };
  pieChart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 6, textStyle: { color: '#9bb0d8' } },
    series: [{
      type: 'pie',
      radius: ['45%', '72%'],
      center: ['50%', '46%'],
      avoidLabelOverlap: true,
      itemStyle: {
        borderColor: '#0a1230', borderWidth: 3,
      },
      label: { color: '#e6f0ff', formatter: '{b}\n{d}%', fontSize: 13 },
      labelLine: { lineStyle: { color: '#9bb0d8' } },
      data: pieData.map(d => ({
        name: d.name, value: d.value,
        itemStyle: {
          color: colorMap[d.name],
          shadowBlur: 18, shadowColor: colorMap[d.name],
        },
      })),
    }],
  });
}

function renderRadar(aspects) {
  ensureCharts();
  if (!aspects || aspects.length === 0) {
    radarChart.clear();
    radarChart.setOption({
      title: { text: '当前样本中未识别到产品维度', left: 'center', top: 'center',
               textStyle: { color: '#9bb0d8', fontSize: 13 } },
    });
    return;
  }
  const indicators = aspects.map(a => ({
    name: a.aspect,
    max: Math.max(3, ...aspects.map(x => x.total)),
  }));
  const posData = aspects.map(a => a.Positive);
  const negData = aspects.map(a => a.Negative);

  radarChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {},
    legend: { bottom: 6, textStyle: { color: '#9bb0d8' }, data: ['正面提及', '负面提及'] },
    radar: {
      indicator: indicators,
      axisName: { color: '#e6f0ff' },
      splitArea: { areaStyle: { color: ['rgba(30,230,255,.04)', 'rgba(30,230,255,.10)'] } },
      splitLine: { lineStyle: { color: 'rgba(120,170,255,.3)' } },
      axisLine: { lineStyle: { color: 'rgba(120,170,255,.3)' } },
    },
    series: [{
      type: 'radar',
      data: [
        { name: '正面提及', value: posData,
          areaStyle: { color: 'rgba(46,212,122,.35)' },
          lineStyle: { color: '#2ed47a', width: 2 },
          itemStyle: { color: '#2ed47a' } },
        { name: '负面提及', value: negData,
          areaStyle: { color: 'rgba(255,94,108,.35)' },
          lineStyle: { color: '#ff5e6c', width: 2 },
          itemStyle: { color: '#ff5e6c' } },
      ],
    }],
  });
}

function renderTable(reviews) {
  const tbody = document.getElementById('t3Table');
  if (!reviews || reviews.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">暂无数据</td></tr>';
    return;
  }
  tbody.innerHTML = reviews.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${escapeHtml(r.text)}</td>
      <td><span class="badge ${r.label}">${r.label}</span></td>
      <td>${(r.confidence * 100).toFixed(2)}%</td>
    </tr>
  `).join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

document.getElementById('t3Btn').addEventListener('click', async () => {
  const btn = document.getElementById('t3Btn');
  btn.disabled = true;
  btn.textContent = '分析中…';
  try {
    const r = await postJSON('/api/batch', {});
    const s = r.summary;
    document.getElementById('kpiTotal').textContent = s.total;
    document.getElementById('kpiPos').textContent   = s.positive;
    document.getElementById('kpiNeg').textContent   = s.negative;
    document.getElementById('kpiNeu').textContent   = s.neutral;
    document.getElementById('kpiPosRate').textContent = (s.positive_rate * 100).toFixed(1) + '%';
    document.getElementById('kpiConf').textContent  = (s.avg_confidence * 100).toFixed(2) + '%';
    renderPie(r.pie);
    renderRadar(r.aspects);
    renderTable(r.reviews);
  } catch (e) {
    alert('批量分析失败：' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ 生成测试舆情数据并分析';
  }
});

window.addEventListener('resize', () => {
  if (pieChart) pieChart.resize();
  if (radarChart) radarChart.resize();
});
