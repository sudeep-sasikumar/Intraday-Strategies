/* Intraday Strategies portal. One master tab per strategy; sub-tabs inside each.
   Routes: #/s/<strategy>/<overview|backtest|trades|stocks|signals|settings>, #/compare, #/data */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const view = $('#view');
const SUBS = [['overview', 'Overview'], ['backtest', 'Backtest'], ['trades', 'Trades'], ['stocks', 'Stocks'],
  ['signals', 'Signals'], ['settings', 'Settings']];
const state = {strategies: [], login: false, detail: {}, run: {}, summary: {}, charts: [], timer: null,
  trades: {page: 0, sort: 'entry_t', desc: 1, symbol: '', side: '', reason: '', result: ''},
  stocks: {sort: 'net', desc: 1}};

// ---------------------------------------------------------------- helpers
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const nf0 = new Intl.NumberFormat('en-IN', {maximumFractionDigits: 0});
const nf2 = new Intl.NumberFormat('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const inr = (n) => n == null ? '–' : (n < 0 ? '−₹' : '₹') + nf0.format(Math.abs(n));
const signed = (n, cls = true) => n == null ? '–' : `<span class="${cls ? (n > 0 ? 'good' : n < 0 ? 'bad' : '') : ''}">${n > 0 ? '+' : ''}${inr(n)}</span>`;
const pct = (x, d = 1) => x == null ? '–' : (x * 100).toFixed(d) + '%';
const rr = (x) => x == null ? '–' : `<span class="${x > 0 ? 'good' : x < 0 ? 'bad' : ''}">${x > 0 ? '+' : ''}${x.toFixed(2)}R</span>`;
const IST = 'Asia/Kolkata';
const fmtDate = (ts) => new Date(ts * 1000).toLocaleDateString('en-GB', {timeZone: IST, day: '2-digit', month: 'short', year: 'numeric'});
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString('en-GB', {timeZone: IST, hour: '2-digit', minute: '2-digit'});
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function api(path, opts = {}) {
  const r = await fetch(path, opts.body ? {method: opts.method || 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(opts.body)} : {method: opts.method || 'GET'});
  if (r.status === 401) { location.href = '/login'; throw new Error('Please sign in'); }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
  return j;
}

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, 3500);
}

function cleanup() {
  state.charts.forEach((c) => c.remove());
  state.charts = [];
  clearInterval(state.timer); state.timer = null;
}

function watchJob(id, onTick, onDone) {
  clearInterval(state.timer);
  state.timer = setInterval(async () => {
    try {
      const j = await api(`/api/job/${id}`);
      onTick(j);
      if (j.status === 'done' || j.status === 'failed') { clearInterval(state.timer); state.timer = null; onDone(j); }
    } catch (e) { /* keep polling: the server may be busy */ }
  }, 1500);
}

const progressHtml = (j) => `<div class="progress"><i style="width:${Math.round((j.progress || 0) * 100)}%"></i></div>
  <div class="muted" id="job-msg">${esc(j.message || 'Starting…')}</div>`;
function tickProgress(j) {
  const bar = $('.progress > i'); if (bar) bar.style.width = Math.round((j.progress || 0) * 100) + '%';
  const m = $('#job-msg'); if (m) m.textContent = j.message || '';
}

// ---------------------------------------------------------------- routing
function route() {
  const p = location.hash.replace(/^#\/?/, '').split('/');
  if (p[0] === 's' && p[1]) return {page: 'strategy', sid: p[1], sub: SUBS.some((s) => s[0] === p[2]) ? p[2] : 'overview'};
  if (p[0] === 'compare' || p[0] === 'data') return {page: p[0]};
  return null;
}

function renderMaster(r) {
  const tabs = state.strategies.map((s) => `<a href="#/s/${s.id}/overview" class="${r.sid === s.id ? 'on' : ''}">${esc(s.name)}</a>`);
  tabs.push('<span class="sep"></span>', `<a href="#/compare" class="${r.page === 'compare' ? 'on' : ''}">Compare</a>`,
    `<a href="#/data" class="${r.page === 'data' ? 'on' : ''}">Data</a>`);
  $('#master').innerHTML = tabs.join('');
}

async function render() {
  cleanup();
  let r = route();
  if (!r) {
    if (state.strategies.length) { location.replace(`#/s/${state.strategies[0].id}/overview`); return; }
    r = {page: 'data'};
  }
  renderMaster(r);
  try {
    if (r.page === 'compare') await viewCompare();
    else if (r.page === 'data') await viewData();
    else await viewStrategy(r);
  } catch (e) {
    view.innerHTML = `<div class="card empty"><h3>Something went wrong</h3><p>${esc(e.message)}</p></div>`;
  }
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------- strategy shell
async function viewStrategy(r) {
  const d = state.detail[r.sid] = await api(`/api/strategy/${r.sid}`);
  const done = d.runs.filter((x) => x.status === 'done');
  if (!done.some((x) => x.id === state.run[r.sid])) state.run[r.sid] = done.length ? done[0].id : null;
  const runId = state.run[r.sid];
  const runSel = done.length > 1 && ['overview', 'backtest', 'trades', 'stocks'].includes(r.sub)
    ? `<label class="muted">Showing <select id="run-sel">${done.map((x) => `<option value="${x.id}" ${x.id === runId ? 'selected' : ''}>Run ${x.id} · ${fmtDate(x.finished)}</option>`).join('')}</select></label>` : '';
  view.innerHTML = `<div class="head"><div><h2>${esc(d.name)}</h2><p class="muted">${esc(d.summary)}</p></div>${runSel}</div>
    <nav class="tabs sub">${SUBS.map(([k, t]) => `<a href="#/s/${d.id}/${k}" class="${k === r.sub ? 'on' : ''}">${t}</a>`).join('')}</nav>
    <div id="sub" class="stack"></div>`;
  const sel = $('#run-sel');
  if (sel) sel.onchange = () => { state.run[r.sid] = +sel.value; render(); };
  const box = $('#sub');
  let sum = null;
  if (runId && r.sub !== 'settings' && r.sub !== 'signals') {
    sum = state.summary[runId] || (state.summary[runId] = await api(`/api/run/${runId}`));
  }
  ({overview: subOverview, backtest: subBacktest, trades: subTrades, stocks: subStocks, signals: subSignals,
    settings: subSettings})[r.sub](box, d, sum);
}

const noRun = (d) => `<div class="card empty"><h3>No backtest yet</h3>
  <p>Download the candles on the Data tab, then run the first backtest for this strategy.</p>
  <a href="#/s/${d.id}/backtest"><button class="primary">Go to Backtest</button></a></div>`;

function verdict(h) {
  if (!h.n) return ['', 'No trades', 'The rules never triggered in this period.'];
  if (h.n < 100) return ['', 'Too few trades to judge', `Only ${h.n} trades. Results this small are mostly luck.`];
  if (h.net > 0) return ['good', 'Made money after costs in this test',
    `On average ${h.exp_r > 0 ? '+' : ''}${h.exp_r.toFixed(2)}R per trade over ${nf0.format(h.n)} trades. A backtest is not a promise; check the Period table on the Backtest tab.`];
  if (h.gross > 0) return ['bad', 'Profitable before costs, loses money after costs',
    `The edge (${inr(h.gross)}) is smaller than brokerage, taxes and slippage (${inr(h.costs)}).`];
  return ['bad', 'Loses money even before costs', `On average ${h.exp_r.toFixed(2)}R per trade over ${nf0.format(h.n)} trades.`];
}

const stat = (k, v, dsc) => `<div class="card stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${dsc}</div></div>`;

function headlineCards(h) {
  return `<div class="grid">
    ${stat('Trades', nf0.format(h.n), `${h.symbols_traded ?? '–'} stocks, on ${h.days ?? '–'} different days`)}
    ${stat('Win rate', pct(h.win_rate), `Trades that made money after costs (${pct(h.win_rate_gross)} before costs)`)}
    ${stat('Net after costs', signed(h.net), 'Total profit or loss, all trades, after every cost')}
    ${stat('Before costs', signed(h.gross), `Costs took ${inr(h.costs)} (brokerage, taxes, slippage)`)}
    ${stat('Average per trade', rr(h.exp_r), `${signed(h.exp, false)} per trade. R = the amount risked (entry to stop)`)}
    ${stat('Profit factor', h.pf == null ? '–' : h.pf.toFixed(2), 'Money won ÷ money lost. Above 1 = profitable')}
    ${stat('Worst drawdown', inr(h.max_dd), 'Biggest fall from a peak in the running total')}
    ${stat('Most trades open at once', nf0.format(h.max_concurrent ?? 0), 'Capital needed = this × capital per trade')}
  </div>`;
}

function subOverview(box, d, sum) {
  const rules = `<div class="card prose"><h3>How it works</h3><div style="margin-top:10px">${d.rules_html}</div></div>`;
  if (!sum) { box.innerHTML = noRun(d) + rules; return; }
  const h = sum.summary.headline, [cls, title, text] = verdict(h);
  box.innerHTML = `<div class="card verdict ${cls}"><b>${title}</b>${esc(text)}
      <div class="muted" style="margin-top:6px">${fmtDate(sum.summary.start_ts)} to ${fmtDate(sum.summary.end_ts)} · ${sum.summary.symbols_tested} stocks tested · ${inr(sum.summary.params.capital_per_trade)} per trade</div></div>
    ${h.n ? headlineCards(h) : ''}${rules}`;
}

// ---------------------------------------------------------------- backtest tab
function breakdownTable(title, rows) {
  return `<div class="card"><h3>${esc(title)}</h3><div class="scroll"><table>
    <thead><tr><th></th><th class="num">Trades</th><th class="num">Win rate</th><th class="num">Before costs</th>
      <th class="num">Net after costs</th><th class="num">Avg per trade</th><th class="num">Profit factor</th></tr></thead>
    <tbody>${rows.map((r) => `<tr><td>${esc(r.key)}</td><td class="num">${nf0.format(r.n)}</td><td class="num">${pct(r.win_rate)}</td>
      <td class="num">${signed(r.gross)}</td><td class="num">${signed(r.net)}</td><td class="num">${rr(r.exp_r)}</td>
      <td class="num">${r.pf == null ? '–' : r.pf.toFixed(2)}</td></tr>`).join('')}</tbody></table></div></div>`;
}

function chartOptions() {
  return {autoSize: true, layout: {background: {color: 'transparent'}, textColor: css('--muted'), attributionLogo: false,
    panes: {separatorColor: css('--line')}},
    grid: {vertLines: {color: css('--line')}, horzLines: {color: css('--line')}},
    rightPriceScale: {borderColor: css('--line')}, timeScale: {borderColor: css('--line'), timeVisible: true, secondsVisible: false},
    localization: {priceFormatter: (p) => nf2.format(p)}};
}

function equityChart(el, eq) {
  const chart = LightweightCharts.createChart(el, {...chartOptions(), localization: {priceFormatter: (p) => inr(p)}});
  state.charts.push(chart);
  const mk = (i, color) => {
    const s = chart.addSeries(LightweightCharts.LineSeries, {color, lineWidth: 2, priceLineVisible: false, lastValueVisible: true});
    s.setData(eq.map((r) => ({time: r[0], value: r[i]})));
  };
  mk(1, css('--muted')); mk(2, css('--accent'));
  chart.timeScale().fitContent();
}

function subBacktest(box, d, sum) {
  const active = d.active;
  const runCard = `<div class="card"><div class="row" style="justify-content:space-between">
      <div><h3>Run a backtest</h3><div class="muted">Uses the downloaded candles and the values on the Settings tab. Past runs are kept.</div></div>
      <button class="primary" id="run-btn" ${active ? 'disabled' : ''}>${active ? 'Running…' : 'Run backtest'}</button></div>
      <div id="run-prog">${active ? progressHtml(active) : ''}</div></div>`;
  let body = '';
  if (sum && sum.summary.headline.n) {
    const s = sum.summary;
    body = `<div class="card"><h3>Running total</h3>
        <div class="legend" style="margin-top:8px"><span><i class="dot" style="background:var(--accent)"></i>After costs</span>
        <span><i class="dot" style="background:var(--muted)"></i>Before costs</span>
        <span>${fmtDate(s.start_ts)} to ${fmtDate(s.end_ts)}, every trade taken with ${inr(s.params.capital_per_trade)}</span></div>
        <div id="eq" class="chart"></div></div>
      ${headlineCards(s.headline)}
      ${Object.entries(s.breakdowns).map(([t, rows]) => breakdownTable(t === 'Period' ? 'Period (does it hold up recently?)' : t, rows)).join('')}
      <div class="card"><div class="row" style="justify-content:space-between"><div class="muted">Run ${sum.job.id}, finished ${fmtDate(sum.job.finished)} ${fmtTime(sum.job.finished)}.
        Stored trades: ${nf0.format(sum.stored.n)}, net ${inr(sum.stored.net)}.</div>
        <button class="ghost" id="del-run">Delete this run</button></div></div>`;
  } else if (!sum) {
    body = `<div class="card empty"><h3>No results yet</h3><p>Press “Run backtest”. A full Nifty 500 run takes a few minutes.</p></div>`;
  } else body = `<div class="card empty"><h3>No trades</h3><p>The rules never triggered in this run.</p></div>`;
  box.innerHTML = runCard + body;
  if ($('#eq')) equityChart($('#eq'), sum.summary.equity);

  const finish = (j) => {
    if (j.status === 'failed') toast('Backtest failed: ' + j.message);
    state.run[d.id] = null; render();
  };
  if (active) watchJob(active.id, tickProgress, finish);
  $('#run-btn').onclick = async () => {
    try {
      const {job} = await api(`/api/strategy/${d.id}/run`, {method: 'POST', body: {}});
      $('#run-btn').disabled = true; $('#run-btn').textContent = 'Running…';
      $('#run-prog').innerHTML = progressHtml({});
      watchJob(job, tickProgress, finish);
    } catch (e) { toast(e.message); }
  };
  const del = $('#del-run');
  if (del) del.onclick = async () => {
    if (!confirm(`Delete run ${sum.job.id} and its trades? This cannot be undone.`)) return;
    await api(`/api/run/${sum.job.id}`, {method: 'DELETE'});
    delete state.summary[sum.job.id]; state.run[d.id] = null; render();
  };
}

// ---------------------------------------------------------------- trades tab
function subTrades(box, d, sum) {
  if (!sum) { box.innerHTML = noRun(d); return; }
  const f = state.trades;
  const reasons = (sum.summary.breakdowns?.['Exit reason'] || []).map((r) => r.key);
  box.innerHTML = `<div class="card"><div class="row">
      <input type="search" id="f-symbol" placeholder="Stock, e.g. RELIANCE" value="${esc(f.symbol)}" style="width:190px">
      <select id="f-side"><option value="">Buys and sells</option><option value="LONG">Buys (long)</option><option value="SHORT">Sells (short)</option></select>
      <select id="f-result"><option value="">Wins and losses</option><option value="win">Wins</option><option value="loss">Losses</option></select>
      <select id="f-reason"><option value="">Any exit</option>${reasons.map((r) => `<option>${esc(r)}</option>`).join('')}</select>
      <span class="muted" id="t-count"></span></div>
    <div class="scroll" style="margin-top:10px"><table><thead><tr>
      <th class="sort" data-s="entry_t">Date</th><th class="sort" data-s="symbol">Stock</th><th>Side</th><th>In</th><th>Out</th>
      <th class="num">Entry</th><th class="num">Stop</th><th class="num">Exit</th><th class="sort" data-s="reason">Exit reason</th>
      <th class="num sort" data-s="net">Net ₹</th><th class="num sort" data-s="r">R</th></tr></thead><tbody id="t-body"></tbody></table></div>
    <div class="pager"><span class="muted">Click a trade to see it on the chart.</span>
      <span class="row"><button class="ghost" id="t-prev">← Newer</button><span id="t-page" class="muted"></span><button class="ghost" id="t-next">Older →</button></span></div></div>`;
  $('#f-side').value = f.side; $('#f-result').value = f.result; $('#f-reason').value = f.reason;
  const size = 50;
  const load = async () => {
    const q = new URLSearchParams({symbol: f.symbol, side: f.side, reason: f.reason, result: f.result, sort: f.sort,
      desc: f.desc, page: f.page, size});
    const res = await api(`/api/run/${sum.job.id}/trades?${q}`);
    $('#t-count').innerHTML = `${nf0.format(res.total)} trades, net ${signed(res.net)}`;
    $('#t-body').innerHTML = res.rows.map((t) => `<tr class="click" data-id="${t.id}"><td>${fmtDate(t.entry_t)}</td><td><b>${esc(t.symbol)}</b></td>
      <td><span class="pill ${t.side === 'SHORT' ? 'short' : ''}">${t.side === 'LONG' ? 'Buy' : 'Sell'}</span></td>
      <td>${fmtTime(t.entry_t)}</td><td>${fmtTime(t.exit_t)}</td><td class="num">${nf2.format(t.entry)}</td><td class="num">${nf2.format(t.stop)}</td>
      <td class="num">${nf2.format(t.exit)}</td><td>${esc(t.reason)}</td><td class="num">${signed(t.net)}</td><td class="num">${rr(t.r)}</td></tr>`).join('')
      || '<tr><td colspan="11" class="muted">No trades match these filters.</td></tr>';
    const pages = Math.max(1, Math.ceil(res.total / size));
    $('#t-page').textContent = `Page ${f.page + 1} of ${nf0.format(pages)}`;
    $('#t-prev').disabled = f.page === 0; $('#t-next').disabled = f.page >= pages - 1;
  };
  const refilter = () => { f.page = 0; load(); };
  let deb;
  $('#f-symbol').oninput = (e) => { clearTimeout(deb); deb = setTimeout(() => { f.symbol = e.target.value.trim(); refilter(); }, 300); };
  $('#f-side').onchange = (e) => { f.side = e.target.value; refilter(); };
  $('#f-result').onchange = (e) => { f.result = e.target.value; refilter(); };
  $('#f-reason').onchange = (e) => { f.reason = e.target.value; refilter(); };
  box.querySelectorAll('th.sort').forEach((th) => th.onclick = () => {
    if (f.sort === th.dataset.s) f.desc = f.desc ? 0 : 1; else { f.sort = th.dataset.s; f.desc = 1; }
    refilter();
  });
  $('#t-prev').onclick = () => { f.page--; load(); };
  $('#t-next').onclick = () => { f.page++; load(); };
  $('#t-body').onclick = (e) => { const tr = e.target.closest('tr[data-id]'); if (tr) openTrade(+tr.dataset.id); };
  load();
}

let modalChart = null;
function closeModal() {
  if (modalChart) { modalChart.remove(); modalChart = null; }
  $('#modal').hidden = true;
}
$('#modal-close').onclick = closeModal;
$('#modal').onclick = (e) => { if (e.target.id === 'modal') closeModal(); };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

async function openTrade(id) {
  let c;
  try { c = await api(`/api/trade/${id}/chart`); } catch (e) { toast(e.message); return; }
  const t = c.trade, long = t.side === 'LONG', tf = c.timeframe_min * 60;
  const utc = (ts, o) => new Date(ts * 1000).toLocaleString('en-GB', {timeZone: 'UTC', ...o});   // times arrive as IST clock time
  $('#modal-title').innerHTML = `<h3>${esc(t.symbol)} · ${long ? 'Buy' : 'Sell'} · ${utc(t.entry_t, {day: '2-digit', month: 'short', year: 'numeric'})}</h3>
    <div class="muted">In ${utc(t.entry_t, {hour: '2-digit', minute: '2-digit'})} at ${nf2.format(t.entry)} · stop ${nf2.format(t.stop)} ·
    out ${utc(t.exit_t, {hour: '2-digit', minute: '2-digit'})} at ${nf2.format(t.exit)} (${esc(t.reason)}) · ${t.qty} shares · net ${signed(t.net)} · ${rr(t.r)}</div>`;
  $('#modal-body').innerHTML = `<div class="legend">${c.panes.flatMap((p) => p.lines).map((l) => `<span><i class="dot" style="background:${l.color}"></i>${esc(l.name)}</span>`).join('')}
    <span>S = setup candle, ▲▼ = entry and exit</span></div><div id="tchart" class="chart tall"></div>`;
  $('#modal').hidden = false;
  const chart = modalChart = LightweightCharts.createChart($('#tchart'), chartOptions());
  const good = css('--good'), bad = css('--bad');
  const cs = chart.addSeries(LightweightCharts.CandlestickSeries, {upColor: good, downColor: bad, borderVisible: false,
    wickUpColor: good, wickDownColor: bad, priceLineVisible: false});
  cs.setData(c.candles.map(([time, open, high, low, close]) => ({time, open, high, low, close})));
  const bar = (ts) => { let b = c.candles[0][0]; for (const k of c.candles) { if (k[0] <= ts) b = k[0]; else break; } return b; };
  const text = css('--text');
  const marks = [
    {time: bar(t.setup_t), position: long ? 'belowBar' : 'aboveBar', color: text, shape: 'circle', text: 'S'},
    {time: bar(t.entry_t), position: long ? 'belowBar' : 'aboveBar', color: long ? good : bad, shape: long ? 'arrowUp' : 'arrowDown', text: long ? 'Buy' : 'Sell'},
    {time: bar(t.exit_t), position: long ? 'aboveBar' : 'belowBar', color: text, shape: long ? 'arrowDown' : 'arrowUp', text: 'Exit'},
  ].sort((a, b) => a.time - b.time);
  LightweightCharts.createSeriesMarkers(cs, marks);
  cs.createPriceLine({price: t.entry, color: css('--accent'), lineStyle: 2, lineWidth: 1, title: 'Entry'});
  cs.createPriceLine({price: t.stop, color: bad, lineStyle: 2, lineWidth: 1, title: 'Stop'});
  c.panes.forEach((p, i) => p.lines.forEach((l) => {
    const s = chart.addSeries(LightweightCharts.LineSeries, {color: l.color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false}, i + 1);
    s.setData(l.data.filter((x) => x[1] != null).map(([time, value]) => ({time, value})));
  }));
  chart.timeScale().fitContent();
}

// ---------------------------------------------------------------- stocks, signals, settings
function subStocks(box, d, sum) {
  if (!sum) { box.innerHTML = noRun(d); return; }
  const st = state.stocks, rows = [...(sum.summary.stocks || [])];
  const cols = [['symbol', 'Stock'], ['n', 'Trades'], ['win_rate', 'Win rate'], ['gross', 'Before costs'], ['net', 'Net after costs'],
    ['exp_r', 'Avg per trade'], ['pf', 'Profit factor'], ['liquidity', 'How heavily traded']];
  rows.sort((a, b) => {
    const x = a[st.sort], y = b[st.sort];
    const c = typeof x === 'string' ? x.localeCompare(y) : (x ?? -Infinity) - (y ?? -Infinity);
    return st.desc ? -c : c;
  });
  const winners = rows.filter((r) => r.net > 0).length;
  box.innerHTML = `<div class="card"><div class="muted">${winners} of ${rows.length} stocks made money after costs. Click a column to sort; click a stock to see its trades.</div>
    <div class="scroll" style="margin-top:10px"><table><thead><tr>${cols.map(([k, t], i) => `<th class="sort ${i && i < 7 ? 'num' : ''}" data-s="${k}">${t}${st.sort === k ? (st.desc ? ' ↓' : ' ↑') : ''}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((r) => `<tr class="click" data-sym="${esc(r.symbol)}"><td><b>${esc(r.symbol)}</b></td><td class="num">${r.n}</td><td class="num">${pct(r.win_rate)}</td>
      <td class="num">${signed(r.gross)}</td><td class="num">${signed(r.net)}</td><td class="num">${rr(r.exp_r)}</td>
      <td class="num">${r.pf == null ? '–' : r.pf.toFixed(2)}</td><td>${esc(r.liquidity)}</td></tr>`).join('')}</tbody></table></div></div>`;
  box.querySelectorAll('th.sort').forEach((th) => th.onclick = () => {
    if (st.sort === th.dataset.s) st.desc = st.desc ? 0 : 1; else { st.sort = th.dataset.s; st.desc = th.dataset.s === 'symbol' ? 0 : 1; }
    subStocks(box, d, sum);
  });
  $('tbody', box).onclick = (e) => {
    const tr = e.target.closest('tr[data-sym]');
    if (tr) { Object.assign(state.trades, {symbol: tr.dataset.sym, page: 0, side: '', reason: '', result: ''}); location.hash = `#/s/${d.id}/trades`; }
  };
}

function subSignals(box) {
  box.innerHTML = `<div class="card empty"><h3>Live signals are not switched on yet</h3>
    <p>This tab will list setups as they form during market hours (checked every 15 minutes) and send them to Telegram. It gets built only if the backtest shows the strategy is worth trading.</p></div>`;
}

function fieldHtml(p) {
  const id = `p-${p.key}`;
  let input;
  if (p.type === 'bool') input = `<input type="checkbox" id="${id}" ${p.value ? 'checked' : ''}>`;
  else if (p.type === 'choice') input = `<select id="${id}">${p.choices.map((c) => `<option ${c === p.value ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
  else if (p.type === 'time') input = `<input type="time" id="${id}" value="${esc(p.value)}">`;
  else input = `<input type="number" id="${id}" value="${esc(p.value)}" min="${p.min ?? ''}" max="${p.max ?? ''}" step="${p.type === 'int' ? 1 : 'any'}">`;
  const changed = String(p.value) !== String(p.default) ? `<span class="changed">Changed. Default: ${esc(p.default)}</span>` : '';
  return `<div class="field"><label for="${id}">${esc(p.label)}</label><div>${input}</div><div class="help">${esc(p.help || '')}${changed}</div></div>`;
}

function subSettings(box, d) {
  box.innerHTML = d.settings.map((g) => `<div class="card"><h3>${esc(g.title)}</h3>${g.params.map(fieldHtml).join('')}</div>`).join('')
    + `<div class="row"><button class="primary" id="save">Save settings</button><button class="ghost" id="reset">Reset to defaults</button>
       <span class="muted">Saved values are used by the next backtest. Old runs keep the values they were run with.</span></div>`;
  $('#save').onclick = async () => {
    const params = {};
    d.settings.flatMap((g) => g.params).forEach((p) => {
      const el = $(`#p-${p.key}`);
      params[p.key] = p.type === 'bool' ? el.checked : el.value;
    });
    try { await api(`/api/strategy/${d.id}/settings`, {body: {params}}); toast('Saved. Run a backtest to see the effect.'); render(); }
    catch (e) { toast(e.message); }
  };
  $('#reset').onclick = async () => {
    await api(`/api/strategy/${d.id}/settings`, {body: {reset: true}}); toast('Back to defaults.'); render();
  };
}

// ---------------------------------------------------------------- compare and data
async function viewCompare() {
  const rows = state.strategies;
  view.innerHTML = `<div class="head"><div><h2>Compare strategies</h2><p class="muted">The latest finished backtest of each strategy, side by side.</p></div></div>
    <div class="card" style="margin-top:16px"><div class="scroll"><table><thead><tr><th>Strategy</th><th>Period</th><th class="num">Trades</th><th class="num">Win rate</th>
      <th class="num">Before costs</th><th class="num">Net after costs</th><th class="num">Avg per trade</th><th class="num">Profit factor</th><th class="num">Worst drawdown</th></tr></thead>
    <tbody>${rows.map((s) => {
      const l = s.latest, h = l && l.headline;
      return `<tr class="click" onclick="location.hash='#/s/${s.id}/overview'"><td><b>${esc(s.name)}</b></td>` + (h && h.n
        ? `<td>${fmtDate(l.start_ts)} to ${fmtDate(l.end_ts)}</td><td class="num">${nf0.format(h.n)}</td><td class="num">${pct(h.win_rate)}</td><td class="num">${signed(h.gross)}</td>
           <td class="num">${signed(h.net)}</td><td class="num">${rr(h.exp_r)}</td><td class="num">${h.pf == null ? '–' : h.pf.toFixed(2)}</td><td class="num">${inr(h.max_dd)}</td>`
        : '<td colspan="8" class="muted">No backtest yet</td>') + '</tr>';
    }).join('') || '<tr><td colspan="9" class="muted">No strategies found.</td></tr>'}</tbody></table></div></div>`;
}

async function viewData() {
  const d = await api('/api/data'), s = d.status, active = d.active;
  const day = (iso) => iso ? new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', {timeZone: 'UTC', day: '2-digit', month: 'short', year: 'numeric'}) : '–';
  view.innerHTML = `<div class="head"><div><h2>Data</h2><p class="muted">5-minute candles for the Nifty 500 from Upstox, shared by every strategy. Bigger candles (15-minute and so on) are built from these.</p></div></div>
    <div class="stack" style="margin-top:16px"><div class="grid">
      ${stat('Stocks downloaded', `${s.downloaded} / ${s.stocks}`, 'Nifty 500 list from NSE')}
      ${stat('From', day(s.from), 'Includes lead-in days for indicators')}
      ${stat('Up to', day(s.to), 'Last day fetched for every stock')}
      ${stat('Size', nf0.format(s.size_mb) + ' MB', 'Stored on this server')}</div>
    <div class="card"><div class="row" style="justify-content:space-between">
      <div><h3>${s.downloaded ? 'Top up to today' : 'Download candles'}</h3><div class="muted">The first download of 3 years takes about 5 hours (Upstox limits the speed). Later top-ups take about 10 minutes. Safe to stop and restart.</div></div>
      <button class="primary" id="dl" ${active ? 'disabled' : ''}>${active ? 'Downloading…' : (s.downloaded ? 'Update data' : 'Download')}</button></div>
      <div id="run-prog">${active ? progressHtml(active) : (d.last ? `<div class="muted" style="margin-top:8px">Last download: ${esc(d.last.message)} (${fmtDate(d.last.finished || d.last.started)})</div>` : '')}</div></div>
    <div class="card prose"><h3>Good to know</h3><ul>
      <li>Prices are adjusted for splits and bonuses, so old candles line up with today's prices.</li>
      <li>The stock list is today's Nifty 500. Stocks that dropped out of the index in the last 3 years are not tested (this flatters results a little).</li>
      <li>Stocks that listed recently simply have a shorter history.</li></ul></div></div>`;
  const finish = (j) => { if (j.status === 'failed') toast('Download failed: ' + j.message); render(); };
  if (active) watchJob(active.id, tickProgress, finish);
  $('#dl').onclick = async () => {
    try {
      const {job} = await api('/api/data/download', {method: 'POST', body: {}});
      $('#dl').disabled = true; $('#dl').textContent = 'Downloading…';
      $('#run-prog').innerHTML = progressHtml({});
      watchJob(job, tickProgress, finish);
    } catch (e) { toast(e.message); }
  };
}

// ---------------------------------------------------------------- start
async function boot() {
  try {
    const s = await api('/api/strategies');
    state.strategies = s.strategies; state.login = s.login;
    $('#logout').hidden = !s.login;
    $('#logout').onclick = async () => { await api('/api/logout', {method: 'POST', body: {}}); location.href = '/login'; };
  } catch (e) { view.innerHTML = `<p class="pad bad">${esc(e.message)}</p>`; return; }
  window.addEventListener('hashchange', async () => {
    if (route()?.page === 'compare') state.strategies = (await api('/api/strategies')).strategies;
    render();
  });
  render();
}
boot();
