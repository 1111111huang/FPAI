"use strict";

/* Standalone ops dashboard -- vanilla JS, no build step. Fetches real data
 * through serve.py's same-origin /api/* proxy (so no CORS config, no token
 * in this file at all). Visual language matches the approved brainstorm
 * mockup: warm palette, Bricolage/Source Serif/IBM Plex Mono, real
 * axis-labeled charts (never a bare bar with no scale -- same rule the
 * real Oddsey /ops dashboard learned the hard way). */

const LEAGUE_LABEL = {
  E0: "Premier League", SWE: "Allsvenskan", SP1: "La Liga",
  I1: "Serie A", D1: "Bundesliga", F1: "Ligue 1",
};
const WEEK_OPTIONS = [4, 8, 12, 26];
const TABS = [
  { key: "failures", label: "Failures" },
  { key: "odds", label: "Odds Availability" },
  { key: "features", label: "Feature Health" },
  { key: "performance", label: "Performance" },
];
const METRICS = [
  { key: "hit_rate", label: "Hit Rate" },
  { key: "roi", label: "ROI" },
  { key: "bets_placed", label: "Bet Frequency" },
  { key: "total_profit", label: "Total Profit" },
];
const STAKING_MODES = [
  { key: "kelly", label: "Kelly" },
  { key: "flat", label: "Flat bet" },
  { key: "ub", label: "UB (app's shown amount)" },
];

const state = {
  tab: "failures",
  weeks: 8,
  league: "",
  perfMetric: "hit_rate",
  perfPickScope: "one_per_match",
  perfIncludeConditional: false,
  perfStakingMode: "kelly",
};

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function apiGet(path, params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const qs = q.toString();
  const res = await fetch(`/api/admin/ops/${path}${qs ? `?${qs}` : ""}`);
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body.message || body.detail || JSON.stringify(body);
    } catch {}
    throw new Error(`Failed to load (${res.status}): ${detail}`);
  }
  return res.json();
}

/* ---------- chrome: filter bar, tabs ---------- */

function renderChrome() {
  const chipsEl = document.getElementById("week-chips");
  chipsEl.innerHTML = "";
  for (const w of WEEK_OPTIONS) {
    const btn = el(`<button type="button" class="chip" aria-pressed="${w === state.weeks}">${w}w</button>`);
    btn.addEventListener("click", () => {
      state.weeks = w;
      renderChrome();
      renderActiveTab();
    });
    chipsEl.appendChild(btn);
  }

  const select = document.getElementById("league-select");
  if (select.options.length <= 1) {
    for (const [code, label] of Object.entries(LEAGUE_LABEL)) {
      select.appendChild(el(`<option value="${code}">${esc(label)}</option>`));
    }
    select.addEventListener("change", () => {
      state.league = select.value;
      renderActiveTab();
    });
  }
  select.value = state.league;

  const tabsEl = document.getElementById("tabs");
  tabsEl.innerHTML = "";
  for (const t of TABS) {
    const btn = el(`<button type="button" class="tab" role="tab" aria-selected="${t.key === state.tab}">${esc(t.label)}</button>`);
    btn.addEventListener("click", () => {
      state.tab = t.key;
      renderChrome();
      renderActiveTab();
    });
    tabsEl.appendChild(btn);
  }

  const panelsEl = document.getElementById("panels");
  if (!panelsEl.children.length) {
    for (const t of TABS) {
      panelsEl.appendChild(el(`<div class="panel" id="panel-${t.key}"></div>`));
    }
  }
  for (const t of TABS) {
    document.getElementById(`panel-${t.key}`).classList.toggle("active", t.key === state.tab);
  }
}

function loadingPanel() {
  return `<div class="state-msg">Loading&hellip;</div>`;
}

function errorPanel(message, onRetry) {
  const wrap = el(`<div class="state-msg error">${esc(message)}<button type="button" class="retry-btn">Retry</button></div>`);
  wrap.querySelector(".retry-btn").addEventListener("click", onRetry);
  return wrap;
}

function renderActiveTab() {
  if (state.tab === "failures") renderFailures();
  else if (state.tab === "odds") renderOddsAvailability();
  else if (state.tab === "features") renderFeatureHealth();
  else if (state.tab === "performance") renderPerformance();
}

/* ---------- shared axis bar chart (SVG, real gridlines + tick labels) ---------- */

/* Rounds a positive magnitude up to a "nice" 1/2/5 x 10^n ceiling, e.g.
 * 847 -> 1000, 18 -> 20, 3.1 -> 5 -- so axis ticks read as round numbers,
 * not raw data maxima. Same algorithm the real Oddsey /ops dashboard's
 * PerformanceTab uses for this exact purpose. */
function niceCeil(magnitude) {
  if (magnitude <= 0) return 0;
  const exp = Math.floor(Math.log10(magnitude));
  const base = Math.pow(10, exp);
  const frac = magnitude / base;
  const niceFrac = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return niceFrac * base;
}

/* A bidirectional axis: domainMin/domainMax may straddle zero (a losing
 * week's negative ROI/profit, say). Bars for a positive value grow
 * rightward from the zero line; bars for a negative value hang leftward
 * from it. Ticks span the real domain, not just [0, max] -- a value
 * below zero must render on the other side of the axis from a positive
 * one, never just a shorter bar growing the same direction. */
function axisBarChartSvg({ rows, width = 620, labelColW = 150, valueColW = 60, barH = 16, gap = 10, tickFormat }) {
  const rawMax = Math.max(...rows.map((r) => r.value), 0);
  const rawMin = Math.min(...rows.map((r) => r.value), 0);
  const domainMax = rawMax > 0 ? niceCeil(rawMax) : niceCeil(1);
  const domainMin = rawMin < 0 ? -niceCeil(-rawMin) : 0;
  const span = domainMax - domainMin || 1;

  const chartW = width - labelColW - valueColW - 16;
  const originX = labelColW;
  const n = rows.length;
  const h = n * (barH + gap) + 30;
  const toX = (v) => originX + ((v - domainMin) / span) * chartW;
  const zeroX = toX(0);

  const tickFractions = [0, 0.25, 0.5, 0.75, 1];
  const ticks = tickFractions.map((f) => ({ x: originX + f * chartW, v: domainMin + f * span }));
  let svg = `<svg width="100%" viewBox="0 0 ${width} ${h}" role="img" aria-label="Bar chart">`;
  for (const t of ticks) {
    const isZero = domainMin < 0 && Math.abs(t.v) < span * 0.001;
    const isEdge = t === ticks[ticks.length - 1];
    svg += `<line class="gridline" x1="${t.x}" y1="4" x2="${t.x}" y2="${h - 22}" ${isZero || isEdge ? 'style="stroke:var(--text-muted)"' : ""}/>`;
    svg += `<text class="tick-label" x="${t.x}" y="${h - 8}" text-anchor="middle">${esc(tickFormat(t.v))}</text>`;
  }
  rows.forEach((row, i) => {
    const y = 8 + i * (barH + gap);
    const barX = toX(row.value);
    const x = Math.min(barX, zeroX);
    const w = Math.max(Math.abs(barX - zeroX), 1);
    svg += `<text class="axis-label" x="${originX - 8}" y="${y + barH * 0.72}" text-anchor="end">${esc(row.label)}</text>`;
    svg += `<rect x="${x}" y="${y}" width="${w}" height="${barH}" rx="3" fill="${row.fill || "var(--accent)"}"/>`;
    const labelX = row.value < 0 ? x - 6 : x + w + 6;
    const anchor = row.value < 0 ? "end" : "start";
    svg += `<text class="axis-label mono" x="${labelX}" y="${y + barH * 0.72}" text-anchor="${anchor}" fill="var(--text)">${esc(row.valueLabel)}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

/* ---------- Failures tab ---------- */

async function renderFailures() {
  const panel = document.getElementById("panel-failures");
  panel.innerHTML = loadingPanel();
  try {
    const data = await apiGet("failures", { league: state.league || undefined });
    const breakers = data.circuit_breakers || {};
    const breakerRow = Object.entries(breakers)
      .map(([vendor, s]) => `<span class="breaker-pill ${esc(s.state)}">${esc(vendor)}: ${esc(s.state)}</span>`)
      .join("");
    const failures = data.failures || [];
    const rows = failures.length
      ? failures.map((f) => `
          <div class="fail-row">
            <div class="fail-head">
              <span class="fail-exc">${esc(f.exc_type)}</span>
              <span class="fail-meta">${esc(f.subsystem)} &middot; ${esc(f.unit_id)} &middot; ${esc(f.league || "—")} &middot; ${esc(f.ts)}</span>
            </div>
            <div class="fail-msg">${esc(f.message)}</div>
          </div>`).join("")
      : `<div class="state-msg">No failures recorded.</div>`;
    panel.innerHTML = `
      <div class="card">
        <h3 class="section-title">Circuit Breakers &amp; Recent Failures</h3>
        <p class="section-dek">Recent run_isolated() failures and live vendor breaker state.</p>
        <div class="breaker-row">${breakerRow}</div>
        ${rows}
      </div>`;
  } catch (err) {
    panel.innerHTML = "";
    panel.appendChild(errorPanel(err.message, renderFailures));
  }
}

/* ---------- Odds Availability tab ---------- */

async function renderOddsAvailability() {
  const panel = document.getElementById("panel-odds");
  panel.innerHTML = loadingPanel();
  try {
    const data = await apiGet("odds-availability-trend", { weeks: state.weeks, league: state.league || undefined });
    const totals = {};
    let grandTotal = 0;
    for (const week of data.weeks || []) {
      grandTotal += week.total_fixtures;
      for (const [market, stats] of Object.entries(week.by_market || {})) {
        totals[market] = (totals[market] || 0) + stats.missing;
      }
    }
    const ranked = Object.entries(totals)
      .map(([market, missing]) => ({ market, pct: grandTotal > 0 ? missing / grandTotal : 0 }))
      .sort((a, b) => b.pct - a.pct);

    if (ranked.length === 0) {
      panel.innerHTML = `<div class="card"><h3 class="section-title">Constantly Missing Markets</h3><div class="state-msg">No missing markets in this window.</div></div>`;
      return;
    }
    const chart = axisBarChartSvg({
      rows: ranked.map((r) => ({
        label: r.market, value: r.pct,
        valueLabel: `${Math.round(r.pct * 100)}%`,
        fill: r.pct >= 0.5 ? "var(--negative)" : r.pct >= 0.2 ? "var(--warning)" : "var(--text-muted)",
      })),
      tickFormat: (v) => `${Math.round(v * 100)}%`,
    });
    panel.innerHTML = `
      <div class="card">
        <h3 class="section-title">Constantly Missing Markets</h3>
        <p class="section-dek">% of fixtures missing each market, aggregated across the ${state.weeks}-week window.</p>
        ${chart}
      </div>`;
  } catch (err) {
    panel.innerHTML = "";
    panel.appendChild(errorPanel(err.message, renderOddsAvailability));
  }
}

/* ---------- Feature Health tab ---------- */

/* Direct user request: show raw rolling min/mean/max per feature instead
 * of a pre-computed drift verdict -- flag_drifting_features's baseline-
 * vs-recent comparison structurally can't catch a feature that's been
 * constantly wrong since before the tracked window starts (a flat
 * min==mean==max line is exactly BUG-096's real signature: one constant
 * value across every match, not a value that changed partway through).
 * A human scanning raw trendlines catches that at a glance; a drift score
 * comparing two already-broken halves of the window never would. */

let featureSearch = "";
let featureVisible = 12;

function featureTrendSvg(weekly) {
  const withData = weekly.filter((w) => w.min !== null && w.max !== null);
  if (withData.length === 0) {
    return `<div class="state-msg" style="padding:8px 0">No numeric values this window (null every time it appeared).</div>`;
  }
  const allMin = Math.min(...withData.map((w) => w.min));
  const allMax = Math.max(...withData.map((w) => w.max));
  const span = (allMax - allMin) || Math.abs(allMax) || 1;
  const domainMin = allMin - span * 0.08;
  const domainMax = allMax + span * 0.08;
  const domainSpan = domainMax - domainMin || 1;

  const width = 420, height = 90, leftPad = 46, rightPad = 10, topPad = 6, bottomPad = 18;
  const chartW = width - leftPad - rightPad;
  const chartH = height - topPad - bottomPad;
  const n = weekly.length;
  const toX = (i) => leftPad + (n === 1 ? chartW / 2 : (i / (n - 1)) * chartW);
  const toY = (v) => topPad + chartH - ((v - domainMin) / domainSpan) * chartH;

  const bandTop = [], bandBottom = [], meanPts = [];
  weekly.forEach((w, i) => {
    if (w.min === null || w.max === null) return;
    bandTop.push(`${toX(i)},${toY(w.max)}`);
    bandBottom.push(`${toX(i)},${toY(w.min)}`);
    if (w.mean !== null) meanPts.push(`${toX(i)},${toY(w.mean)}`);
  });
  const band = bandTop.length ? `${bandTop.join(" ")} ${bandBottom.reverse().join(" ")}` : "";

  const ticks = [allMin, (allMin + allMax) / 2, allMax];
  let svg = `<svg width="100%" viewBox="0 0 ${width} ${height}" role="img" aria-label="Weekly min/mean/max trend">`;
  ticks.forEach((v) => {
    const y = toY(v);
    svg += `<line class="gridline" x1="${leftPad}" y1="${y}" x2="${width - rightPad}" y2="${y}"/>`;
    svg += `<text class="tick-label" x="${leftPad - 6}" y="${y + 3}" text-anchor="end">${v.toFixed(2)}</text>`;
  });
  if (band) svg += `<polygon points="${band}" fill="var(--accent-soft)" opacity="0.7"/>`;
  if (meanPts.length) svg += `<polyline points="${meanPts.join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
  weekly.forEach((w, i) => {
    svg += `<text class="tick-label" x="${toX(i)}" y="${height - 4}" text-anchor="middle">${esc(w.week_start.slice(5))}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

async function renderFeatureHealth() {
  const panel = document.getElementById("panel-features");
  panel.innerHTML = loadingPanel();
  try {
    const data = await apiGet("feature-health-trend", { weeks: state.weeks, league: state.league || undefined });
    const all = data.all_features || [];
    const flagByName = new Map((data.flagged_features || []).map((f) => [f.name, f.flags]));

    if (all.length === 0) {
      panel.innerHTML = `<div class="card"><h3 class="section-title">Feature Trends</h3><div class="state-msg">No feature data in this window (tracked_feature_count: ${data.tracked_feature_count}).</div></div>`;
      return;
    }

    function renderList() {
      const q = featureSearch.trim().toLowerCase();
      const filtered = q ? all.filter((f) => f.name.toLowerCase().includes(q)) : all;
      const visible = filtered.slice(0, featureVisible);
      const cards = visible.map((f) => {
        const flags = flagByName.get(f.name) || [];
        const badges = flags.map((fl) => `<span class="feat-flag ${fl}">${fl === "drift" ? "DRIFT" : "NULL RATE &uarr;"}</span>`).join("");
        return `<div class="feat-trend-card">
          <div class="feat-trend-head">
            <div class="feat-id"><span class="feat-name">${esc(f.name)}</span><span class="feat-cat">${esc(f.category)}</span></div>
            <div>${badges}</div>
          </div>
          ${featureTrendSvg(f.weekly)}
        </div>`;
      }).join("");
      const more = filtered.length > featureVisible
        ? `<button type="button" class="show-more">Show ${filtered.length - featureVisible} more &darr;</button>` : "";
      const listCard = document.getElementById("feat-list-card");
      listCard.innerHTML = `
        <div class="feat-toolbar">
          <h3 class="section-title" style="margin-bottom:0">Feature Trends &mdash; ${esc(String(all.length))} tracked</h3>
          <input class="feat-search" placeholder="Filter by name, e.g. XOC" value="${esc(featureSearch)}">
        </div>
        <p class="section-dek">Rolling min/mean/max per week, every tracked feature &mdash; a flat line (min &asymp; mean &asymp; max, unmoving across weeks) is the signature of a stuck/placeholder value, same as BUG-096's goal-market features. Badges are an optional hint (drift-vs-baseline), not a filter.</p>
        <div class="feat-trend-grid">${cards || '<div class="state-msg">No features match that filter.</div>'}</div>
        ${more}`;
      listCard.querySelector(".feat-search").addEventListener("input", (e) => {
        featureSearch = e.target.value;
        featureVisible = 12;
        renderList();
      });
      const moreBtn = listCard.querySelector(".show-more");
      if (moreBtn) moreBtn.addEventListener("click", () => { featureVisible = filtered.length; renderList(); });
    }

    panel.innerHTML = `<div class="card" id="feat-list-card"></div>`;
    renderList();
  } catch (err) {
    panel.innerHTML = "";
    panel.appendChild(errorPanel(err.message, renderFeatureHealth));
  }
}

/* ---------- Performance tab ---------- */

function formatMetric(metric, week) {
  const v = week[metric];
  if (metric === "hit_rate" || metric === "roi") return `${Math.round(v * 100)}%`;
  if (metric === "bets_placed") return String(v);
  return v.toFixed(1);
}

async function renderPerformance() {
  const panel = document.getElementById("panel-performance");
  panel.innerHTML = loadingPanel();
  try {
    const data = await apiGet("performance-trend", {
      weeks: state.weeks, league: state.league || undefined,
      pick_scope: state.perfPickScope, include_conditional: state.perfIncludeConditional ? "true" : undefined,
      staking_mode: state.perfStakingMode,
    });
    const weeks = data.weeks || [];

    const metricToggles = METRICS.map((m) =>
      `<button type="button" class="toggle-btn" data-metric="${m.key}" aria-pressed="${m.key === state.perfMetric}">${esc(m.label)}</button>`
    ).join("");
    const strategyToggles = `
      <button type="button" class="toggle-btn" id="toggle-pickscope" aria-pressed="${state.perfPickScope === "all_direct"}">
        ${state.perfPickScope === "one_per_match" ? "One pick per match" : "All direct bets"}
      </button>
      <button type="button" class="toggle-btn" id="toggle-conditional" aria-pressed="${state.perfIncludeConditional}">
        ${state.perfIncludeConditional ? "Including conditional" : "Include conditional"}
      </button>`;
    const stakingToggles = STAKING_MODES.map((m) =>
      `<button type="button" class="toggle-btn" data-staking="${m.key}" aria-pressed="${m.key === state.perfStakingMode}">${esc(m.label)}</button>`
    ).join("");

    let chartAndDelta = `<div class="state-msg">No weeks in this window.</div>`;
    if (weeks.length) {
      const isPercent = state.perfMetric === "hit_rate" || state.perfMetric === "roi";
      const chart = axisBarChartSvg({
        rows: weeks.map((w) => ({
          label: w.week_start, value: w[state.perfMetric],
          valueLabel: formatMetric(state.perfMetric, w),
          fill: "var(--accent)",
        })),
        tickFormat: (v) => (isPercent ? `${Math.round(v * 100)}%` : v.toFixed(0)),
      });
      let deltaHtml = "";
      if (weeks.length >= 2 && isPercent) {
        const last = weeks[weeks.length - 1], prev = weeks[weeks.length - 2];
        const delta = Math.round((last[state.perfMetric] - prev[state.perfMetric]) * 100);
        deltaHtml = `<span class="wow-delta ${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "+" : ""}${delta}pp vs previous week</span>`;
      }
      chartAndDelta = `${chart}${deltaHtml}`;
    }

    panel.innerHTML = `
      <div class="card">
        <div class="toggle-row">${metricToggles}</div>
        <div class="toggle-row">${strategyToggles}</div>
        <div class="toggle-row">${stakingToggles}</div>
        <h3 class="section-title">Week over Week</h3>
        <p class="section-dek">Driven by the Window/League filters above.</p>
        ${chartAndDelta}
      </div>`;

    panel.querySelectorAll(".toggle-btn[data-metric]").forEach((btn) => {
      btn.addEventListener("click", () => { state.perfMetric = btn.dataset.metric; renderPerformance(); });
    });
    const pickBtn = panel.querySelector("#toggle-pickscope");
    pickBtn.addEventListener("click", () => {
      state.perfPickScope = state.perfPickScope === "one_per_match" ? "all_direct" : "one_per_match";
      renderPerformance();
    });
    const condBtn = panel.querySelector("#toggle-conditional");
    condBtn.addEventListener("click", () => {
      state.perfIncludeConditional = !state.perfIncludeConditional;
      renderPerformance();
    });
    panel.querySelectorAll(".toggle-btn[data-staking]").forEach((btn) => {
      btn.addEventListener("click", () => { state.perfStakingMode = btn.dataset.staking; renderPerformance(); });
    });
  } catch (err) {
    panel.innerHTML = "";
    panel.appendChild(errorPanel(err.message, renderPerformance));
  }
}

/* ---------- init ---------- */

renderChrome();
renderActiveTab();
