"""Create a self-contained, coach-facing explorer for prospect speed profiles."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FEATURES = [
    ("Build", "40-yard · 0–1s acceleration", "forty_early_accel_0_1s_ydps2_mean", "yd/s²"),
    ("Build", "40-yard · 10–20 yd acceleration", "forty_late_accel_10_20yd_ydps2_mean", "yd/s²"),
    ("Carry", "40-yard · sustained speed", "forty_peak_sustained_speed_ydps_mean", "yd/s"),
    ("Maintain", "40-yard · speed change, 20–40 yd", "forty_speed_change_20_40yd_ydps_mean", "yd/s"),
    ("Maintain", "40-yard · speed retention, 20–40 yd", "forty_speed_retention_20_40yd_ratio_mean", "ratio"),
    ("Redirect", "Short shuttle · 0–1s acceleration", "short_shuttle_early_accel_0_1s_ydps2_mean", "yd/s²"),
    ("Redirect", "Short shuttle · sustained speed", "short_shuttle_peak_sustained_speed_ydps_mean", "yd/s"),
    ("Brake", "Short shuttle · braking magnitude", "short_shuttle_braking_magnitude_ydps2", "yd/s²"),
    ("Cross-axis motion", "Short shuttle · cross-axis speed gap", "short_shuttle_lateral_speed_balance_abs_ydps", "yd/s"),
    ("Redirect", "Three-cone · 0–1s acceleration", "three_cone_early_accel_0_1s_ydps2_mean", "yd/s²"),
    ("Redirect", "Three-cone · sustained speed", "three_cone_peak_sustained_speed_ydps_mean", "yd/s"),
    ("Brake", "Three-cone · braking magnitude", "three_cone_braking_magnitude_ydps2", "yd/s²"),
    ("Cross-axis motion", "Three-cone · cross-axis speed gap", "three_cone_lateral_speed_balance_abs_ydps", "yd/s"),
]


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs"
    profiles = pd.read_csv(out / "player_speed_signatures.csv")
    drills = pd.read_csv(out / "player_drill_speed_profiles.csv")
    selected_drills = {"SHORT_SHUTTLE", "THREE_CONE_DRILL", "PASS_PRO-MIRROR_DRILL",
                       "PASS_PRO_MIRROR_DRILL", "RUN_THE_HOOP_DRILL"}
    drill_mask = drills.drill_type.astype(str).str.startswith("SKILL_DRILLS_") | drills.drill_name.isin(selected_drills)
    drills = drills.loc[drill_mask].copy()
    early_gain = "forty_early_accel_0_1s_ydps2"
    profile_cols = ["nfl_id", "display_name", "draft_year", "nfl_position", "combine_position", "forty", "ten_yd_split",
                    f"{early_gain}_std", f"{early_gain}_count"]
    for _, _, raw, _ in FEATURES:
        profile_cols += [raw, f"{raw}_position_percentile"]
    def script_safe_json(frame: pd.DataFrame) -> str:
        # Prevent any input value from terminating the inline script element.
        return (frame.to_json(orient="records", double_precision=4, force_ascii=True)
                .replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026"))

    p_json = script_safe_json(profiles[profile_cols])

    drill_cols = ["nfl_id", "draft_year", "drill_type", "drill_name",
                  "mean_accel_first_1s_ydps2_mean", "peak_sustained_speed_ydps_mean",
                  "peak_braking_ydps2_mean", "left_right_lateral_speed_difference_by_drill_ydps"]
    d_json = script_safe_json(drills[drill_cols])
    metadata = [{"domain": domain, "label": label, "raw": raw,
                 "pct": f"{raw}_position_percentile", "unit": unit}
                for domain, label, raw, unit in FEATURES]

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Combine Speed Profile Explorer</title>
<style>
:root {{ color-scheme: light; --ink:#142b2b; --muted:#617574; --line:#d9e2dd; --paper:#f5f7f3; --white:#ffffff; --accent:#117c68; --soft:#e6f1ed; --gold:#bb8a35; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 Inter,Segoe UI,Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 26px 44px; }}
header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-end; padding-bottom:22px; border-bottom:1px solid var(--line); }}
.eyebrow {{ color:var(--accent); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; }}
h1 {{ margin:5px 0 4px; font-size:clamp(25px,4vw,36px); letter-spacing:-.035em; line-height:1.1; }}
.sub {{ margin:0; color:var(--muted); max-width:700px; }}
.scope {{ text-align:right; color:var(--muted); font-size:13px; white-space:nowrap; }}
.controls {{ display:grid; grid-template-columns:minmax(145px,.8fr) minmax(210px,1.25fr) minmax(210px,1.25fr) auto; gap:12px; align-items:end; padding:18px 0; }}
label {{ display:block; color:var(--muted); font-size:12px; font-weight:650; margin-bottom:5px; }}
select,input {{ width:100%; min-height:42px; padding:9px 11px; background:var(--white); color:var(--ink); border:1px solid var(--line); border-radius:7px; font:inherit; }}
select:focus-visible,input:focus-visible,button:focus-visible {{ outline:3px solid var(--gold); outline-offset:2px; border-color:var(--ink); }}
.picked {{ border-left:3px solid var(--accent); padding:10px 14px; background:var(--soft); border-radius:0 7px 7px 0; min-width:185px; }}
.picked strong {{ display:block; font-size:16px; }}
.picked span {{ font-size:12px; color:var(--muted); }}
.testline {{ display:flex; flex-wrap:wrap; gap:10px 20px; margin:0 0 20px; color:var(--muted); font-size:13px; }}
.testline b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
section {{ margin-top:20px; }}
.section-head {{ display:flex; justify-content:space-between; gap:16px; align-items:baseline; border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:5px; }}
h2 {{ font-size:18px; margin:0; }}
.section-head p {{ margin:0; color:var(--muted); font-size:12px; }}
.axis {{ display:grid; grid-template-columns:minmax(210px,34%) minmax(150px,1fr) 90px 75px; gap:10px; color:var(--muted); font-size:11px; padding:5px 0 7px; }}
.ticks {{ display:flex; justify-content:space-between; position:relative; padding:0 3px; }}
.domain {{ margin:12px 0 3px; color:var(--accent); text-transform:uppercase; font-size:10px; letter-spacing:.1em; font-weight:750; }}
.row {{ display:grid; grid-template-columns:minmax(210px,34%) minmax(150px,1fr) 90px 75px; gap:10px; align-items:center; min-height:34px; border-bottom:1px solid #e9eeea; }}
.metric {{ font-size:12px; }}
.track {{ height:19px; position:relative; }}
.track::before {{ content:""; position:absolute; left:0; right:0; top:9px; height:1px; background:#d9e2dd; }}
.track::after {{ content:""; position:absolute; top:1px; bottom:1px; left:50%; border-left:1px dashed var(--gold); opacity:.8; }}
.dot {{ position:absolute; top:5px; width:9px; height:9px; border-radius:50%; transform:translateX(-50%); background:var(--accent); box-shadow:0 0 0 2px var(--white); }}
.dot-a {{ top:2px; background:var(--accent); }}
.dot-b {{ top:11px; background:var(--gold); }}
.series-a {{ color:var(--accent); font-weight:700; }}
.series-b {{ color:#8b641c; font-weight:700; }}
.comparison {{ color:var(--muted); font-size:11px; }}
.print-actions {{ display:flex; align-items:center; gap:12px; margin:0 0 14px; }}
.print-actions button {{ min-height:38px; padding:8px 12px; border:1px solid var(--accent); border-radius:6px; background:var(--accent); color:white; font:inherit; font-weight:650; }}
.print-actions button:disabled {{ opacity:.55; cursor:not-allowed; }}
.print-actions span {{ color:var(--muted); font-size:12px; }}
#print-brief {{ display:none; }}
@media print {{
  @page {{ size:letter landscape; margin:.42in; }}
  body {{ background:#fff; color:#111; font-size:9pt; }}
  main {{ max-width:none; margin:0; padding:0; }}
  main > header, .controls, .testline, main > section, main > footer, .print-actions {{ display:none !important; }}
  #print-brief {{ display:block; }}
  #print-brief h1 {{ margin:0 0 2px; font-size:19pt; line-height:1.1; }}
  #print-brief .brief-kicker {{ margin:0 0 3px; color:#333; font-size:8pt; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
  #print-brief .brief-sub {{ margin:0 0 8px; color:#333; }}
  #print-brief .brief-head {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-end; margin-bottom:8px; }}
  #print-brief table {{ width:100%; border-collapse:collapse; }}
  #print-brief th, #print-brief td {{ padding:3px 5px; border-bottom:1px solid #aaa; text-align:left; vertical-align:top; }}
  #print-brief th {{ font-size:8pt; }}
  #print-brief .brief-meta {{ margin:0 0 8px; }}
  #print-brief .brief-meta th {{ width:16%; }}
  #print-brief .brief-meta td {{ width:17%; }}
  #print-brief .brief-measures {{ font-size:8pt; }}
  #print-brief .brief-measures td:nth-child(1) {{ width:17%; }}
  #print-brief .brief-measures td:nth-child(2) {{ width:31%; }}
  #print-brief .brief-measures td:nth-child(3), #print-brief .brief-measures td:nth-child(4) {{ width:21%; }}
  #print-brief .brief-measures td:nth-child(5) {{ width:10%; text-align:right; white-space:nowrap; }}
  #print-brief .brief-domain td {{ padding-top:5px; font-weight:700; }}
  #print-brief .brief-caveats {{ margin:7px 0 0; padding-top:5px; border-top:1px solid #777; font-size:7.5pt; line-height:1.25; }}
  #print-brief tr {{ break-inside:avoid; }}
}}
.raw,.pct {{ font-variant-numeric:tabular-nums; font-size:12px; text-align:right; white-space:nowrap; }}
.missing {{ color:#84918e; font-size:11px; text-align:right; grid-column:2 / 5; }}
.note {{ color:var(--muted); font-size:12px; margin:11px 0 0; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; min-width:700px; }}
th {{ text-align:left; color:var(--muted); font-weight:650; padding:9px 10px 8px 0; border-bottom:1px solid var(--line); }}
td {{ padding:9px 10px 9px 0; border-bottom:1px solid #e9eeea; font-variant-numeric:tabular-nums; }}
td.num {{ text-align:right; }}
.empty {{ color:var(--muted); padding:16px 0; font-size:13px; }}
footer {{ border-top:1px solid var(--line); margin-top:24px; padding-top:12px; color:var(--muted); font-size:11px; }}
@media(max-width:720px) {{ main {{ padding:20px 14px 30px; }} header {{ display:block; }} .scope {{ text-align:left; margin-top:12px; }} .controls {{ grid-template-columns:1fr; }} .picked {{ min-width:0; }} .axis,.row {{ grid-template-columns:minmax(135px,38%) minmax(85px,1fr) 72px 62px; gap:6px; }} .metric,.raw,.pct {{ font-size:10px; }} }}
@media(max-width:390px) {{ .axis,.row {{ grid-template-columns:minmax(118px,40%) minmax(70px,1fr) 58px 50px; gap:4px; }} .raw,.pct {{ font-size:9px; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">NFL Combine · Tracking profiles</div><h1>Speed profile explorer</h1><p class="sub">Build, carry, braking and redirection measures for each prospect, compared with players at the same Combine position.</p></div>
    <div class="scope">2023–2025 cohorts<br><strong>510 prospects</strong> · 13 tracking measures</div>
  </header>
  <div class="controls">
    <div><label for="position">Combine position</label><select id="position"></select></div>
    <div><label for="player">Prospect A</label><select id="player"></select></div>
    <div><label for="compare-player">Compare with · same position</label><select id="compare-player"><option value="">No comparison</option></select></div>
    <div class="picked" aria-live="polite"><strong id="picked-name">—</strong><span id="picked-meta">—</span></div>
  </div>
  <div class="testline" id="traditional" aria-live="polite"></div>
  <div class="print-actions"><button id="print-button" type="button" disabled>Print comparison brief</button><span id="print-status" aria-live="polite">Choose a second prospect to enable the brief.</span></div>
  <section aria-labelledby="profile-heading">
    <div class="section-head"><h2 id="profile-heading">Movement measures</h2><p id="profile-legend">Dot: position percentile · Raw value shown at right</p></div>
    <div class="axis"><div>Measure</div><div class="ticks" aria-label="Percentile scale"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div><div class="raw">Raw</div><div class="pct">Percentile</div></div>
    <div id="profile-rows" role="region" aria-label="Selected prospect's movement measures compared with the same Combine position"></div>
    <p class="note">Percentiles rank the numeric measurement within Combine position across all three cohorts. A higher percentile means a larger value; it does not mean better performance. Missing tests remain blank.</p>
  </section>
  <section aria-labelledby="drills-heading">
    <div class="section-head"><h2 id="drills-heading">Position and skill drills</h2><p>Each drill stays separate</p></div>
    <div id="drills" class="table-wrap"></div>
  </section>
  <section id="print-brief" aria-label="Printable player comparison brief"></section>
  <footer>Movement onset is a sampled-speed convention, not reaction time. The displayed cross-axis gap is the absolute difference between mean positive and negative velocity projected perpendicular to the initial movement path, averaged across available attempts when both signs were observed. Without body orientation or drill-template side labels, it is not anatomical left/right or pure shuttle-direction speed. The 40-yard within-session SD describes spread across at most two available attempts; it is not a stable reliability estimate.</footer>
</main>
<script>
const players = {p_json};
const drillRows = {d_json};
const features = {json.dumps(metadata, ensure_ascii=False)};
const positionSel = document.getElementById('position');
const playerSel = document.getElementById('player');
const compareSel = document.getElementById('compare-player');
const printButton = document.getElementById('print-button');
const printStatus = document.getElementById('print-status');
const rowHost = document.getElementById('profile-rows');
const drillHost = document.getElementById('drills');
let selectedId = null;

function pctFormat(v) {{
  if (!Number.isFinite(v)) return '—';
  const n = Math.round(v * 100), lastTwo = n % 100;
  const suffix = lastTwo >= 11 && lastTwo <= 13 ? 'th' : (n % 10 === 1 ? 'st' : n % 10 === 2 ? 'nd' : n % 10 === 3 ? 'rd' : 'th');
  return `${{n}}${{suffix}}`;
}}
function escapeHTML(value) {{
  return String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;')
    .replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
}}
function rawFormat(v, unit) {{
  if (!Number.isFinite(v)) return '—';
  const places = unit === 'ratio' ? 2 : 1;
  return `${{v.toFixed(places)}} ${{unit}}`;
}}
function attemptSpread(p, label) {{
  const n = p.forty_early_accel_0_1s_ydps2_count;
  const sd = p.forty_early_accel_0_1s_ydps2_std;
  if (!Number.isFinite(n) || n < 1) return `40 early-gain attempts · ${{label}}: <b>No tracked estimate</b>`;
  const count = `${{n}} ${{n === 1 ? 'attempt' : 'attempts'}}`;
  const spread = Number.isFinite(sd)
    ? `within-session SD ${{sd.toFixed(2)}} yd/s²`
    : 'repeat spread unavailable';
  return `40 early-gain attempts · ${{label}}: <b>${{count}} · ${{spread}}</b>`;
}}
function seriesValue(value, label, unit) {{
  const shown = Number.isFinite(value) ? rawFormat(value, unit) : '—';
  return `<span class="series-${{label.toLowerCase()}}">${{label}}</span> ${{shown}}`;
}}
function seriesPercentile(value, label) {{
  return `<span class="series-${{label.toLowerCase()}}">${{label}}</span> ${{pctFormat(value)}}`;
}}
function briefMetricValue(value, unit) {{
  if (!Number.isFinite(value)) return '—';
  return `${{value.toFixed(unit === 'ratio' ? 3 : 2)}} ${{unit}}`;
}}
function briefPlayerCell(player, feature) {{
  const raw = player[feature.raw], percentile = player[feature.pct];
  return `<td><b>${{escapeHTML(briefMetricValue(raw,feature.unit))}}</b><br>Position percentile: ${{escapeHTML(pctFormat(percentile))}}</td>`;
}}
function briefAttemptCell(player) {{
  const count = player.forty_early_accel_0_1s_ydps2_count;
  const sd = player.forty_early_accel_0_1s_ydps2_std;
  if (!Number.isFinite(count) || count < 1) return 'No tracked estimate';
  const spread = Number.isFinite(sd) ? `within-session SD ${{sd.toFixed(2)}} yd/s²` : 'repeat spread unavailable';
  return `${{count}} ${{count === 1 ? 'attempt' : 'attempts'}} · ${{spread}}`;
}}
function renderPrintBrief(a, b) {{
  const aForty = Number.isFinite(a.forty) ? `${{a.forty.toFixed(2)}} s` : '—';
  const bForty = Number.isFinite(b.forty) ? `${{b.forty.toFixed(2)}} s` : '—';
  const aSplit = Number.isFinite(a.ten_yd_split) ? `${{a.ten_yd_split.toFixed(2)}} s` : '—';
  const bSplit = Number.isFinite(b.ten_yd_split) ? `${{b.ten_yd_split.toFixed(2)}} s` : '—';
  const paired = features.filter(f => Number.isFinite(a[f.pct]) && Number.isFinite(b[f.pct])).length;
  let lastDomain = '';
  const rows = features.map(f => {{
    const group = f.domain !== lastDomain ? `<tr class="brief-domain"><td colspan="5">${{escapeHTML(f.domain)}}</td></tr>` : '';
    lastDomain = f.domain;
    const delta = Number.isFinite(a[f.raw]) && Number.isFinite(b[f.raw])
      ? `${{a[f.raw] - b[f.raw] > 0 ? '+' : ''}}${{(a[f.raw] - b[f.raw]).toFixed(f.unit === 'ratio' ? 3 : 2)}} ${{escapeHTML(f.unit)}}`
      : '—';
    return `${{group}}<tr><td>${{escapeHTML(f.label)}}</td>${{briefPlayerCell(a,f)}}${{briefPlayerCell(b,f)}}<td>${{escapeHTML(delta)}}</td></tr>`;
  }}).join('');
  document.getElementById('print-brief').innerHTML = `
    <div class="brief-head"><div><p class="brief-kicker">Combine movement profile · same-position comparison</p><h1>${{escapeHTML(a.display_name)}} vs. ${{escapeHTML(b.display_name)}}</h1><p class="brief-sub">${{escapeHTML(a.combine_position)}} · Combine cohorts 2023–2025 · ${{paired}} of ${{features.length}} measures have percentiles for both players</p></div><div><b>A</b> ${{escapeHTML(a.display_name)}}<br><b>B</b> ${{escapeHTML(b.display_name)}}</div></div>
    <table class="brief-meta"><caption>Player and traditional test context</caption><thead><tr><th scope="col">Context</th><th scope="col">A · ${{escapeHTML(a.display_name)}}</th><th scope="col">B · ${{escapeHTML(b.display_name)}}</th></tr></thead><tbody>
      <tr><th scope="row">Position · draft year</th><td>${{escapeHTML(a.nfl_position)}} · ${{escapeHTML(a.draft_year)}}</td><td>${{escapeHTML(b.nfl_position)}} · ${{escapeHTML(b.draft_year)}}</td></tr>
      <tr><th scope="row">Official 40-yard dash</th><td>${{escapeHTML(aForty)}}</td><td>${{escapeHTML(bForty)}}</td></tr>
      <tr><th scope="row">10-yard split</th><td>${{escapeHTML(aSplit)}}</td><td>${{escapeHTML(bSplit)}}</td></tr>
      <tr><th scope="row">40 early-gain tracking</th><td>${{escapeHTML(briefAttemptCell(a))}}</td><td>${{escapeHTML(briefAttemptCell(b))}}</td></tr>
    </tbody></table>
    <table class="brief-measures"><caption>Movement measures; A − B is a descriptive raw-value difference</caption><thead><tr><th scope="col">Domain</th><th scope="col">Measure</th><th scope="col">A · raw and percentile</th><th scope="col">B · raw and percentile</th><th scope="col">A − B</th></tr></thead><tbody>${{rows}}</tbody></table>
    <p class="brief-caveats"><b>How to read this page:</b> Position percentiles rank numeric values among players at the same Combine position in the 2023–2025 cohorts; a higher percentile means a larger measured value, not better performance. Dashes mean a test or percentile is unavailable; no missing value is treated as zero. Early acceleration begins at detected movement onset; no cue timestamp is available, so it is not reaction time. Within-session SD describes spread across at most two attempts and is not a confidence interval or stable reliability estimate. Cross-axis speed gap is path-relative, not anatomical left/right. These descriptive Combine measurements are not NFL performance predictions, a draft grade, or a combined talent score.</p>`;
}}
function fillPlayerOptions() {{
  const pos = positionSel.value;
  const current = players.find(p => String(p.nfl_id) === String(selectedId));
  const currentCompare = players.find(p => String(p.nfl_id) === String(compareSel.value));
  const list = players.filter(p => pos === 'All' || p.combine_position === pos)
    .sort((a,b) => String(a.display_name).localeCompare(String(b.display_name)));
  playerSel.innerHTML = list.map(p => `<option value="${{escapeHTML(p.nfl_id)}}">${{escapeHTML(p.display_name)}} · ${{escapeHTML(p.draft_year)}}</option>`).join('');
  const keep = current && list.some(p => String(p.nfl_id) === String(current.nfl_id));
  selectedId = keep ? current.nfl_id : (list[0]?.nfl_id ?? null);
  if (selectedId !== null) playerSel.value = String(selectedId);
  fillCompareOptions(list, currentCompare?.nfl_id);
  render();
}}
function fillCompareOptions(list, requestedId) {{
  const first = players.find(p => String(p.nfl_id) === String(playerSel.value));
  const previous = requestedId ?? compareSel.value;
  const options = list.filter(p => String(p.nfl_id) !== String(first?.nfl_id) && p.combine_position === first?.combine_position);
  compareSel.innerHTML = '<option value="">No comparison</option>' + options.map(p => `<option value="${{escapeHTML(p.nfl_id)}}">${{escapeHTML(p.display_name)}} · ${{escapeHTML(p.draft_year)}}</option>`).join('');
  const keep = options.some(p => String(p.nfl_id) === String(previous));
  compareSel.value = keep ? String(previous) : '';
}}
function render() {{
  const p = players.find(x => String(x.nfl_id) === String(playerSel.value));
  if (!p) return;
  const candidate = players.find(x => String(x.nfl_id) === String(compareSel.value));
  const q = candidate && candidate.combine_position === p.combine_position ? candidate : null;
  selectedId = p.nfl_id;
  document.getElementById('picked-name').textContent = p.display_name;
  document.getElementById('picked-meta').textContent = `${{p.nfl_position}} · Combine ${{p.combine_position}} · Draft ${{p.draft_year}}`;
  rowHost.setAttribute('aria-label', q ? `${{p.display_name}} compared with ${{q.display_name}} across movement measures` : `${{p.display_name}} movement measures compared with the same Combine position`);
  const compareTraditional = q ? `<span>Official 40 · <span class="series-b">B</span>: <b>${{Number.isFinite(q.forty) ? q.forty.toFixed(2) + ' s' : '—'}}</b></span><span>10-yard split · <span class="series-b">B</span>: <b>${{Number.isFinite(q.ten_yd_split) ? q.ten_yd_split.toFixed(2) + ' s' : '—'}}</b></span><span>${{attemptSpread(q,'B')}}</span>` : '';
  const firstTraditional = `<span>Official 40 · <span class="series-a">A</span>: <b>${{Number.isFinite(p.forty) ? p.forty.toFixed(2) + ' s' : '—'}}</b></span><span>10-yard split · <span class="series-a">A</span>: <b>${{Number.isFinite(p.ten_yd_split) ? p.ten_yd_split.toFixed(2) + ' s' : '—'}}</b></span><span>${{attemptSpread(p,'A')}}</span>`;
  document.getElementById('traditional').innerHTML = firstTraditional + compareTraditional;
  printButton.disabled = !q;
  printStatus.textContent = q ? 'Print a descriptive comparison; no overall player grade.' : 'Choose a second prospect to enable the brief.';
  if (q) renderPrintBrief(p, q);
  else document.getElementById('print-brief').replaceChildren();
  document.getElementById('profile-legend').innerHTML = q ? `<span class="series-a">A · ${{escapeHTML(p.display_name)}}</span> · <span class="series-b">B · ${{escapeHTML(q.display_name)}}</span> · position percentile scale` : 'Dot: position percentile · Raw value shown at right';
  rowHost.innerHTML = '';
  let domain = '';
  for (const f of features) {{
    if (f.domain !== domain) {{
      domain = f.domain;
      const heading = document.createElement('div'); heading.className='domain'; heading.textContent=domain; rowHost.appendChild(heading);
    }}
    const raw = p[f.raw], rank = p[f.pct];
    const rawB = q ? q[f.raw] : null, rankB = q ? q[f.pct] : null;
    const line = document.createElement('div'); line.className='row';
    const label = document.createElement('div'); label.className='metric'; label.textContent=f.label;
    line.appendChild(label);
    if (!q && Number.isFinite(raw) && Number.isFinite(rank)) {{
      const track = document.createElement('div'); track.className='track';
      const dot = document.createElement('span'); dot.className='dot'; dot.style.left=`${{Math.max(0,Math.min(1,rank))*100}}%`; dot.setAttribute('aria-hidden','true');
      track.appendChild(dot); line.appendChild(track);
      const value = document.createElement('div'); value.className='raw'; value.textContent=rawFormat(raw,f.unit); line.appendChild(value);
      const percentile = document.createElement('div'); percentile.className='pct'; percentile.textContent=pctFormat(rank); line.appendChild(percentile);
    }} else if (q) {{
      const track = document.createElement('div'); track.className='track';
      for (const [series, value] of [['a', rank], ['b', rankB]]) {{
        if (!Number.isFinite(value)) continue;
        const dot = document.createElement('span'); dot.className=`dot dot-${{series}}`; dot.style.left=`${{Math.max(0,Math.min(1,value))*100}}%`;
        dot.setAttribute('aria-hidden','true'); track.appendChild(dot);
      }}
      line.appendChild(track);
      const value = document.createElement('div'); value.className='raw comparison'; value.innerHTML=`${{seriesValue(raw,'A',f.unit)}}<br>${{seriesValue(rawB,'B',f.unit)}}`; line.appendChild(value);
      const percentile = document.createElement('div'); percentile.className='pct comparison'; percentile.innerHTML=`${{seriesPercentile(rank,'A')}}<br>${{seriesPercentile(rankB,'B')}}`; line.appendChild(percentile);
    }} else {{ const missing=document.createElement('div'); missing.className='missing'; missing.textContent='No tracked result'; line.appendChild(missing); }}
    rowHost.appendChild(line);
  }}
  const athleteDrills = drillRows.filter(d => String(d.nfl_id)===String(p.nfl_id) && d.drill_type && (String(d.drill_type).startsWith('SKILL_DRILLS_') || String(d.drill_name).includes('SHUTTLE') || String(d.drill_name).includes('CONE') || String(d.drill_name).includes('MIRROR') || String(d.drill_name).includes('HOOP')));
  if (!athleteDrills.length) {{ drillHost.innerHTML='<div class="empty">No tracked position or agility drills for this prospect.</div>'; return; }}
  const sorted = athleteDrills.sort((a,b)=>String(a.drill_name).localeCompare(String(b.drill_name)));
  drillHost.innerHTML = `<table><caption>Available position and skill drill measures for ${{escapeHTML(p.display_name)}}</caption><thead><tr><th scope="col">Drill</th><th scope="col">Drill type</th><th scope="col" class="num">0–1s accel<br>(yd/s²)</th><th scope="col" class="num">Sustained speed<br>(yd/s)</th><th scope="col" class="num">Braking magnitude<br>(yd/s²)</th><th scope="col" class="num">Cross-axis speed gap<br>(yd/s)</th></tr></thead><tbody>${{sorted.map(d=>`<tr><td>${{escapeHTML(d.drill_name)}}</td><td>${{escapeHTML(String(d.drill_type).replaceAll('_',' '))}}</td><td class="num">${{Number.isFinite(d.mean_accel_first_1s_ydps2_mean)?d.mean_accel_first_1s_ydps2_mean.toFixed(1):'—'}}</td><td class="num">${{Number.isFinite(d.peak_sustained_speed_ydps_mean)?d.peak_sustained_speed_ydps_mean.toFixed(1):'—'}}</td><td class="num">${{Number.isFinite(d.peak_braking_ydps2_mean)?(-d.peak_braking_ydps2_mean).toFixed(1):'—'}}</td><td class="num">${{Number.isFinite(d.left_right_lateral_speed_difference_by_drill_ydps)?Math.abs(d.left_right_lateral_speed_difference_by_drill_ydps).toFixed(1):'—'}}</td></tr>`).join('')}}</tbody></table>`;
}}
for (const pos of ['All', ...new Set(players.map(p=>p.combine_position).filter(Boolean).sort())]) {{
  const option=document.createElement('option'); option.value=pos; option.textContent=pos==='All'?'All positions':pos; positionSel.appendChild(option);
}}
positionSel.addEventListener('change', fillPlayerOptions);
playerSel.addEventListener('change', () => {{ fillCompareOptions(players.filter(p => positionSel.value === 'All' || p.combine_position === positionSel.value)); render(); }});
compareSel.addEventListener('change', render);
printButton.addEventListener('click', () => {{
  const a = players.find(x => String(x.nfl_id) === String(playerSel.value));
  const b = players.find(x => String(x.nfl_id) === String(compareSel.value));
  if (a && b && a.combine_position === b.combine_position) window.print();
}});
selectedId = players[0]?.nfl_id ?? null;
fillPlayerOptions();
</script>
</body>
</html>'''
    path = out / "speed_profile_explorer.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
