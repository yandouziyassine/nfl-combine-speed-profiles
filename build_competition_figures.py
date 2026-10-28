from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WIDTH, HEIGHT = 1200, 620

def esc(value: str) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

def svg_start(title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="1200" height="620" fill="#f7f8fa"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17212b}.title{font-size:26px;font-weight:700}.sub{font-size:15px;fill:#52606d}.label{font-size:17px}.small{font-size:13px;fill:#52606d}.axis{stroke:#8b98a5;stroke-width:1}.grid{stroke:#dce2e8;stroke-width:1}.ci{stroke-width:5;stroke-linecap:round}.dot{stroke:#fff;stroke-width:2}</style>',
        f'<text x="64" y="54" class="title">{esc(title)}</text>',
        f'<text x="64" y="82" class="sub">{esc(desc)}</text>',
    ]

def write_transfer():
    j=json.loads((OUT/'dl_game_speed_ablation_results_100plays.json').read_text(encoding='utf-8'))
    defs=[
        ('Early acceleration only','early_acceleration','#087e8b'),
        ('Late acceleration only','late_acceleration','#718096'),
        ('20–40 yd speed retention','speed_retention','#718096'),
        ('All three 40 yd phases','all_three_phases','#718096'),
    ]
    items=[]
    for label,key,color in defs:
        row=next(x for x in j['specifications'][key] if x['alpha']==5.0)
        items.append((label,row['rmse_change'],row['player_bootstrap_95pct_ci'][0],row['player_bootstrap_95pct_ci'][1],color))
    x0,x1=410,1080; lo,hi=-.10,.05
    px=lambda v:x0+(v-lo)/(hi-lo)*(x1-x0)
    ns={key:value['n_players'] for key,value in j['samples'].items()}
    lines=svg_start('Early acceleration is the clearest DL phase signal at 100 plays',
        'Change in leave-one-draft-cohort-out RMSE (candidate − baseline), yd/s · ridge alpha 5 · feature-specific complete cases')
    lines+=['<text x="64" y="130" class="small">Lower error  ←</text>']
    for tick in [-.10,-.075,-.05,-.025,0,.025,.05]:
        x=px(tick); lines.append(f'<line x1="{x:.1f}" y1="166" x2="{x:.1f}" y2="470" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="500" text-anchor="middle" class="small">{tick:+.3f}</text>')
    lines.append(f'<line x1="{px(0):.1f}" y1="154" x2="{px(0):.1f}" y2="472" stroke="#334155" stroke-width="2"/>')
    for i,(label,est,low,high,color) in enumerate(items):
        y=205+i*75
        feature_key={'Early acceleration only':'early_acceleration','Late acceleration only':'late_acceleration','20–40 yd speed retention':'speed_retention','All three 40 yd phases':'all_three_phases'}[label]
        lines.append(f'<text x="64" y="{y+6}" class="label">{esc(label)} · n={ns[feature_key]}</text>')
        lines.append(f'<line x1="{px(low):.1f}" y1="{y}" x2="{px(high):.1f}" y2="{y}" class="ci" stroke="{color}"/>')
        lines.append(f'<line x1="{px(low):.1f}" y1="{y-8}" x2="{px(low):.1f}" y2="{y+8}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{px(high):.1f}" y1="{y-8}" x2="{px(high):.1f}" y2="{y+8}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{px(est):.1f}" cy="{y}" r="9" class="dot" fill="{color}"/>')
        lines.append(f'<text x="1100" y="{y+5}" class="small">{est:+.4f} [{low:+.4f}, {high:+.4f}]</text>')
    lines+=['<text x="64" y="552" class="small">Negative values favor the added feature. Whiskers are player-bootstrap intervals over fixed OOF predictions; they do not refit models.</text>',
            '<text x="64" y="576" class="small">Early acceleration followed a broader exploratory screen; treat this as a discovery result, not external validation.</text>','</svg>']
    (OUT/'fig_dl_early_acceleration_transfer.svg').write_text('\n'.join(lines),encoding='utf-8')

def write_reliability():
    data=json.loads((OUT/'forty_repeatability_by_position_results.json').read_text(encoding='utf-8'))['results']
    order=['DL','OL','DB','TE','WR']
    rows={r['combine_position']:r for r in data if r['measure']=='early_acceleration'}
    x0,x1=370,1000; lo,hi=-.4,1.0; px=lambda v:x0+(v-lo)/(hi-lo)*(x1-x0)
    lines=svg_start('Early 40-yard acceleration repeats best in the DL group',
        'Single-attempt ICC(A,1), paired Combine attempts · same-session reliability, not NFL transfer')
    for tick in [-.4,-.2,0,.2,.4,.6,.8,1.0]:
        x=px(tick); lines.append(f'<line x1="{x:.1f}" y1="150" x2="{x:.1f}" y2="480" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="510" text-anchor="middle" class="small">{tick:.1f}</text>')
    for i,pos in enumerate(order):
        r=rows[pos]; est=r['icc_absolute_agreement_single_attempt']; low,high=r['icc_bootstrap_95pct_ci']; y=190+i*64
        color='#087e8b' if pos=='DL' else '#718096'
        lines.append(f'<text x="100" y="{y+6}" class="label">{pos} · n={r["players_with_both_attempts"]}</text>')
        lines.append(f'<line x1="{px(low):.1f}" y1="{y}" x2="{px(high):.1f}" y2="{y}" class="ci" stroke="{color}"/>')
        lines.append(f'<line x1="{px(low):.1f}" y1="{y-8}" x2="{px(low):.1f}" y2="{y+8}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{px(high):.1f}" y1="{y-8}" x2="{px(high):.1f}" y2="{y+8}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{px(est):.1f}" cy="{y}" r="9" class="dot" fill="{color}"/>')
        lines.append(f'<text x="1015" y="{y+5}" class="small">{est:.3f} [{low:.3f}, {high:.3f}]</text>')
    lines.append(f'<line x1="{px(0):.1f}" y1="142" x2="{px(0):.1f}" y2="480" stroke="#334155" stroke-width="2"/>')
    lines+=['<text x="100" y="560" class="small">ICC depends on the spread of measured athletes; intervals are player-bootstrap and may be unstable at small n.</text>',
            '<text x="100" y="584" class="small">DL repeatability supports measurement use for follow-up; it does not by itself establish predictive or coaching value.</text>','</svg>']
    (OUT/'fig_dl_early_acceleration_repeatability.svg').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    write_transfer()
    write_reliability()

