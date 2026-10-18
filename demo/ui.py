"""Visual system for the GAT-ETM demo: design tokens, global CSS, HTML components.

All colours, sizes and radii live in the :root tokens below; components use
classes only, so app.py never writes hex values or inline styles.
"""

from __future__ import annotations

from html import escape

# Categorical tones for disease clusters, ranked per case (0 = most prominent).
# Red is reserved for danger states, so it is not a cluster tone.
N_TONES = 4
OTHER_TONE = "other"

_FONTS = (
    "https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700"
    "&family=IBM+Plex+Mono:wght@500&display=swap"
)
# Icons in HTML reuse the "Material Symbols Rounded" face Streamlit already loads.
# Do not @import a Google subset of it: same family name → it replaces Streamlit's full set.

CSS = f"""
<style>
@import url('{_FONTS}');

:root {{
  --gt-ink: #0f172a;
  --gt-ink-2: #334155;
  --gt-muted: #5b6b80;          /* ≥ 4.9:1 on every surface below */
  --gt-bg: #f5f7f9;
  --gt-surface: #ffffff;
  --gt-surface-2: #f1f5f9;
  --gt-border: #e2e8f0;
  --gt-border-strong: #cbd5e1;
  --gt-primary: #0e7490;
  --gt-primary-ink: #155e75;
  --gt-primary-soft: #ecfeff;
  --gt-warn-ink: #92400e;
  --gt-warn-bg: #fffbeb;
  --gt-warn-bd: #fde68a;
  --gt-danger-ink: #991b1b;
  --gt-danger-bg: #fef2f2;
  --gt-danger-bd: #fecaca;

  --gt-fs-xs: 12px;
  --gt-fs-sm: 13px;
  --gt-fs-md: 14px;
  --gt-fs-lg: 16px;
  --gt-fs-xl: 22px;

  --gt-radius-sm: 8px;
  --gt-radius: 12px;
  --gt-font: 'Be Vietnam Pro', system-ui, sans-serif;
  --gt-mono: 'IBM Plex Mono', ui-monospace, monospace;
}}

/* Cluster tones: --c main, --c-soft background, --c-ink text on soft */
.tone-0 {{ --c: #0e7490; --c-soft: #ecfeff; --c-ink: #155e75; }}
.tone-1 {{ --c: #4f46e5; --c-soft: #eef2ff; --c-ink: #3730a3; }}
.tone-2 {{ --c: #b45309; --c-soft: #fffbeb; --c-ink: #92400e; }}
.tone-3 {{ --c: #047857; --c-soft: #ecfdf5; --c-ink: #065f46; }}
.tone-other {{ --c: #94a3b8; --c-soft: #f1f5f9; --c-ink: #334155; }}

/* ---------- Streamlit shell ---------- */
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[role="dialog"], [role="dialog"] p, [role="dialog"] h2, [role="dialog"] h5, [role="dialog"] div {{ font-family: var(--gt-font); }}
/* Icon của Streamlit dùng font ligature riêng — không được ghi đè */
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: var(--gt-bg); }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"] {{ display: none; }}
h1, h2, h3 {{ font-family: var(--gt-font); letter-spacing: -0.01em; }}
code {{ font-family: var(--gt-mono); }}

.stButton > button, .stDownloadButton > button {{
  border-radius: var(--gt-radius-sm); font-weight: 600; min-height: 42px;
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
  outline: 2px solid var(--gt-primary); outline-offset: 2px;
}}

[data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--gt-border); }}
[data-baseweb="tab"] {{ padding: 12px 18px; }}
[data-baseweb="tab"] p {{ font-size: var(--gt-fs-lg); font-weight: 600; color: var(--gt-muted); }}
[data-baseweb="tab"][aria-selected="true"] p {{ color: var(--gt-primary-ink); }}

div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="textarea"] {{
  background: var(--gt-surface); border: 1px solid var(--gt-border-strong); border-radius: var(--gt-radius-sm);
}}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"]:focus-within {{
  border-color: var(--gt-primary); box-shadow: 0 0 0 3px rgba(14, 116, 144, .15);
}}
span[data-baseweb="tag"] {{
  background: var(--gt-primary-soft) !important; color: var(--gt-primary-ink) !important;
  border-radius: 6px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
  background: var(--gt-surface);
}}
[data-testid="stExpander"] details {{ border-radius: var(--gt-radius-sm); border-color: var(--gt-border); }}
[data-testid="stCaptionContainer"] {{ color: var(--gt-muted); }}

/* ---------- Icons ---------- */
.msr {{
  font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
  font-size: 20px; line-height: 1; display: inline-block; flex: none;
  font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
}}

/* ---------- Page header ---------- */
.gt-header h1 {{ margin: 0; padding: 0; font-size: var(--gt-fs-xl); font-weight: 700; color: var(--gt-ink); }}
.gt-header p {{ margin: 4px 0 0; font-size: var(--gt-fs-md); color: var(--gt-muted); }}
.gt-disclaimer {{
  display: flex; gap: 8px; align-items: center; margin: 4px 0 6px;
  font-size: var(--gt-fs-sm); color: var(--gt-warn-ink);
}}

/* ---------- Section titles ---------- */
.gt-section {{ margin: 0 0 10px; }}
.gt-section h3 {{ margin: 0; padding: 0; font-size: var(--gt-fs-lg); font-weight: 700; color: var(--gt-ink); }}
.gt-section p {{ margin: 3px 0 0; font-size: var(--gt-fs-sm); color: var(--gt-muted); }}

/* ---------- Banner (one system, colour = severity) ---------- */
.gt-banner {{
  display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; margin: 0 0 12px;
  border-radius: var(--gt-radius-sm); border: 1px solid; font-size: var(--gt-fs-sm); line-height: 1.5;
}}
.gt-banner--info {{ background: var(--gt-primary-soft); border-color: #a5f3fc; color: var(--gt-primary-ink); }}
.gt-banner--warn {{ background: var(--gt-warn-bg); border-color: var(--gt-warn-bd); color: var(--gt-warn-ink); }}
.gt-banner--danger {{ background: var(--gt-danger-bg); border-color: var(--gt-danger-bd); color: var(--gt-danger-ink); }}
.gt-banner p {{ margin: 0; }}

/* ---------- Code chips ---------- */
.gt-chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.gt-chip {{
  display: inline-flex; gap: 6px; align-items: baseline; max-width: 100%;
  padding: 3px 9px; border-radius: 6px; background: var(--gt-surface-2);
  font-size: var(--gt-fs-sm); color: var(--gt-ink-2);
}}
.gt-chip code {{ background: none; padding: 0; font-size: var(--gt-fs-xs); color: var(--gt-ink); }}

/* ---------- Drug table ---------- */
.gt-drugs {{ border: 1px solid var(--gt-border); border-radius: var(--gt-radius); background: var(--gt-surface); overflow: hidden; container-type: inline-size; }}
.gt-drug {{
  display: grid; grid-template-columns: 28px minmax(0, 1fr) 112px minmax(0, 250px);
  gap: 12px; align-items: center; padding: 12px 14px; border-top: 1px solid var(--gt-border);
  transition: opacity .15s ease, background .15s ease;
}}
.gt-drug--head {{
  border-top: 0; padding-top: 10px; padding-bottom: 8px; background: var(--gt-surface-2);
  font-size: var(--gt-fs-xs); font-weight: 600; color: var(--gt-muted);
}}
.gt-drug--dim {{ opacity: .4; }}
.gt-drug--ring {{ background: var(--c-soft); box-shadow: inset 3px 0 0 var(--c); }}
.gt-rank {{ font-size: var(--gt-fs-lg); font-weight: 700; color: var(--gt-muted); text-align: center; }}
.gt-drug-name {{ font-size: var(--gt-fs-md); font-weight: 600; color: var(--gt-ink); overflow-wrap: anywhere; }}
.gt-drug-meta {{ margin-top: 2px; font-size: var(--gt-fs-xs); color: var(--gt-muted); overflow-wrap: anywhere; }}
.gt-drug-meta code {{ font-size: var(--gt-fs-xs); color: var(--gt-ink-2); background: none; padding: 0; margin-right: 6px; }}

.gt-level {{ display: inline-flex; align-items: center; gap: 6px; font-size: var(--gt-fs-sm); font-weight: 600; }}
.gt-dots {{ display: inline-flex; gap: 3px; }}
.gt-dots i {{ width: 7px; height: 7px; border-radius: 50%; background: var(--gt-border-strong); }}
.gt-level--3 {{ color: #065f46; }} .gt-level--3 .gt-dots i {{ background: #047857; }}
.gt-level--2 {{ color: var(--gt-ink-2); }} .gt-level--2 .gt-dots i:nth-child(-n+2) {{ background: #475569; }}
.gt-level--1 {{ color: var(--gt-muted); }} .gt-level--1 .gt-dots i:first-child {{ background: #94a3b8; }}

.gt-reason {{
  display: inline-flex; align-items: center; gap: 7px; max-width: 100%;
  padding: 4px 9px; border-radius: 6px; background: var(--c-soft); color: var(--c-ink);
  font-size: var(--gt-fs-xs); font-weight: 600;
}}
.gt-reason span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.gt-reason b {{ font-weight: 700; flex: none; }}
.gt-swatch {{ width: 8px; height: 8px; border-radius: 2px; background: var(--c); flex: none; }}

.gt-footnote {{ margin: 10px 2px 4px; font-size: var(--gt-fs-xs); line-height: 1.55; color: var(--gt-muted); }}

/* ---------- Cluster profile ---------- */
.gt-bars {{ display: flex; flex-direction: column; gap: 12px; }}
.gt-bar-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: var(--gt-fs-sm); }}
.gt-bar-label {{ font-weight: 600; color: var(--gt-ink-2); min-width: 0; }}
.gt-bar-pct {{ margin-left: auto; font-weight: 700; color: var(--gt-ink); font-variant-numeric: tabular-nums; }}
.gt-lift {{
  flex: none; padding: 1px 7px; border-radius: 999px; background: var(--c-soft); color: var(--c-ink);
  font-size: var(--gt-fs-xs); font-weight: 600;
}}
.gt-track {{ height: 10px; border-radius: 6px; background: var(--gt-surface-2); overflow: hidden; }}
.gt-fill {{ height: 100%; border-radius: 6px; background: var(--c); }}

.gt-card {{
  border: 1px solid var(--gt-border); border-radius: var(--gt-radius); background: var(--gt-surface);
  overflow: hidden; container-type: inline-size;
}}
.gt-card-head {{ padding: 12px 14px; background: var(--c-soft); border-bottom: 1px solid var(--gt-border); }}
.gt-card-head h4 {{ margin: 0; padding: 0; font-size: var(--gt-fs-md); font-weight: 700; color: var(--c-ink); }}
.gt-card-head p {{ margin: 2px 0 0; font-size: var(--gt-fs-xs); color: var(--c-ink); }}
.gt-card-body {{ padding: 12px 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.gt-card-body h5 {{ margin: 0 0 6px; padding: 0; font-size: var(--gt-fs-xs); font-weight: 600; color: var(--gt-muted); }}
.gt-card-body ul {{ margin: 0; padding-left: 16px; font-size: var(--gt-fs-sm); color: var(--gt-ink-2); line-height: 1.6; }}
.gt-card-body code {{ font-size: var(--gt-fs-xs); background: none; padding: 0; color: var(--gt-muted); }}

/* ---------- Empty state ---------- */
.gt-empty {{
  display: flex; gap: 12px; align-items: flex-start; padding: 22px; border-radius: var(--gt-radius);
  border: 1px dashed var(--gt-border-strong); background: var(--gt-surface); color: var(--gt-ink-2);
}}
.gt-empty .msr {{ color: var(--gt-primary); font-size: 24px; }}
.gt-empty h4 {{ margin: 0; padding: 0; font-size: var(--gt-fs-lg); font-weight: 700; color: var(--gt-ink); }}
.gt-empty p {{ margin: 4px 0 0; font-size: var(--gt-fs-md); color: var(--gt-muted); }}

/* ---------- KG table (Tab 2) ---------- */
.gt-kg {{ border: 1px solid var(--gt-border); border-radius: var(--gt-radius); background: var(--gt-surface); overflow: hidden; container-type: inline-size; }}
.gt-kg-row {{
  display: grid; grid-template-columns: 76px minmax(0, 1fr) 150px 64px; gap: 12px; align-items: center;
  padding: 9px 14px; border-top: 1px solid var(--gt-border); font-size: var(--gt-fs-sm); color: var(--gt-ink);
}}
.gt-kg-row--head {{ border-top: 0; background: var(--gt-surface-2); font-size: var(--gt-fs-xs); font-weight: 600; color: var(--gt-muted); }}
.gt-kg-row--thin {{ color: var(--gt-muted); }}
.gt-kg-row code {{ font-size: var(--gt-fs-xs); background: none; padding: 0; color: var(--gt-ink-2); }}
.gt-kg-share {{ display: flex; align-items: center; gap: 8px; font-variant-numeric: tabular-nums; }}
.gt-kg-share .gt-track {{ flex: 1; height: 6px; }}
.gt-kg-share .gt-fill {{ background: var(--gt-primary); }}
.gt-kg-row--thin .gt-fill {{ background: var(--gt-border-strong); }}
.gt-kg-n {{ text-align: right; font-variant-numeric: tabular-nums; }}
.gt-tag {{
  margin-left: 6px; padding: 0 6px; border-radius: 4px; background: var(--gt-surface-2);
  font-size: 11px; font-weight: 600; color: var(--gt-muted); white-space: nowrap;
}}

/* ---------- About dialog ---------- */
.gt-metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }}
.gt-metric {{ padding: 12px 14px; border: 1px solid var(--gt-border); border-radius: var(--gt-radius-sm); background: var(--gt-surface); }}
.gt-metric b {{ display: block; font-size: 20px; font-weight: 700; color: var(--gt-ink); }}
.gt-metric span {{ display: block; margin-top: 2px; font-size: var(--gt-fs-sm); font-weight: 600; color: var(--gt-ink-2); }}
.gt-metric small {{ display: block; margin-top: 2px; font-size: var(--gt-fs-xs); color: var(--gt-muted); }}
.gt-compare {{ display: grid; grid-template-columns: 240px 1fr 52px; gap: 10px; align-items: center; margin: 6px 0; font-size: var(--gt-fs-sm); color: var(--gt-ink-2); }}
.gt-compare .gt-track {{ height: 14px; }}
.gt-compare b {{ text-align: right; font-variant-numeric: tabular-nums; }}

/* ---------- Narrow containers (column width, not viewport) ---------- */
@container (max-width: 600px) {{
  .gt-drug {{ grid-template-columns: 24px minmax(0, 1fr); row-gap: 6px; }}
  .gt-drug > :nth-child(3), .gt-drug > :nth-child(4) {{ grid-column: 2; }}
  .gt-drug--head {{ display: none; }}
}}
@container (max-width: 520px) {{
  .gt-kg-row {{ grid-template-columns: 64px minmax(0, 1fr) 52px; }}
  .gt-kg-row > .gt-kg-share {{ display: none; }}
}}
@container (max-width: 420px) {{
  .gt-card-body {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 640px) {{
  .gt-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .gt-compare {{ grid-template-columns: 1fr 44px; }}
  .gt-compare .gt-track {{ display: none; }}
}}
</style>
"""


def _e(text) -> str:
    return escape(str(text))


def tone_class(rank: int | None) -> str:
    return f"tone-{rank}" if rank is not None and rank < N_TONES else f"tone-{OTHER_TONE}"


def fmt_pct(x: float) -> str:
    return f"{x:.0%}"


def fmt_num(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_ratio(x: float) -> str:
    return f"×{x:.1f}".replace(".", ",")


# ─────────────────────────────────────────────────────────────────────────────
# Page-level pieces
# ─────────────────────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str) -> str:
    return f'<div class="gt-header"><h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>'


def disclaimer(text: str) -> str:
    return f'<div class="gt-disclaimer"><span class="msr">info</span><span>{_e(text)}</span></div>'


def section(title: str, subtitle: str | None = None) -> str:
    sub = f"<p>{_e(subtitle)}</p>" if subtitle else ""
    return f'<div class="gt-section"><h3>{_e(title)}</h3>{sub}</div>'


def banner(body_html: str, level: str = "info") -> str:
    """level: info | warn | danger. body_html is trusted markup built by the caller."""
    icon = {"info": "info", "warn": "warning", "danger": "error"}[level]
    return f'<div class="gt-banner gt-banner--{level}"><span class="msr">{icon}</span><p>{body_html}</p></div>'


def empty_state(title: str, body: str, icon: str = "lightbulb") -> str:
    return (f'<div class="gt-empty"><span class="msr">{icon}</span>'
            f'<div><h4>{_e(title)}</h4><p>{_e(body)}</p></div></div>')


def code_chips(items: list[tuple[str, str]]) -> str:
    """items: [(code, name)]"""
    chips = "".join(f'<span class="gt-chip"><code>{_e(c)}</code>{_e(n)}</span>' for c, n in items)
    return f'<div class="gt-chips">{chips}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — drug suggestions and cluster profile
# ─────────────────────────────────────────────────────────────────────────────

_LEVELS = {3: "Cao", 2: "Trung bình", 1: "Thấp"}


def drug_table(rows: list[dict]) -> str:
    """rows: [{rank, name, atc, group, level (1-3), tone, reason, share, score, dim, ring}]"""
    head = ('<div class="gt-drug gt-drug--head"><div>#</div><div>Hoạt chất</div>'
            '<div>Mức gợi ý</div><div>Lý do chính</div></div>')
    body = []
    for r in rows:
        cls = "gt-drug " + r["tone"]
        if r.get("dim"):
            cls += " gt-drug--dim"
        if r.get("ring"):
            cls += " gt-drug--ring"
        dots = "<i></i>" * 3
        tip = f"{r['reason']} — đóng góp {fmt_pct(r['share'])} điểm gợi ý của thuốc"
        body.append(
            f'<div class="{cls}">'
            f'<div class="gt-rank">{r["rank"]}</div>'
            f'<div><div class="gt-drug-name">{_e(r["name"])}</div>'
            f'<div class="gt-drug-meta"><code>{_e(r["atc"])}</code>{_e(r["group"])}</div></div>'
            f'<div class="gt-level gt-level--{r["level"]}" title="Điểm mô hình: {r["score"]:.3f}">'
            f'<span class="gt-dots">{dots}</span>{_LEVELS[r["level"]]}</div>'
            f'<div><span class="gt-reason" title="{_e(tip)}"><i class="gt-swatch"></i>'
            f'<span>{_e(r["reason"])}</span><b>{fmt_pct(r["share"])}</b></span></div>'
            f'</div>'
        )
    return f'<div class="gt-drugs">{head}{"".join(body)}</div>'


def cluster_bars(rows: list[dict]) -> str:
    """rows: [{label, pct, tone, lift (float | None)}]"""
    out = []
    for r in rows:
        lift = (f'<span class="gt-lift" title="So với một lượt khám trung bình trong dữ liệu">'
                f'{fmt_ratio(r["lift"])} so với TB</span>') if r.get("lift") else ""
        out.append(
            f'<div class="{r["tone"]}"><div class="gt-bar-top"><i class="gt-swatch"></i>'
            f'<span class="gt-bar-label">{_e(r["label"])}</span>{lift}'
            f'<span class="gt-bar-pct">{fmt_pct(r["pct"])}</span></div>'
            f'<div class="gt-track"><div class="gt-fill" style="width:{max(r["pct"] * 100, 0.5):.1f}%"></div></div></div>'
        )
    return f'<div class="gt-bars">{"".join(out)}</div>'


def cluster_card(label: str, tone: str, topics: list[int], icds: list[dict], atcs: list[dict]) -> str:
    topic_txt = ", ".join(str(t + 1) for t in topics)
    note = (f"Gộp {len(topics)} chủ đề gần trùng nhau: {topic_txt}" if len(topics) > 1
            else f"Chủ đề {topic_txt}")
    icd_li = "".join(f"<li>{_e(i['name'])} <code>{_e(i['code'])}</code></li>" for i in icds)
    atc_li = "".join(f"<li>{_e(a['name'])} <code>{_e(a['code'])}</code></li>" for a in atcs)
    return (
        f'<div class="gt-card {tone}"><div class="gt-card-head"><h4>{_e(label)}</h4><p>{_e(note)}</p></div>'
        f'<div class="gt-card-body"><div><h5>Bệnh đặc trưng</h5><ul>{icd_li}</ul></div>'
        f'<div><h5>Thuốc thường dùng</h5><ul>{atc_li}</ul></div></div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — co-occurrence table
# ─────────────────────────────────────────────────────────────────────────────

def kg_table(rows: list[dict], col_name: str, thin_below: int = 10) -> str:
    """rows: [{code, name, freq, count}]; rows with count < thin_below are flagged as thin data."""
    head = (f'<div class="gt-kg-row gt-kg-row--head"><div>Mã</div><div>{_e(col_name)}</div>'
            f'<div>Tỉ lệ cùng xuất hiện</div><div class="gt-kg-n">Số lượt</div></div>')
    body = []
    for r in rows:
        thin = r["count"] < thin_below
        tag = '<span class="gt-tag">ít dữ liệu</span>' if thin else ""
        body.append(
            f'<div class="gt-kg-row{" gt-kg-row--thin" if thin else ""}">'
            f'<code>{_e(r["code"])}</code><div>{_e(r["name"])}{tag}</div>'
            f'<div class="gt-kg-share"><div class="gt-track"><div class="gt-fill" '
            f'style="width:{max(r["freq"] * 100, 0.5):.1f}%"></div></div>{fmt_pct(r["freq"])}</div>'
            f'<div class="gt-kg-n">{fmt_num(r["count"])}</div></div>'
        )
    return f'<div class="gt-kg">{head}{"".join(body)}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# About dialog
# ─────────────────────────────────────────────────────────────────────────────

def metric_tiles(items: list[tuple[str, str, str]]) -> str:
    """items: [(value, label, note)]"""
    tiles = "".join(f'<div class="gt-metric"><b>{_e(v)}</b><span>{_e(l)}</span><small>{_e(n)}</small></div>'
                    for v, l, n in items)
    return f'<div class="gt-metrics">{tiles}</div>'


def compare_bars(rows: list[tuple[str, float, str]]) -> str:
    """rows: [(label, value, tone)] — horizontal bars scaled to the max value."""
    top = max(v for _, v, _ in rows) or 1
    return "".join(
        f'<div class="gt-compare {tone}"><span>{_e(label)}</span>'
        f'<div class="gt-track"><div class="gt-fill" style="width:{v / top * 100:.1f}%"></div></div>'
        f'<b>{f"{v:.3f}".replace(".", ",")}</b></div>'
        for label, v, tone in rows
    )


def footnote(text_html: str) -> str:
    return f'<div class="gt-footnote">{text_html}</div>'
