"""GAT-ETM Demo — Clinical Decision Support
Run:  streamlit run demo/app.py

Layout and flow only; the visual system (tokens, CSS, HTML components) lives in
ui.py and topic clustering / drug attribution in topic_utils.py.
"""

import sys
from pathlib import Path

# Allow imports from demo/ and project root
_DEMO_DIR = Path(__file__).parent
_ROOT     = _DEMO_DIR.parent
sys.path.insert(0, str(_DEMO_DIR))
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data_utils import (
    get_atc_frequency, get_atc_options, get_icd_frequency, get_icd_name, get_icd_options, get_icd_rarity_info,
    get_n_train_visits, parse_atc_option, parse_icd_option,
    query_comorbidities, query_disease_drugs, query_drug_diseases,
)
from inference import load_mean_theta, load_resources, predict
from sample_cases import SAMPLE_CASES
from topic_utils import (
    PLACEHOLDER_ATC, cluster_topics, cluster_weights, describe_clusters, drug_attribution,
    drug_display_name, drug_group_name,
)

st.set_page_config(
    page_title="GAT-ETM · Hỗ trợ kê đơn",
    page_icon=":material/medication:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(ui.CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Resources (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang tải mô hình GAT-ETM …")
def _load():
    encoder, beta_atc, beta_icd, vocab = load_resources()
    clusters = cluster_topics(beta_icd, beta_atc)
    cluster_info = describe_clusters(clusters, beta_icd, beta_atc, vocab["icd_vocab"], vocab["atc_vocab"])
    ref_weights = cluster_weights(load_mean_theta(encoder, vocab), clusters)
    return encoder, beta_atc, beta_icd, vocab, clusters, cluster_info, ref_weights


encoder, beta_atc, beta_icd, vocab, clusters, cluster_info, ref_weights = _load()
icd_vocab = vocab["icd_vocab"]
atc_vocab = vocab["atc_vocab"]
ICD_OPTIONS = get_icd_options()
OPTION_BY_CODE = {parse_icd_option(o): o for o in ICD_OPTIONS}
N_TRAIN = get_n_train_visits()


@st.cache_data(show_spinner=False)
def _predict(codes: tuple) -> dict:
    return predict(encoder, beta_atc, beta_icd, vocab, list(codes))


# ─────────────────────────────────────────────────────────────────────────────
# "Về mô hình" — số liệu cho hội đồng, mở bằng 1 cú bấm
# ─────────────────────────────────────────────────────────────────────────────

@st.dialog("Về mô hình GAT-ETM", width="large")
def about_dialog():
    st.markdown(ui.metric_tiles([
        ("70.264", "lượt khám", "từ 83.761 dòng kê đơn · 1 bệnh viện · TP.HCM · 1 tháng"),
        ("0,338", "F1@5 gợi ý thuốc", "so với đơn thuốc thật trên tập kiểm thử"),
        ("+27%", "Topic Quality nhờ đồ thị tri thức", "0,150 so với 0,118 (ETM thuần)"),
        ("20", "chủ đề bệnh", f"hiển thị thành {len(clusters)} cụm"),
    ]), unsafe_allow_html=True)

    st.markdown("##### Đóng góp của đồ thị tri thức (Topic Quality)")
    st.markdown(ui.compare_bars([
        ("ETM thuần (không đồ thị tri thức)", 0.118, "tone-other"),
        ("GAT-ETM (có đồ thị tri thức)", 0.150, "tone-0"),
    ]), unsafe_allow_html=True)

    st.markdown("##### Số liệu kỹ thuật")
    st.dataframe(pd.DataFrame([
        ("NLL", "↓", "14,08"), ("Topic Coherence", "↑", "0,355"), ("Topic Diversity", "↑", "0,412"),
        ("Topic Quality", "↑", "0,150"), ("F1@5", "↑", "0,338"),
    ], columns=["Chỉ số", "Tốt hơn khi", "GAT-ETM"]), hide_index=True, width="stretch")
    st.caption(
        "Trung bình 5 seed, đồ thị tri thức chỉ rút từ tập huấn luyện (không rò rỉ). "
        "Topic Quality = Topic Coherence × Topic Diversity. Baseline MLP có giám sát đạt F1@5 = 0,5195 — "
        "đánh đổi giữa độ chính xác dự đoán và chất lượng chủ đề."
    )

    st.markdown("##### Cách demo hiển thị chủ đề")
    big = max(clusters, key=len)
    st.markdown(
        f"Nhiều chủ đề có phân bố gần như trùng nhau (cosine ≥ 0,8 trên β): ví dụ "
        f"{len(big)} chủ đề {', '.join(str(t + 1) for t in big)} cùng xoay quanh tim mạch – chuyển hoá. "
        f"Demo gộp chúng thành **{len(clusters)} cụm**, tỉ trọng cụm = tổng θ các chủ đề trong cụm. "
        "Nhãn cụm sinh tự động từ chương ICD-10 của các mã đặc trưng. "
        "θ, β và điểm thuốc của mô hình không thay đổi."
    )
    st.markdown(
        "Lý do chính của một thuốc = cụm đóng góp nhiều nhất vào điểm của thuốc đó, "
        "vì điểm(d) = Σₖ θₖ·βₖ,d. \"×N so với TB\" so tỉ trọng cụm với một lượt khám trung bình "
        "trong tập huấn luyện."
    )
    st.caption("Mô hình dựa trên Zou et al. (2022), Scientific Reports — điều chỉnh cho EHR Việt Nam. Luận văn ThS · 2026.")


HOW_IT_WORKS = """
**GAT-ETM gợi ý thuốc theo 3 bước**

1. Đọc các mã ICD-10 của lượt khám.
2. Ước lượng tỉ trọng các **cụm bệnh** mà lượt khám thuộc về, học từ hàng chục nghìn hồ sơ
   và đồ thị tri thức bệnh – thuốc.
3. Gợi ý thuốc thường đi cùng các cụm đó, kèm **lý do chính**: cụm đóng góp nhiều nhất cho từng thuốc.

Vai trò: *ý kiến thứ hai* giúp rà soát thuốc có thể bị bỏ sót và nhận ra bệnh đồng mắc.
Không thay thế quyết định của bác sĩ.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

h_title, h_actions = st.columns([2.4, 1.6], vertical_alignment="center")
with h_title:
    st.markdown(ui.page_header(
        "Hỗ trợ kê đơn GAT-ETM",
        "Gợi ý thuốc từ chẩn đoán ICD-10, kèm lý giải theo cụm bệnh — học từ dữ liệu EHR Việt Nam",
    ), unsafe_allow_html=True)
with h_actions:
    a1, a2 = st.columns(2)
    with a1:
        with st.popover("Cách hoạt động", icon=":material/help:", width="stretch"):
            st.markdown(HOW_IT_WORKS)
    with a2:
        if st.button("Về mô hình", icon=":material/analytics:", width="stretch"):
            about_dialog()

st.markdown(ui.disclaimer(
    "Công cụ tham khảo, không thay thế chỉ định của bác sĩ. "
    "Mô hình học từ dữ liệu 1 bệnh viện tại TP.HCM trong 1 tháng."
), unsafe_allow_html=True)

tab_rx, tab_kg = st.tabs([":material/medication: Gợi ý thuốc", ":material/hub: Tra cứu bệnh – thuốc"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Gợi ý thuốc
# ═════════════════════════════════════════════════════════════════════════════

def _run(codes: list):
    st.session_state["rx_codes"]  = list(codes)
    st.session_state["rx_result"] = _predict(tuple(codes))
    st.session_state["rx_focus"]  = None


def _on_case():
    name = st.session_state.get("rx_case")
    if not name:
        return
    codes = SAMPLE_CASES[name]["codes"]
    st.session_state["rx_select"] = [OPTION_BY_CODE[c] for c in codes if c in OPTION_BY_CODE]
    st.session_state["rx_manual"] = ""
    st.session_state["rx_case_name"] = name
    _run(codes)
    st.session_state["rx_case"] = None


def _input_codes() -> list:
    picked = [parse_icd_option(o) for o in st.session_state.get("rx_select", [])]
    manual = [c.strip().upper() for c in st.session_state.get("rx_manual", "").replace(",", "\n").splitlines()
              if c.strip()]
    return picked + [c for c in manual if c not in picked]


def render_input():
    with st.container(border=True):
        st.markdown(ui.section("Chẩn đoán của lượt khám (ICD-10)"), unsafe_allow_html=True)
        c_in, c_btn = st.columns([4, 1], vertical_alignment="bottom")
        with c_in:
            st.multiselect(
                "Chẩn đoán", options=ICD_OPTIONS, key="rx_select", label_visibility="collapsed",
                placeholder="Gõ tên bệnh hoặc mã, ví dụ: tăng huyết áp, I10",
            )
        with c_btn:
            if st.button("Gợi ý thuốc", type="primary", icon=":material/arrow_forward:", width="stretch"):
                codes = _input_codes()
                if codes:
                    st.session_state.pop("rx_case_name", None)
                    _run(codes)
                else:
                    st.session_state["rx_need_input"] = True

        c_case, c_paste = st.columns([3, 2], vertical_alignment="center")
        with c_case:
            st.pills("Ca mẫu", options=list(SAMPLE_CASES), key="rx_case", on_change=_on_case,
                     help="Chọn để điền sẵn chẩn đoán và chạy ngay")
        with c_paste:
            with st.expander("Dán danh sách mã"):
                st.text_area("Mỗi dòng 1 mã hoặc cách nhau bởi dấu phẩy", key="rx_manual",
                             placeholder="E11.9, I10, E78.2", height=90)

        if st.session_state.pop("rx_need_input", False):
            st.markdown(ui.banner("Chưa có mã chẩn đoán nào. Chọn ít nhất một mã ICD-10 hoặc một ca mẫu.",
                                  "warn"), unsafe_allow_html=True)


def render_input_notes(result: dict):
    notes = []
    if result["icd_oov"]:
        notes.append(f"Bỏ qua {len(result['icd_oov'])} mã không có trong từ điển mã của mô hình: "
                     f"<b>{', '.join(result['icd_oov'])}</b>.")
    info = {c: get_icd_rarity_info(c) for c in result["icd_found"]}
    unseen = [c for c, r in info.items() if r["count"] == 0]
    rare = [c for c, r in info.items() if r["count"] > 0 and r["is_rare"]]
    if unseen:
        notes.append(f"Mô hình chưa gặp mã này trong dữ liệu huấn luyện, phần đóng góp của nó không đáng tin: "
                     f"<b>{', '.join(unseen)}</b>.")
    if rare:
        notes.append(f"Mã ít gặp trong dữ liệu, gợi ý liên quan kém chắc chắn hơn: <b>{', '.join(rare)}</b>.")
    if notes:
        st.markdown(ui.banner(" ".join(notes), "warn"), unsafe_allow_html=True)


def render_results(result: dict):
    theta = result["theta"]
    weights = cluster_weights(theta, clusters)
    order = [int(i) for i in np.argsort(weights)[::-1]]
    shown = [c for c in order[:5] if weights[c] >= 0.03]
    tone_of = {cid: ui.tone_class(rank if rank < ui.N_TONES else None) for rank, cid in enumerate(order)}
    label_of = {c["id"]: c["label"] for c in cluster_info}

    case_name = st.session_state.get("rx_case_name")
    head = f"Ca mẫu “{case_name}”: {SAMPLE_CASES[case_name]['description']}" if case_name else None
    st.markdown(ui.section("Kết quả cho lượt khám", head), unsafe_allow_html=True)
    st.markdown(ui.code_chips([(c, get_icd_name(c)) for c in result["icd_found"]]), unsafe_allow_html=True)
    st.write("")
    render_input_notes(result)

    focus = st.session_state.get("rx_focus")
    col_drugs, col_why = st.columns([7, 5], gap="large")

    # ── Thuốc gợi ý ───────────────────────────────────────────────────────────
    with col_drugs:
        st.markdown(ui.section("Thuốc gợi ý", "5 hoạt chất có điểm cao nhất · lý do chính là cụm bệnh đóng góp nhiều nhất"),
                    unsafe_allow_html=True)
        ranked = sorted(((a, p) for a, p in result["drug_probs"].items() if a not in PLACEHOLDER_ATC),
                        key=lambda x: -x[1])[:5]
        top_score = ranked[0][1] if ranked else 1.0
        rows, csv_rows = [], []
        for rank, (atc, score) in enumerate(ranked, 1):
            cid, share = drug_attribution(theta, beta_atc, clusters, atc_vocab.index(atc))
            rel = score / top_score if top_score > 0 else 0
            level = 3 if rel >= 0.85 else 2 if rel >= 0.6 else 1
            name, group = drug_display_name(atc), drug_group_name(atc)
            rows.append(dict(rank=rank, name=name, atc=atc, group=group, level=level, score=score,
                             tone=tone_of[cid], reason=label_of[cid], share=share, cid=cid))
            csv_rows.append({"#": rank, "Hoạt chất": name, "ATC": atc, "Nhóm thuốc": group,
                             "Điểm mô hình": round(score, 4), "Lý do chính": label_of[cid],
                             "Tỉ lệ đóng góp": f"{share:.0%}"})
        # Chỉ làm nổi / làm mờ khi cụm đang chọn thật sự là lý do chính của ít nhất 1 thuốc
        if focus is not None and any(r["cid"] == focus for r in rows):
            for r in rows:
                r["ring"], r["dim"] = r["cid"] == focus, r["cid"] != focus
        elif focus is not None:
            st.markdown(ui.banner(
                f"Cụm <b>{label_of[focus]}</b> không phải lý do chính của thuốc nào trong 5 gợi ý. "
                "Thuốc thường dùng của cụm này nằm ở thẻ bên phải.", "info"), unsafe_allow_html=True)
        st.markdown(ui.drug_table(rows), unsafe_allow_html=True)
        st.markdown(
            ui.footnote("Mức gợi ý so sánh 5 thuốc trong cùng ca (di chuột để xem điểm). "
            "Điểm phản ánh mức thuốc thường được kê ở hồ sơ tương tự, không phải xác suất điều trị đúng — "
            "luôn kiểm tra chống chỉ định và tương tác thuốc."),
            unsafe_allow_html=True,
        )
        st.download_button("Tải bảng gợi ý (CSV)", data=pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name="goi_y_thuoc.csv", mime="text/csv", icon=":material/download:",
                           type="tertiary", on_click="ignore")

    # ── Vì sao: hồ sơ cụm bệnh ────────────────────────────────────────────────
    with col_why:
        st.markdown(ui.section("Vì sao — hồ sơ cụm bệnh", "Tỉ trọng các cụm bệnh mô hình nhận ra ở lượt khám này"),
                    unsafe_allow_html=True)
        bar_rows = []
        for cid in shown:
            lift = weights[cid] / ref_weights[cid] if ref_weights[cid] > 0 else None
            bar_rows.append(dict(label=label_of[cid], pct=weights[cid], tone=tone_of[cid],
                                 lift=lift if lift and lift >= 1.5 else None))
        rest = 1.0 - sum(weights[c] for c in shown)
        if rest > 0.005:
            bar_rows.append(dict(label="Các cụm khác", pct=rest, tone="tone-other", lift=None))
        st.markdown(ui.cluster_bars(bar_rows), unsafe_allow_html=True)
        st.write("")

        picked = st.pills(
            "Làm nổi thuốc theo cụm", options=shown, format_func=lambda c: label_of[c],
            key="rx_focus", help="Chọn một cụm để làm nổi các thuốc mà cụm đó là lý do chính",
        )
        detail = picked if picked is not None else shown[0]
        info = cluster_info[detail]
        st.markdown(ui.cluster_card(info["label"], tone_of[detail], info["topics"],
                                    info["top_icds"], info["top_atcs"]), unsafe_allow_html=True)


def render_tab_rx():
    render_input()
    result = st.session_state.get("rx_result")
    if result is None:
        st.markdown(ui.empty_state(
            "Chưa có lượt khám nào được phân tích",
            "Nhập chẩn đoán ICD-10 hoặc chọn một ca mẫu ở trên để xem thuốc gợi ý và lý giải.",
        ), unsafe_allow_html=True)
        return
    if _input_codes() and _input_codes() != st.session_state.get("rx_codes"):
        st.markdown(ui.banner("Chẩn đoán đã thay đổi. Bấm <b>Gợi ý thuốc</b> để cập nhật kết quả bên dưới.", "info"),
                    unsafe_allow_html=True)
    render_results(result)


with tab_rx:
    render_tab_rx()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tra cứu bệnh – thuốc (thống kê mô tả, không chạy mô hình)
# ═════════════════════════════════════════════════════════════════════════════

def _seen_icd_options() -> tuple:
    freq = get_icd_frequency()
    return tuple(o for o in get_icd_options() if freq.get(parse_icd_option(o), 0) > 0)


def _seen_atc_options() -> tuple:
    freq = get_atc_frequency()
    return tuple(o for o in get_atc_options() if freq.get(parse_atc_option(o), 0) > 0)


KG_MODES = {
    "disease_drug": dict(
        label="Bệnh → thuốc", options=_seen_icd_options, parse=parse_icd_option, query=query_disease_drugs,
        placeholder="Gõ tên bệnh hoặc mã, ví dụ: đái tháo đường, E11", code_col="atc_code",
        col_name="Thuốc", title="Thuốc thường kê cho", tone="tone-0",
    ),
    "drug_disease": dict(
        label="Thuốc → bệnh", options=_seen_atc_options, parse=parse_atc_option, query=query_drug_diseases,
        placeholder="Gõ tên thuốc hoặc mã, ví dụ: metformin, A10BA02", code_col="icd_code",
        col_name="Bệnh", title="Chẩn đoán thường gặp khi kê", tone="tone-1",
    ),
    "comorbidity": dict(
        label="Bệnh → bệnh đi kèm", options=_seen_icd_options, parse=parse_icd_option, query=query_comorbidities,
        placeholder="Gõ tên bệnh hoặc mã, ví dụ: tăng huyết áp, I10", code_col="icd_code",
        col_name="Bệnh đi kèm", title="Bệnh thường đi kèm", tone="tone-3",
    ),
}
_TONE_HEX = {"tone-0": "#0e7490", "tone-1": "#4f46e5", "tone-3": "#047857"}  # plotly cần mã màu


def _code_name(mode: str, code: str) -> str:
    return get_icd_name(code) if mode != "drug_disease" else drug_display_name(code)


def _kg_network(center: str, center_name: str, df: pd.DataFrame, mode: str, max_nodes: int = 12):
    cfg = KG_MODES[mode]
    df = df.head(max_nodes).reset_index(drop=True)
    if df.empty:
        return None
    angles = np.linspace(0, 2 * np.pi, len(df), endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    node_name = drug_display_name if cfg["code_col"] == "atc_code" else get_icd_name
    names = [node_name(c) for c in df[cfg["code_col"]]]
    short = [n if len(n) <= 18 else n[:17].rsplit(" ", 1)[0] + "…" for n in names]
    color = _TONE_HEX[cfg["tone"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sum(([0, x, None] for x in xs), []), y=sum(([0, y, None] for y in ys), []), mode="lines",
        line=dict(color="rgba(100,116,139,.35)", width=1.2), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text", marker=dict(size=46, color="#0f172a"),
        text=[center], textfont=dict(color="white", size=11), hovertext=[center_name], hoverinfo="text",
        showlegend=False,
    ))
    sizes = 14 + 26 * (df["count"] / max(int(df["count"].max()), 1))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", marker=dict(size=sizes, color=color, line=dict(width=1, color="white")),
        text=short, textposition="bottom center", textfont=dict(size=11, color="#334155"),
        hovertext=[f"{n} ({c})<br>{cnt} lượt khám" for n, c, cnt in zip(names, df[cfg["code_col"]], df["count"])],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(visible=False, range=[-1.55, 1.55]), yaxis=dict(visible=False, range=[-1.45, 1.35]),
                      font=dict(family="Be Vietnam Pro, sans-serif"))
    return fig


def render_tab_kg():
    st.markdown(ui.section(
        "Tra cứu thống kê bệnh – thuốc",
        f"Đếm trực tiếp trên {ui.fmt_num(N_TRAIN)} lượt khám của tập huấn luyện — không chạy mô hình AI. "
        "Dùng để tra nhanh và đối chiếu với gợi ý ở tab Gợi ý thuốc.",
    ), unsafe_allow_html=True)

    with st.container(border=True):
        mode = st.segmented_control("Loại tra cứu", options=list(KG_MODES), format_func=lambda m: KG_MODES[m]["label"],
                                    default="disease_drug", key="kg_mode")
        mode = mode or "disease_drug"
        cfg = KG_MODES[mode]
        c_q, c_n, c_btn = st.columns([4, 1.1, 1.1], vertical_alignment="bottom")
        with c_q:
            sel = st.selectbox("Mã tra cứu", options=cfg["options"](), index=None, placeholder=cfg["placeholder"],
                               key=f"kg_sel_{mode}")
        with c_n:
            top_k = st.number_input("Số dòng", min_value=5, max_value=50, value=15, step=5, key="kg_top_n")
        with c_btn:
            if st.button("Tra cứu", type="primary", icon=":material/search:", width="stretch", disabled=sel is None):
                st.session_state["kg_query"] = (mode, cfg["parse"](sel))

    query = st.session_state.get("kg_query")
    if not query or query[0] != mode:
        st.markdown(ui.empty_state("Chọn một mã để tra cứu",
                                   "Kết quả gồm bảng tỉ lệ cùng xuất hiện và sơ đồ quan hệ.", icon="info"),
                    unsafe_allow_html=True)
        return

    _, code = query
    name = _code_name(mode, code)
    df = cfg["query"](code, top_k=int(top_k))
    n_visits = df.attrs.get("n_visits", 0)
    st.markdown(ui.section(f"{cfg['title']} {code} — {name}",
                           f"n = {ui.fmt_num(n_visits)} lượt khám có mã này trong tập huấn luyện"),
                unsafe_allow_html=True)
    if df.empty:
        st.markdown(ui.banner(f"Không có dữ liệu cho mã <b>{code}</b> trong tập huấn luyện.", "warn"),
                    unsafe_allow_html=True)
        return

    col_tbl, col_net = st.columns([7, 5], gap="large")
    with col_tbl:
        rows = [dict(code=r[cfg["code_col"]],
                     name=drug_display_name(r[cfg["code_col"]]) if cfg["code_col"] == "atc_code" else r["name"],
                     freq=float(r["freq"]), count=int(r["count"])) for _, r in df.iterrows()]
        st.markdown(ui.kg_table(rows, cfg["col_name"]), unsafe_allow_html=True)
        st.markdown(
            ui.footnote("Tỉ lệ = số lượt khám có cả hai mã ÷ số lượt khám có mã tra cứu. "
            "Dòng “ít dữ liệu” dựa trên dưới 10 lượt khám. Phản ánh thói quen kê đơn và đồng mắc trong dữ liệu, "
            "không phải khuyến cáo điều trị."),
            unsafe_allow_html=True,
        )
        st.download_button("Tải kết quả (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"tra_cuu_{code}.csv", mime="text/csv", icon=":material/download:",
                           type="tertiary", on_click="ignore")
    with col_net:
        st.markdown(ui.section("Sơ đồ quan hệ", "Kích thước nút theo số lượt khám cùng xuất hiện"),
                    unsafe_allow_html=True)
        fig = _kg_network(code, name, df, mode)
        if fig is not None:
            st.plotly_chart(fig, config={"displayModeBar": False})


with tab_kg:
    render_tab_kg()
