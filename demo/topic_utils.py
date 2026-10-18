"""Topic interpretation for the demo: topic clusters, cluster labels, drug attribution.

Many of the K=20 topics learned by GAT-ETM have near-identical distributions
(e.g. 10 cardio-metabolic topics with cosine ≥ 0.99). Shown one by one they
split a patient's mass into ~5% slivers that look like a flat profile, so the
demo groups near-duplicate topics into clusters and sums theta within each
cluster. This is presentation only — theta, beta and drug scores are unchanged.
"""

from __future__ import annotations

import numpy as np

from data_utils import get_atc_info, get_icd_name

NUM_TOPICS = 20

# ATC codes that are catch-all groups rather than a drug (e.g. "V" = VARIOUS)
PLACEHOLDER_ATC = {"V"}

# ICD-10 chapters → short clinical area names used in cluster labels
_ICD_CHAPTERS = [
    ("A00", "B99", "Nhiễm trùng"),
    ("C00", "D48", "Khối u"),
    ("D50", "D89", "Huyết học"),
    ("E00", "E90", "Nội tiết – chuyển hoá"),
    ("F00", "F99", "Tâm thần"),
    ("G00", "G99", "Thần kinh"),
    ("H00", "H59", "Mắt"),
    ("H60", "H95", "Tai – tiền đình"),
    ("I00", "I99", "Tim mạch"),
    ("J00", "J99", "Hô hấp"),
    ("K00", "K93", "Tiêu hoá"),
    ("L00", "L99", "Da liễu"),
    ("M00", "M99", "Cơ xương khớp"),
    ("N00", "N99", "Thận – tiết niệu – sinh dục"),
    ("O00", "O99", "Thai sản"),
    ("R00", "R99", "Triệu chứng chung"),
    ("S00", "T98", "Chấn thương"),
    ("Z00", "Z99", "Yếu tố sức khoẻ"),
]


def icd_chapter(code: str) -> str:
    head = code.strip().upper()[:3]
    for lo, hi, name in _ICD_CHAPTERS:
        if lo <= head <= hi:
            return name
    return "Khác"


def _sentence_case(name: str) -> str:
    """WHO ATC names are lower-case INN or UPPER-CASE group names → sentence case."""
    return name[:1].upper() + (name[1:].lower() if name.isupper() else name[1:])


def drug_display_name(atc: str) -> str:
    return _sentence_case(get_atc_info(atc)["name"])


def drug_group_name(atc: str) -> str:
    """Pharmacological subgroup (ATC level 4), sentence case."""
    return _sentence_case(get_atc_info(atc)["group"])


# ─────────────────────────────────────────────────────────────────────────────
# Clusters
# ─────────────────────────────────────────────────────────────────────────────

def cluster_topics(beta_icd: np.ndarray, beta_atc: np.ndarray, threshold: float = 0.8) -> list[list[int]]:
    """Greedy grouping of topics whose joint (ICD+ATC) distributions have cosine ≥ threshold."""
    b = np.hstack([beta_icd, beta_atc])
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = b @ b.T
    assigned = np.full(len(b), -1)
    clusters = []
    for i in range(len(b)):
        if assigned[i] >= 0:
            continue
        members = [j for j in range(i, len(b)) if assigned[j] < 0 and sim[i, j] >= threshold]
        assigned[members] = len(clusters)
        clusters.append(members)
    return clusters


def describe_clusters(clusters: list[list[int]], beta_icd: np.ndarray, beta_atc: np.ndarray,
                      icd_vocab: list[str], atc_vocab: list[str], n_codes: int = 4) -> list[dict]:
    """Label each cluster by the ICD chapters of its top codes, plus its characteristic codes.

    Returns [{id, topics, label, top_icds: [{code, name}], top_atcs: [{code, name}]}].
    """
    out = []
    for cid, members in enumerate(clusters):
        b_icd = beta_icd[members].mean(axis=0)
        b_atc = beta_atc[members].mean(axis=0)

        top10 = np.argsort(b_icd)[::-1][:10]
        mass: dict[str, float] = {}
        for i in top10:
            ch = icd_chapter(icd_vocab[i])
            mass[ch] = mass.get(ch, 0.0) + float(b_icd[i])
        chapters = sorted(mass.items(), key=lambda kv: -kv[1])
        label = chapters[0][0]
        if len(chapters) > 1 and chapters[1][1] >= 0.6 * chapters[0][1]:
            label = f"{label} · {chapters[1][0]}"

        top_icds = [{"code": icd_vocab[i], "name": get_icd_name(icd_vocab[i])}
                    for i in np.argsort(b_icd)[::-1][:n_codes]]
        top_atcs = [{"code": atc_vocab[i], "name": drug_display_name(atc_vocab[i])}
                    for i in np.argsort(b_atc)[::-1]
                    if atc_vocab[i] not in PLACEHOLDER_ATC][:n_codes]
        out.append({"id": cid, "topics": members, "label": label,
                    "top_icds": top_icds, "top_atcs": top_atcs})

    # Same chapter label for two clusters → disambiguate with the leading disease
    counts: dict[str, int] = {}
    for c in out:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    for c in out:
        if counts[c["label"]] > 1:
            c["label"] = f"{c['label']} ({c['top_icds'][0]['name'].split(',')[0].strip().lower()})"
    return out


def cluster_weights(theta: np.ndarray, clusters: list[list[int]]) -> np.ndarray:
    return np.array([theta[m].sum() for m in clusters])


def drug_attribution(theta: np.ndarray, beta_atc: np.ndarray, clusters: list[list[int]],
                     atc_idx: int) -> tuple[int, float]:
    """The drug score is Σ_k θ_k·β_k,d. Return (cluster id contributing most, its share of the score)."""
    contrib = theta * beta_atc[:, atc_idx]
    per_cluster = np.array([contrib[m].sum() for m in clusters])
    best = int(np.argmax(per_cluster))
    total = per_cluster.sum()
    return best, float(per_cluster[best] / total) if total > 0 else 0.0
