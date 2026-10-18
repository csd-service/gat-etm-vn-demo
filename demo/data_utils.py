"""Reference data loaders: ICD-10 names, ATC names, BoW co-occurrence stats."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).parent.parent
_ATC_CSV    = PROJECT_ROOT / "data_vn/atc_reference/who_atc_ddd.csv"
_VOCAB      = PROJECT_ROOT / "data_vn/vocab_info.pkl"
_BOW_TRAIN  = PROJECT_ROOT / "data_vn/bow_train.npy"
_COOC       = PROJECT_ROOT / "data_vn/cooccurrence_train.npz"  # optional precomputed C = XᵀX
_ICD_VN_MAP = PROJECT_ROOT / "data_vn/icd_vn_names.json"

try:
    import simple_icd_10 as _icd_lib
    _ICD_LIB_OK = True
except ImportError:
    _ICD_LIB_OK = False


# ---------------------------------------------------------------------------
# ICD-10 names — ưu tiên tên tiếng Việt thật trích từ dữ liệu huấn luyện
# (xem scripts/build_icd_vn_names.py), fallback tên tiếng Anh (simple_icd_10)
# cho ~3% mã không đủ dữ liệu để trích tên Việt đáng tin cậy.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_icd_vn_lookup() -> dict:
    """Returns dict: icd_code -> tên tiếng Việt (rỗng nếu chưa build mapping)."""
    if not _ICD_VN_MAP.exists():
        return {}
    with open(_ICD_VN_MAP, encoding="utf-8") as f:
        return json.load(f)


def get_icd_name(code: str) -> str:
    """Return Vietnamese description for an ICD-10 code when available,
    else English description (simple_icd_10), else the raw code."""
    code = code.strip().upper()
    vn = get_icd_vn_lookup()
    if code in vn:
        return vn[code]
    if "." in code and code.split(".")[0] in vn:
        return vn[code.split(".")[0]]

    if not _ICD_LIB_OK:
        return code
    try:
        return _icd_lib.get_description(code)
    except Exception:
        pass
    if "." in code:
        try:
            return _icd_lib.get_description(code.split(".")[0])
        except Exception:
            pass
    return code


# ---------------------------------------------------------------------------
# ATC names
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_atc_lookup() -> dict:
    """Returns dict: atc_code (str) → atc_name (str), all hierarchy levels."""
    df = pd.read_csv(_ATC_CSV, dtype=str)
    return {
        str(row["atc_code"]).strip(): str(row["atc_name"]).strip()
        for _, row in df.iterrows()
    }


def get_atc_info(code: str) -> dict:
    """Return {name, group, group_code} for an ATC code string.

    name       — leaf-level name (7-char if available, else closest parent)
    group      — pharmacological subgroup name (4-char ATC level)
    group_code — 4-char ATC code
    """
    lookup = get_atc_lookup()
    code = str(code).strip().upper()

    # Leaf name: try from longest to shortest
    name = code
    for length in [7, 5, 4, 3, 1]:
        prefix = code[:length] if len(code) >= length else code
        if prefix in lookup:
            name = lookup[prefix]
            break

    # Group: 4-char pharmacological subgroup
    group_code = code[:4] if len(code) >= 4 else code
    group = lookup.get(group_code, lookup.get(code[:3], "Unknown"))

    return {"name": name, "group": group, "group_code": group_code}


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_vocab() -> dict:
    with open(_VOCAB, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Training co-occurrence statistics (Tab 2, code frequencies, rarity)
#
# Everything here derives from C = XᵀX, where X is the binary training BoW
# (visits × codes): C[i, j] = number of training visits carrying both codes,
# C[i, i] = visits carrying code i. A deployment can ship C precomputed
# (data_vn/cooccurrence_train.npz) instead of the visit-level bow_train.npy,
# so no individual visit record leaves the project.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_bow_train() -> csr_matrix:
    return np.load(_BOW_TRAIN, allow_pickle=True).item()


def build_cooccurrence() -> tuple[np.ndarray, int]:
    """Compute (C, n_visits) from the visit-level training BoW."""
    x = (load_bow_train() > 0).astype(np.int32).tocsr()
    return (x.T @ x).toarray().astype(np.int32), int(x.shape[0])


@lru_cache(maxsize=1)
def load_cooccurrence() -> tuple[np.ndarray, int]:
    """(C, n_visits): precomputed file if present, else built from bow_train.npy."""
    if _COOC.exists():
        data = np.load(_COOC)
        return data["C"], int(data["n_visits"])
    return build_cooccurrence()


def get_n_train_visits() -> int:
    """Number of training visits — the base of every Tab 2 statistic."""
    return load_cooccurrence()[1]


def _code_counts() -> np.ndarray:
    return np.diag(load_cooccurrence()[0])


# ---------------------------------------------------------------------------
# ICD search options + rarity lookup (for friendlier input UI)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_icd_frequency() -> dict:
    """Returns dict: icd_code -> number of training visits carrying that code."""
    icd_vocab = load_vocab()["icd_vocab"]
    counts = _code_counts()
    return {icd_vocab[i]: int(counts[i]) for i in range(len(icd_vocab))}


@lru_cache(maxsize=1)
def get_icd_rank_lookup() -> dict:
    """Returns dict: icd_code -> rank (1 = most common code in train set)."""
    freq = get_icd_frequency()
    ordered = sorted(freq.items(), key=lambda kv: -kv[1])
    return {code: i + 1 for i, (code, _cnt) in enumerate(ordered)}


@lru_cache(maxsize=1)
def get_icd_options() -> tuple:
    """Cached tuple of 'CODE — Tên bệnh' strings, sorted by frequency desc (common codes first)."""
    vocab     = load_vocab()
    icd_vocab = vocab["icd_vocab"]
    freq      = get_icd_frequency()
    ordered   = sorted(icd_vocab, key=lambda c: -freq.get(c, 0))
    return tuple(f"{c} — {get_icd_name(c)}" for c in ordered)


def parse_icd_option(option: str) -> str:
    """Extract the ICD code from a 'CODE — Tên bệnh' option string."""
    return option.split(" — ")[0].strip()


@lru_cache(maxsize=1)
def get_atc_frequency() -> dict:
    """Returns dict: atc_code -> number of training visits carrying that drug."""
    vocab     = load_vocab()
    atc_vocab = vocab["atc_vocab"]
    counts    = _code_counts()[len(vocab["icd_vocab"]):]
    return {atc_vocab[i]: int(counts[i]) for i in range(len(atc_vocab))}


@lru_cache(maxsize=1)
def get_atc_options() -> tuple:
    """Cached tuple of 'CODE — Hoạt chất' strings, sorted by frequency desc."""
    vocab     = load_vocab()
    atc_vocab = vocab["atc_vocab"]
    freq      = get_atc_frequency()
    ordered   = sorted(atc_vocab, key=lambda c: -freq.get(c, 0))
    return tuple(f"{c} — {get_atc_info(c)['name']}" for c in ordered)


def parse_atc_option(option: str) -> str:
    """Extract the ATC code from a 'CODE — Hoạt chất' option string."""
    return option.split(" — ")[0].strip()


def get_icd_rarity_info(code: str) -> dict:
    """Return {count, rank, total, is_rare} describing how common `code` is in train data."""
    rank_lookup = get_icd_rank_lookup()
    freq        = get_icd_frequency()
    total       = len(rank_lookup)
    rank        = rank_lookup.get(code, total)
    count       = freq.get(code, 0)
    return {
        "count":   count,
        "rank":    rank,
        "total":   total,
        "is_rare": rank > total * 0.5,
    }


# ---------------------------------------------------------------------------
# Tab 2 queries — one row of C each
# ---------------------------------------------------------------------------

def _top_rows(counts: np.ndarray, n_visits: int, top_k: int, make_row) -> pd.DataFrame:
    rows = []
    for i in np.argsort(-counts, kind="stable")[:top_k]:
        if counts[i] == 0:
            break
        rows.append({**make_row(int(i)), "count": int(counts[i]), "freq": round(int(counts[i]) / n_visits, 3)})
    return pd.DataFrame(rows)


def _icd_index(icd_code: str):
    icd2idx = load_vocab()["icd2idx"]
    code = icd_code.upper().strip()
    idx = icd2idx.get(code)
    if idx is None and "." in code:
        idx = icd2idx.get(code.split(".")[0])
    return idx


def query_disease_drugs(icd_code: str, top_k: int = 20) -> pd.DataFrame:
    """Top drugs co-occurring with icd_code in train data.

    Returns DataFrame with columns: atc_code, name, group, count, freq; attrs["n_visits"].
    """
    vocab = load_vocab()
    C, _ = load_cooccurrence()
    V_icd, atc_vocab = len(vocab["icd_vocab"]), vocab["atc_vocab"]
    idx = _icd_index(icd_code)
    n = int(C[idx, idx]) if idx is not None else 0
    if n == 0:
        df = pd.DataFrame(columns=["atc_code", "name", "group", "count", "freq"])
    else:
        def row(j):
            info = get_atc_info(atc_vocab[j])
            return {"atc_code": atc_vocab[j], "name": info["name"], "group": info["group"]}
        df = _top_rows(C[idx, V_icd:], n, top_k, row)
    df.attrs["n_visits"] = n
    return df


def query_drug_diseases(atc_code: str, top_k: int = 20) -> pd.DataFrame:
    """Top ICD codes co-occurring with atc_code in train data.

    Returns DataFrame with columns: icd_code, name, count, freq; attrs["n_visits"].
    """
    vocab = load_vocab()
    C, _ = load_cooccurrence()
    icd_vocab, atc2idx = vocab["icd_vocab"], vocab["atc2idx"]
    V_icd = len(icd_vocab)

    code = atc_code.upper().strip()
    idx = atc2idx.get(code)
    if idx is None:
        for length in [5, 4, 3]:
            prefix = code[:length] if len(code) >= length else None
            if prefix and prefix in atc2idx:
                idx = atc2idx[prefix]
                break
    n = int(C[V_icd + idx, V_icd + idx]) if idx is not None else 0
    if n == 0:
        df = pd.DataFrame(columns=["icd_code", "name", "count", "freq"])
    else:
        df = _top_rows(C[V_icd + idx, :V_icd], n, top_k,
                       lambda i: {"icd_code": icd_vocab[i], "name": get_icd_name(icd_vocab[i])})
    df.attrs["n_visits"] = n
    return df


def query_comorbidities(icd_code: str, top_k: int = 20) -> pd.DataFrame:
    """Top ICD codes that co-occur with icd_code in the same visit (query code excluded).

    Returns DataFrame with columns: icd_code, name, count, freq; attrs["n_visits"].
    """
    vocab = load_vocab()
    C, _ = load_cooccurrence()
    icd_vocab = vocab["icd_vocab"]
    idx = _icd_index(icd_code)
    n = int(C[idx, idx]) if idx is not None else 0
    if n == 0:
        df = pd.DataFrame(columns=["icd_code", "name", "count", "freq"])
    else:
        counts = C[idx, :len(icd_vocab)].copy()
        counts[idx] = 0
        df = _top_rows(counts, n, top_k, lambda i: {"icd_code": icd_vocab[i], "name": get_icd_name(icd_vocab[i])})
    df.attrs["n_visits"] = n
    return df
