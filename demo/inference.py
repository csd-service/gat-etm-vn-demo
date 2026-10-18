"""Lightweight inference for GAT-ETM demo.

Loads only the VAE encoder from the checkpoint (no graph/GAT needed at inference
time) plus pre-saved beta matrices. This keeps the demo fast on CPU.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent

# E1.5 GAT-ETM full, K=20, tau=3, KG TRAIN-ONLY (leak-free, seed 42) — nhất quán với số liệu
# báo cáo trong luận văn sau hiệu đính leakage (mục 3.4.3, Chương 5).
_CHECKPOINT = PROJECT_ROOT / "results_vn/gat_etm_full_trainonly_K20/model_gat_etm_full_K20.pt"
_BETA_ATC   = PROJECT_ROOT / "results_vn/gat_etm_full_trainonly_K20/beta_atc.npy"
_BETA_ICD   = PROJECT_ROOT / "results_vn/gat_etm_full_trainonly_K20/beta_icd.npy"
_VOCAB      = PROJECT_ROOT / "data_vn/vocab_info.pkl"
_THETA_MEAN = PROJECT_ROOT / "data_vn/theta_mean_train.npy"  # optional precomputed mean_theta()

VOCAB_ICD   = 1493
VOCAB_ATC   = 377
VOCAB_TOTAL = VOCAB_ICD + VOCAB_ATC
NUM_TOPICS  = 20
T_HIDDEN    = 128


class _LightEncoder(nn.Module):
    """VAE encoder extracted from GETM. Mirrors the exact layer structure so that
    state-dict keys (q_theta.0, q_theta.2, mu_q_theta, ...) match the checkpoint.
    """

    def __init__(self):
        super().__init__()
        act = nn.ReLU()
        # Must use nn.Sequential with same indexing as GETM.q_theta
        self.q_theta = nn.Sequential(
            nn.Linear(VOCAB_TOTAL, T_HIDDEN),  # index 0
            act,                                # index 1
            nn.Linear(T_HIDDEN, T_HIDDEN),      # index 2
            act,                                # index 3
        )
        self.mu_q_theta = nn.Linear(T_HIDDEN, NUM_TOPICS)
        self.logsigma_q_theta = nn.Linear(T_HIDDEN, NUM_TOPICS)  # loaded but unused at inference
        self.t_drop = nn.Dropout(0.1)

    def get_theta(self, norm_bows: torch.Tensor) -> torch.Tensor:
        """Returns theta (K-dim topic distribution) for a batch of BoW vectors."""
        h = self.t_drop(self.q_theta(norm_bows))
        mu = self.mu_q_theta(h)
        return F.softmax(mu, dim=-1)


def load_resources():
    """Load and return (encoder, beta_atc, beta_icd, vocab).
    Designed to be wrapped with @st.cache_resource — runs once per process.
    """
    encoder = _LightEncoder()
    state = torch.load(_CHECKPOINT, map_location="cpu", weights_only=True)
    encoder_state = {
        k: v for k, v in state.items()
        if k.startswith(("q_theta.", "mu_q_theta.", "logsigma_q_theta.", "t_drop."))
    }
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    beta_atc = np.load(_BETA_ATC)  # (NUM_TOPICS, VOCAB_ATC)
    beta_icd = np.load(_BETA_ICD)  # (NUM_TOPICS, VOCAB_ICD)

    with open(_VOCAB, "rb") as f:
        vocab = pickle.load(f)

    return encoder, beta_atc, beta_icd, vocab


def _normalize_bow(bow: np.ndarray, vocab_cum: list) -> np.ndarray:
    """Per-code-type L1 normalisation (mirrors dataset.normalize)."""
    norm = np.zeros_like(bow, dtype=np.float32)
    for i in range(len(vocab_cum) - 1):
        lo, hi = vocab_cum[i], vocab_cum[i + 1]
        s = bow[lo:hi].sum() + 1e-6
        norm[lo:hi] = bow[lo:hi] / s
    return norm


def mean_theta(encoder, vocab, bow_train, n_samples: int = 20000, seed: int = 0) -> np.ndarray:
    """Average theta over training visits encoded the same way as the demo (ICD codes only).

    Used as the reference profile when saying a cluster is "×N above an average visit".
    """
    n_icd = len(vocab["icd_vocab"])
    rng = np.random.default_rng(seed)
    rows = rng.choice(bow_train.shape[0], size=min(n_samples, bow_train.shape[0]), replace=False)
    thetas = []
    for start in range(0, len(rows), 5000):
        icd = bow_train[rows[start:start + 5000], :n_icd].toarray().astype(np.float32)
        x = np.zeros((len(icd), VOCAB_TOTAL), dtype=np.float32)
        x[:, :n_icd] = icd / (icd.sum(axis=1, keepdims=True) + 1e-6)
        with torch.no_grad():
            thetas.append(encoder.get_theta(torch.from_numpy(x)).numpy())
    return np.concatenate(thetas).mean(axis=0)


def load_mean_theta(encoder, vocab) -> np.ndarray:
    """mean_theta() over the training BoW, or its precomputed copy when the BoW is not shipped."""
    if _THETA_MEAN.exists():
        return np.load(_THETA_MEAN)
    from data_utils import load_bow_train
    return mean_theta(encoder, vocab, load_bow_train())


def predict(encoder, beta_atc, beta_icd, vocab, icd_codes: list) -> dict:
    """Run inference for one patient given a list of ICD-10 code strings.

    Returns
    -------
    dict with keys:
        theta       : np.array (NUM_TOPICS,) — topic distribution
        drug_probs  : dict[atc_code, float]  — drug probability scores
        icd_probs   : dict[icd_code, float]  — ICD reconstruction scores
        icd_found   : list[str]              — codes resolved in vocab
        icd_oov     : list[str]              — codes not found in vocab
    """
    icd2idx   = vocab["icd2idx"]
    icd_vocab = vocab["icd_vocab"]
    atc_vocab = vocab["atc_vocab"]
    vocab_cum = vocab["vocab_cum"]

    bow = np.zeros(VOCAB_TOTAL, dtype=np.float32)
    icd_found, icd_oov = [], []

    for raw in icd_codes:
        code = str(raw).upper().strip()
        if code in icd2idx:
            bow[icd2idx[code]] += 1
            icd_found.append(code)
        elif "." in code:
            # fall back to 3-char parent (e.g. E11.9 → E11)
            parent = code.split(".")[0]
            if parent in icd2idx:
                bow[icd2idx[parent]] += 1
                icd_found.append(parent)
            else:
                icd_oov.append(code)
        else:
            icd_oov.append(code)

    norm_bow = _normalize_bow(bow, vocab_cum)

    with torch.no_grad():
        t = torch.tensor(norm_bow, dtype=torch.float32).unsqueeze(0)
        theta = encoder.get_theta(t).squeeze(0).numpy()

    drug_vec = theta @ beta_atc  # (VOCAB_ATC,)
    icd_vec  = theta @ beta_icd  # (VOCAB_ICD,)

    return {
        "theta":      theta,
        "drug_probs": {atc_vocab[i]: float(drug_vec[i]) for i in range(len(atc_vocab))},
        "icd_probs":  {icd_vocab[i]: float(icd_vec[i])  for i in range(len(icd_vocab))},
        "icd_found":  icd_found,
        "icd_oov":    icd_oov,
    }
