"""Preset demo cases for the Tab 1 quick-load pills.

Chosen from code combinations seen in training visits so that each case lands
clearly in one disease cluster (except the multimorbidity case, which spans
several on purpose). Every code is in the model vocabulary.
"""

SAMPLE_CASES = {
    "Tim mạch – chuyển hoá": {
        "codes":       ["E11.9", "I10", "E78.2", "I25"],
        "description": "Đái tháo đường type 2 + tăng huyết áp + rối loạn lipid máu + bệnh tim thiếu máu cục bộ",
    },
    "Cơ xương khớp": {
        "codes":       ["M47", "M54", "M81.0", "E58"],
        "description": "Thoái hoá cột sống + đau lưng + loãng xương sau mãn kinh + thiếu calci",
    },
    "Thần kinh – tâm thần": {
        "codes":       ["F41.2", "F51.0", "G43", "H81"],
        "description": "Rối loạn hỗn hợp lo âu – trầm cảm + mất ngủ + migraine + rối loạn tiền đình",
    },
    "Đa bệnh": {
        "codes":       ["E11.9", "I10", "M17", "K21", "F51.0"],
        "description": "Đái tháo đường + tăng huyết áp + thoái hoá khớp gối + trào ngược dạ dày – thực quản + mất ngủ",
    },
}
