# GAT-ETM Demo — Hỗ trợ kê đơn từ chẩn đoán ICD-10

Demo minh hoạ luận văn thạc sĩ **GAT-ETM** (Graph Attention Network + Embedded Topic Model)
trên dữ liệu EHR của một bệnh viện tại TP.HCM. Viết bằng [Streamlit](https://streamlit.io).

- **Gợi ý thuốc:** nhập chẩn đoán ICD-10 → mô hình ước lượng các cụm bệnh tiềm ẩn → Top-5 thuốc,
  mỗi thuốc kèm cụm bệnh đóng góp nhiều nhất vào gợi ý.
- **Tra cứu bệnh – thuốc:** thống kê đồng xuất hiện trên tập huấn luyện (không chạy mô hình).
- Số liệu đánh giá mô hình nằm trong nút **Về mô hình** của app.

> Công cụ minh hoạ nghiên cứu, không thay thế chỉ định của bác sĩ.

## Dữ liệu trong repo

Repo này **không chứa hồ sơ từng lượt khám**. Chỉ có:

| File | Nội dung |
|---|---|
| `results_vn/gat_etm_full_trainonly_K20/` | Checkpoint mô hình và ma trận β |
| `data_vn/vocab_info.pkl` | Danh sách 1.493 mã ICD-10 và 377 mã ATC |
| `data_vn/cooccurrence_train.npz` | Số lượt khám có từng cặp mã (thống kê tổng hợp trên tập huấn luyện) |
| `data_vn/theta_mean_train.npy` | Phân bố chủ đề trung bình của tập huấn luyện (20 số) |
| `data_vn/icd_vn_names.json`, `data_vn/atc_reference/` | Tên bệnh tiếng Việt, tên thuốc theo WHO ATC |

## Chạy và deploy

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Streamlit Community Cloud: **Main file path** = `demo/app.py`, Python 3.11.

Repo này được sinh tự động từ repo nghiên cứu bằng `demo/deploy/build_deploy.py`;
sửa code ở repo gốc rồi build lại, không sửa trực tiếp tại đây.
