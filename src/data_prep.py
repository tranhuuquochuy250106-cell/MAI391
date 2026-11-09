"""Câu 1 - Chuẩn bị dữ liệu (TV2). Chạy: python src/data_prep.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/Folds5x2_pp.xlsx"
PROC = ROOT / "data/processed"
RES = ROOT / "results"
SEED, TEST_FRAC, DDOF = 42, 0.20, 0
COLS = ["AT", "V", "AP", "RH", "PE"]


def main():
    PROC.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(RAW, sheet_name="Sheet1")           # chỉ đọc Sheet1
    df.insert(0, "source_row", np.arange(2, len(df) + 2))   # số dòng Excel (header = dòng 1)
    audit = []
    def log(item, value, note=""):
        audit.append({"item": item, "value": value, "note": note})

    log("rows_raw", len(df), "Sheet1 của Folds5x2_pp.xlsx")
    for c in COLS:
        log(f"dtype_{c}", str(df[c].dtype))
        log(f"missing_{c}", int(df[c].isna().sum()))
        log(f"nonfinite_{c}", int((~np.isfinite(df[c])).sum()))
    log("invalid_AT_range", int(((df.AT < -50) | (df.AT > 60)).sum()), "AT ngoài [-50,60] °C")
    log("invalid_PE_nonpositive", int((df.PE <= 0).sum()), "PE <= 0 MW")
    log("invalid_RH_range", int(((df.RH < 0) | (df.RH > 100.5)).sum()), "RH ngoài [0,100.5] %")
    dup_mask = df.duplicated(subset=COLS, keep="first")
    log("duplicate_rows_full5", int(dup_mask.sum()), "trùng hoàn toàn cả 5 cột (giữ bản đầu)")
    log("rows_with_repeated_AT", int(df.duplicated(subset=["AT"], keep=False).sum()), "AT lặp - KHÔNG loại")
    log("outliers_removed", 0, "Không bỏ điểm cực trị")

    clean = df.loc[~dup_mask].reset_index(drop=True)
    log("rows_removed", int(dup_mask.sum()), "source_row bị loại: " + ",".join(map(str, df.source_row[dup_mask])))
    log("rows_clean", len(clean))
    assert len(clean) >= 500

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(clean))
    n_test = int(round(TEST_FRAC * len(clean)))
    test = clean.iloc[np.sort(perm[:n_test])].reset_index(drop=True)
    train = clean.iloc[np.sort(perm[n_test:])].reset_index(drop=True)
    assert set(train.source_row).isdisjoint(test.source_row)
    log("rows_train", len(train)); log("rows_test", len(test))
        # Tính trung bình AT trên tập train
    n_train = len(train)
    total_at = 0.0
    for at in train["AT"]:
        total_at += at
    mean_at = total_at / n_train

    # Tính độ lệch chuẩn AT trên tập train
    squared_diff_sum = 0.0
    for at in train["AT"]:
        diff = at - mean_at
        squared_diff_sum += diff ** 2

    variance_at = squared_diff_sum / (n_train - DDOF)
    std_at = variance_at ** 0.5

    # Dùng cùng trung bình và độ lệch chuẩn để chuẩn hóa train và test
    for d in (train, test):
        d["AT_scaled"] = (d.AT - mean_at) / std_at

    mean_at = float(train.AT.mean()); std_at = float(train.AT.std(ddof=DDOF))
    for d in (train, test):
        d["AT_scaled"] = (d.AT - mean_at) / std_at
    clean.to_csv(PROC / "clean.csv", index=False)
    train.to_csv(PROC / "train.csv", index=False)
    test.to_csv(PROC / "test.csv", index=False)
    scaling = {"variable": "AT", "target": "PE", "mean_AT": mean_at, "std_AT": std_at,
               "ddof": DDOF, "seed": SEED, "test_fraction": TEST_FRAC,
               "split_method": "numpy default_rng(seed).permutation trên clean.csv; giữ thứ tự source_row trong mỗi tập",
               "n_clean": len(clean), "n_train": len(train), "n_test": len(test),
               "units": {"AT": "°C", "PE": "MW"}}
    (RES / "scaling.json").write_text(json.dumps(scaling, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(audit).to_csv(RES / "data_audit.csv", index=False)
    pd.DataFrame({"variable": COLS,
        "meaning": ["Nhiệt độ môi trường TB theo giờ", "Chân không xả (Exhaust Vacuum)",
                    "Áp suất môi trường", "Độ ẩm tương đối", "Công suất điện ròng theo giờ"],
        "unit": ["°C", "cm Hg", "mbar", "%", "MW"]}).to_csv(RES / "q1_variables.csv", index=False)
    print(pd.DataFrame(audit).to_string(index=False)); print(scaling)

if __name__ == "__main__":
    main()
