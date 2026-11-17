from pathlib import Path
import pandas as pd

from src.config import TRAIN_FILE, TEST_FILE, LABELS


def load_fasttext_file(file_path):
    """Đọc file dạng __label__ten_nhan noi_dung van_ban."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {file_path}")

    records = []
    invalid_lines = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                invalid_lines.append((line_no, "Dòng thiếu nội dung"))
                continue

            raw_label, text = parts
            if not raw_label.startswith("__label__"):
                invalid_lines.append((line_no, f"Thiếu tiền tố nhãn: {raw_label}"))
                continue

            label = raw_label[len("__label__"):]
            if label not in LABELS:
                invalid_lines.append((line_no, f"Nhãn không hợp lệ: {label}"))
                continue

            if not text.strip():
                invalid_lines.append((line_no, "Văn bản rỗng"))
                continue

            records.append({"label": label, "text": text.strip()})

    if invalid_lines:
        preview = "; ".join(
            f"dòng {n}: {reason}" for n, reason in invalid_lines[:5]
        )
        raise ValueError(
            f"{file_path.name} có {len(invalid_lines)} dòng không hợp lệ. "
            f"Một số lỗi: {preview}"
        )

    if not records:
        raise ValueError(f"Không có dữ liệu hợp lệ trong {file_path}")

    return pd.DataFrame(records)


def load_train_data():
    return load_fasttext_file(TRAIN_FILE)


def load_test_data():
    return load_fasttext_file(TEST_FILE)


if __name__ == "__main__":
    train_df = load_train_data()
    test_df = load_test_data()
    print(f"Train: {len(train_df):,} văn bản")
    print(f"Test : {len(test_df):,} văn bản")
    print("\nPhân bố nhãn Train:")
    print(train_df["label"].value_counts().reindex(LABELS, fill_value=0))
    print("\nPhân bố nhãn Test:")
    print(test_df["label"].value_counts().reindex(LABELS, fill_value=0))
