import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from data_loader import load_test_data
from src.config import (
    LABELS, LABEL_NAMES, MODEL_FILE, VECTORIZER_FILE, RESULT_DIR,
    TEST_METRICS_FILE, CLASSIFICATION_REPORT_FILE,
    CONFUSION_MATRIX_FILE, NORMALIZED_CONFUSION_MATRIX_FILE,
    MISCLASSIFIED_FILE,
)
from src.preprocessing import preprocess_texts


def main():
    print("=" * 68)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 68)

    if not Path(MODEL_FILE).exists() or not Path(VECTORIZER_FILE).exists():
        raise FileNotFoundError(
            "Chưa có mô hình hoặc TF-IDF. Hãy chạy `python train.py` trước."
        )

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    test_df = load_test_data()
    texts = preprocess_texts(test_df["text"].tolist())
    y_true = test_df["label"].tolist()

    print(f"Số văn bản Test: {len(y_true):,}")
    print("Đang vector hóa và dự đoán...")
    X_test = vectorizer.transform(texts)

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_seconds = time.perf_counter() - start

    metrics = {
        "model": type(model).__name__,
        "test_documents": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, labels=LABELS,
            average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, labels=LABELS,
            average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=LABELS,
            average="macro", zero_division=0
        )),
        "precision_weighted": float(precision_score(
            y_true, y_pred, labels=LABELS,
            average="weighted", zero_division=0
        )),
        "recall_weighted": float(recall_score(
            y_true, y_pred, labels=LABELS,
            average="weighted", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_true, y_pred, labels=LABELS,
            average="weighted", zero_division=0
        )),
        "prediction_time_seconds": round(float(prediction_seconds), 3),
        "misclassified_documents": int(sum(a != b for a, b in zip(y_true, y_pred))),
    }

    target_names = [LABEL_NAMES[x] for x in LABELS]
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=target_names,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        CLASSIFICATION_REPORT_FILE, encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(
        cm, index=target_names, columns=target_names
    ).to_csv(CONFUSION_MATRIX_FILE, encoding="utf-8-sig")

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )
    pd.DataFrame(
        normalized, index=target_names, columns=target_names
    ).to_csv(NORMALIZED_CONFUSION_MATRIX_FILE, encoding="utf-8-sig")

    errors = test_df.loc[
        np.asarray(y_true) != np.asarray(y_pred)
    ].copy()
    errors["true_label_name"] = errors["label"].map(LABEL_NAMES)
    errors["predicted_label"] = np.asarray(y_pred)[
        np.asarray(y_true) != np.asarray(y_pred)
    ]
    errors["predicted_label_name"] = errors["predicted_label"].map(LABEL_NAMES)
    errors.rename(columns={"label": "true_label"}, inplace=True)
    errors.to_csv(MISCLASSIFIED_FILE, index=False, encoding="utf-8-sig")

    with open(TEST_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nKẾT QUẢ TEST")
    print("-" * 68)
    for key in [
        "model", "test_documents", "accuracy", "precision_macro",
        "recall_macro", "f1_macro", "f1_weighted",
        "prediction_time_seconds", "misclassified_documents"
    ]:
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key:28s}: {value:.4f}")
        else:
            print(f"{key:28s}: {value}")
    print("\nĐã lưu báo cáo, ma trận nhầm lẫn và các mẫu dự đoán sai trong results/.")


if __name__ == "__main__":
    main()
