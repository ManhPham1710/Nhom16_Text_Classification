
"""
NHÓM 16 - PHÂN LOẠI CHỦ ĐỀ VĂN BẢN TIẾNG VIỆT

Quy trình:
1. Đọc tập Train.
2. Chia Train thành Train nội bộ và Validation có phân tầng.
3. Huấn luyện, so sánh các mô hình.
4. Chọn mô hình theo Macro-F1 trên Validation.
5. Huấn luyện lại mô hình tốt nhất trên toàn bộ Train.
6. Lưu mô hình, vectorizer và báo cáo thực nghiệm.

Lưu ý: Không sử dụng tập Test trong quá trình lựa chọn mô hình.
"""

import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from data_loader import load_train_data
from src.preprocessing import preprocess_texts
from src.config import (
    LABELS,
    RANDOM_STATE,
    VALIDATION_SIZE,
    TFIDF_CONFIG,
    MODEL_FILE,
    VECTORIZER_FILE,
    RESULT_DIR,
    METADATA_FILE,
    VALIDATION_METRICS_FILE,
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

FOCUS_LABEL = "doi_song"

VALIDATION_REPORT_FILE = (
    RESULT_DIR / "validation_classification_report.csv"
)


def create_vectorizer():
    """Khởi tạo TF-IDF vectorizer theo cấu hình dự án."""

    return TfidfVectorizer(
        max_features=TFIDF_CONFIG["max_features"],
        ngram_range=tuple(TFIDF_CONFIG["ngram_range"]),
        min_df=TFIDF_CONFIG["min_df"],
        max_df=TFIDF_CONFIG["max_df"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        dtype=np.float32,
    )


# ============================================================
# 2. XÂY DỰNG DANH SÁCH MÔ HÌNH
# ============================================================

def create_models():

    models = [
        (
            "MultinomialNB_alpha_0.1",
            MultinomialNB(alpha=0.1),
            "Multinomial Naive Bayes",
        ),

        (
            "DecisionTree_depth100",
            DecisionTreeClassifier(
                max_depth=100,
                min_samples_split=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Decision Tree",
        ),
    ]

    # Thử nghiệm nhiều giá trị C và class_weight.
    for c_value in [0.5, 1.0, 2.0]:

        for class_weight in [None, "balanced"]:

            weight_name = (
                "none" if class_weight is None else "balanced"
            )

            model_name = (
                f"LinearSVC_C{c_value}_weight_{weight_name}"
            )

            model = LinearSVC(
                C=c_value,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                max_iter=10000,
                dual="auto",
            )

            models.append(
                (model_name, model, "LinearSVC")
            )

    return models


# ============================================================
# 3. ĐÁNH GIÁ MÔ HÌNH
# ============================================================

def evaluate_model(model, x_val, y_val):

    predictions = model.predict(x_val)

    class_scores = f1_score(
        y_val,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    result = {
        "accuracy": accuracy_score(y_val, predictions),

        "precision_macro": precision_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "f1_weighted": f1_score(
            y_val,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    for label, score in zip(LABELS, class_scores):
        result[f"f1_{label}"] = float(score)

    return result, predictions


# ============================================================
# 4. QUY TRÌNH HUẤN LUYỆN CHÍNH
# ============================================================

def main():

    print("=" * 75)
    print("NHÓM 16 - HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI VĂN BẢN")
    print("=" * 75)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Bước 1: Đọc dữ liệu Train
    # --------------------------------------------------------

    print("\n[1/6] Đang đọc dữ liệu Train...")

    df = load_train_data()

    if df.empty:
        raise ValueError("Dữ liệu Train không có bản ghi.")

    texts = preprocess_texts(df["text"].tolist())
    labels = df["label"].astype(str).tolist()

    invalid_labels = set(labels) - set(LABELS)

    if invalid_labels:
        raise ValueError(
            f"Phát hiện nhãn không hợp lệ: {invalid_labels}"
        )

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Có văn bản rỗng sau bước tiền xử lý."
        )

    print(f"Tổng số văn bản: {len(texts):,}")
    print(f"Số lượng nhãn: {len(set(labels))}")

    # --------------------------------------------------------
    # Bước 2: Chia Train và Validation
    # --------------------------------------------------------

    print("\n[2/6] Đang chia dữ liệu...")

    x_train_text, x_val_text, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train nội bộ: {len(y_train):,}")
    print(f"Validation: {len(y_val):,}")

    # --------------------------------------------------------
    # Bước 3: Trích xuất đặc trưng TF-IDF
    # --------------------------------------------------------

    print("\n[3/6] Đang trích xuất đặc trưng TF-IDF...")

    selection_vectorizer = create_vectorizer()

    x_train = selection_vectorizer.fit_transform(
        x_train_text
    )

    x_val = selection_vectorizer.transform(
        x_val_text
    )

    print(f"Số đặc trưng: {x_train.shape[1]:,}")

    # --------------------------------------------------------
    # Bước 4: Huấn luyện và so sánh
    # --------------------------------------------------------

    print("\n[4/6] Bắt đầu thử nghiệm các mô hình...")

    results = []
    fitted_models = {}

    model_list = create_models()

    for index, (name, model, family) in enumerate(
        model_list, start=1
    ):

        print(f"\nĐang huấn luyện {index}/{len(model_list)}: {name}")

        start_time = time.perf_counter()

        model.fit(x_train, y_train)

        training_time = time.perf_counter() - start_time

        metrics, predictions = evaluate_model(
            model,
            x_val,
            y_val,
        )

        metrics["model"] = name
        metrics["model_family"] = family
        metrics["training_time_seconds"] = round(
            training_time, 3
        )

        results.append(metrics)

        fitted_models[name] = (
            model,
            predictions,
        )

        print(
            f"Macro-F1: {metrics['f1_macro']:.4f} | "
            f"Weighted-F1: {metrics['f1_weighted']:.4f} | "
            f"F1 Đời sống: "
            f"{metrics[f'f1_{FOCUS_LABEL}']:.4f}"
        )

    # --------------------------------------------------------
    # Bước 5: Lựa chọn mô hình tốt nhất
    # --------------------------------------------------------

    print("\n[5/6] Đang lựa chọn mô hình tốt nhất...")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "f1_macro",
            f"f1_{FOCUS_LABEL}",
            "f1_weighted",
            "accuracy",
        ],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "validation_rank",
        range(1, len(results_df) + 1),
    )

    results_df.to_csv(
        VALIDATION_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    best_name = results_df.iloc[0]["model"]

    best_model, best_predictions = fitted_models[best_name]

    best_metrics = results_df.iloc[0].to_dict()

    print("\nBẢNG XẾP HẠNG MÔ HÌNH")
    print("-" * 75)

    print(
        results_df[
            [
                "validation_rank",
                "model",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                f"f1_{FOCUS_LABEL}",
            ]
        ].to_string(index=False)
    )

    print("\nMÔ HÌNH ĐƯỢC LỰA CHỌN:")
    print(best_name)

    # Báo cáo chi tiết từng lớp trên Validation.
    report = classification_report(
        y_val,
        best_predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.index.name = "label"

    report_df.to_csv(
        VALIDATION_REPORT_FILE,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Bước 6: Huấn luyện lại trên toàn bộ Train
    # --------------------------------------------------------

    print("\n[6/6] Đang huấn luyện lại trên toàn bộ Train...")

    final_vectorizer = create_vectorizer()

    x_full = final_vectorizer.fit_transform(texts)

    final_model = best_model.__class__(
        **best_model.get_params()
    )

    start_time = time.perf_counter()

    final_model.fit(x_full, labels)

    final_training_time = (
        time.perf_counter() - start_time
    )

    # Lưu mô hình và vectorizer.
    joblib.dump(
        final_model,
        MODEL_FILE,
        compress=3,
    )

    joblib.dump(
        final_vectorizer,
        VECTORIZER_FILE,
        compress=3,
    )

    # --------------------------------------------------------
    # Lưu metadata
    # --------------------------------------------------------

    metadata = {
        "selected_model": best_name,

        "selected_model_family": best_metrics[
            "model_family"
        ],

        "selection_metric": "f1_macro",

        "selection_tie_breakers": [
            f"f1_{FOCUS_LABEL}",
            "f1_weighted",
            "accuracy",
        ],

        "test_used_during_training_or_selection": False,

        "random_state": RANDOM_STATE,

        "validation_size": VALIDATION_SIZE,

        "train_documents": len(df),

        "internal_train_documents": len(y_train),

        "validation_documents": len(y_val),

        "number_of_classes": len(LABELS),

        "labels": LABELS,

        "tfidf_config": {
            **TFIDF_CONFIG,
            "ngram_range": list(
                TFIDF_CONFIG["ngram_range"]
            ),
        },

        "selected_validation_metrics": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (float, int, np.floating, np.integer),
                )
                else value
            )
            for key, value in best_metrics.items()
        },

        "all_validation_results": results,

        "final_training_time_seconds": round(
            final_training_time,
            3,
        ),

        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # --------------------------------------------------------
    # Kết quả cuối cùng
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("HUẤN LUYỆN HOÀN TẤT")
    print("=" * 75)

    print(f"Mô hình được chọn: {best_name}")

    print(
        f"Macro-F1 Validation: "
        f"{best_metrics['f1_macro']:.4f}"
    )

    print(
        f"F1 lớp Đời sống: "
        f"{best_metrics[f'f1_{FOCUS_LABEL}']:.4f}"
    )

    print(f"Thời gian huấn luyện cuối: {final_training_time:.2f}s")

    print("\nCác tệp đã lưu:")
    print(f"- {MODEL_FILE}")
    print(f"- {VECTORIZER_FILE}")
    print(f"- {VALIDATION_METRICS_FILE}")
    print(f"- {VALIDATION_REPORT_FILE}")
    print(f"- {METADATA_FILE}")

    print("\nLƯU Ý:")
    print("Tập Test không được sử dụng để lựa chọn mô hình.")
    print("Chỉ đánh giá Test sau khi đã chốt mô hình cuối cùng.")


if __name__ == "__main__":
    main()
