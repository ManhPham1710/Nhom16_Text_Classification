from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_FILE = RAW_DATA_DIR / "train.txt"
TEST_FILE = RAW_DATA_DIR / "test.txt"

MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "the_thao",
    "chinh_tri_xa_hoi",
    "phap_luat",
    "suc_khoe",
    "doi_song",
    "van_hoa",
    "the_gioi",
    "kinh_doanh",
    "vi_tinh",
    "khoa_hoc",
]

LABEL_NAMES = {
    "the_thao": "Thể thao",
    "chinh_tri_xa_hoi": "Chính trị - Xã hội",
    "phap_luat": "Pháp luật",
    "suc_khoe": "Sức khỏe",
    "doi_song": "Đời sống",
    "van_hoa": "Văn hóa",
    "the_gioi": "Thế giới",
    "kinh_doanh": "Kinh doanh",
    "vi_tinh": "Vi tính",
    "khoa_hoc": "Khoa học",
}

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

TFIDF_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "dtype": "float32",
}

MODEL_FILE = MODEL_DIR / "best_model.joblib"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
METADATA_FILE = RESULT_DIR / "experiment_metadata.json"
VALIDATION_METRICS_FILE = RESULT_DIR / "validation_metrics.csv"
TEST_METRICS_FILE = RESULT_DIR / "test_metrics.json"
CLASSIFICATION_REPORT_FILE = RESULT_DIR / "classification_report.csv"
CONFUSION_MATRIX_FILE = RESULT_DIR / "confusion_matrix.csv"
NORMALIZED_CONFUSION_MATRIX_FILE = RESULT_DIR / "confusion_matrix_normalized.csv"
MISCLASSIFIED_FILE = RESULT_DIR / "misclassified_samples.csv"
