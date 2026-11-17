import html
import re
import unicodedata


def clean_text(text):
    """Chuẩn hóa văn bản tiếng Việt, giữ lại chữ và số."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = html.unescape(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_texts(texts):
    return [clean_text(text) for text in texts]
