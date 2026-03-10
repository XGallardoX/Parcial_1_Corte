import re

def clean_raw_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess_batch(texts):
    return [clean_raw_text(t) for t in texts]
