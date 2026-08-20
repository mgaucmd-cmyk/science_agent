"""
本地题库模块。
支持上传 pdf / docx / txt 格式的卷子，自动切分成一道道题存到本地 JSON 文件里，
之后可以按主题检索出最相关的几道题，用于生成更有针对性的巩固练习。

检索使用 TF-IDF + 余弦相似度，全部本地计算，不调用任何外部 API，不产生额外费用。
"""
import os
import re
import json
import time
from typing import List, Dict

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 题库文件存放在 app.py 同级目录下，随项目一起走，不用额外配置路径
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_store.json")

# 检索相关度阈值：低于这个分数的匹配不采用，宁缺毋滥
MATCH_SCORE_THRESHOLD = 0.12


# ============================================================
# 文件文本提取
# ============================================================

def extract_text(filename: str, file_bytes: bytes) -> str:
    """根据文件后缀提取纯文本内容"""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == "docx":
        import docx
        import io
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if ext == "pdf":
        import pdfplumber
        import io
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)

    raise ValueError(f"不支持的文件格式：.{ext}（目前支持 pdf / docx / txt）")


# ============================================================
# 文本切分：尽量按题号切，切不出来就按段落/固定窗口切
# ============================================================

_QUESTION_NUM_PATTERN = re.compile(r'(?=\n\s*\d{1,3}\s*[\.\、．])')


def split_into_chunks(text: str, min_len: int = 20, max_len: int = 800) -> List[str]:
    text = text.strip()
    if not text:
        return []

    # 优先尝试按题号切分（"1." "2、" 这种格式）
    parts = _QUESTION_NUM_PATTERN.split(text)
    parts = [p.strip() for p in parts if len(p.strip()) >= min_len]

    if len(parts) < 2:
        # 题号切不出来，退化为按空行分段
        parts = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) >= min_len]

    if len(parts) < 2:
        # 还是切不出来（比如整篇没有明显分段），用固定窗口滑动切分兜底
        parts = []
        step = max_len - 100
        for i in range(0, len(text), step):
            piece = text[i:i + max_len].strip()
            if len(piece) >= min_len:
                parts.append(piece)

    # 超长的段落再二次切分，避免单个chunk过长影响检索效果
    final_chunks = []
    for p in parts:
        if len(p) <= max_len:
            final_chunks.append(p)
        else:
            for i in range(0, len(p), max_len):
                sub = p[i:i + max_len].strip()
                if len(sub) >= min_len:
                    final_chunks.append(sub)

    return final_chunks


# ============================================================
# 存储
# ============================================================

def _load_store() -> List[Dict]:
    if not os.path.exists(KB_PATH):
        return []
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_store(chunks: List[Dict]) -> None:
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def add_document(filename: str, file_bytes: bytes) -> int:
    """解析并存入一个文件，返回新增的题目/片段数量"""
    text = extract_text(filename, file_bytes)
    pieces = split_into_chunks(text)
    if not pieces:
        return 0

    store = _load_store()
    next_id = (max((c["id"] for c in store), default=0)) + 1
    now = time.strftime("%Y-%m-%d %H:%M")

    for piece in pieces:
        store.append({
            "id": next_id,
            "source": filename,
            "text": piece,
            "added_at": now,
        })
        next_id += 1

    _save_store(store)
    return len(pieces)


def list_documents() -> List[Dict]:
    """返回已上传的文件列表及每个文件包含的题目数"""
    store = _load_store()
    summary: Dict[str, Dict] = {}
    for c in store:
        src = c["source"]
        if src not in summary:
            summary[src] = {"source": src, "count": 0, "added_at": c["added_at"]}
        summary[src]["count"] += 1
    return list(summary.values())


def remove_document(filename: str) -> None:
    store = _load_store()
    store = [c for c in store if c["source"] != filename]
    _save_store(store)


def clear_kb() -> None:
    _save_store([])


# ============================================================
# 检索
# ============================================================

def _jieba_tokenize(text: str) -> List[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def search_kb(query: str, top_k: int = 3, min_score: float = MATCH_SCORE_THRESHOLD) -> List[Dict]:
    """
    在题库中检索与 query 最相关的题目片段。
    返回：[{"text":, "source":, "score":}, ...]，按相关度从高到低排列。
    没有匹配到高质量结果时返回空列表。
    """
    store = _load_store()
    if not store:
        return []

    texts = [c["text"] for c in store]
    try:
        vectorizer = TfidfVectorizer(tokenizer=_jieba_tokenize, token_pattern=None)
        matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, matrix).flatten()
    except Exception:
        return []

    ranked = sorted(zip(sims, store), key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in ranked[:top_k]:
        if score < min_score:
            continue
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "score": round(float(score), 3),
        })
    return results
