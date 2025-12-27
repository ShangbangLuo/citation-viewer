# app.py —— Python 3.8 兼容版，后端缓存 paragraph embeddings

import re
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

# ---------------- 基本路径 ----------------
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_ROOT = STATIC_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -------------- lazy 加载 embedding 模型 ----------------
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        # 和前端一致的模型
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model


def embed_sentences(texts: List[str]) -> List[List[float]]:
    """
    返回 list[list[float]]，方便写到 JSON 中。
    """
    model = get_embed_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


# -------------- PDF → 段落 + figure label --------------


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def extract_paragraphs_from_pdf(pdf_path: Path):
    """
    用 PyMuPDF blocks 抽取较长的段落
    返回: [{id, page, text}]
    """
    doc = fitz.open(str(pdf_path))
    paragraphs = []
    pid = 0

    for page_i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text) < 80:
                continue  # 跳过 caption / 杂讯
            pid += 1
            paragraphs.append(
                {
                    "id": "p{}".format(pid),
                    "page": page_i + 1,
                    "text": text,
                }
            )

    doc.close()
    return paragraphs


def extract_figure_labels(text: str):
    """
    抓 Figure / Table 编号，比如：
      Fig. 1, FIGURE 3, table 2
    """
    labels = set()
    for m in re.finditer(r"(figure|fig\.|table)\s*\d+", text, re.I):
        labels.add(m.group(0).upper().replace("FIG.", "FIGURE"))
    return sorted(labels)


def build_cited_index_with_embeddings(pdf_path: Path) -> dict:
    """
    读 cited_paper.pdf → 抽段落 → 算 embedding → 生成 index dict
    """
    paragraphs = extract_paragraphs_from_pdf(pdf_path)
    texts = [p["text"] for p in paragraphs]

    embeddings = embed_sentences(texts)  # list[list[float]]

    idx_paras = []
    for p, emb in zip(paragraphs, embeddings):
        idx_paras.append(
            {
                "id": p["id"],
                "page": p["page"],
                "text": p["text"],
                "norm": normalize(p["text"]),
                "figures": extract_figure_labels(p["text"]),
                "embedding": emb,  # ✅ 每段一个 embedding 向量
            }
        )

    return {
        "source_pdf": pdf_path.name,
        "created_at": datetime.utcnow().isoformat(),
        "paragraphs": idx_paras,
    }


# -------------- 路由：根路径 → 上传页 --------------


@app.get("/")
def root():
    # 打开时默认跳转到上传页面（你之前的 upload.html 保持不动）
    return RedirectResponse(url="/static/upload.html")


# -------------- API: 上传一对 PDF + 生成 index + 缓存 embedding --------------


@app.post("/api/upload_pair")
async def upload_pair(
    original: UploadFile = File(...),
    cited: UploadFile = File(...),
):
    """
    前端 upload.html 调这个：
      - original: 原文 PDF
      - cited: 被引用 PDF
    后端：
      - 为本次上传生成一个 file_id
      - 保存到 static/uploads/<file_id>/original.pdf / cited.pdf
      - 对 cited.pdf 生成 cited_index.json（包含每段 embedding）
      - 返回 file_id + viewer_url
    """
    if not original.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Original must be PDF")
    if not cited.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Cited must be PDF")

    file_id = secrets.token_hex(8)
    folder = UPLOAD_ROOT / file_id
    folder.mkdir(parents=True, exist_ok=True)

    orig_path = folder / "original.pdf"
    cited_path = folder / "cited.pdf"
    index_path = folder / "cited_index.json"

    # 保存 PDF
    with open(orig_path, "wb") as f:
        f.write(await original.read())
    with open(cited_path, "wb") as f:
        f.write(await cited.read())

    # 构建 index + embedding
    try:
        index_obj = build_cited_index_with_embeddings(cited_path)
    except Exception as e:
        # 解析失败时把文件删掉
        try:
            if orig_path.exists():
                orig_path.unlink()
            if cited_path.exists():
                cited_path.unlink()
            if index_path.exists():
                index_path.unlink()
            # 如果目录空了尝试删目录
            if folder.exists():
                folder.rmdir()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to parse cited PDF: {}".format(e))

    # 写 index JSON
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, indent=2, ensure_ascii=False)

    return {
        "file_id": file_id,
        "viewer_url": "/static/viewer.html?file_id={}".format(file_id),
    }


# -------------- API: viewer.html 根据 file_id 获取文件路径 --------------


@app.get("/api/file_info/{file_id}")
def file_info(file_id: str):
    folder = UPLOAD_ROOT / file_id
    orig_path = folder / "original.pdf"
    cited_path = folder / "cited.pdf"
    index_path = folder / "cited_index.json"

    if not (orig_path.exists() and cited_path.exists() and index_path.exists()):
        raise HTTPException(status_code=404, detail="file_id not found")

    return {
        "file_id": file_id,
        "original_url": "/static/uploads/{}/original.pdf".format(file_id),
        "cited_url": "/static/uploads/{}/cited.pdf".format(file_id),
        "index_url": "/static/uploads/{}/cited_index.json".format(file_id),
    }
