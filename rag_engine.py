"""
RAG Engine for Kakarot Study
Supports: PDF + Images (JPG, PNG, WEBP, HEIC)
Uses TF-IDF retrieval + Google Gemini vision + generation
"""

import re
import base64
import numpy as np
from typing import List, Tuple, Optional

import pypdf
from PIL import Image
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage


# ─── Supported file types ───────────────────────────────────────────────────

PDF_TYPES  = ["pdf"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif"]
ALL_TYPES   = PDF_TYPES + IMAGE_TYPES


# ─── Text Extraction from PDF ───────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    reader = pypdf.PdfReader(pdf_file)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(pages_text)


# ─── Text Extraction from Image (via Gemini Vision) ─────────────────────────

def extract_text_from_image(image_file, llm) -> str:
    """Use Gemini vision to OCR and extract all text/questions from an image."""
    image_bytes = image_file.read()
    image_file.seek(0)  # reset for re-use

    # Detect mime type
    ext = image_file.name.split(".")[-1].lower()
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "webp": "image/webp",
        "heic": "image/heic", "heif": "image/heif"
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"}
        },
        {
            "type": "text",
            "text": (
                "This is a question paper or study material image. "
                "Please extract ALL text from this image exactly as it appears — "
                "including every question, sub-question, answer, diagram label, "
                "formula, table, and any other text. "
                "Preserve numbering and structure. Output only the extracted text."
            )
        }
    ])

    response = llm.invoke([message])
    return response.content


# ─── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if current_len + len(words) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk[:]
            current_chunk = overlap_words
            current_len = len(current_chunk)
        current_chunk.extend(words)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c.split()) > 10]


# ─── TF-IDF Retriever ───────────────────────────────────────────────────────

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=10000,
            stop_words='english', sublinear_tf=True
        )
        self.chunks: List[str] = []
        self.matrix = None
        self.is_fitted = False

    def fit(self, chunks: List[str]):
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform(chunks)
        self.is_fitted = True

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.is_fitted or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[idx], float(scores[idx]))
            for idx in top_k_idx if scores[idx] > 0.05
        ]


# ─── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are **Kakarot Study AI** — an expert academic tutor that makes complex exam questions feel simple.

## Your Teaching Style:
1. **CLARITY FIRST** — Explain as if the student is hearing this for the first time
2. **REAL ANALOGIES** — Anchor abstract ideas to everyday things
3. **STEP-BY-STEP** — Break multi-part questions into numbered micro-steps
4. **TRICKS & SHORTCUTS** — Always share a memory trick, pattern, or shortcut
5. **PRECISION ON IMPORTANCE** — When flagged important, give exhaustive exam-ready depth

## Response Format:
- Use **bold** for key terms
- Numbered lists for steps/processes
- Bullet points for properties/features
- Include 💡 **Quick Trick** or mnemonic whenever possible
- End complex answers with ✅ **Summary**
- Add ⚠️ **Exam Tip** for exam-critical content

## Tone: Friendly, confident, like a helpful senior student.
Always base answers on the provided context. If something is not in the context, say so honestly."""


# ─── Prompt Builder ─────────────────────────────────────────────────────────

def build_messages(
    query: str,
    context_chunks: List[Tuple[str, float]],
    chat_history: List[dict],
    importance_level: str = "normal",
    image_b64: Optional[str] = None,
    image_mime: Optional[str] = None
) -> List:
    context_text = ""
    for i, (chunk, score) in enumerate(context_chunks, 1):
        context_text += f"\n--- Excerpt {i} (relevance: {score:.2f}) ---\n{chunk}\n"

    importance_note = {
        "high":   "\n🎯 PRIORITY MODE: VERY IMPORTANT — Give exhaustive answer: deep explanation, "
                  "multiple examples, common mistakes, memory tricks, exam tips.",
        "medium": "\n📌 Important to user — Give complete answer with at least one example.",
        "normal": ""
    }.get(importance_level, "")

    messages = []
    for turn in chat_history[-4:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    text_content = f"""{SYSTEM_PROMPT}

---

## Document Context:
{context_text if context_text else "No specific context found — answering from general knowledge."}

## Student Question:
{query}
{importance_note}

Please give a clear, structured, student-friendly answer."""

    # If image query — send image alongside text for visual context
    if image_b64 and image_mime:
        messages.append(HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            {"type": "text", "text": text_content}
        ]))
    else:
        messages.append(HumanMessage(content=text_content))

    return messages


# ─── Importance Detector ────────────────────────────────────────────────────

def detect_importance(query: str) -> str:
    q = query.lower()
    high   = ["very important", "most important", "highly important", "critical",
              "crucial", "key concept", "must know", "exam important",
              "frequently asked", "important topic", "!!!", "***"]
    medium = ["important", "focus on", "pay attention", "notable",
              "significant", "need to understand", "need to know"]
    for kw in high:
        if kw in q: return "high"
    for kw in medium:
        if kw in q: return "medium"
    return "normal"


# ─── Main RAG Pipeline ──────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, api_key: str):
        self.retriever   = TFIDFRetriever()
        self.llm         = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )
        self.document_loaded = False
        self.total_chunks    = 0
        self.doc_summary     = ""
        self.file_type       = None      # "pdf" or "image"
        # Store raw image for visual queries
        self._image_b64      = None
        self._image_mime     = None

    # ── Ingest PDF ──────────────────────────────────────────────────────────
    def ingest_pdf(self, pdf_file) -> dict:
        self.file_type = "pdf"
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text.strip():
            return {"success": False, "error": "Could not extract text from this PDF."}

        chunks = chunk_text(raw_text, chunk_size=400, overlap=80)
        if not chunks:
            return {"success": False, "error": "No usable text chunks found."}

        self.retriever.fit(chunks)
        self.document_loaded = True
        self.total_chunks    = len(chunks)
        self.doc_summary     = self._summarize(raw_text[:3000])

        return {
            "success": True, "chunks": len(chunks),
            "pages": raw_text.count("[Page "),
            "summary": self.doc_summary, "file_type": "pdf"
        }

    # ── Ingest Image ────────────────────────────────────────────────────────
    def ingest_image(self, image_file) -> dict:
        self.file_type = "image"

        # Store image for visual queries
        image_bytes = image_file.read()
        image_file.seek(0)
        ext = image_file.name.split(".")[-1].lower()
        mime_map = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                    "png":"image/png","webp":"image/webp",
                    "heic":"image/heic","heif":"image/heif"}
        self._image_mime = mime_map.get(ext, "image/jpeg")
        self._image_b64  = base64.b64encode(image_bytes).decode("utf-8")

        # Extract text via Gemini vision OCR
        try:
            raw_text = extract_text_from_image(image_file, self.llm)
        except Exception as e:
            return {"success": False, "error": f"Image reading failed: {str(e)}"}

        if not raw_text.strip():
            return {"success": False, "error": "Could not extract text from image."}

        chunks = chunk_text(raw_text, chunk_size=400, overlap=80)
        if not chunks:
            # Image had very little text — still usable via visual queries
            chunks = [raw_text]

        self.retriever.fit(chunks)
        self.document_loaded = True
        self.total_chunks    = len(chunks)
        self.doc_summary     = self._summarize(raw_text[:2000])

        return {
            "success": True, "chunks": len(chunks),
            "pages": 1, "summary": self.doc_summary,
            "file_type": "image", "extracted_text": raw_text[:500]
        }

    # ── Smart ingest (auto-detect type) ─────────────────────────────────────
    def ingest(self, uploaded_file) -> dict:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            return self.ingest_pdf(uploaded_file)
        elif ext in IMAGE_TYPES:
            return self.ingest_image(uploaded_file)
        else:
            return {"success": False, "error": f"Unsupported file type: .{ext}"}

    # ── Summarize ────────────────────────────────────────────────────────────
    def _summarize(self, text_snippet: str) -> str:
        try:
            msg = self.llm.invoke([HumanMessage(
                content=f"In 2 sentences, describe what subject/topic this document covers:\n\n{text_snippet}"
            )])
            return msg.content
        except Exception:
            return "Document loaded and ready for questions."

    # ── Query ────────────────────────────────────────────────────────────────
    def query(self, user_query: str, chat_history: List[dict], top_k: int = 5) -> dict:
        if not self.document_loaded:
            return {
                "answer": "⚠️ Please upload a file first before asking questions.",
                "context_used": [], "importance": "normal"
            }

        importance     = detect_importance(user_query)
        k              = 7 if importance == "high" else top_k
        context_chunks = self.retriever.retrieve(user_query, k=k)

        # Pass image bytes for image files (gives Gemini visual context)
        messages = build_messages(
            query          = user_query,
            context_chunks = context_chunks,
            chat_history   = chat_history,
            importance_level = importance,
            image_b64      = self._image_b64 if self.file_type == "image" else None,
            image_mime     = self._image_mime if self.file_type == "image" else None
        )

        response = self.llm.invoke(messages)
        return {
            "answer":       response.content,
            "context_used": context_chunks,
            "importance":   importance
        }
