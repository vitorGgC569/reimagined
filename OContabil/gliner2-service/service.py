"""Microsserviço FastAPI que expõe extração de entidades via GLiNER2.

Endpoints:
- POST /extract   { text, doc_type, fields[] } → { entities, avg_confidence }
- POST /extract-file (multipart/form-data: file, doc_type, fields)
- GET  /health    → { status: "ok" }

Pode ser consumido pela API .NET (OContabil.API) ou diretamente pelo desktop
quando o Python local não estiver disponível. Modelo é carregado lazy na
primeira requisição e mantido em memória.
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Importação lazy para acelerar startup
_model: Any = None
_model_name = os.environ.get("OCONTABIL_MODEL", "fastino/gliner2-base-v1")


def get_model():
    global _model
    if _model is None:
        from gliner2 import GLiNER2

        _model = GLiNER2.from_pretrained(_model_name)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo no startup para evitar latência na primeira requisição
    try:
        get_model()
    except Exception as exc:
        print(f"[gliner2-service] aviso: modelo não carregado no boot ({exc})")
    yield


app = FastAPI(title="OContabil GLiNER2 Service", version="1.0.0", lifespan=lifespan)


class ExtractRequest(BaseModel):
    text: str
    doc_type: str = "Outro"
    fields: list[str] = []
    threshold: float = 0.5


class Entity(BaseModel):
    label: str
    text: str
    score: float


class ExtractResponse(BaseModel):
    entities: list[Entity]
    avg_confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _model_name, "loaded": _model is not None}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    if not req.text.strip():
        raise HTTPException(400, "Texto vazio.")
    if not req.fields:
        raise HTTPException(400, "Informe ao menos um campo para extrair.")

    model = get_model()
    raw = model.extract_entities(req.text, req.fields, threshold=req.threshold)
    ents = [
        Entity(label=e["label"], text=e["text"], score=float(e.get("score", 0.0)))
        for e in raw.get("entities", [])
    ]
    avg = (sum(e.score for e in ents) / len(ents)) if ents else 0.0
    return ExtractResponse(entities=ents, avg_confidence=avg)


@app.post("/extract-file", response_model=ExtractResponse)
async def extract_file(
    file: UploadFile = File(...),
    doc_type: str = Form("Outro"),
    fields: str = Form(""),
    threshold: float = Form(0.5),
) -> ExtractResponse:
    """Aceita PDF/imagem, faz OCR (PyMuPDF / Tesseract) e roda extração."""
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    if not field_list:
        raise HTTPException(400, "Informe ao menos um campo via 'fields' (CSV).")

    data = await file.read()
    text = _extract_text(data, file.filename or "")
    if not text.strip():
        raise HTTPException(422, "Não foi possível extrair texto do arquivo.")

    model = get_model()
    raw = model.extract_entities(text, field_list, threshold=threshold)
    ents = [
        Entity(label=e["label"], text=e["text"], score=float(e.get("score", 0.0)))
        for e in raw.get("entities", [])
    ]
    avg = (sum(e.score for e in ents) / len(ents)) if ents else 0.0
    return ExtractResponse(entities=ents, avg_confidence=avg)


def _extract_text(data: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=data, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            return ""
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            return pytesseract.image_to_string(img, lang="por+eng")
        except Exception:
            return ""
    # Assume texto puro
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")
