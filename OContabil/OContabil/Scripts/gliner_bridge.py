"""
OContabil — GLiNER 2 Bridge (CORRIGIDO)
Processes documents via GLiNER 2 and returns JSON extraction results.

Usage:
    python gliner_bridge.py <file_path> [document_type] [model_name] [threshold]

Output (stdout):
    JSON with extracted entities, metadata, and average confidence.
"""

import sys
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# ── CRITICAL: Force UTF-8 on Windows to prevent charmap encode errors ──
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or image file with optimized error handling."""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext == ".pdf":
            return _extract_pdf(file_path)
        elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            return _extract_image(file_path)
        elif ext in (".txt", ".xml", ".csv", ".json"):
            return _extract_text_file(file_path)
        else:
            return f"[ERRO: Formato nao suportado: {ext}]"
    except Exception as e:
        return f"[ERRO: Falha na extracao ({ext}): {str(e)}]"


def _extract_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF with layout preservation."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[ERRO: PyMuPDF nao instalado. Execute: pip install pymupdf]"
    
    doc = fitz.open(file_path)
    text_parts = []
    
    try:
        for page_num, page in enumerate(doc, 1):
            # Preservar layout melhora NER em documentos estruturados
            text = page.get_text("text", sort=True)  # sort=True mantém ordem de leitura
            if text.strip():
                text_parts.append(f"--- Pagina {page_num} ---\n{text}")
    finally:
        doc.close()
    
    return "\n\n".join(text_parts).strip()


def _extract_image(file_path: str) -> str:
    """Extract text from image using Tesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return "[ERRO: Pillow/pytesseract nao instalado]"
    
    img = Image.open(file_path)
    
    # Pré-processamento para melhorar OCR em documentos contábeis
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Configuração específica para documentos (modo 6 = assume único bloco de texto uniforme)
    custom_config = r'--oem 3 --psm 6 -l por'
    text = pytesseract.image_to_string(img, config=custom_config)
    
    return text.strip()


def _extract_text_file(file_path: str) -> str:
    """Extract text from plain text files."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    return "[ERRO: Nao foi possivel decodificar arquivo]"


def apply_threshold_filter(
    extraction_result: Dict[str, Any], 
    threshold: float
) -> Dict[str, Any]:
    """
    Filtra entidades abaixo do threshold e marca como null.
    Preserva estrutura do schema.
    """
    if not isinstance(extraction_result, dict):
        return extraction_result
    
    filtered = {}
    
    for key, value in extraction_result.items():
        if isinstance(value, dict) and "confidence" in value:
            # É uma entidade com confidence
            if float(value.get("confidence", 0)) >= threshold:
                filtered[key] = value
            else:
                # Abaixo do threshold: manter chave mas valor null (consistência de schema)
                filtered[key] = {} # Usar dict vazio no lugar de None para evitar erro de atribuição
        elif isinstance(value, list):
            # Lista de entidades ou valores aninhados
            filtered_list = []
            for item in value:
                if isinstance(item, dict) and "confidence" in item:
                    if float(item.get("confidence", 0)) >= threshold:
                        filtered_list.append(item)
                else:
                    filtered_list.append(apply_threshold_filter(item, threshold))
            filtered[key] = filtered_list
        elif isinstance(value, dict):
            # Dicionário aninhado (recursão)
            filtered[key] = apply_threshold_filter(value, threshold)
        else:
            filtered[key] = value
    
    return filtered


def process_with_gliner(
    text: str, 
    doc_type: str = "", 
    model_name: str = "fastino/gliner2-base-v1", 
    threshold: float = 0.5
) -> dict:
    """
    Process extracted text with GLiNER 2 model.
    
    CORREÇÃO: Aplica threshold via filtragem post-extraction (API GLiNER2 
    nao suporta threshold direto em extract_json).
    """
    try:
        from gliner2 import GLiNER2
        
        # Inicialização com cache de modelo (evita reload)
        extractor = GLiNER2.from_pretrained(model_name)
        
        # Schema contábil otimizado para notas fiscais brasileiras
        schema = {
            "nota_fiscal": [
                "cnpj_emitente::str::CNPJ da empresa emitente (formato XX.XXX.XXX/XXXX-XX)",
                "nome_emitente::str::Razao social completa do emitente",
                "cnpj_destinatario::str::CNPJ do destinatario",
                "numero_nota::str::Numero da nota fiscal (serie e numero)",
                "data_emissao::str::Data de emissao (DD/MM/AAAA)",
                "valor_total::str::Valor total em reais (R$ X.XXX,XX)",
                "valor_icms::str::Valor do ICMS destacado",
                "valor_ipi::str::Valor do IPI",
                "chave_acesso::str::Chave de acesso NF-e (44 digitos)",
                "natureza_operacao::str::Natureza da operacao",
                "descricao_produtos::str::Descricao dos produtos ou servicos"
            ]
        }

        # Extração com confidence scores
        raw_result = extractor.extract_json(
            text,
            schema,
            include_confidence=True
        )
        
        # 🎯 CORREÇÃO CRÍTICA: Aplicar threshold configurável
        filtered_result = apply_threshold_filter(raw_result, threshold)
        
        # Coletar estatísticas de confiança APÓS filtragem
        confidences = []
        entity_count = 0
        
        def collect_stats(obj):
            nonlocal entity_count
            if isinstance(obj, dict):
                if "confidence" in obj and "text" in obj:
                    confidences.append(float(obj["confidence"]))
                    if obj.get("text"):  # Só conta se tem valor
                        entity_count += 1
                else:
                    for v in obj.values():
                        collect_stats(v)
            elif isinstance(obj, list):
                for item in obj:
                    collect_stats(item)
        
        collect_stats(filtered_result)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        
        # Validar CNPJs extraídos (formato brasileiro)
        validation = validate_extraction(filtered_result)
        
        return {
            "success": True,
            "extraction": filtered_result,
            "avg_confidence": round(float(avg_confidence), 4),
            "max_confidence": round(float(max_confidence), 4),
            "min_confidence": round(float(min_confidence), 4),
            "entity_count": entity_count,
            "threshold_applied": threshold,  # ✅ Transparência para debug
            "text_length": len(text),
            "model": model_name,
            "validation": validation,
            "processing_method": "gliner2-json"
        }

    except (ImportError, OSError) as e:
        return extract_basic_fallback(text, str(e), threshold)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "text_length": len(text),
            "avg_confidence": 0.0,
            "threshold_requested": threshold
        }


def validate_extraction(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """Valida campos críticos da extração (CNPJs, datas, valores)."""
    validation = {
        "cnpj_emitente_valido": False,
        "cnpj_destinatario_valido": False,
        "data_valida": False,
        "campos_obrigatorios_presentes": 0,
        "campos_obrigatorios_total": 3  # CNPJ emitente, data, valor
    }
    
    try:
        nf = extraction.get("nota_fiscal", {})
        if isinstance(nf, list) and nf:
            nf = nf[0]  # Pegar primeira NF se for lista
        
        # Validar CNPJ emitente
        cnpj_emit = _extract_field_value(nf, "cnpj_emitente")
        if cnpj_emit and _validar_cnpj(cnpj_emit):
            validation["cnpj_emitente_valido"] = True
            validation["campos_obrigatorios_presentes"] += 1
        
        # Validar data
        data = _extract_field_value(nf, "data_emissao")
        if data and re.match(r"\d{2}/\d{2}/\d{4}", str(data)):
            validation["data_valida"] = True
            validation["campos_obrigatorios_presentes"] += 1
        
        # Validar valor
        valor = _extract_field_value(nf, "valor_total")
        if valor and re.search(r"\d+[\.,]\d{2}", str(valor)):
            validation["campos_obrigatorios_presentes"] += 1
            
    except Exception:
        pass
    
    return validation


def _extract_field_value(nf_data: Any, field: str) -> Optional[str]:
    """Extrai valor textual de campo que pode ser dict ou string."""
    if isinstance(nf_data, dict):
        val = nf_data.get(field)
        if isinstance(val, dict):
            return val.get("text", "")
        return str(val) if val else None
    return None


def _validar_cnpj(cnpj: str) -> bool:
    """Valida dígitos verificadores do CNPJ brasileiro."""
    cnpj = re.sub(r'[^0-9]', '', str(cnpj))
    if len(cnpj) != 14 or cnpj == cnpj[0] * 14:
        return False
    
    # Algoritmo de validação de CNPJ
    def calc_digit(pos):
        weights = [5,4,3,2,9,8,7,6,5,4,3,2] if pos == 1 else [6,5,4,3,2,9,8,7,6,5,4,3,2]
        total = sum(int(cnpj[i]) * w for i, w in enumerate(weights))
        digit = 11 - (total % 11)
        return 0 if digit > 9 else digit
    
    return (calc_digit(1) == int(cnpj[12]) and 
            calc_digit(2) == int(cnpj[13]))


def extract_basic_fallback(
    text: str, 
    reason: str = "", 
    threshold: float = 0.5
) -> dict:
    """Fallback: extração por regex quando GLiNER indisponível."""
    import re
    
    # Padrões otimizados para documentos contábeis brasileiros
    patterns = {
        "cnpj_emitente": r"(?:CNPJ[:\s]+|Emitente.*?)(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})",
        "cnpj_destinatario": r"(?:Destinat[aá]rio.*?CNPJ[:\s]+)(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})",
        "numero_nota": r"(?:N[°º]\s*|Nota Fiscal.*?)(\d{3,}\s*(?:S[eé]rie\s*\d+)?)",
        "data_emissao": r"(\d{2}/\d{2}/\d{4})",
        "valor_total": r"(?:VALOR TOTAL|Total.*?)[R$]\s*([\d\.,]+)",
        "valor_icms": r"(?:ICMS.*?)[R$]\s*([\d\.,]+)",
        "chave_acesso": r"(\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\s+\d{4})"
    }
    
    extraction = {}
    found_count = 0
    
    for field, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            extraction[field] = {
                "text": matches[0] if isinstance(matches[0], str) else matches[0][0],
                "confidence": 0.6  # Base confidence para regex
            }
            found_count += 1
    
    # Ajustar confidence baseado em quantos campos encontrou vs threshold
    base_confidence = min(0.75, 0.40 + (found_count * 0.08))
    meets_threshold = base_confidence >= threshold
    
    return {
        "success": True,
        "extraction": {"nota_fiscal": extraction if meets_threshold else {}},
        "avg_confidence": round(base_confidence, 4) if meets_threshold else 0.0,
        "entity_count": found_count if meets_threshold else 0,
        "text_length": len(text),
        "model": "regex-fallback",
        "threshold_applied": threshold,
        "threshold_met": meets_threshold,
        "note": f"GLiNER 2 indisponivel ({reason}). Usando extração por regex."
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False, 
            "error": "Uso: python gliner_bridge.py <arquivo> [tipo] [modelo] [threshold]"
        }))
        sys.exit(1)

    file_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "nota_fiscal"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "fastino/gliner2-base-v1"
    
    # Validação robusta do threshold
    try:
        threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        threshold = max(0.0, min(1.0, threshold))  # Clamp entre 0 e 1
    except ValueError:
        threshold = 0.5

    if not os.path.exists(file_path):
        print(json.dumps({
            "success": False, 
            "error": f"Arquivo nao encontrado: {file_path}"
        }))
        sys.exit(1)

    # Extração de texto
    text = extract_text_from_file(file_path)
    if text.startswith("[ERRO"):
        print(json.dumps({"success": False, "error": text}))
        sys.exit(1)

    # Processamento GLiNER
    result = process_with_gliner(text, doc_type, model_name, threshold)
    
    # Incluir amostra do texto para debug (limitado)
    result["ocr_sample"] = text[:1500] + "..." if len(text) > 1500 else text
    
    # Saída JSON estrita
    print(json.dumps(result, ensure_ascii=False, indent=None))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
