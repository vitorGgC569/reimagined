#!/usr/bin/env python3
"""Treino conversacional PT-BR do NSOS Mamba-only.

Este script orquestra o motor autoral existente. Ele não usa PyTorch e não
reimplementa o modelo. O fluxo completo é:

    prepare -> corpus SQLite auditável -> tokenizer OX3 -> shards uint16
    train   -> NSOS C++/HIP -> checkpoints transacionais -> avaliação
    run     -> prepare + train

Exemplos:

    # Validação rápida e offline do pipeline:
    python scripts/train_ptbr_conversational.py run --preset smoke --fixture

    # Piloto real de 71M / 100M tokens vistos:
    python scripts/train_ptbr_conversational.py run --preset pilot --device gpu

    # Continuar automaticamente do último checkpoint íntegro:
    python scripts/train_ptbr_conversational.py train --preset pilot --device gpu

O preset ``full`` materializa a receita de 1,4 bilhão de tokens vistos:
1,15B base + 150M continuação conversacional + 50M SFT por duas épocas.
"""

from __future__ import annotations

import argparse
import ast
from array import array
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_module import load_native_module, resolve_native_build_dir


SCRIPT_VERSION = 4
DB_SCHEMA_VERSION = 1
PACK_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 3
CORPUS_IDENTITY_FORMAT_VERSION = 2
CORPUS_RECIPE_VERSION = 1
CORPUS_MANIFEST_FORMAT_VERSION = 1
SFT_ESTIMATE_POLICY_VERSION = "assistant-target-capped-v1"
SFT_INGEST_HEADROOM = 1.05

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|bos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|sep|>",
]
EOS_TOKEN = SPECIAL_TOKENS[0]
BOS_TOKEN = SPECIAL_TOKENS[2]
DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente conversacional brasileiro. Responda somente em "
    "português do Brasil, com clareza, honestidade e naturalidade. Quando não "
    "souber algo, diga que não sabe em vez de inventar."
)

DATASET_REPOS = {
    "base": "Polygl0t/gigaverbo-v2",
    "sft": "Polygl0t/gigaverbo-v2-sft",
    "ultrachat": "recogna-nlp/UltrachatBR",
}

# A lista é deliberadamente conservadora. "conditional" só é aceito quando o
# operador escolhe explicitamente a política research-reviewed.
BASE_SOURCE_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("carolina", "deny", "CC BY-NC-SA na fonte oficial"),
    ("xl-sum", "deny", "CC BY-NC-SA"),
    ("xlsum", "deny", "CC BY-NC-SA"),
    ("bactrian", "deny", "CC BY-NC"),
    ("brwac", "deny", "licença não declarada"),
    ("instruct-ptbr", "deny", "termos derivados de Llama"),
    ("hplt", "allow", "CC0 conforme card do GigaVerbo-v2"),
    ("crawlpt", "allow", "CC0 conforme card do GigaVerbo-v2"),
    ("oscar", "allow", "CC0 conforme card do GigaVerbo-v2"),
    ("blogset", "allow", "Apache-2.0 conforme card do GigaVerbo-v2"),
    ("quati", "allow", "CC BY 4.0 conforme card do GigaVerbo-v2"),
    ("legalpt", "allow", "CC BY 4.0 conforme card do GigaVerbo-v2"),
    ("gpt4all", "allow", "Apache-2.0 conforme card do GigaVerbo-v2"),
    ("ultrachat", "allow", "MIT conforme card do GigaVerbo-v2"),
    ("fineweb", "conditional", "ODC-By + Common Crawl ToU"),
    ("culturax", "conditional", "ODC-By + Common Crawl ToU"),
    ("mc4", "conditional", "ODC-By + Common Crawl ToU"),
    ("common_crawl", "conditional", "Common Crawl ToU e direitos por página"),
    ("common-crawl", "conditional", "Common Crawl ToU e direitos por página"),
    ("wikipedia", "conditional", "CC BY-SA 3.0 e obrigações de atribuição"),
)

# --- Mistura de fontes da fase base ---------------------------------------
#
# A auditoria de 2026-08-06 encontrou 100% do corpus base vindo de HPLT: as
# demais fontes do gigaverbo-v2 são `conditional` e a política
# commercial-strict as rejeita. Uma família única de crawl expõe o modelo aos
# mesmos padrões editoriais, de extração e de domínio.
#
# As cotas são tetos por fonte, aplicados sobre tokens estimados. A soma passa
# de 1,0 de propósito: se uma fonte se esgota antes da meta, as outras podem
# preencher o restante sem travar a preparação.
# Famílias de dados, não repositórios.
#
# O dry-run de 2026-08-07 mostrou que o gigaverbo-v2 traz mC4 por DOIS
# repositórios distintos (`legacy-datasets/mc4` e `thegoodfellas/mc4-pt`,
# ambos subset `mc4_pt`). Cotas por repositório dariam a duas versões do mesmo
# Common Crawl orçamentos independentes, e o corpus receberia ~30% de mC4
# achando que recebeu duas fontes diferentes.
#
# A cota atua sobre a família. A ordem importa: o primeiro marcador que casar
# define a família.
SOURCE_FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("hplt", "hplt"),
    ("fineweb", "fineweb"),
    ("mc4", "mc4"),          # cobre os dois repositórios de mC4
    ("quati", "quati"),
    ("culturax", "culturax"),
    ("oscar", "oscar"),
    ("wikipedia", "wikipedia"),
)

# Tetos provisórios, redesenhados sobre a composição REAL medida no dry-run
# (FineWeb-2 50,4% / mC4 29,5% / HPLT 19,4%). A soma passa de 1,0 de
# propósito: são tetos, não alvos, e uma família esgotada não deve travar a
# preparação.
#
# FineWeb-2 continua o maior componente — já vem filtrado por qualidade
# educacional — mas deixa de poder virar quase todo o corpus. mC4 bruto fica
# limitado a um quarto. Nenhum destes números é validado: o próximo dry-run
# existe para tentar invalidá-los.
#
# `quati` entrou na v3. A amostra densa por shard de 2026-08-07 (36.000 linhas
# consecutivas de 9 shards) encontrou FineWeb 41,3% / HPLT 29,4% / quati 18,0%
# / mC4 11,4%, e revelou que `quati` não casava com nenhum marcador de família:
# `mix_bucket_for_source` devolvia None e a verificação de cota era pulada por
# inteiro. As duas corridas de 10M nunca o viram porque param antes daquela
# região do stream — ele estrearia sem teto direto nos 80M.
#
# Não é lixo incidental: o card do GigaVerbo-v2 traz `unicamp-dl/quati` como
# CC BY-4.0, montado a partir de sites brasileiros selecionados por qualidade
# e relevância, o que o torna material desejável para um LLM PT-BR. O teto de
# 20% não afirma que 20% seja ótimo; é fusível contra uma região posterior do
# stream fazer o quati dominar em silêncio.
BASE_SOURCE_MIX = {
    "fineweb": 0.60,
    "mc4": 0.25,
    "hplt": 0.30,
    "quati": 0.20,
}

# Faixa condicional de qualidade.
#
# `edu_int_score` 3 concentra 79% do corpus. Descartá-lo trocaria um corpus
# grande e heterogêneo por um pequeno, limpo e estreito; aceitá-lo sem
# distinção mantém o problema. Score 3 entra sob exigências adicionais e com
# teto de participação, para que seu valor possa ser medido separadamente.
QUOTA_POLICY_VERSION = "ptbr-quota-source-family-post-dedup-v3"
# Versão do algoritmo de empacotamento. Muda quando a forma de transformar
# documentos selecionados em shards de tokens muda, mesmo que os documentos
# sejam exatamente os mesmos.
PACKING_VERSION = "ptbr-packing-v1"
QUALITY_MIN_SCORE = 3
QUALITY_CONDITIONAL_SCORE = 3
QUALITY_CONDITIONAL_MAX_SHARE = 0.40
# Exigências extras aplicadas apenas à faixa condicional.
CONDITIONAL_MIN_CHARS = 600
CONDITIONAL_MIN_WORDS = 120
CONDITIONAL_MAX_REPEATED_LINE_RATIO = 0.15
CONDITIONAL_MAX_NON_LATIN_RATIO = 0.02

SFT_CONFIG_WEIGHTS = {
    "general": 0.65,
    "system_prompts": 0.10,
    "rewriting": 0.10,
    "summarization": 0.10,
    "ultrachat": 0.05,
}


PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "description": "Validação funcional curta; não mede qualidade.",
        "layers": 2,
        "d_model": 96,
        "target_vocab": 768,
        "seq_len": 64,
        "causal_batch": 1,
        "sft_batch": 2,
        "base_tokens": 8_192,
        "continuation_tokens": 4_096,
        "sft_unique_tokens": 1_024,
        "sft_epochs": 1,
        "tokenizer_chars": 200_000,
        "token_shard_size": 8_192,
        "sft_records_per_shard": 64,
        "lr": 8.0e-4,
        "warmup_steps": 4,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "use_gradient_checkpointing": False,
        "progressive_qat": False,
        "checkpoint_every_steps": 2,
        "checkpoint_every_minutes": 5.0,
        "validation_windows": 2,
    },
    "pilot": {
        "description": "Mamba-only 71M; piloto de 100M tokens vistos.",
        "layers": 16,
        "d_model": 768,
        "target_vocab": 16_384,
        "seq_len": 512,
        "causal_batch": 1,
        "sft_batch": 1,
        "base_tokens": 80_000_000,
        "continuation_tokens": 12_000_000,
        "sft_unique_tokens": 4_000_000,
        "sft_epochs": 2,
        "tokenizer_chars": 80_000_000,
        "token_shard_size": 2_000_000,
        "sft_records_per_shard": 2_000,
        "lr": 3.0e-4,
        "warmup_steps": 2_000,
        "weight_decay": 0.10,
        "dropout": 0.0,
        "use_gradient_checkpointing": False,
        "progressive_qat": False,
        "checkpoint_every_steps": 1_000,
        "checkpoint_every_minutes": 30.0,
        "validation_windows": 8,
    },
    "dryrun": {
        # Calibração de POLÍTICAS de dados, não de aprendizado. Existe para
        # observar distribuição real — sobreposição entre fontes, cauda dos
        # clusters, fronteira do near-dedup, comportamento das cotas e do
        # score 3 condicional — sem gastar os 80M tokens do pilot.
        #
        # Toda política é idêntica à do pilot de propósito: mesmos filtros,
        # mesmas cotas, mesmo dedup, mesmo schema, mesma vocab-alvo. Só o
        # volume muda, para que o que se aprender aqui transfira.
        "description": (
            "Calibração de políticas de dados sobre 10M tokens reais; "
            "arquitetura idêntica ao pilot."
        ),
        "layers": 16,
        "d_model": 768,
        "target_vocab": 16_384,
        "seq_len": 512,
        "causal_batch": 1,
        "sft_batch": 1,
        "base_tokens": 10_000_000,
        "continuation_tokens": 1_500_000,
        "sft_unique_tokens": 500_000,
        "sft_epochs": 2,
        "tokenizer_chars": 10_000_000,
        "token_shard_size": 2_000_000,
        "sft_records_per_shard": 2_000,
        "lr": 3.0e-4,
        "warmup_steps": 2_000,
        "weight_decay": 0.10,
        "dropout": 0.0,
        "use_gradient_checkpointing": False,
        "progressive_qat": False,
        "checkpoint_every_steps": 1_000,
        "checkpoint_every_minutes": 30.0,
        "validation_windows": 8,
    },
    "full": {
        "description": "Receita integral de 1,4B tokens vistos.",
        "layers": 16,
        "d_model": 768,
        "target_vocab": 16_384,
        "seq_len": 512,
        "causal_batch": 1,
        "sft_batch": 1,
        "base_tokens": 1_150_000_000,
        "continuation_tokens": 150_000_000,
        "sft_unique_tokens": 50_000_000,
        "sft_epochs": 2,
        "tokenizer_chars": 512_000_000,
        "token_shard_size": 4_000_000,
        "sft_records_per_shard": 4_000,
        "lr": 3.0e-4,
        "warmup_steps": 4_000,
        "weight_decay": 0.10,
        "dropout": 0.0,
        "use_gradient_checkpointing": False,
        "progressive_qat": False,
        "checkpoint_every_steps": 1_000,
        "checkpoint_every_minutes": 30.0,
        "validation_windows": 16,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename deliberately short. Checkpoint generation
    # paths are already descriptive and can approach Win32's legacy 260-char
    # boundary; embedding the target name, PID, thread id and timestamp in the
    # temporary basename made an otherwise valid checkpoint fail with ENOENT.
    # mkstemp gives an exclusive, same-directory file (therefore an atomic
    # os.replace target) without consuming that path-length budget.
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            descriptor = -1
            json.dump(
                payload, handle, indent=2, ensure_ascii=False, sort_keys=True
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            # Cleanup is best-effort because the target may already be
            # durably published, but the residual temporary file is still an
            # observable operational warning rather than a silent failure.
            print(
                f"[io] aviso: falha ao remover temporário {temporary} "
                f"de {path}: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )


def fsync_file(path: Path) -> None:
    # Windows rejects FlushFileBuffers/os.fsync for a descriptor opened with
    # read-only access (EBADF).  Every caller owns a newly written checkpoint
    # artifact, so reopen it read/write without truncation and flush the same
    # bytes durably on both Windows and POSIX.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Persist directory entries where the platform exposes directory fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_component(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")
    return cleaned[:80] or "checkpoint"


BOILERPLATE_PATTERNS = [
    r"(?i)\bleia\s+mais\b.*$",
    r"(?i)\bleia\s+també[mm]\b.*$",
    r"(?i)\bveja\s+també[mm]\b.*$",
    r"(?i)\bclique\s+aqui\b.*$",
    r"(?i)\binscreva-se\s+no\s+canal\b.*$",
    r"(?i)\bconfira\s+també[mm]\b.*$",
    r"(?i)\btodos\s+os\s+direitos\s+reservados\b.*$",
    r"(?i)\bpolítica\s+de\s+privacidade\b.*$",
    r"(?i)\btermo(?:s)?\s+de\s+uso\b.*$",
    r"(?i)\baceitar\s+all?\s+cookies?\b.*$",
    r"(?i)\bcompartilhe\s+(?:no|via)\s+(?:whatsapp|facebook|twitter|x|telegram)\b.*$",
    r"(?i)\bfoto:\s*reprodução.*$",
    r"(?i)\bimagem:\s*divulgação.*$",
    r"(?i)\bpublicado\s+em\s+\d{2}/\d{2}/\d{4}.*$",
    r"(?i)<[^>]+>",
    r"&nbsp;|&amp;|&quot;|&lt;|&gt;",
]

COMPILED_BOILERPLATE = [re.compile(p) for p in BOILERPLATE_PATTERNS]

# Versão dos filtros. Entra no manifesto de proveniência: um corpus preparado
# com filtros diferentes não é comparável a outro, e o wind tunnel exige que
# todos os braços vejam exatamente o mesmo material.
FILTER_VERSION = "ptbr-filters-v2-2026-08-07"

# --- PII e links -----------------------------------------------------------
#
# A auditoria de 2026-08-06 mediu, na fase base já filtrada, 4,20% de documentos
# com e-mail e 7,32% com URL. E-mail é PII e sai do corpus; URL raramente
# carrega sinal linguístico e frequentemente marca bloco de navegação.
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
# Telefone BR em formatos comuns; PII pelo mesmo motivo do e-mail.
PHONE_BR_RE = re.compile(
    r"(?<!\d)(?:\+55\s?)?(?:\(?\d{2}\)?[\s.-]?)?9?\d{4}[\s.-]?\d{4}(?!\d)"
)
# Documentos brasileiros. Nunca deveriam estar num corpus de treino.
CPF_RE = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
CNPJ_RE = re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}(?!\d)")

# Padrões que a auditoria encontrou sobrevivendo aos filtros originais.
EXTRA_BOILERPLATE_PATTERNS = [
    r"(?i)\b\d+\s+coment[áa]rios?\b.*$",
    r"(?i)©\s*\d{4}.*$",
    r"(?i)\b(assine|cadastre-se|receba)\s+.{0,30}\bnewsletter\b.*$",
    r"(?i)\b(siga-nos|curta nossa página|nos siga)\b.*$",
    r"(?i)\b(página inicial|home)\s*[>|»/]\s*\w+.*$",
    r"(?i)\b(conteúdo exclusivo|assine para continuar|"
    r"faça login para ler)\b.*$",
    r"(?i)^\s*(tags?|assuntos?|palavras-chave)\s*:",
    r"(?i)\b\d+\s*min(utos?)?\s+de\s+leitura\b.*$",
    r"(?i)\b(aceit\w+|gerenciar)\s+cookies?\b.*$",
]
COMPILED_EXTRA_BOILERPLATE = [
    re.compile(p) for p in EXTRA_BOILERPLATE_PATTERNS
]


def strip_pii_and_links(line: str) -> str:
    """Remove PII e links de uma linha, devolvendo '' se sobrar só ruído.

    Remoção em vez de máscara: um marcador como ``[EMAIL]`` viraria token
    aprendido e reapareceria na geração.
    """
    if CPF_RE.search(line) or CNPJ_RE.search(line):
        return ""  # linha com documento não é recuperável
    cleaned = EMAIL_RE.sub(" ", line)
    cleaned = PHONE_BR_RE.sub(" ", cleaned)
    without_links = URL_RE.sub(" ", cleaned)
    # Uma linha que era majoritariamente link não sobrevive à remoção.
    if len(without_links.strip()) < 0.4 * len(cleaned.strip() or " "):
        return ""
    return " ".join(without_links.split())


def normalize_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u00a0": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t\f\v]+", " ", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        line_clean = re.sub(r"<[^>]+>", "", line_clean)
        line_clean = re.sub(r"&nbsp;|&amp;|&quot;|&lt;|&gt;", " ", line_clean)
        skip = False
        for pat in COMPILED_BOILERPLATE:
            if pat.search(line_clean):
                skip = True
                break
        if not skip:
            for pat in COMPILED_EXTRA_BOILERPLATE:
                if pat.search(line_clean):
                    skip = True
                    break
        if not skip:
            # PII e links saem depois do boilerplate: uma linha que já era
            # navegação foi descartada inteira, e o que sobra é texto real do
            # qual só o contato precisa ser removido.
            line_clean = strip_pii_and_links(line_clean)
        if not skip and line_clean:
            cleaned_lines.append(line_clean)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def stable_document_id(
    source: str, subset: str, revision: str, record_id: str
) -> str:
    """Identidade de ORIGEM do documento, estável entre reconstruções.

    Deliberadamente independente do texto: derivar do conteúdo limpo faria a
    identidade mudar a cada FILTER_VERSION, impedindo comparar duas
    reconstruções do mesmo material.
    """
    return stable_hash_text(
        f"doc\0{source}\0{subset}\0{revision}\0{record_id}"
    )


def content_hash_of(text: str) -> str:
    """Hash do conteúdo já normalizado — base do dedup exato."""
    collapsed = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return stable_hash_text(collapsed)


def pii_flags(text: str) -> str:
    """Flags compactas de PII presente, para auditoria posterior."""
    flags = []
    if EMAIL_RE.search(text):
        flags.append("email")
    if PHONE_BR_RE.search(text):
        flags.append("phone")
    if CPF_RE.search(text):
        flags.append("cpf")
    if CNPJ_RE.search(text):
        flags.append("cnpj")
    if URL_RE.search(text):
        flags.append("url")
    return ",".join(flags)


def latin_language_score(text: str) -> float:
    """Proxy barato de confiança de idioma: fração de letras latinas."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return round(latin / len(letters), 4)


def dedup_shingles(text: str, size: int = 8, stride: int = 4) -> List[int]:
    """Shingles de palavras usados como assinatura de quase-duplicata."""
    words = re.findall(r"\w+", text.casefold(), re.UNICODE)
    if len(words) < size:
        return []
    return [
        int.from_bytes(
            hashlib.blake2b(
                " ".join(words[i:i + size]).encode("utf-8"), digest_size=8
            ).digest(),
            "big",
        )
        for i in range(0, len(words) - size + 1, stride)
    ]


# Confiança editorial por fonte, usada para eleger o representante de um
# cluster de duplicatas. Wikipedia é curada; HPLT e CulturaX são crawls.
SOURCE_EDITORIAL_RANK = {"wikipedia": 3, "culturax": 2, "hplt": 1}
# Licença menos restritiva vence: um cluster representado por um documento
# `allow` mantém o corpus utilizável sob a política mais estrita possível.
LICENSE_RANK = {"allow": 3, "conditional": 2, "unknown": 1, "deny": 0}


def source_family_of(source: str, subset: str = "") -> str | None:
    """Família de dados da linha, ou None se não reconhecida.

    Famílias agrupam repositórios diferentes do mesmo upstream — os dois
    repositórios de mC4 são uma família só.
    """
    haystack = f"{source} {subset}".lower()
    for marker, family in SOURCE_FAMILIES:
        if marker in haystack:
            return family
    return None


def mix_bucket_for_source(source: str, subset: str = "") -> str | None:
    """Bucket de cota da linha, ou None quando a família não tem teto."""
    family = source_family_of(source, subset)
    if family is None or family not in BASE_SOURCE_MIX:
        return None
    return family


def conditional_quality_accepts(text: str) -> Tuple[bool, str]:
    """Exigências extras para a faixa de qualidade condicional (score 3).

    Devolve (aceito, motivo_da_rejeição).
    """
    if len(text) < CONDITIONAL_MIN_CHARS:
        return False, "curto_em_caracteres"
    words = text.split()
    if len(words) < CONDITIONAL_MIN_WORDS:
        return False, "curto_em_palavras"
    if EMAIL_RE.search(text) or PHONE_BR_RE.search(text):
        return False, "pii_residual"
    if CPF_RE.search(text) or CNPJ_RE.search(text):
        return False, "documento_residual"
    if URL_RE.search(text):
        return False, "url_residual"
    for pat in COMPILED_BOILERPLATE:
        if pat.search(text):
            return False, "boilerplate_residual"
    for pat in COMPILED_EXTRA_BOILERPLATE:
        if pat.search(text):
            return False, "boilerplate_extra_residual"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        counts: Dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        repeated = sum(n for n in counts.values() if n > 1)
        if repeated / len(lines) > CONDITIONAL_MAX_REPEATED_LINE_RATIO:
            return False, "linhas_repetidas"
    letters = [c for c in text if c.isalpha()]
    if letters:
        non_latin = sum(
            1 for c in letters
            if "LATIN" not in unicodedata.name(c, "")
        )
        if non_latin / len(letters) > CONDITIONAL_MAX_NON_LATIN_RATIO:
            return False, "nao_latino"
    return True, ""


def classify_base_source(source: str, subset: str = "") -> Tuple[str, str]:
    haystack = f"{source} {subset}".lower()
    for marker, decision, reason in BASE_SOURCE_RULES:
        if marker in haystack:
            return decision, reason
    return "unknown", "fonte sem regra explícita"


def source_allowed(source: str, subset: str, policy: str) -> Tuple[bool, str, str]:
    decision, reason = classify_base_source(source, subset)
    if decision == "deny" or decision == "unknown":
        return False, decision, reason
    if decision == "conditional" and policy == "commercial-strict":
        return False, decision, reason
    return True, decision, reason


def deterministic_split(fingerprint: str, eval_basis_points: int = 50) -> str:
    bucket = int(fingerprint[:8], 16) % 10_000
    return "eval" if bucket < eval_basis_points else "train"


def estimate_tokens_from_text(text: str) -> int:
    # Estimativa apenas para interromper o download. O pack final registra a
    # contagem exata produzida pelo tokenizer NSOS.
    return max(1, len(text.encode("utf-8")) // 4)


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    normalized: List[Dict[str, str]] = []
    aliases = {
        "human": "user",
        "humano": "user",
        "usuario": "user",
        "usuário": "user",
        "gpt": "assistant",
        "assistente": "assistant",
        "system": "system",
        "sistema": "system",
        "user": "user",
        "assistant": "assistant",
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = aliases.get(str(message.get("role", "")).strip().lower())
        content = normalize_text(str(message.get("content", "")))
        if role and content:
            normalized.append({"role": role, "content": content})
    if not normalized:
        return []
    if normalized[0]["role"] != "system":
        normalized.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    return normalized


def parse_ultrachat_conversation(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, str) or not value or len(value) > 2_000_000:
        return []
    parsed: Any
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, list):
        return []
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]
    for turn in parsed:
        if not isinstance(turn, dict):
            continue
        user = normalize_text(str(turn.get("humano", "")))
        assistant = normalize_text(str(turn.get("assistente", "")))
        if user and assistant:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
    return messages if len(messages) >= 3 else []


def format_chat(messages: Sequence[Dict[str, str]], include_eos: bool = True) -> str:
    pieces = [BOS_TOKEN]
    for message in messages:
        role = message["role"]
        pieces.append(f"<|{role}|>\n{message['content']}\n")
    if include_eos:
        pieces.append(EOS_TOKEN)
    return "".join(pieces)


def supervised_examples(
    messages: Sequence[Dict[str, str]],
) -> Iterator[Tuple[str, str]]:
    history: List[Dict[str, str]] = []
    for message in messages:
        if message["role"] == "assistant":
            prompt = format_chat(history, include_eos=False) + "<|assistant|>\n"
            yield prompt, message["content"]
        history.append(message)


def estimate_sft_target_tokens(
    messages: Sequence[Dict[str, str]], seq_len: int
) -> int:
    """Estima somente tokens supervisionados, respeitando o corte do pack."""
    max_answer = max(8, int(seq_len) // 2)
    estimate = 0
    for _prompt, answer in supervised_examples(messages):
        answer_estimate = estimate_tokens_from_text(answer) + 1
        estimate += min(answer_estimate, max_answer)
    return max(estimate, 1)


@dataclass
class CorpusCounts:
    documents: int = 0
    estimated_tokens: int = 0


class CorpusStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Hugging Face tokenizers consumes train_from_iterator on a worker
        # thread. The preparation stage has already finished all writes when
        # that happens, so cross-thread read access is safe and intentional.
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                fingerprint TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                split TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                subset_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                estimated_tokens INTEGER NOT NULL,
                quality REAL NOT NULL,
                license_decision TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS records_phase_split
                ON records(phase, split, fingerprint);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._migrate_provenance_columns()
        current = self.get_meta("schema_version")
        if current is None:
            self.set_meta("schema_version", str(DB_SCHEMA_VERSION))
        elif int(current) != DB_SCHEMA_VERSION:
            raise RuntimeError(
                f"Corpus DB schema {current} != esperado {DB_SCHEMA_VERSION}"
            )

    # Colunas de proveniência. Adicionadas por ALTER TABLE para que um corpus
    # já preparado continue legível: as linhas antigas ficam com NULL, o que é
    # informação honesta ("não registrado") e não um valor inventado.
    #
    # document_id deriva da identidade de ORIGEM, nunca do texto limpo: se
    # dependesse do conteúdo, mudar FILTER_VERSION mudaria a identidade do
    # documento e destruiria a comparabilidade entre reconstruções.
    # content_hash cobre o conteúdo normalizado, separadamente.
    PROVENANCE_COLUMNS = (
        ("document_id", "TEXT"),
        ("source_revision", "TEXT"),
        ("source_url_hash", "TEXT"),
        ("license_class", "TEXT"),
        ("quality_score", "INTEGER"),
        ("language_score", "REAL"),
        ("filter_version", "TEXT"),
        ("content_hash", "TEXT"),
        ("dedup_cluster", "TEXT"),
        # Duas propriedades distintas, deliberadamente separadas:
        # is_canonical  -> qual cópia representa o cluster de duplicatas;
        # is_training_selected -> se esse representante entra no treino.
        # Um documento pode ser canônico e válido e ainda ficar de fora porque
        # a cota da sua fonte fechou. O packer exige as duas.
        ("is_canonical", "INTEGER"),
        ("is_training_selected", "INTEGER"),
        ("original_char_count", "INTEGER"),
        ("clean_char_count", "INTEGER"),
        # Denominador estável para NLL normalizada no A/B de tokenizer.
        # "caractere" é ambíguo em Unicode (grafemas compostos, marcas
        # combinantes), então bits/byte evita a discussão: o byte UTF-8 é a
        # mesma unidade para qualquer vocabulário comparado.
        ("clean_utf8_byte_count", "INTEGER"),
        ("raw_pii_flags", "TEXT"),
        ("residual_pii_flags", "TEXT"),
        ("rejection_reason", "TEXT"),
    )

    def _migrate_provenance_columns(self) -> None:
        existing = {
            row[1] for row in self.connection.execute(
                "PRAGMA table_info(records)"
            )
        }
        added = []
        for name, sql_type in self.PROVENANCE_COLUMNS:
            if name in existing:
                continue
            self.connection.execute(
                f"ALTER TABLE records ADD COLUMN {name} {sql_type}"
            )
            added.append(name)
        if added:
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS records_content_hash "
                "ON records(content_hash)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS records_dedup_cluster "
                "ON records(dedup_cluster)"
            )
            self.connection.commit()
            print(
                f"[corpus] proveniência migrada: +{len(added)} colunas "
                f"({', '.join(added[:4])}…)",
                flush=True,
            )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
        self.connection.commit()

    def insert(
        self,
        *,
        fingerprint: str,
        phase: str,
        split: str,
        category: str,
        source: str,
        subset: str,
        payload: str,
        estimated_tokens: int,
        quality: float,
        license_decision: str,
        provenance: Dict[str, Any] | None = None,
    ) -> bool:
        prov = provenance or {}
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO records(
                fingerprint, phase, split, category, source, subset_name,
                payload, estimated_tokens, quality, license_decision,
                document_id, source_revision, source_url_hash, license_class,
                quality_score, language_score, filter_version, content_hash,
                dedup_cluster, is_canonical, original_char_count,
                clean_char_count, clean_utf8_byte_count, raw_pii_flags, residual_pii_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fingerprint,
                phase,
                split,
                category,
                source,
                subset,
                payload,
                max(int(estimated_tokens), 1),
                float(quality),
                license_decision,
                prov.get("document_id"),
                prov.get("source_revision"),
                prov.get("source_url_hash"),
                prov.get("license_class"),
                prov.get("quality_score"),
                prov.get("language_score"),
                prov.get("filter_version", FILTER_VERSION),
                prov.get("content_hash"),
                # dedup_cluster e is_canonical são preenchidos pelo passe de
                # deduplicação global, que só pode rodar depois que todas as
                # fontes passaram pela mesma normalização.
                None,
                None,
                prov.get("original_char_count"),
                prov.get("clean_char_count"),
                prov.get("clean_utf8_byte_count"),
                prov.get("raw_pii_flags"),
                prov.get("residual_pii_flags"),
            ),
        )
        return cursor.rowcount == 1

    def counts(self, phase: str, split: str = "train") -> CorpusCounts:
        row = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(estimated_tokens), 0)
            FROM records WHERE phase = ? AND split = ?
            """,
            (phase, split),
        ).fetchone()
        return CorpusCounts(int(row[0]), int(row[1]))

    def category_counts(self, phase: str, split: str = "train") -> Dict[str, CorpusCounts]:
        rows = self.connection.execute(
            """
            SELECT category, COUNT(*), COALESCE(SUM(estimated_tokens), 0)
            FROM records WHERE phase = ? AND split = ?
            GROUP BY category ORDER BY category
            """,
            (phase, split),
        )
        return {
            str(category): CorpusCounts(int(documents), int(tokens))
            for category, documents, tokens in rows
        }

    def iter_records(
        self, phase: str, split: str = "train"
    ) -> Iterator[Tuple[str, str, str]]:
        """Documentos que entram no pack.

        Fail-closed nas duas propriedades: um documento só é empacotado se for
        o representante canônico do seu cluster E tiver sido selecionado pela
        cota final. Ausência de rejeição não implica seleção — um corpus que
        ainda não passou pelo dedup/seleção tem esses campos em NULL e não
        deve produzir pack algum, em vez de empacotar tudo silenciosamente.

        O split de avaliação não passa por cota (é holdout, não material de
        treino), mas ainda exige canonicidade para não avaliar contra
        duplicatas.
        """
        if split == "train":
            condition = "is_canonical = 1 AND is_training_selected = 1"
        else:
            condition = "is_canonical = 1"
        rows = self.connection.execute(
            f"""
            SELECT fingerprint, category, payload
            FROM records
            WHERE phase = ? AND split = ? AND {condition}
            ORDER BY fingerprint
            """,
            (phase, split),
        )
        for fingerprint, category, payload in rows:
            yield str(fingerprint), str(category), str(payload)

    def commit(self) -> None:
        self.connection.commit()


def migrate_sft_estimates_to_supervised_targets(
    store: CorpusStore, seq_len: int
) -> None:
    """Converte workspaces antigos de tokens da conversa para tokens-alvo."""
    policy = f"{SFT_ESTIMATE_POLICY_VERSION}:seq-{int(seq_len)}"
    if store.get_meta("sft_estimate_policy") == policy:
        return
    updates: List[Tuple[int, str]] = []
    rows = store.connection.execute(
        "SELECT fingerprint, payload FROM records WHERE phase = 'sft'"
    )
    for fingerprint, payload in rows:
        messages = json.loads(str(payload))
        updates.append(
            (
                estimate_sft_target_tokens(messages, int(seq_len)),
                str(fingerprint),
            )
        )
    if updates:
        store.connection.executemany(
            "UPDATE records SET estimated_tokens = ? WHERE fingerprint = ?",
            updates,
        )
    store.connection.execute(
        """
        INSERT INTO metadata(key, value) VALUES('sft_estimate_policy', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (policy,),
    )
    store.commit()
    print(
        "[prepare:sft] estimativas migradas para tokens-alvo do assistente "
        f"(registros={len(updates):,})",
        flush=True,
    )


def resolve_dataset_revisions() -> Dict[str, str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Preparação real requer huggingface_hub: pip install huggingface_hub"
        ) from exc
    api = HfApi()
    return {
        name: str(api.dataset_info(repo, files_metadata=False).sha)
        for name, repo in DATASET_REPOS.items()
    }


def insert_fixture_corpus(store: CorpusStore, preset: Dict[str, Any]) -> None:
    base_templates = [
        (
            "O Brasil possui uma enorme diversidade cultural, linguística e regional. "
            "Uma conversa útil respeita essa diversidade e evita generalizações."
        ),
        (
            "Planejar a semana começa por listar compromissos, definir prioridades "
            "realistas e reservar tempo para descanso."
        ),
        (
            "Quando uma informação não puder ser confirmada, a resposta mais segura "
            "é reconhecer a incerteza e explicar como verificá-la."
        ),
        (
            "Uma alimentação equilibrada combina variedade, moderação e orientação "
            "profissional quando existem necessidades específicas."
        ),
        (
            "A comunicação clara usa frases naturais, exemplos concretos e um tom "
            "adequado ao contexto da pessoa."
        ),
    ]
    index = 0
    while store.counts("base").estimated_tokens < int(preset["base_tokens"]):
        text = (
            base_templates[index % len(base_templates)]
            + f" Este é o documento brasileiro de validação número {index}."
        )
        fingerprint = stable_hash_text(f"fixture-base:{index}:{text}")
        store.insert(
            fingerprint=fingerprint,
            phase="base",
            split="train" if index % 20 else "eval",
            category="fixture",
            source="fixture_autoral",
            subset="fixture",
            payload=text,
            estimated_tokens=estimate_tokens_from_text(text),
            quality=5.0,
            license_decision="autoral",
        )
        index += 1

    dialogues = [
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Como posso organizar melhor a minha semana?"},
            {
                "role": "assistant",
                "content": (
                    "Comece anotando os compromissos fixos. Depois escolha até três "
                    "prioridades por dia e deixe intervalos para imprevistos."
                ),
            },
        ],
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Você tem certeza dessa informação?"},
            {
                "role": "assistant",
                "content": (
                    "Sem uma fonte verificável, eu não deveria afirmar certeza. "
                    "Posso explicar o que sei e indicar o que precisa ser confirmado."
                ),
            },
        ],
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Estou cansado. O que posso fazer agora?"},
            {
                "role": "assistant",
                "content": (
                    "Se puder, faça uma pausa curta, beba água e reduza a próxima "
                    "tarefa a um passo pequeno. Se o cansaço persistir, vale procurar "
                    "orientação de um profissional de saúde."
                ),
            },
        ],
    ]
    for phase in ("continuation", "sft"):
        target = int(
            preset[
                "continuation_tokens" if phase == "continuation" else "sft_unique_tokens"
            ]
        )
        index = 0
        while store.counts(phase).estimated_tokens < target:
            messages = dialogues[index % len(dialogues)]
            payload = stable_json(messages)
            fingerprint = stable_hash_text(f"fixture-{phase}:{index}:{payload}")
            store.insert(
                fingerprint=fingerprint,
                phase=phase,
                split="train" if index % 20 else "eval",
                category="fixture",
                source="fixture_autoral",
                subset="fixture",
                payload=payload,
                estimated_tokens=estimate_tokens_from_text(format_chat(messages)),
                quality=5.0,
                license_decision="autoral",
            )
            index += 1
    store.commit()


def ingest_base(
    store: CorpusStore,
    preset: Dict[str, Any],
    revision: str,
    seed: int,
    license_policy: str,
    shuffle_buffer: int,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Preparação requer datasets: pip install datasets") from exc

    target_train = int(preset["base_tokens"])
    target_eval = max(20_000, target_train // 200)
    current_train = store.counts("base", "train")
    current_eval = store.counts("base", "eval")
    if (
        current_train.estimated_tokens >= target_train
        and current_eval.estimated_tokens >= target_eval
    ):
        return

    cursor_key = f"cursor:base:{revision}:{seed}:{license_policy}"
    rows_seen = int(store.get_meta(cursor_key) or "0")
    dataset = load_dataset(
        DATASET_REPOS["base"],
        "default",
        split="train",
        streaming=True,
        revision=revision,
    )
    # shuffle_buffer=0 desliga o embaralhamento e entrega linhas consecutivas.
    # Amostra ALEATÓRIA e amostra DENSA respondem perguntas diferentes: com
    # shuffle, duas cópias de um documento duplicado quase nunca caem na mesma
    # fatia, e o near-dedup fica sem candidatos para exercitar (o dry-run de
    # 2026-08-07 viu 153 candidatos em 10 mil documentos). Linhas consecutivas
    # preservam a vizinhança do crawl, onde as duplicatas de fato coocorrem.
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(
            seed=seed, buffer_size=max(shuffle_buffer, 1_000)
        )
    if rows_seen:
        dataset = dataset.skip(rows_seen)

    # Tetos por fonte e por faixa de qualidade, em tokens estimados.
    mix_budget = {
        bucket: int(target_train * share)
        for bucket, share in BASE_SOURCE_MIX.items()
    }
    mix_used: Dict[str, int] = {bucket: 0 for bucket in BASE_SOURCE_MIX}
    conditional_budget = int(target_train * QUALITY_CONDITIONAL_MAX_SHARE)
    conditional_used = 0
    rejected: Dict[str, int] = {}

    accepted_since_commit = 0
    for row in dataset:
        rows_seen += 1
        text = normalize_text(str(row.get("text", "")))
        if len(text) < 200:
            rejected["curto"] = rejected.get("curto", 0) + 1
            continue
        quality = int(row.get("edu_int_score", 0) or 0)
        toxic = int(row.get("toxic_int_score", 5) or 5)
        if quality < QUALITY_MIN_SCORE or toxic > 2:
            rejected["qualidade_ou_toxicidade"] = (
                rejected.get("qualidade_ou_toxicidade", 0) + 1
            )
            continue
        source = str(row.get("source", ""))
        subset = str(row.get("subset", ""))
        allowed, decision, _reason = source_allowed(
            source, subset, license_policy
        )
        if not allowed:
            rejected["licenca"] = rejected.get("licenca", 0) + 1
            continue

        token_estimate_early = int(row.get("token_count", 0) or 0)
        if token_estimate_early <= 0:
            token_estimate_early = estimate_tokens_from_text(text)

        # Cota por fonte: impede que uma única família de crawl domine.
        bucket = mix_bucket_for_source(source, subset)
        if bucket is not None:
            if mix_used[bucket] + token_estimate_early > mix_budget[bucket]:
                rejected[f"cota_{bucket}"] = (
                    rejected.get(f"cota_{bucket}", 0) + 1
                )
                continue

        # Faixa condicional: score no mínimo aceitável entra só se passar por
        # exigências mais duras, e com teto de participação.
        if quality <= QUALITY_CONDITIONAL_SCORE:
            if conditional_used + token_estimate_early > conditional_budget:
                rejected["cota_score3"] = (
                    rejected.get("cota_score3", 0) + 1
                )
                continue
            ok, why = conditional_quality_accepts(text)
            if not ok:
                rejected[f"score3_{why}"] = (
                    rejected.get(f"score3_{why}", 0) + 1
                )
                continue
        row_id = str(row.get("id", ""))
        fingerprint = stable_hash_text(
            f"base\0{source}\0{subset}\0{row_id}\0{text}"
        )
        raw_text = str(row.get("text", ""))
        provenance = {
            "document_id": stable_document_id(
                source, subset, revision, row_id
            ),
            "source_revision": revision,
            "source_url_hash": stable_hash_text(str(row.get("url", ""))),
            "license_class": decision,
            "quality_score": quality,
            "language_score": latin_language_score(text),
            "filter_version": FILTER_VERSION,
            "content_hash": content_hash_of(text),
            "original_char_count": len(raw_text),
            "clean_char_count": len(text),
            "clean_utf8_byte_count": len(text.encode("utf-8")),
            "raw_pii_flags": pii_flags(raw_text),
            "residual_pii_flags": pii_flags(text),
        }
        split = deterministic_split(fingerprint)
        token_estimate = int(row.get("token_count", 0) or 0)
        if token_estimate <= 0:
            token_estimate = estimate_tokens_from_text(text)
        if store.insert(
            fingerprint=fingerprint,
            phase="base",
            split=split,
            category="pretraining",
            source=source,
            subset=subset,
            payload=text,
            estimated_tokens=token_estimate,
            quality=float(quality),
            license_decision=decision,
            provenance=provenance,
        ):
            accepted_since_commit += 1
            if bucket is not None:
                mix_used[bucket] += token_estimate
            if quality <= QUALITY_CONDITIONAL_SCORE:
                conditional_used += token_estimate
        if accepted_since_commit >= 250:
            store.commit()
            store.set_meta(cursor_key, str(rows_seen))
            accepted_since_commit = 0
            current_train = store.counts("base", "train")
            current_eval = store.counts("base", "eval")
            mix_text = " ".join(
                f"{name}={mix_used[name] * 100 // max(mix_budget[name], 1)}%"
                for name in sorted(mix_used)
            )
            top_rejections = " ".join(
                f"{k}={v:,}"
                for k, v in sorted(
                    rejected.items(), key=lambda kv: -kv[1]
                )[:4]
            )
            print(
                "[prepare:base] "
                f"train={current_train.estimated_tokens:,}/{target_train:,} "
                f"eval={current_eval.estimated_tokens:,}/{target_eval:,} "
                f"seen={rows_seen:,} | cotas {mix_text} "
                f"score3={conditional_used * 100 // max(conditional_budget, 1)}%"
                f" | rejeitados {top_rejections}",
                flush=True,
            )
            if (
                current_train.estimated_tokens >= target_train
                and current_eval.estimated_tokens >= target_eval
            ):
                break
    store.commit()
    store.set_meta(cursor_key, str(rows_seen))
    # Contadores de rejeição do ingest: os estágios anteriores ao banco não
    # são reconstituíveis depois, então precisam ser persistidos agora para o
    # funil poder responder "por que esta fonte encolheu?".
    store.set_meta(
        "ingest_counters:base",
        json.dumps(
            {
                "rows_seen": rows_seen,
                "rejected": rejected,
                "source_tokens_at_ingest": mix_used,
                "conditional_tokens_at_ingest": conditional_used,
                "filter_version": FILTER_VERSION,
                "license_policy": license_policy,
            },
            sort_keys=True,
        ),
    )
    store.commit()


def _sft_phase_bucket(fingerprint: str) -> str:
    # Partição disjunta 60/40. Continuação e SFT nunca reutilizam o mesmo
    # diálogo, mesmo quando vêm do mesmo config.
    return "continuation" if int(fingerprint[8:16], 16) % 5 < 3 else "sft"


def ingest_sft_config(
    store: CorpusStore,
    *,
    config: str,
    revision: str,
    seed: int,
    continuation_target: int,
    sft_target: int,
    seq_len: int,
    shuffle_buffer: int,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Preparação requer datasets: pip install datasets") from exc

    cursor_key = f"cursor:sft:{config}:{revision}:{seed}"
    rows_seen = int(store.get_meta(cursor_key) or "0")
    dataset = load_dataset(
        DATASET_REPOS["sft"],
        config,
        split="train",
        streaming=True,
        revision=revision,
    ).shuffle(seed=seed, buffer_size=max(shuffle_buffer, 1_000))
    if rows_seen:
        dataset = dataset.skip(rows_seen)

    accepted_since_commit = 0
    for row in dataset:
        rows_seen += 1
        score = float(row.get("instruct_score", 0.0) or 0.0)
        int_score = int(row.get("instruct_int_score", 0) or 0)
        if score < 4.0 and int_score < 4:
            continue
        messages = normalize_messages(row.get("messages"))
        if not messages or not any(m["role"] == "assistant" for m in messages):
            continue
        payload = stable_json(messages)
        fingerprint = stable_hash_text(f"sft\0{config}\0{payload}")
        phase = _sft_phase_bucket(fingerprint)
        phase_target = continuation_target if phase == "continuation" else sft_target
        phase_category_tokens = store.category_counts(phase, "train").get(
            config, CorpusCounts()
        ).estimated_tokens
        if phase_category_tokens >= phase_target:
            other = "sft" if phase == "continuation" else "continuation"
            other_target = sft_target if other == "sft" else continuation_target
            other_category_tokens = store.category_counts(other, "train").get(
                config, CorpusCounts()
            ).estimated_tokens
            if other_category_tokens < other_target:
                phase = other
                phase_target = other_target
            else:
                break
        split = deterministic_split(fingerprint, eval_basis_points=100)
        if phase == "sft":
            token_estimate = estimate_sft_target_tokens(messages, seq_len)
        else:
            token_estimate = int(row.get("token_count", 0) or 0)
            if token_estimate <= 0:
                token_estimate = estimate_tokens_from_text(format_chat(messages))
        if store.insert(
            fingerprint=fingerprint,
            phase=phase,
            split=split,
            category=config,
            source=DATASET_REPOS["sft"],
            subset=config,
            payload=payload,
            estimated_tokens=token_estimate,
            quality=max(score, float(int_score)),
            license_decision="Apache-2.0 aggregate SFT",
        ):
            accepted_since_commit += 1
        if accepted_since_commit >= 100:
            store.commit()
            store.set_meta(cursor_key, str(rows_seen))
            accepted_since_commit = 0
            print(
                f"[prepare:sft:{config}] "
                f"continuation={store.counts('continuation').estimated_tokens:,}/"
                f"{continuation_target:,} "
                f"sft={store.counts('sft').estimated_tokens:,}/{sft_target:,}",
                flush=True,
            )
    store.commit()
    store.set_meta(cursor_key, str(rows_seen))


def ingest_ultrachat(
    store: CorpusStore,
    *,
    revision: str,
    seed: int,
    continuation_target: int,
    sft_target: int,
    seq_len: int,
    shuffle_buffer: int,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Preparação requer datasets: pip install datasets") from exc

    cursor_key = f"cursor:ultrachat:{revision}:{seed}"
    rows_seen = int(store.get_meta(cursor_key) or "0")
    dataset = load_dataset(
        DATASET_REPOS["ultrachat"],
        "default",
        split="train",
        streaming=True,
        revision=revision,
    ).shuffle(seed=seed, buffer_size=max(shuffle_buffer, 1_000))
    if rows_seen:
        dataset = dataset.skip(rows_seen)
    accepted_since_commit = 0
    for row in dataset:
        rows_seen += 1
        messages = parse_ultrachat_conversation(row.get("conversa"))
        if not messages:
            continue
        payload = stable_json(messages)
        fingerprint = stable_hash_text(f"ultrachat\0{payload}")
        phase = _sft_phase_bucket(fingerprint)
        phase_target = continuation_target if phase == "continuation" else sft_target
        phase_category_tokens = store.category_counts(phase, "train").get(
            "ultrachat", CorpusCounts()
        ).estimated_tokens
        if phase_category_tokens >= phase_target:
            other = "sft" if phase == "continuation" else "continuation"
            other_target = sft_target if other == "sft" else continuation_target
            other_category_tokens = store.category_counts(other, "train").get(
                "ultrachat", CorpusCounts()
            ).estimated_tokens
            if other_category_tokens < other_target:
                phase = other
            else:
                break
        split = deterministic_split(fingerprint, eval_basis_points=100)
        token_estimate = (
            estimate_sft_target_tokens(messages, seq_len)
            if phase == "sft"
            else estimate_tokens_from_text(format_chat(messages))
        )
        if store.insert(
            fingerprint=fingerprint,
            phase=phase,
            split=split,
            category="ultrachat",
            source=DATASET_REPOS["ultrachat"],
            subset="default",
            payload=payload,
            estimated_tokens=token_estimate,
            quality=4.0,
            license_decision="MIT conforme card do GigaVerbo-v2",
        ):
            accepted_since_commit += 1
        if accepted_since_commit >= 100:
            store.commit()
            store.set_meta(cursor_key, str(rows_seen))
            accepted_since_commit = 0
    store.commit()
    store.set_meta(cursor_key, str(rows_seen))


def ingest_sft(
    store: CorpusStore,
    preset: Dict[str, Any],
    revisions: Dict[str, str],
    seed: int,
    shuffle_buffer: int,
) -> None:
    continuation_total = int(preset["continuation_tokens"])
    sft_total = int(preset["sft_unique_tokens"])
    seq_len = int(preset["seq_len"])
    migrate_sft_estimates_to_supervised_targets(store, seq_len)
    for index, (config, weight) in enumerate(SFT_CONFIG_WEIGHTS.items()):
        continuation_target = math.ceil(continuation_total * weight)
        sft_target = math.ceil(
            sft_total * weight * SFT_INGEST_HEADROOM
        )
        if config == "ultrachat":
            ingest_ultrachat(
                store,
                revision=revisions["ultrachat"],
                seed=seed + 500 + index,
                continuation_target=continuation_target,
                sft_target=sft_target,
                seq_len=seq_len,
                shuffle_buffer=shuffle_buffer,
            )
        else:
            ingest_sft_config(
                store,
                config=config,
                revision=revisions["sft"],
                seed=seed + 500 + index,
                continuation_target=continuation_target,
                sft_target=sft_target,
                seq_len=seq_len,
                shuffle_buffer=shuffle_buffer,
            )


# --- Parâmetros do near-dedup ---------------------------------------------
#
# Dois shingles em comum não bastam: citações, rodapés, texto jurídico,
# fórmulas editoriais e notícias republicadas parcialmente compartilham
# trechos longos sem serem duplicatas. O critério é proporcional ao tamanho
# do menor documento (containment), com piso absoluto.
DEDUP_MIN_SHARED_SHINGLES = 4
DEDUP_CONTAINMENT_THRESHOLD = 0.60
# Teto de candidatos examinados por documento: protege o custo no corpus real
# sem alterar o resultado nos casos típicos (o índice devolve os candidatos em
# ordem determinística).
DEDUP_MAX_CANDIDATES = 32
DEDUP_VERSION = "ptbr-dedup-anchored-containment-v1"


def deduplicate_corpus_globally(
    store: CorpusStore, phase: str = "base"
) -> Dict[str, Any]:
    """Dedup exato + quase-duplicata ENTRE fontes, após a mesma normalização.

    Precisa rodar depois de toda a ingestão: HPLT, CulturaX e Wikipedia podem
    conter a mesma página vinda de pipelines diferentes, e isso só é visível
    quando todas passaram pelo mesmo `normalize_text`.

    Não apaga nada. Marca `dedup_cluster` e elege um representante canônico
    por regra determinística, de modo que uma reconstrução produza exatamente
    os mesmos clusters e os mesmos representantes.
    """
    rows = store.connection.execute(
        "SELECT fingerprint, document_id, source, subset_name, payload, "
        "       license_decision, quality, estimated_tokens, content_hash "
        "FROM records WHERE phase=? ORDER BY fingerprint",
        (phase,),
    ).fetchall()
    if not rows:
        return {"documents": 0}

    # Cluster por ÂNCORA, não por union-find transitivo.
    #
    # Union-find puro encadeia: A~B, B~C, C~D colapsa A e D num só cluster
    # ainda que A e D não se pareçam. Em corpus grande isso produz
    # megaclusters falsos que removeriam material legítimo em massa.
    #
    # Aqui cada cluster tem uma âncora fixa (o primeiro documento a criá-lo, na
    # ordem determinística de fingerprint) e um candidato só entra se for
    # semelhante À ÂNCORA. A semelhança não é transitiva por construção.
    cluster_of: List[int] = [-1] * len(rows)
    anchors: List[int] = []
    anchor_shingles: List[set] = []
    exact_pairs = 0
    near_pairs = 0
    # Instrumentação da decisão de near-dedup. A ancoragem evita chaining mas
    # pode fragmentar famílias reais (B entra por A, C não entra porque se
    # parece com B e não com A). Estes contadores mostram se a fragmentação
    # está em nível aceitável ou se o critério está rejeitando candidatos
    # plausíveis em massa.
    candidates_considered = 0
    candidates_accepted = 0
    accepted_containment: List[float] = []
    borderline_rejected: List[float] = []
    near_by_size: Dict[str, int] = {}
    docs_by_size: Dict[str, int] = {}

    def size_bucket_of(text: str) -> str:
        words = len(text.split())
        if words < 200:
            return "<200"
        if words < 500:
            return "200-500"
        if words < 1000:
            return "500-1000"
        return ">1000"

    for row in rows:
        bucket_name = size_bucket_of(row[4] or "")
        docs_by_size[bucket_name] = docs_by_size.get(bucket_name, 0) + 1

    # Camada 1: dedup exato pelo conteúdo normalizado — sempre agrupa.
    by_content: Dict[str, int] = {}
    for index, row in enumerate(rows):
        digest = row[8] or content_hash_of(row[4] or "")
        first = by_content.get(digest)
        if first is None:
            by_content[digest] = index
        else:
            cluster_of[index] = cluster_of[first]
            exact_pairs += 1

    # Camada 2: quase-duplicata por containment sobre o menor documento.
    shingle_index: Dict[int, List[int]] = {}
    for index, row in enumerate(rows):
        if cluster_of[index] != -1:
            continue  # já resolvido como duplicata exata
        shingles = set(dedup_shingles(row[4] or ""))
        if not shingles:
            cluster_of[index] = len(anchors)
            anchors.append(index)
            anchor_shingles.append(shingles)
            continue

        # Candidatos: clusters que compartilham ao menos um shingle.
        counts: Dict[int, int] = {}
        for shingle in shingles:
            for candidate in shingle_index.get(shingle, ())[
                :DEDUP_MAX_CANDIDATES
            ]:
                counts[candidate] = counts.get(candidate, 0) + 1

        best_cluster = -1
        best_containment = 0.0
        for candidate, shared in sorted(counts.items()):
            if shared < DEDUP_MIN_SHARED_SHINGLES:
                continue
            candidates_considered += 1
            reference = anchor_shingles[candidate]
            denominator = min(len(shingles), len(reference))
            if denominator <= 0:
                continue
            containment = shared / denominator
            # Amostra da fronteira do classificador: os pares logo abaixo do
            # limiar são os que dizem se 0,60 está calibrado.
            if 0.50 <= containment < DEDUP_CONTAINMENT_THRESHOLD:
                borderline_rejected.append(round(containment, 3))
            if containment > best_containment:
                best_containment = containment
                best_cluster = candidate

        if (best_cluster >= 0
                and best_containment >= DEDUP_CONTAINMENT_THRESHOLD):
            cluster_of[index] = best_cluster
            near_pairs += 1
            candidates_accepted += 1
            accepted_containment.append(round(best_containment, 3))
            size_bucket = size_bucket_of(row[4] or "")
            near_by_size[size_bucket] = near_by_size.get(size_bucket, 0) + 1
        else:
            cluster_of[index] = len(anchors)
            anchors.append(index)
            anchor_shingles.append(shingles)
            for shingle in shingles:
                shingle_index.setdefault(shingle, []).append(
                    cluster_of[index]
                )

    # Duplicatas exatas herdaram o cluster do primeiro; as que apareceram antes
    # de o primeiro receber cluster ficam resolvidas aqui.
    for index, row in enumerate(rows):
        if cluster_of[index] == -1:
            digest = row[8] or content_hash_of(row[4] or "")
            cluster_of[index] = cluster_of[by_content[digest]]

    clusters: Dict[int, List[int]] = {}
    for index in range(len(rows)):
        clusters.setdefault(cluster_of[index], []).append(index)

    def canonical_key(index: int):
        row = rows[index]
        source_l = f"{row[2]} {row[3]}".lower()
        editorial = max(
            (rank for name, rank in SOURCE_EDITORIAL_RANK.items()
             if name in source_l),
            default=0,
        )
        # Ordem de desempate declarada e total: licença menos restritiva,
        # maior qualidade, maior confiança editorial, mais conteúdo útil e,
        # por fim, o document_id — que garante determinismo absoluto.
        return (
            -LICENSE_RANK.get(row[5], 0),
            -float(row[6] or 0.0),
            -editorial,
            -len(row[4] or ""),
            row[1] or row[0],
        )

    updates = []
    duplicates = 0
    for members in clusters.values():
        representative = min(members, key=canonical_key)
        cluster_id = rows[representative][1] or rows[representative][0]
        for index in members:
            is_canonical = 1 if index == representative else 0
            if not is_canonical:
                duplicates += 1
            updates.append((cluster_id, is_canonical, rows[index][0]))

    store.connection.executemany(
        "UPDATE records SET dedup_cluster=?, is_canonical=? "
        "WHERE fingerprint=?",
        updates,
    )
    store.commit()

    kept = len(rows) - duplicates
    sizes = sorted(len(m) for m in clusters.values())
    tokens_removed = sum(
        int(rows[i][7] or 0)
        for members in clusters.values()
        for i in members
        if i != min(members, key=canonical_key)
    )
    # Matriz de sobreposição por par de fontes, em documentos e tokens. Revela
    # quanto de uma fonte é informação NOVA: se CulturaX perde massa por
    # overlap com HPLT, "30% CulturaX" antes do dedup significa outra coisa.
    cross_source = 0
    pair_docs: Dict[str, int] = {}
    pair_tokens: Dict[str, int] = {}

    def source_bucket(raw: str, sub: str) -> str:
        return mix_bucket_for_source(raw or "", sub or "") or "outra"

    for members in clusters.values():
        buckets = {source_bucket(rows[i][2], rows[i][3]) for i in members}
        if len(buckets) > 1:
            cross_source += 1
        representative = min(members, key=canonical_key)
        rep_bucket = source_bucket(
            rows[representative][2], rows[representative][3]
        )
        for i in members:
            if i == representative:
                continue
            key = f"{source_bucket(rows[i][2], rows[i][3])}->{rep_bucket}"
            pair_docs[key] = pair_docs.get(key, 0) + 1
            pair_tokens[key] = pair_tokens.get(key, 0) + int(rows[i][7] or 0)

    def pct(seq, p):
        if not seq:
            return 0
        return seq[min(len(seq) - 1, int(round(p * (len(seq) - 1))))]

    # Clusters grandes são o sinal de alarme: um cluster com milhares de
    # documentos quase sempre indica critério permissivo demais, não um corpus
    # que realmente repete tanto.
    largest = sorted(
        (
            {
                "cluster": rows[min(m, key=canonical_key)][1]
                or rows[min(m, key=canonical_key)][0],
                "size": len(m),
                "sources": sorted({rows[i][2] for i in m}),
                "sample": (rows[m[0]][4] or "")[:120],
            }
            for m in clusters.values() if len(m) > 1
        ),
        key=lambda c: -c["size"],
    )[:20]

    return {
        "phase": phase,
        "dedup_version": DEDUP_VERSION,
        "containment_threshold": DEDUP_CONTAINMENT_THRESHOLD,
        "min_shared_shingles": DEDUP_MIN_SHARED_SHINGLES,
        "documents": len(rows),
        "clusters": len(clusters),
        "singleton_clusters": sum(1 for s in sizes if s == 1),
        "cross_source_clusters": cross_source,
        "same_source_clusters": len(clusters) - cross_source,
        "exact_duplicate_links": exact_pairs,
        "near_duplicate_links": near_pairs,
        "duplicates_marked": duplicates,
        "canonical_documents": kept,
        "duplicate_rate": duplicates / len(rows),
        "tokens_removed": tokens_removed,
        "cluster_size_p50": pct(sizes, 0.50),
        "cluster_size_p90": pct(sizes, 0.90),
        "cluster_size_p99": pct(sizes, 0.99),
        "cluster_size_max": sizes[-1] if sizes else 0,
        "largest_clusters": largest,
        # Instrumentação da decisão de near-dedup.
        "near_candidates_considered": candidates_considered,
        "near_candidates_accepted": candidates_accepted,
        "near_candidates_rejected_by_anchor": (
            candidates_considered - candidates_accepted
        ),
        "near_accept_rate": (
            candidates_accepted / candidates_considered
            if candidates_considered else 0.0
        ),
        # Fronteira do classificador: pares aceitos e pares rejeitados logo
        # abaixo do limiar. O limiar de 0,60 é HIPÓTESE a calibrar com estes
        # números no dry-run real, não constante validada.
        "accepted_containment_sample": sorted(accepted_containment)[:50],
        "borderline_rejected_sample": sorted(borderline_rejected)[-50:],
        "borderline_rejected_count": len(borderline_rejected),
        # Taxa de near-dedup por faixa de tamanho: se documentos curtos forem
        # eliminados em proporção muito maior, o piso de shingles precisa ser
        # adaptativo ao comprimento.
        "documents_by_size": docs_by_size,
        "near_dedup_by_size": near_by_size,
        "near_dedup_rate_by_size": {
            k: round(near_by_size.get(k, 0) / v, 4)
            for k, v in docs_by_size.items() if v
        },
        # Sobreposição entre fontes, em documentos e tokens.
        "duplicate_docs_by_source_pair": pair_docs,
        "duplicate_tokens_by_source_pair": pair_tokens,
    }


def corpus_funnel(
    store: CorpusStore, phase: str = "base"
) -> Dict[str, Any]:
    """Funil documentos/tokens por fonte e por estágio.

    Existe para responder, meses depois e sem reconstruir nada, perguntas do
    tipo "por que CulturaX caiu de 26M para 17M tokens?". Os estágios
    anteriores à ingestão vêm de contadores gravados em `metadata` durante o
    ingest; os posteriores são computados sobre o próprio banco.
    """
    ingest_counters = {}
    raw = store.get_meta(f"ingest_counters:{phase}")
    if raw:
        try:
            ingest_counters = json.loads(raw)
        except json.JSONDecodeError:
            ingest_counters = {"_parse_error": True}

    stages: List[Dict[str, Any]] = []

    def collect(label: str, condition: str) -> None:
        rows = store.connection.execute(
            f"SELECT source, subset_name, COUNT(*), "
            f"       COALESCE(SUM(estimated_tokens), 0) "
            f"FROM records WHERE phase=? AND split='train' AND {condition} "
            f"GROUP BY source, subset_name",
            (phase,),
        ).fetchall()
        by_bucket: Dict[str, Dict[str, int]] = {}
        for source, subset, docs, tokens in rows:
            bucket = mix_bucket_for_source(source or "", subset or "") or "outra"
            entry = by_bucket.setdefault(bucket, {"documents": 0, "tokens": 0})
            entry["documents"] += int(docs)
            entry["tokens"] += int(tokens or 0)
        stages.append({
            "stage": label,
            "by_source": by_bucket,
            "documents": sum(v["documents"] for v in by_bucket.values()),
            "tokens": sum(v["tokens"] for v in by_bucket.values()),
        })

    collect("ingerido", "1=1")
    collect("canonico_pos_dedup", "is_canonical = 1")
    collect(
        "selecionado_por_quota",
        "is_canonical = 1 AND is_training_selected = 1",
    )

    # Percentual de cada estágio em relação ao anterior e ao final.
    final_tokens = stages[-1]["tokens"] if stages else 0
    previous = None
    for stage in stages:
        stage["pct_of_previous"] = (
            round(stage["tokens"] / previous, 4)
            if previous else 1.0
        )
        stage["pct_of_final"] = (
            round(stage["tokens"] / final_tokens, 4) if final_tokens else 0.0
        )
        previous = stage["tokens"] or None

    rejections = {
        reason: count for reason, count in store.connection.execute(
            "SELECT rejection_reason, COUNT(*) FROM records "
            "WHERE phase=? AND rejection_reason IS NOT NULL "
            "GROUP BY rejection_reason",
            (phase,),
        ).fetchall()
    }

    return {
        "phase": phase,
        "ingest_rejections": ingest_counters,
        "stages": stages,
        "post_dedup_rejections": rejections,
    }


def canonical_corpus_manifest(
    store: CorpusStore, phase: str = "base"
) -> Dict[str, Any]:
    """Manifesto canônico do corpus e seu hash de CONTEÚDO.

    O hash não pode ser o SHA do arquivo JSON: timestamp, ordem de campos,
    whitespace e caminho local mudariam o hash sem que o corpus mudasse. Aqui
    ele é calculado sobre os registros serializados de forma canônica e
    ordenados por document_id, de modo que:

        mesmo corpus lógico -> mesmo hash, onde e quando for preparado.

    É este valor que entra na identidade do treino: `license_policy` e
    `filter_version` são categorias, mas só o hash responde QUAIS documentos
    exatamente produziram um checkpoint.
    """
    rows = store.connection.execute(
        "SELECT document_id, content_hash, source, source_revision, "
        "       license_class, quality_score, filter_version, dedup_cluster, "
        "       is_canonical, is_training_selected, estimated_tokens, split "
        "FROM records WHERE phase=? ORDER BY document_id, content_hash",
        (phase,),
    ).fetchall()

    digest = hashlib.sha256()
    selected_docs = 0
    selected_tokens = 0
    for row in rows:
        record = "\x1f".join(
            "" if value is None else str(value) for value in row
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\x1e")
        if row[8] == 1 and row[9] == 1 and row[11] == "train":
            selected_docs += 1
            selected_tokens += int(row[10] or 0)

    return {
        "phase": phase,
        "records": len(rows),
        "selected_documents": selected_docs,
        "selected_tokens": selected_tokens,
        "manifest_content_sha256": digest.hexdigest(),
        "filter_version": FILTER_VERSION,
        "dedup_version": DEDUP_VERSION,
        "quota_policy_version": QUOTA_POLICY_VERSION,
        "source_mix_policy": dict(BASE_SOURCE_MIX),
        "conditional_quality_max_share": QUALITY_CONDITIONAL_MAX_SHARE,
    }


def select_training_documents(
    store: CorpusStore, phase: str = "base"
) -> Dict[str, Any]:
    """Cota FINAL sobre tokens canônicos, depois do dedup.

    Separada da canonicidade de propósito: `is_canonical` decide qual cópia
    representa um cluster; `is_training_selected` decide se esse representante
    entra no treino. Um documento pode ser canônico, válido e ainda assim
    ficar de fora porque a cota da sua fonte já fechou.

    Aplicar a cota só no ingest seria ilusório: o dedup pode remover uma fonte
    desproporcionalmente e a composição final não seria a pretendida.

    As frações são TETOS, não alvos. Nenhuma fonte é preenchida
    artificialmente para atingir percentual.
    """
    rows = store.connection.execute(
        "SELECT fingerprint, document_id, source, subset_name, "
        "       estimated_tokens, quality_score "
        "FROM records WHERE phase=? AND split='train' AND is_canonical=1 "
        "ORDER BY document_id, fingerprint",
        (phase,),
    ).fetchall()
    if not rows:
        return {"documents": 0}

    # Cota de fonte e de qualidade descrevem a mistura da web na fase base.
    # `continuation` e `sft` vêm de datasets de instrução, com estrutura de
    # fonte e escala de qualidade diferentes — ali a certificação é apenas a
    # canonicidade, e todo representante entra. Marcar explicitamente em vez
    # de deixar NULL, porque o packer é fail-closed e NULL significa
    # "não certificado", não "aprovado por omissão".
    if phase != "base":
        store.connection.executemany(
            "UPDATE records SET is_training_selected=1 WHERE fingerprint=?",
            [(row[0],) for row in rows],
        )
        store.commit()
        return {
            "phase": phase,
            "policy": "canonical_only_no_quota",
            "canonical_documents": len(rows),
            "selected_documents": len(rows),
            "selected_tokens": sum(int(r[4] or 0) for r in rows),
        }

    total_tokens = sum(int(r[4] or 0) for r in rows)
    source_budget = {
        bucket: int(total_tokens * share)
        for bucket, share in BASE_SOURCE_MIX.items()
    }
    conditional_budget = int(total_tokens * QUALITY_CONDITIONAL_MAX_SHARE)
    source_used = {bucket: 0 for bucket in BASE_SOURCE_MIX}
    conditional_used = 0
    selected_tokens = 0
    rejected: Dict[str, int] = {}
    updates = []

    for fingerprint, _doc_id, source, subset, tokens, quality in rows:
        tokens = int(tokens or 0)
        bucket = mix_bucket_for_source(source or "", subset or "")
        reason = None
        if bucket is not None and (
            source_used[bucket] + tokens > source_budget[bucket]
        ):
            reason = f"quota_{bucket}"
        elif (quality is not None
              and int(quality) <= QUALITY_CONDITIONAL_SCORE
              and conditional_used + tokens > conditional_budget):
            reason = "quota_score3"
        if reason is None:
            updates.append((1, None, fingerprint))
            if bucket is not None:
                source_used[bucket] += tokens
            if quality is not None and int(quality) <= QUALITY_CONDITIONAL_SCORE:
                conditional_used += tokens
            selected_tokens += tokens
        else:
            updates.append((0, reason, fingerprint))
            rejected[reason] = rejected.get(reason, 0) + 1

    store.connection.executemany(
        "UPDATE records SET is_training_selected=?, rejection_reason=? "
        "WHERE fingerprint=?",
        updates,
    )
    store.commit()
    return {
        "phase": phase,
        "canonical_documents": len(rows),
        "selected_documents": sum(1 for u in updates if u[0] == 1),
        "canonical_tokens": total_tokens,
        "selected_tokens": selected_tokens,
        "source_tokens": dict(source_used),
        "source_share": {
            k: round(v / max(selected_tokens, 1), 4)
            for k, v in source_used.items()
        },
        "conditional_tokens": conditional_used,
        "conditional_share": round(
            conditional_used / max(selected_tokens, 1), 4
        ),
        "rejected": rejected,
    }


def corpus_manifest(
    store: CorpusStore,
    preset_name: str,
    preset: Dict[str, Any],
    seed: int,
    fixture: bool,
    license_policy: str,
    revisions: Dict[str, str],
) -> Dict[str, Any]:
    phases: Dict[str, Any] = {}
    for phase in ("base", "continuation", "sft"):
        phases[phase] = {
            "train": store.counts(phase, "train").__dict__,
            "eval": store.counts(phase, "eval").__dict__,
            "train_categories": {
                name: counts.__dict__
                for name, counts in store.category_counts(phase, "train").items()
            },
        }
    return {
        "format_version": CORPUS_MANIFEST_FORMAT_VERSION,
        "created_at": utc_now(),
        "preset": preset_name,
        "preset_config": preset,
        "seed": seed,
        "fixture": fixture,
        "license_policy": license_policy,
        "dataset_revisions": revisions,
        "sources": {
            "base": {
                "repo": DATASET_REPOS["base"],
                "license": "por fonte; consulte source_rules",
            },
            "sft": {
                "repo": DATASET_REPOS["sft"],
                "license": "Apache-2.0 no card do aggregate",
                "included_configs": list(SFT_CONFIG_WEIGHTS),
                "excluded_configs": [
                    "code",
                    "function_call",
                    "math",
                    "math_cot",
                    "reasoning",
                    "retrieval",
                    "structured",
                    "translation",
                ],
            },
            "ultrachat": {
                "repo": DATASET_REPOS["ultrachat"],
                "maximum_mix": SFT_CONFIG_WEIGHTS["ultrachat"],
                "warning": "tradução automática; uso limitado para evitar translationese",
            },
        },
        "source_rules": [
            {"marker": marker, "decision": decision, "reason": reason}
            for marker, decision, reason in BASE_SOURCE_RULES
        ],
        "phases": phases,
        "legal_notice": (
            "O manifesto registra proveniência e política técnica, mas não "
            "substitui revisão jurídica antes da comercialização."
        ),
    }


def _bytes_to_unicode_decoder() -> Dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    missing = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + missing)
            missing += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


def _tokenizer_pieces(text: str) -> Iterator[str]:
    raw = normalize_text(text).encode("utf-8")
    cursor = 0
    while cursor < len(raw):
        value = raw[cursor]
        if chr(value).isspace():
            cursor += 1
            continue
        is_word = (
            48 <= value <= 57
            or 65 <= value <= 90
            or 97 <= value <= 122
            or value in (ord("_"), ord("-"), ord("/"))
            or value >= 0x80
        )
        end = cursor + 1
        if is_word:
            while end < len(raw):
                next_value = raw[end]
                if not (
                    48 <= next_value <= 57
                    or 65 <= next_value <= 90
                    or 97 <= next_value <= 122
                    or next_value in (ord("_"), ord("-"), ord("/"))
                    or next_value >= 0x80
                ):
                    break
                end += 1
        piece = raw[cursor:end]
        if len(piece) >= 2:
            yield piece.decode("utf-8", errors="surrogateescape")
        cursor = end


def tokenizer_training_batches(
    store: CorpusStore, char_budget: int, batch_size: int = 1_000
) -> Iterator[List[str]]:
    consumed = 0
    batch: List[str] = []
    for phase in ("base", "continuation", "sft"):
        for _fingerprint, _category, payload in store.iter_records(phase, "train"):
            if phase == "base":
                text = payload
            else:
                messages = json.loads(payload)
                text = format_chat(messages)
            for piece in _tokenizer_pieces(text):
                batch.append(piece)
                consumed += len(piece)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                if consumed >= char_budget:
                    if batch:
                        yield batch
                    return
    if batch:
        yield batch


def write_ox3_from_tokenizers_merges(merges_path: Path, ox3_path: Path) -> int:
    decoder = _bytes_to_unicode_decoder()
    merges: List[Tuple[bytes, bytes]] = []
    with merges_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                raise RuntimeError(f"Merge BPE inválido: {line!r}")
            try:
                left = bytes(decoder[ch] for ch in parts[0])
                right = bytes(decoder[ch] for ch in parts[1])
            except KeyError as exc:
                raise RuntimeError(
                    f"Merge contém símbolo fora do alfabeto byte-level: {line!r}"
                ) from exc
            if not left or not right or len(left) + len(right) > 4096:
                raise RuntimeError("Merge OX3 fora dos limites")
            merges.append((left, right))
    temporary = ox3_path.with_name(ox3_path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(b"OX3\x00")
        output.write(struct.pack("<I", 1))
        for rank, (left, right) in enumerate(merges):
            output.write(struct.pack("<II", rank, len(left)))
            output.write(left)
            output.write(struct.pack("<I", len(right)))
            output.write(right)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, ox3_path)
    return len(merges)


def train_tokenizer(
    store: CorpusStore,
    artifact_dir: Path,
    target_vocab: int,
    char_budget: int,
) -> Path:
    ox3_path = artifact_dir / f"tokenizer_{target_vocab}.ox3"
    metadata_path = artifact_dir / f"tokenizer_{target_vocab}.json"
    if ox3_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") == sha256_file(ox3_path):
            return ox3_path
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer industrial requer tokenizers: pip install tokenizers"
        ) from exc

    artifact_dir.mkdir(parents=True, exist_ok=True)
    core_vocab = target_vocab - len(SPECIAL_TOKENS)
    if core_vocab <= 256:
        raise ValueError("target_vocab pequeno demais para byte vocab + especiais")
    model = BPE(unk_token=None, byte_fallback=False)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    trainer = BpeTrainer(
        vocab_size=core_vocab,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=[],
    )
    print(
        f"[tokenizer] treinando target={target_vocab} chars<={char_budget:,}",
        flush=True,
    )
    tokenizer.train_from_iterator(
        tokenizer_training_batches(store, char_budget),
        trainer=trainer,
    )
    with tempfile.TemporaryDirectory(prefix="nsos_ptbr_tokenizer_") as temp_dir:
        model.save(temp_dir, "ptbr")
        merges_path = Path(temp_dir) / "ptbr-merges.txt"
        learned_merges = write_ox3_from_tokenizers_merges(merges_path, ox3_path)
    metadata = {
        "format_version": 1,
        "target_vocab": target_vocab,
        "core_vocab_target": core_vocab,
        "base_vocab": 256,
        "special_tokens": SPECIAL_TOKENS,
        "learned_merges": learned_merges,
        "training_char_budget": char_budget,
        "sha256": sha256_file(ox3_path),
        "created_at": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    return ox3_path


def detect_build_dir(explicit: Path | None) -> Path:
    nsos_root = Path(__file__).resolve().parents[1]
    candidates: List[Path] = []
    for name in (
        "build-codex-hip",
        "build-validation-hip",
        "build-validation-cpu-min",
        "build-codex-cpu",
        "build_cuda129",
        "build_v1",
        "build_full",
        "build",
    ):
        candidates.extend([nsos_root / name, nsos_root / name / "Release"])
    return resolve_native_build_dir(explicit, candidates)


def configure_gpu_training_profile(args, config, is_gpu: bool, wave_size: int = 32):
    """Select a versioned execution policy before model/Trainer construction.

    Inherit preserves manual policies and existing checkpoints. Redesign is
    opt-in and keeps the truncated TTT derivative unless full-sequence-v1 is
    selected separately. Its semantic selector is stable; native identity
    also versions the concrete recurrence/reduction implementation.
    """
    profile = getattr(args, "gpu_training_profile", "inherit")
    switches = ("NSOS_MAMBA_BOUNDARY_HISTORY", "NSOS_DEVICE_GRAD_CLIP",
                "NSOS_ATTN_TILED_TRAINING", "NSOS_MOE_ORDERED_DEVICE",
                "NSOS_TTT_DEVICE_RECURRENCE")
    if profile not in ("inherit", "legacy", "redesign-v1"):
        raise ValueError("Unknown GPU training profile")
    if profile == "redesign-v1":
        if not is_gpu:
            raise ValueError("redesign-v1 requires explicit GPU execution")
        if (not config.mamba2_faithful or config.mamba_d_state > 64 or wave_size <= 0
                or config.mamba_head_dim % wave_size):
            raise ValueError("redesign-v1 requires faithful Mamba N<=64 and a native-wave-aligned head")
        for name, forbidden in (("NSOS_MAMBA_FAITHFUL_LINEAR_GEOMETRY", "1"),
                                ("NSOS_MAMBA_DETERMINISTIC_HEAD_WAVE_GEOMETRY", "0")):
            if os.environ.get(name) == forbidden:
                raise ValueError(f"redesign-v1 conflicts with {name}={forbidden}")
        os.environ.update({name: "1" for name in switches})
        os.environ.update({"NSOS_MAMBA_CHUNKED_BACKWARD": "1",
                           "NSOS_MAMBA_BACKWARD_CHUNK_SIZE": "32",
                           "NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD": "1",
                           "NSOS_MAMBA_FORWARD_CHUNK_SIZE": "128"})
    elif profile == "legacy":
        os.environ.update({name: "0" for name in switches})
        os.environ["NSOS_MOE_GROUPED_TRAINING"] = "0"
        os.environ["NSOS_MOE_WMMA_TRAINING"] = "0"
        os.environ["NSOS_KAN_RECOMPUTE_TRAINING"] = "0"
        os.environ["NSOS_KAN_WMMA_TRAINING"] = "0"
    kan_compute = getattr(args, "kan_compute_policy", "inherit")
    if kan_compute not in ("inherit", "legacy", "tiled-v1", "wmma-v1"):
        raise ValueError("Unknown KAN compute policy")
    if kan_compute in ("tiled-v1", "wmma-v1"):
        if not is_gpu or not getattr(config, "use_kan", False):
            raise ValueError("KAN tiled training requires GPU and use_kan")
        os.environ["NSOS_KAN_RECOMPUTE_TRAINING"] = "1"
        os.environ["NSOS_KAN_WMMA_TRAINING"] = "1" if kan_compute == "wmma-v1" else "0"
    elif kan_compute == "legacy":
        os.environ["NSOS_KAN_RECOMPUTE_TRAINING"] = "0"
        os.environ["NSOS_KAN_WMMA_TRAINING"] = "0"
    kan_recompute = os.environ.get("NSOS_KAN_RECOMPUTE_TRAINING", "0")
    if kan_recompute not in ("", "0", "1"):
        raise ValueError("NSOS_KAN_RECOMPUTE_TRAINING must be 0 or 1")
    if kan_recompute == "1" and (not is_gpu or not getattr(config, "use_kan", False)):
        raise ValueError("NSOS_KAN_RECOMPUTE_TRAINING requires GPU and use_kan")
    kan_wmma = os.environ.get("NSOS_KAN_WMMA_TRAINING", "0")
    if kan_wmma not in ("", "0", "1"):
        raise ValueError("NSOS_KAN_WMMA_TRAINING must be 0 or 1")
    if kan_wmma == "1" and (kan_recompute != "1" or wave_size != 32):
        raise ValueError("NSOS_KAN_WMMA_TRAINING requires recompute KAN and native wave32; native identity also checks RDNA3")
    moe_compute = getattr(args, "moe_compute_policy", "inherit")
    if moe_compute not in ("inherit", "legacy", "grouped-v1", "wmma-v1"):
        raise ValueError("Unknown MoE compute policy")
    if moe_compute in ("grouped-v1", "wmma-v1"):
        if not is_gpu or not getattr(config, "use_moe", False):
            raise ValueError("Grouped MoE training requires GPU and an architecture with use_moe enabled")
        os.environ.update({"NSOS_MOE_GROUPED_TRAINING": "1", "NSOS_MOE_ORDERED_DEVICE": "1"})
        os.environ["NSOS_MOE_WMMA_TRAINING"] = "1" if moe_compute == "wmma-v1" else "0"
    elif moe_compute == "legacy":
        os.environ["NSOS_MOE_GROUPED_TRAINING"] = "0"
        os.environ["NSOS_MOE_WMMA_TRAINING"] = "0"
    grouped = os.environ.get("NSOS_MOE_GROUPED_TRAINING", "0")
    if grouped not in ("", "0", "1"):
        raise ValueError("NSOS_MOE_GROUPED_TRAINING must be 0 or 1")
    if grouped == "1" and (not is_gpu or not getattr(config, "use_moe", False)
                           or os.environ.get("NSOS_MOE_ORDERED_DEVICE") != "1"):
        raise ValueError("NSOS_MOE_GROUPED_TRAINING requires GPU, use_moe and ordered routing")
    wmma = os.environ.get("NSOS_MOE_WMMA_TRAINING", "0")
    if wmma not in ("", "0", "1"):
        raise ValueError("NSOS_MOE_WMMA_TRAINING must be 0 or 1")
    if wmma == "1" and (grouped != "1" or wave_size != 32):
        raise ValueError("NSOS_MOE_WMMA_TRAINING requires grouped MoE and native wave32; native identity also checks RDNA3")
    derivative = getattr(args, "ttt_gradient_policy", "inherit")
    if derivative not in ("inherit", "truncated", "full-sequence-v1"):
        raise ValueError("Unknown TTT gradient policy")
    if derivative == "full-sequence-v1":
        if not config.use_ttt:
            raise ValueError("Full TTT sequence BPTT requires an architecture with use_ttt enabled")
        os.environ["NSOS_TTT_FULL_BPTT"] = "1"
    elif derivative == "truncated" or profile == "legacy":
        os.environ["NSOS_TTT_FULL_BPTT"] = "0"
    full = os.environ.get("NSOS_TTT_FULL_BPTT", "0")
    if full not in ("", "0", "1"):
        raise ValueError("NSOS_TTT_FULL_BPTT must be 0 or 1")
    switches += ("NSOS_TTT_FULL_BPTT", "NSOS_MOE_GROUPED_TRAINING", "NSOS_MOE_WMMA_TRAINING", "NSOS_KAN_RECOMPUTE_TRAINING", "NSOS_KAN_WMMA_TRAINING")
    return {"profile": profile, "switches": {name: os.environ.get(name, "default") for name in switches},
            "ttt_derivative": ("full_sequence_bptt_isolated_boundary32_column_q_v2" if full == "1"
                               else "truncated_stop_gradient_adaptation"),
            "moe_compute": ("device_segmented_tile16_active_qat_v2" if grouped == "1" else "legacy"),
            "moe_wmma": ("rdna3_lowp_tile32_minrows16_mindim32_v1" if wmma == "1" else "off"),
            "moe_sparse_gradient": ("explicit_group_contribution_host_v1" if getattr(config, "use_moe", False) else None),
            "kan_compute": ("device_qat_tree256_rbf_tile16_recompute_v1" if kan_recompute == "1" else "legacy"),
            "kan_wmma": ("rdna3_implicit_rbf_tile32_minrows16_mindim32_v1" if kan_wmma == "1" else "off"),
            "promotion": "explicit_candidate_not_global_default", "native_verified": False}


def validate_native_training_policy(policy, identity):
    """Reject a stale native binary that silently ignores requested switches.

    Environment requests are not execution evidence. The Trainer's versioned
    identity is authoritative, including the concrete full-TTT implementation.
    """
    requested = policy["switches"]
    checks = (
        ("NSOS_MAMBA_BOUNDARY_HISTORY", "mamba.history_retention", "entering_boundary_chunk32_recompute_v1"),
        ("NSOS_DEVICE_GRAD_CLIP", "optimizer.clip_policy", "device_fp64_chunk8192_tree256_v1"),
        ("NSOS_ATTN_TILED_TRAINING", "attention.training_policy", "exact_fp32_online_tile8_ordered_v1"),
        ("NSOS_MOE_ORDERED_DEVICE", "moe.dispatch_policy", "stable_expert_row_order_device_combine_v1"),
        ("NSOS_MOE_GROUPED_TRAINING", "moe.training_compute_policy", "device_segmented_tile16_active_qat_v2"),
        ("NSOS_MOE_WMMA_TRAINING", "moe.training_wmma_policy", "rdna3_lowp_tile32_minrows16_mindim32_v1"),
        ("NSOS_KAN_RECOMPUTE_TRAINING", "kan.training_policy", "device_qat_tree256_rbf_tile16_recompute_v1"),
        ("NSOS_KAN_WMMA_TRAINING", "kan.wmma_policy", "rdna3_implicit_rbf_tile32_minrows16_mindim32_v1"),
    )
    for switch, field, expected in checks:
        if requested.get(switch) == "1" and identity.get(field) != expected:
            raise RuntimeError(f"Native binary does not honor {switch}: {field}={identity.get(field)!r}, expected {expected!r}")
    if requested.get("NSOS_TTT_FULL_BPTT") == "1":
        expected = policy["ttt_derivative"]
    elif requested.get("NSOS_TTT_DEVICE_RECURRENCE") == "1":
        expected = "device_recurrence_fp64_norm_truncated_v1"
    else:
        expected = None
    if expected is not None and identity.get("ttt.training_policy") != expected:
        raise RuntimeError(f"Native TTT training policy mismatch: {identity.get('ttt.training_policy')!r}, expected {expected!r}")
    sparse_policy = policy.get("moe_sparse_gradient")
    if sparse_policy is not None and identity.get("optimizer.sparse_gradient_policy") != sparse_policy:
        raise RuntimeError(f"Native sparse gradient policy mismatch: {identity.get('optimizer.sparse_gradient_policy')!r}, expected {sparse_policy!r}")
    policy["native_verified"] = True
    policy["native_ttt_policy"] = identity.get("ttt.training_policy")
    policy["native_moe_compute"] = identity.get("moe.training_compute_policy", "legacy")
    policy["native_sparse_gradient_policy"] = identity.get("optimizer.sparse_gradient_policy")
    policy["native_kan_compute"] = identity.get("kan.training_policy", "legacy")
    policy["native_kan_wmma"] = identity.get("kan.wmma_policy", "off")


def load_nsos(build_dir: Path):
    return load_native_module(build_dir)


def load_product_tokenizer(nsos, ox3_path: Path, pack_path: Path):
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(ox3_path))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.save_pack(str(pack_path))
    return tokenizer


class TokenShardWriter:
    def __init__(
        self,
        directory: Path,
        prefix: str,
        shard_token_limit: int,
    ) -> None:
        self.directory = directory
        self.prefix = prefix
        self.limit = shard_token_limit
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer = array("H")
        self.index = 0
        self.shards: List[Dict[str, Any]] = []
        self.total_tokens = 0

    def add(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            if token < 0 or token > 65535:
                raise RuntimeError(f"Token {token} não cabe em uint16")
            self.buffer.append(int(token))
            if len(self.buffer) >= self.limit:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.directory / f"{self.prefix}-{self.index:05d}.u16"
        temporary = path.with_name(path.name + ".tmp")
        output_values = array("H", self.buffer)
        if sys.byteorder != "little":
            output_values.byteswap()
        with temporary.open("wb") as handle:
            output_values.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        count = len(self.buffer)
        self.shards.append(
            {
                "file": path.name,
                "tokens": count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self.total_tokens += count
        self.buffer = array("H")
        self.index += 1

    def finish(self) -> List[Dict[str, Any]]:
        self.flush()
        return self.shards


class SFTShardWriter:
    MAGIC = b"NSOSSFT1"

    def __init__(self, directory: Path, records_per_shard: int) -> None:
        self.directory = directory
        self.limit = records_per_shard
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: List[Tuple[List[int], List[int]]] = []
        self.index = 0
        self.shards: List[Dict[str, Any]] = []
        self.total_target_tokens = 0

    def add(self, prompt: List[int], answer: List[int]) -> None:
        if not prompt or not answer:
            return
        if max(prompt + answer) > 65535 or min(prompt + answer) < 0:
            raise RuntimeError("Registro SFT contém token fora de uint16")
        self.records.append((prompt, answer))
        if len(self.records) >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.directory / f"sft-{self.index:05d}.bin"
        temporary = path.with_name(path.name + ".tmp")
        target_tokens = 0
        with temporary.open("wb") as handle:
            handle.write(self.MAGIC)
            handle.write(struct.pack("<II", PACK_FORMAT_VERSION, len(self.records)))
            for prompt, answer in self.records:
                handle.write(struct.pack("<II", len(prompt), len(answer)))
                values = array("H", prompt + answer)
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(handle)
                target_tokens += len(answer)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self.shards.append(
            {
                "file": path.name,
                "records": len(self.records),
                "target_tokens": target_tokens,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self.total_target_tokens += target_tokens
        self.records = []
        self.index += 1

    def finish(self) -> List[Dict[str, Any]]:
        self.flush()
        return self.shards


class U16TokenShard(Sequence[int]):
    """Read-only uint16-le shard; only requested windows become Python ints."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._size = path.stat().st_size
        if self._size % 2 != 0:
            raise RuntimeError(f"Shard uint16 truncado: {path}")
        self._handle = path.open("rb")
        try:
            self._mapping = (
                mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
                if self._size
                else None
            )
        except Exception:
            self._handle.close()
            self._handle = None
            raise

    def __len__(self) -> int:
        return self._size // 2

    def __getitem__(self, key):
        if self._handle is None:
            raise ValueError("uint16 shard is closed")
        length = len(self)
        if isinstance(key, slice):
            start, stop, step = key.indices(length)
            if step != 1:
                return [self[index] for index in range(start, stop, step)]
            if stop <= start:
                return []
            assert self._mapping is not None
            values = array("H")
            values.frombytes(self._mapping[start * 2 : stop * 2])
            if sys.byteorder != "little":
                values.byteswap()
            return list(values)
        index = int(key)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("uint16 shard index out of range")
        assert self._mapping is not None
        return struct.unpack_from("<H", self._mapping, index * 2)[0]

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "U16TokenShard":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            print(
                f"[dataset] falha ao fechar shard uint16 {self.path}: "
                f"{cleanup_error}",
                file=sys.stderr,
                flush=True,
            )


class SFTRecordShard(Sequence[Tuple[List[int], List[int]]]):
    """Validated mmap plus compact offsets; payloads decode only on demand."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("rb")
        size = path.stat().st_size
        if size < 16:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"Header SFT truncado: {path}")
        try:
            self._mapping = mmap.mmap(
                self._handle.fileno(), 0, access=mmap.ACCESS_READ
            )
        except Exception:
            self._handle.close()
            self._handle = None
            raise
        if self._mapping[:8] != SFTShardWriter.MAGIC:
            self.close()
            raise RuntimeError(f"Magic SFT inválido: {path}")
        version, count = struct.unpack_from("<II", self._mapping, 8)
        if version != PACK_FORMAT_VERSION or count > 10_000_000:
            self.close()
            raise RuntimeError(f"Header SFT incompatível: {path}")
        self._records: List[Tuple[int, int, int]] = []
        cursor = 16
        for _ in range(count):
            if cursor + 8 > size:
                self.close()
                raise RuntimeError(f"Registro SFT truncado: {path}")
            prompt_len, answer_len = struct.unpack_from(
                "<II", self._mapping, cursor
            )
            cursor += 8
            total = prompt_len + answer_len
            if prompt_len == 0 or answer_len == 0 or total > 1_000_000:
                self.close()
                raise RuntimeError(f"Comprimento SFT inválido: {path}")
            payload_bytes = total * 2
            if cursor + payload_bytes > size:
                self.close()
                raise RuntimeError(f"Payload SFT truncado: {path}")
            self._records.append((cursor, prompt_len, answer_len))
            cursor += payload_bytes
        if cursor != size:
            self.close()
            raise RuntimeError(f"Bytes extras no shard SFT: {path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, key):
        if self._mapping is None:
            raise ValueError("SFT shard is closed")
        if isinstance(key, slice):
            return [self[index] for index in range(*key.indices(len(self)))]
        index = int(key)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("SFT shard index out of range")
        offset, prompt_len, answer_len = self._records[index]
        values = array("H")
        values.frombytes(
            self._mapping[offset : offset + 2 * (prompt_len + answer_len)]
        )
        if sys.byteorder != "little":
            values.byteswap()
        return (list(values[:prompt_len]), list(values[prompt_len:]))

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def __enter__(self) -> "SFTRecordShard":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            print(
                f"[dataset] falha ao fechar shard SFT {self.path}: "
                f"{cleanup_error}",
                file=sys.stderr,
                flush=True,
            )


def read_u16_tokens(path: Path) -> U16TokenShard:
    return U16TokenShard(path)


def read_sft_records(path: Path) -> SFTRecordShard:
    return SFTRecordShard(path)


def truncate_supervised(
    tokenizer,
    prompt_text: str,
    answer_text: str,
    eos_token_id: int,
    seq_len: int,
) -> Tuple[List[int], List[int]]:
    prompt = list(tokenizer.encode(prompt_text))
    answer = list(tokenizer.encode(answer_text)) + [eos_token_id]
    max_answer = max(8, seq_len // 2)
    if len(answer) > max_answer:
        answer = answer[: max_answer - 1] + [eos_token_id]
    max_prompt = seq_len + 1 - len(answer)
    if max_prompt < 1:
        return [], []
    if len(prompt) > max_prompt:
        bos_ids = list(tokenizer.encode(BOS_TOKEN))
        system_prefix = (
            f"{BOS_TOKEN}<|system|>\n{DEFAULT_SYSTEM_PROMPT}\n"
        )
        system_ids = list(tokenizer.encode(system_prefix))
        if len(system_ids) < max_prompt // 2:
            tail = prompt[-(max_prompt - len(system_ids)) :]
            prompt = system_ids + tail
        else:
            prompt = (bos_ids + prompt[-(max_prompt - len(bos_ids)) :])[:max_prompt]
    return prompt, answer


def pack_causal_phase(
    store: CorpusStore,
    tokenizer,
    phase: str,
    split: str,
    out_dir: Path,
    token_target: int,
    shard_size: int,
    eos_token_id: int,
) -> Dict[str, Any]:
    writer = TokenShardWriter(out_dir, f"{phase}-{split}", shard_size)
    for _fingerprint, _category, payload in store.iter_records(phase, split):
        if phase == "base":
            text = payload
        else:
            text = format_chat(json.loads(payload))
        tokens = list(tokenizer.encode(text))
        if not tokens or tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)
        remaining = token_target - (writer.total_tokens + len(writer.buffer))
        if split == "train" and remaining <= 0:
            break
        if split == "train" and len(tokens) > remaining:
            tokens = tokens[:remaining]
        writer.add(tokens)
    shards = writer.finish()
    return {
        "phase": phase,
        "split": split,
        "dtype": "uint16-le",
        "target_tokens": token_target if split == "train" else None,
        "tokens": sum(int(shard["tokens"]) for shard in shards),
        "shards": shards,
    }


def pack_sft_phase(
    store: CorpusStore,
    tokenizer,
    split: str,
    out_dir: Path,
    target_tokens: int,
    records_per_shard: int,
    eos_token_id: int,
    seq_len: int,
) -> Dict[str, Any]:
    writer = SFTShardWriter(out_dir, records_per_shard)
    stop = False
    for _fingerprint, _category, payload in store.iter_records("sft", split):
        messages = json.loads(payload)
        for prompt_text, answer_text in supervised_examples(messages):
            prompt, answer = truncate_supervised(
                tokenizer,
                prompt_text,
                answer_text,
                eos_token_id,
                seq_len,
            )
            if not prompt or not answer:
                continue
            remaining = target_tokens - (
                writer.total_target_tokens
                + sum(len(record[1]) for record in writer.records)
            )
            if split == "train" and remaining <= 0:
                stop = True
                break
            if split == "train" and len(answer) > remaining:
                answer = answer[:remaining]
                if answer:
                    answer[-1] = eos_token_id
            writer.add(prompt, answer)
        if stop:
            break
    shards = writer.finish()
    return {
        "phase": "sft",
        "split": split,
        "format": "NSOSSFT1",
        "target_tokens": target_tokens if split == "train" else None,
        "target_tokens_packed": sum(
            int(shard["target_tokens"]) for shard in shards
        ),
        "records": sum(int(shard["records"]) for shard in shards),
        "shards": shards,
    }


def corpus_recipe_identity(
    preset_name: str,
    preset: Dict[str, Any],
    seed: int,
    fixture: bool,
    license_policy: str,
) -> Dict[str, Any]:
    """Return the immutable data-recipe identity, independent of code releases."""
    return {
        "format_version": CORPUS_IDENTITY_FORMAT_VERSION,
        "recipe_version": CORPUS_RECIPE_VERSION,
        "preset": preset_name,
        "preset_config": preset,
        "seed": seed,
        "fixture": fixture,
        "license_policy": license_policy,
    }


def corpus_identity_compatible(
    serialized_identity: str,
    expected_identity: Dict[str, Any],
) -> bool:
    """Accept only the current identity or the exact auditable legacy schema."""
    try:
        stored = json.loads(serialized_identity)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(stored, dict):
        return False

    current_keys = set(expected_identity)
    if set(stored) == current_keys:
        return stable_json(stored) == stable_json(expected_identity)

    legacy_keys = {
        "script_version",
        "preset",
        "preset_config",
        "seed",
        "fixture",
        "license_policy",
    }
    if set(stored) != legacy_keys:
        return False
    script_version = stored["script_version"]
    if (
        isinstance(script_version, bool)
        or not isinstance(script_version, int)
        or not 1 <= script_version <= SCRIPT_VERSION
    ):
        return False
    recipe_keys = (
        "preset",
        "preset_config",
        "seed",
        "fixture",
        "license_policy",
    )
    legacy_recipe = {key: stored[key] for key in recipe_keys}
    expected_recipe = {key: expected_identity[key] for key in recipe_keys}
    return stable_json(legacy_recipe) == stable_json(expected_recipe)


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} invalido: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} deve conter um objeto JSON: {path}")
    return value


def _resolve_workspace_artifact(
    workspace: Path,
    relative_path: Any,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError(f"{label} possui caminho ausente ou invalido")
    workspace_root = workspace.resolve()
    resolved = (workspace_root / Path(relative_path)).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapa do workspace: {relative_path}") from exc
    if resolved != expected_path.resolve():
        raise RuntimeError(
            f"{label} aponta para artefato inesperado: {relative_path}"
        )
    return resolved


def verify_pack_manifest(pack_root: Path, manifest: Dict[str, Any]) -> None:
    phases = manifest.get("phases")
    if not isinstance(phases, dict) or not phases:
        raise RuntimeError("Pack manifest sem fases validas")
    pack_root = pack_root.resolve()
    for phase_name, phase in phases.items():
        if not isinstance(phase_name, str) or not isinstance(phase, dict):
            raise RuntimeError("Pack manifest contem fase invalida")
        directory = phase.get("directory")
        shards = phase.get("shards")
        if not isinstance(directory, str) or not directory:
            raise RuntimeError(f"Diretorio invalido na fase {phase_name}")
        if not isinstance(shards, list) or not shards:
            raise RuntimeError(f"Fase {phase_name} sem shards validos")
        phase_dir = (pack_root / Path(directory)).resolve()
        try:
            phase_dir.relative_to(pack_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Diretorio da fase {phase_name} escapa do pack"
            ) from exc
        for shard in shards:
            if not isinstance(shard, dict):
                raise RuntimeError(f"Shard invalido na fase {phase_name}")
            shard_name = shard.get("file")
            expected_bytes = shard.get("bytes")
            expected_sha256 = shard.get("sha256")
            if not isinstance(shard_name, str) or not shard_name:
                raise RuntimeError(f"Shard sem nome na fase {phase_name}")
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes <= 0
            ):
                raise RuntimeError(f"Tamanho invalido no shard {shard_name}")
            if not isinstance(expected_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", expected_sha256
            ) is None:
                raise RuntimeError(f"SHA-256 invalido no shard {shard_name}")
            path = (phase_dir / Path(shard_name)).resolve()
            try:
                path.relative_to(phase_dir)
            except ValueError as exc:
                raise RuntimeError(
                    f"Shard da fase {phase_name} escapa do diretorio"
                ) from exc
            if not path.is_file():
                raise RuntimeError(f"Shard ausente: {path}")
            if path.stat().st_size != expected_bytes:
                raise RuntimeError(f"Tamanho divergente: {path}")
            if sha256_file(path) != expected_sha256:
                raise RuntimeError(f"SHA-256 divergente: {path}")


def _require_exact_keys(
    payload: Dict[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RuntimeError(
            f"{label} possui esquema divergente; ausentes={missing}, "
            f"desconhecidos={unknown}"
        )


def _validated_count_payload(value: Any, label: str) -> Dict[str, int]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} deve ser um objeto")
    _require_exact_keys(value, {"documents", "estimated_tokens"}, label)
    for key in ("documents", "estimated_tokens"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RuntimeError(f"{label}.{key} deve ser inteiro nao negativo")
    return {
        "documents": value["documents"],
        "estimated_tokens": value["estimated_tokens"],
    }


def validate_prepared_workspace(
    store: CorpusStore,
    workspace: Path,
    preset_name: str,
    preset: Dict[str, Any],
    seed: int,
    fixture: bool,
    license_policy: str,
) -> Path | None:
    """Validate the complete immutable corpus-to-pack provenance chain."""
    corpus_manifest_path = workspace / "corpus" / "corpus_manifest.json"
    pack_manifest_path = workspace / "packs" / "pack_manifest.json"
    corpus_exists = corpus_manifest_path.is_file()
    pack_exists = pack_manifest_path.is_file()
    if not corpus_exists and not pack_exists:
        return None
    if corpus_exists != pack_exists:
        raise RuntimeError(
            "Workspace de dados parcial: corpus_manifest.json e "
            "pack_manifest.json devem existir juntos"
        )

    corpus = _read_json_object(corpus_manifest_path, "Corpus manifest")
    _require_exact_keys(
        corpus,
        {
            "format_version",
            "created_at",
            "preset",
            "preset_config",
            "seed",
            "fixture",
            "license_policy",
            "dataset_revisions",
            "sources",
            "source_rules",
            "phases",
            "legal_notice",
        },
        "Corpus manifest",
    )
    if corpus["format_version"] != CORPUS_MANIFEST_FORMAT_VERSION:
        raise RuntimeError(
            "Versao incompatível do corpus manifest: "
            f"{corpus['format_version']}"
        )
    if (
        corpus["preset"] != preset_name
        or corpus["preset_config"] != preset
        or isinstance(corpus["seed"], bool)
        or corpus["seed"] != seed
        or corpus["fixture"] is not fixture
        or corpus["license_policy"] != license_policy
    ):
        raise RuntimeError("Corpus manifest pertence a outra receita")
    if not isinstance(corpus["created_at"], str) or not corpus["created_at"]:
        raise RuntimeError("Corpus manifest sem created_at valido")

    revisions = corpus["dataset_revisions"]
    if not isinstance(revisions, dict):
        raise RuntimeError("Corpus manifest sem dataset_revisions validas")
    expected_revision_keys = {"fixture"} if fixture else set(
        preset.get("dataset_repos", DATASET_REPOS)
    )
    _require_exact_keys(revisions, expected_revision_keys, "dataset_revisions")
    for source, revision in revisions.items():
        if not isinstance(revision, str) or re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", revision
        ) is None:
            raise RuntimeError(f"Revisao invalida para {source}")
    if fixture:
        expected_fixture_revision = stable_hash_text("nsos-ptbr-fixture-v1")
        if revisions["fixture"] != expected_fixture_revision:
            raise RuntimeError("Revisao do corpus fixture divergente")
    else:
        revisions_path = workspace / "corpus" / "dataset_revisions.json"
        if not revisions_path.is_file():
            raise RuntimeError(f"Dataset revisions ausente: {revisions_path}")
        stored_revisions = _read_json_object(
            revisions_path, "Dataset revisions"
        )
        if stored_revisions != revisions:
            raise RuntimeError(
                "dataset_revisions.json diverge do corpus manifest"
            )

    phases = corpus["phases"]
    if not isinstance(phases, dict):
        raise RuntimeError("Corpus manifest sem fases validas")
    _require_exact_keys(phases, {"base", "continuation", "sft"}, "Fases")
    for phase_name in ("base", "continuation", "sft"):
        phase = phases[phase_name]
        if not isinstance(phase, dict):
            raise RuntimeError(f"Fase de corpus invalida: {phase_name}")
        _require_exact_keys(
            phase,
            {"train", "eval", "train_categories"},
            f"Fase {phase_name}",
        )
        for split in ("train", "eval"):
            recorded = _validated_count_payload(
                phase[split], f"Fase {phase_name}.{split}"
            )
            actual = store.counts(phase_name, split).__dict__
            if recorded != actual:
                raise RuntimeError(
                    f"Contagens SQLite divergem em {phase_name}.{split}: "
                    f"manifest={recorded}, sqlite={actual}"
                )
        categories = phase["train_categories"]
        if not isinstance(categories, dict):
            raise RuntimeError(
                f"Categorias de treino invalidas na fase {phase_name}"
            )
        recorded_categories = {
            name: _validated_count_payload(
                value, f"Categoria {phase_name}.{name}"
            )
            for name, value in categories.items()
            if isinstance(name, str) and name
        }
        if len(recorded_categories) != len(categories):
            raise RuntimeError(
                f"Nome de categoria invalido na fase {phase_name}"
            )
        actual_categories = {
            name: counts.__dict__
            for name, counts in store.category_counts(
                phase_name, "train"
            ).items()
        }
        if recorded_categories != actual_categories:
            raise RuntimeError(
                f"Categorias SQLite divergem na fase {phase_name}"
            )

    pack = _read_json_object(pack_manifest_path, "Pack manifest")
    _require_exact_keys(
        pack,
        {
            "format_version",
            "created_at",
            "preset",
            # Identidade lógica do corpus (quais documentos) e do artefato
            # empacotado (quais tokens, em que ordem).
            "dataset",
            "packing",
            "tokenizer",
            "corpus_manifest",
            "phases",
        },
        "Pack manifest",
    )
    if pack["format_version"] != PACK_FORMAT_VERSION:
        raise RuntimeError(
            f"Versao incompatível do pack manifest: {pack['format_version']}"
        )
    if pack["preset"] != preset_name:
        raise RuntimeError("Pack manifest pertence a outro preset")

    corpus_link = pack["corpus_manifest"]
    if not isinstance(corpus_link, dict):
        raise RuntimeError("Pack manifest sem vinculo de corpus")
    _require_exact_keys(corpus_link, {"path", "sha256"}, "Vinculo de corpus")
    linked_corpus = _resolve_workspace_artifact(
        workspace,
        corpus_link["path"],
        corpus_manifest_path,
        "Vinculo de corpus",
    )
    if corpus_link["sha256"] != sha256_file(linked_corpus):
        raise RuntimeError("SHA-256 do corpus manifest diverge do pack")

    tokenizer = pack["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise RuntimeError("Pack manifest sem tokenizer valido")
    _require_exact_keys(
        tokenizer,
        {
            "path",
            "sha256",
            "vocab_size",
            "special_tokens",
            "eos_token_id",
        },
        "Tokenizer do pack",
    )
    tokenizer_path = _resolve_workspace_artifact(
        workspace,
        tokenizer["path"],
        workspace / "tokenizer.nsos",
        "Tokenizer do pack",
    )
    if not tokenizer_path.is_file():
        raise RuntimeError(f"Tokenizer ausente: {tokenizer_path}")
    if tokenizer["sha256"] != sha256_file(tokenizer_path):
        raise RuntimeError("SHA-256 do tokenizer diverge do pack")
    vocab_size = tokenizer["vocab_size"]
    eos_token_id = tokenizer["eos_token_id"]
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= len(SPECIAL_TOKENS)
        or vocab_size > int(preset["target_vocab"])
    ):
        raise RuntimeError("Vocabulario invalido no pack manifest")
    if tokenizer["special_tokens"] != SPECIAL_TOKENS:
        raise RuntimeError("Tokens especiais divergem do contrato")
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < vocab_size
    ):
        raise RuntimeError("EOS invalido no pack manifest")

    pack_phases = pack["phases"]
    if not isinstance(pack_phases, dict):
        raise RuntimeError("Pack manifest sem fases validas")
    expected_pack_phases = {
        "base_train": ("base", "train"),
        "base_eval": ("base", "eval"),
        "continuation_train": ("continuation", "train"),
        "continuation_eval": ("continuation", "eval"),
        "sft_train": ("sft", "train"),
        "sft_eval": ("sft", "eval"),
    }
    _require_exact_keys(pack_phases, set(expected_pack_phases), "Fases do pack")
    for key, (phase_name, split) in expected_pack_phases.items():
        phase = pack_phases[key]
        if not isinstance(phase, dict):
            raise RuntimeError(f"Fase invalida no pack: {key}")
        if (
            phase.get("phase") != phase_name
            or phase.get("split") != split
            or phase.get("directory") != key
        ):
            raise RuntimeError(f"Contrato divergente na fase do pack: {key}")
    expected_targets = {
        "base_train": int(preset["base_tokens"]),
        "continuation_train": int(preset["continuation_tokens"]),
        "sft_train": int(preset["sft_unique_tokens"]),
    }
    for key, target in expected_targets.items():
        if pack_phases[key].get("target_tokens") != target:
            raise RuntimeError(f"Target divergente na fase do pack: {key}")

    verify_pack_manifest(workspace / "packs", pack)
    return pack_manifest_path


def validate_training_workspace(
    workspace: Path,
    preset_name: str,
    preset: Dict[str, Any],
    seed: int,
    fixture: bool,
    license_policy: str,
) -> Path:
    database_path = workspace / "corpus" / "corpus.sqlite3"
    if not database_path.is_file():
        raise RuntimeError(f"Corpus SQLite ausente: {database_path}")
    expected_identity = corpus_recipe_identity(
        preset_name, preset, seed, fixture, license_policy
    )
    store = CorpusStore(database_path)
    try:
        stored_identity = store.get_meta("run_identity")
        if stored_identity is None or not corpus_identity_compatible(
            stored_identity, expected_identity
        ):
            raise RuntimeError(
                "Identidade do corpus ausente ou incompatível com o treino"
            )
        manifest_path = validate_prepared_workspace(
            store,
            workspace,
            preset_name,
            preset,
            seed,
            fixture,
            license_policy,
        )
        if manifest_path is None:
            raise RuntimeError("Packs ausentes; execute prepare ou run primeiro")
        return manifest_path
    finally:
        store.close()


def build_packs(
    store: CorpusStore,
    nsos,
    ox3_path: Path,
    workspace: Path,
    preset_name: str,
    preset: Dict[str, Any],
    corpus_manifest_path: Path,
    *,
    causal_packer=None,
    sft_packer=None,
    dataset_metadata: Dict[str, Any] | None = None,
    packing_version: str = PACKING_VERSION,
) -> Path:
    causal_packer = causal_packer or pack_causal_phase
    sft_packer = sft_packer or pack_sft_phase
    pack_root = workspace / "packs"
    manifest_path = pack_root / "pack_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_pack_manifest(pack_root, manifest)
        return manifest_path
    pack_root.mkdir(parents=True, exist_ok=True)
    tokenizer_pack = workspace / "tokenizer.nsos"
    tokenizer = load_product_tokenizer(nsos, ox3_path, tokenizer_pack)
    actual_vocab = int(tokenizer.vocab_size)
    target_vocab = int(preset["target_vocab"])
    minimum_fill = 0.75 if preset_name == "smoke" else 0.95
    if actual_vocab < int(target_vocab * minimum_fill):
        raise RuntimeError(
            f"Tokenizer subpreenchido: vocab={actual_vocab}, target={target_vocab}"
        )
    eos_ids = list(tokenizer.encode(EOS_TOKEN))
    if len(eos_ids) != 1:
        raise RuntimeError("EOS deve codificar em exatamente um token")
    eos_token_id = int(eos_ids[0])

    phase_specs: Dict[str, Dict[str, Any]] = {}
    for phase, target in (
        ("base", int(preset["base_tokens"])),
        ("continuation", int(preset["continuation_tokens"])),
    ):
        for split in ("train", "eval"):
            key = f"{phase}_{split}"
            directory = key
            eval_target = max(
                int(preset["seq_len"]) * 4,
                min(target // 200, int(preset["token_shard_size"])),
            )
            phase_specs[key] = {
                **causal_packer(
                    store,
                    tokenizer,
                    phase,
                    split,
                    pack_root / directory,
                    target if split == "train" else eval_target,
                    int(preset["token_shard_size"]),
                    eos_token_id,
                ),
                "directory": directory,
            }
    for split in ("train", "eval"):
        key = f"sft_{split}"
        directory = key
        phase_specs[key] = {
            **sft_packer(
                store,
                tokenizer,
                split,
                pack_root / directory,
                int(preset["sft_unique_tokens"])
                if split == "train"
                else max(2_000, int(preset["sft_unique_tokens"]) // 100),
                int(preset["sft_records_per_shard"]),
                eos_token_id,
                int(preset["seq_len"]),
            ),
            "directory": directory,
        }

    corpus_minimum_fill = 0.25 if preset_name == "smoke" else 0.95
    for key in ("base_train", "continuation_train"):
        packed = int(phase_specs[key]["tokens"])
        target = int(phase_specs[key]["target_tokens"])
        if packed < int(target * corpus_minimum_fill):
            raise RuntimeError(
                f"Corpus {key} insuficiente após tokenização: {packed:,}/{target:,}"
            )
    packed_sft = int(phase_specs["sft_train"]["target_tokens_packed"])
    sft_target = int(preset["sft_unique_tokens"])
    if packed_sft < int(sft_target * corpus_minimum_fill):
        raise RuntimeError(
            f"Corpus SFT insuficiente: {packed_sft:,}/{sft_target:,}"
        )

    # Identidade LÓGICA do corpus: quais documentos foram escolhidos, sem
    # depender de formatação, timestamp ou caminho. Distinta da identidade do
    # artefato empacotado logo abaixo — os mesmos documentos repackados com
    # outro tokenizer produziriam tokens diferentes e não seriam o mesmo
    # dataset de treino.
    dataset_identity = canonical_corpus_manifest(store, "base")
    manifest = {
        "format_version": PACK_FORMAT_VERSION,
        "created_at": utc_now(),
        "preset": preset_name,
        "dataset": {
            "manifest_content_sha256": dataset_identity[
                "manifest_content_sha256"
            ],
            "selected_documents": dataset_identity["selected_documents"],
            "selected_tokens_estimated": dataset_identity["selected_tokens"],
            "filter_version": FILTER_VERSION,
            "dedup_version": DEDUP_VERSION,
            "quota_policy_version": QUOTA_POLICY_VERSION,
            "source_mix_policy": dict(BASE_SOURCE_MIX),
            "conditional_quality_max_share": QUALITY_CONDITIONAL_MAX_SHARE,
            **(dataset_metadata or {}),
        },
        "packing": {
            "version": packing_version,
        },
        "tokenizer": {
            "path": str(tokenizer_pack.relative_to(workspace)),
            "sha256": sha256_file(tokenizer_pack),
            "vocab_size": actual_vocab,
            "special_tokens": SPECIAL_TOKENS,
            "eos_token_id": eos_token_id,
        },
        "corpus_manifest": {
            "path": str(corpus_manifest_path.relative_to(workspace)),
            "sha256": sha256_file(corpus_manifest_path),
        },
        "phases": phase_specs,
    }
    # Identidade de CONTEÚDO do artefato empacotado.
    #
    # `sha256_file(pack_manifest.json)` não serve como identidade: o manifesto
    # carrega `created_at`, então dois packs idênticos preparados em momentos
    # diferentes teriam hashes diferentes. Pior, um hash da descrição não
    # cobre o binário descrito.
    #
    # Este hash é calculado sobre os artefatos realmente consumidos — o
    # sha256 de cada shard, suas contagens, o tokenizer e a versão do
    # empacotador — ordenados deterministicamente. Uma alteração acidental no
    # conteúdo de um shard muda a identidade mesmo que o metadado externo
    # permaneça igual.
    packing_digest = hashlib.sha256()
    packing_digest.update(packing_version.encode("utf-8"))
    packing_digest.update(b"\x1e")
    packing_digest.update(
        str(manifest["tokenizer"]["sha256"]).encode("utf-8")
    )
    packing_digest.update(b"\x1e")
    # Nomes de fase ordenados: a fase é escolhida por código
    # (`phases = ["base", "continuation", "sft"]` em train), não pela ordem do
    # dicionário, então ordenar aqui é canonicalização legítima.
    #
    # Shards NÃO são ordenados: o laço de treino consome
    # `for shard_index, shard in enumerate(spec["shards"])`, ou seja, a ordem
    # da lista é a ordem de consumo. Dois manifestos com os mesmos shards em
    # ordens diferentes entregam o mesmo conteúdo em sequências diferentes de
    # updates do AdamW e produzem modelos diferentes — precisam de identidades
    # diferentes. A posição entra no hash.
    for phase_name in sorted(phase_specs):
        spec = phase_specs[phase_name]
        packing_digest.update(phase_name.encode("utf-8"))
        packing_digest.update(b"\x1d")
        for position, shard in enumerate(spec.get("shards", [])):
            packing_digest.update("\x1f".join((
                str(position),
                str(shard.get("file", "")),
                str(shard.get("sha256", "")),
                str(shard.get("tokens", shard.get("records", 0))),
            )).encode("utf-8"))
            packing_digest.update(b"\x1e")
    manifest["packing"]["content_sha256"] = packing_digest.hexdigest()

    atomic_write_json(manifest_path, manifest)
    verify_pack_manifest(pack_root, manifest)
    return manifest_path


def prepare(
    args: argparse.Namespace,
    preset_name: str,
    preset: Dict[str, Any],
    workspace: Path,
    nsos,
) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    corpus_dir = workspace / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    store = CorpusStore(corpus_dir / "corpus.sqlite3")
    try:
        identity_value = corpus_recipe_identity(
            preset_name,
            preset,
            args.seed,
            args.fixture,
            args.license_policy,
        )
        identity = stable_json(identity_value)
        previous_identity = store.get_meta("run_identity")
        if previous_identity is not None and not corpus_identity_compatible(
            previous_identity, identity_value
        ):
            raise RuntimeError(
                "O workspace contém corpus de outra receita. Use outro "
                "--workspace para preservar a auditoria."
            )
        prepared_manifest = validate_prepared_workspace(
            store,
            workspace,
            preset_name,
            preset,
            args.seed,
            args.fixture,
            args.license_policy,
        )
        if prepared_manifest is not None:
            if previous_identity is None:
                raise RuntimeError(
                    "Workspace preparado sem identidade de receita no SQLite; "
                    "a proveniencia nao pode ser migrada automaticamente"
                )
            if previous_identity != identity:
                store.set_meta("run_identity", identity)
                print(
                    "[prepare] identidade legada do corpus migrada apos "
                    "validacao integral dos artefatos",
                    flush=True,
                )
            return prepared_manifest

        if args.fixture:
            if preset_name != "smoke":
                raise RuntimeError("--fixture só é permitido com --preset smoke")
            revisions = {"fixture": stable_hash_text("nsos-ptbr-fixture-v1")}
            insert_fixture_corpus(store, preset)
        else:
            revisions_path = corpus_dir / "dataset_revisions.json"
            pinned = getattr(args, "dataset_revisions", None)
            if revisions_path.is_file():
                revisions = json.loads(revisions_path.read_text(encoding="utf-8"))
            elif pinned is not None:
                # Reconstrução A/B usa workspaces diferentes, então a segunda
                # resolveria a revisão de novo e pegaria outro commit se o
                # dataset tivesse mudado nesse intervalo — divergência por
                # causa externa ao pipeline. O pin torna a fonte parte
                # declarada do experimento.
                revisions = _read_json_object(
                    Path(pinned), "dataset_revisions fixadas"
                )
                print(
                    f"[prepare] revisões fixadas de {pinned}: "
                    f"{ {k: v[:12] for k, v in revisions.items()} }",
                    flush=True,
                )
                atomic_write_json(revisions_path, revisions)
            else:
                revisions = resolve_dataset_revisions()
                atomic_write_json(revisions_path, revisions)
                print(
                    "[prepare] revisões resolvidas agora: "
                    f"{ {k: v[:12] for k, v in revisions.items()} } "
                    "— passe --dataset-revisions com este arquivo na "
                    "reconstrução B",
                    flush=True,
                )
            ingest_base(
                store,
                preset,
                revisions["base"],
                args.seed,
                args.license_policy,
                args.shuffle_buffer,
            )
            ingest_sft(
                store,
                preset,
                revisions,
                args.seed,
                args.shuffle_buffer,
            )

        manifest = corpus_manifest(
            store,
            preset_name,
            preset,
            args.seed,
            args.fixture,
            args.license_policy,
            revisions,
        )
        corpus_manifest_path = corpus_dir / "corpus_manifest.json"
        atomic_write_json(corpus_manifest_path, manifest)

        # Dedup global e seleção por cota ANTES do tokenizer e do packing:
        # o tokenizer deve ser treinado sobre o material que realmente vai ao
        # treino, e o packer é fail-closed nas duas marcas.
        # Todas as fases precisam ser certificadas: o packer é fail-closed nas
        # duas marcas, e uma fase sem dedup/seleção não produz pack algum.
        dedup_stats: Dict[str, Any] = {}
        selection: Dict[str, Any] = {}
        for corpus_phase in ("base", "continuation", "sft"):
            stats = deduplicate_corpus_globally(store, corpus_phase)
            dedup_stats[corpus_phase] = stats
            if stats.get("documents"):
                print(
                    f"[prepare:dedup:{corpus_phase}] "
                    f"clusters={stats.get('clusters', 0):,} "
                    f"duplicatas={stats.get('duplicates_marked', 0):,} "
                    f"({stats.get('duplicate_rate', 0) * 100:.2f}%) "
                    f"maior_cluster={stats.get('cluster_size_max', 0):,} "
                    f"cross_source={stats.get('cross_source_clusters', 0):,} "
                    f"candidatos={stats.get('near_candidates_considered', 0):,}",
                    flush=True,
                )
            chosen = select_training_documents(store, corpus_phase)
            selection[corpus_phase] = chosen
            if chosen.get("selected_documents"):
                print(
                    f"[prepare:quota:{corpus_phase}] selecionados="
                    f"{chosen.get('selected_documents', 0):,} docs / "
                    f"{chosen.get('selected_tokens', 0):,} tokens"
                    + (
                        f" | mistura={chosen.get('source_share')}"
                        f" | score3={chosen.get('conditional_share')}"
                        if corpus_phase == "base" else
                        f" | politica={chosen.get('policy')}"
                    ),
                    flush=True,
                )
        atomic_write_json(
            workspace / "corpus" / "dedup_report.json",
            {
                "dedup": dedup_stats,
                "selection": selection,
                "funnel": {
                    p: corpus_funnel(store, p)
                    for p in ("base", "continuation", "sft")
                },
            },
        )

        ox3_path = train_tokenizer(
            store,
            workspace / "tokenizer",
            int(preset["target_vocab"]),
            int(preset["tokenizer_chars"]),
        )
        pack_manifest_path = build_packs(
            store,
            nsos,
            ox3_path,
            workspace,
            preset_name,
            preset,
            corpus_manifest_path,
        )
        validated_manifest = validate_prepared_workspace(
            store,
            workspace,
            preset_name,
            preset,
            args.seed,
            args.fixture,
            args.license_policy,
        )
        if validated_manifest != pack_manifest_path:
            raise RuntimeError("Validacao retornou pack manifest inesperado")
        store.set_meta("run_identity", identity)
        return pack_manifest_path
    finally:
        store.close()


class GradientHealth:
    """Agrega norma de gradiente e clipping entre checkpoints.

    O treino de 2026-08-08 rodou 11 horas sem nenhum registro disso: os campos
    existem no Trainer e nos bindings, mas nada os persistia, então não havia
    como saber a posteriori se o clipping esteve ativo. Clipping constante é o
    sintoma clássico de LR alto demais, e era justamente o diagnóstico
    impossível de fazer.

    Guardar só o valor do último passo não resolveria — ele diz o que
    aconteceu num passo arbitrário. O que informa decisão é a FRAÇÃO de
    updates cortados e a cauda da distribuição da norma.
    """

    __slots__ = ("_n", "_clipped", "_sum", "_max", "_min", "_last_pre",
                 "_last_post", "_supported")

    def __init__(self) -> None:
        self._n = 0
        self._clipped = 0
        self._sum = 0.0
        self._max = 0.0
        self._min = float("inf")
        self._last_pre = None
        self._last_post = None
        self._supported = None

    def observe(self, trainer) -> None:
        if self._supported is None:
            self._supported = hasattr(trainer, "last_grad_norm_pre_clip")
        if not self._supported:
            return
        pre = float(trainer.last_grad_norm_pre_clip)
        post = float(trainer.last_grad_norm_post_clip)
        # Norma zero/não-finita significa que nenhum update aconteceu neste
        # microbatch (acumulação pendente) ou que o passo foi rejeitado.
        # Contá-la achataria a média e esconderia a cauda que importa.
        if not math.isfinite(pre) or pre <= 0.0:
            return
        self._n += 1
        self._sum += pre
        self._max = max(self._max, pre)
        self._min = min(self._min, pre)
        if bool(trainer.last_update_was_clipped):
            self._clipped += 1
        self._last_pre = pre
        self._last_post = post

    def snapshot(self) -> Dict[str, Any]:
        if self._supported is False:
            return {"instrumented": False,
                    "reason": "runtime não expõe last_grad_norm_pre_clip"}
        if self._n == 0:
            return {"instrumented": True, "updates_observed": 0}
        payload = {
            "instrumented": True,
            "updates_observed": self._n,
            "clipped_updates": self._clipped,
            "clipped_fraction": round(self._clipped / self._n, 6),
            "grad_norm_mean_pre_clip": round(self._sum / self._n, 6),
            "grad_norm_max_pre_clip": round(self._max, 6),
            "grad_norm_min_pre_clip": round(self._min, 6),
            "last_pre_clip": round(self._last_pre, 6),
            "last_post_clip": round(self._last_post, 6),
        }
        # Janela por checkpoint: sem o reset a média viraria a média da
        # corrida inteira e deixaria de mostrar tendência.
        self._n = self._clipped = 0
        self._sum = 0.0
        self._max = 0.0
        self._min = float("inf")
        return payload


def build_model_config(
    nsos, preset: Dict[str, Any], vocab_size: int, device, args=None
):
    config = nsos.ModelConfig()
    config.num_layers = int(preset["layers"])
    config.d_model = int(preset["d_model"])
    config.vocab_size = int(vocab_size)
    config.n_heads = 12 if int(preset["d_model"]) == 768 else 4
    config.n_kv_heads = 4 if int(preset["d_model"]) == 768 else 2
    config.sliding_window = 1024
    config.attention_period = 64
    config.attention_slot = 63
    config.force_mamba_last_layer = True
    config.use_moe = False
    config.moe_period = 64
    config.moe_slot = 63
    config.use_ttt = False
    config.ttt_period = 64
    config.ttt_slot = 63
    config.use_chrass = False
    config.use_kan = False
    config.use_slender_embedding = False
    config.mamba_proper_ssm = True
    config.mamba_state_expansion = True
    config.mamba_d_state = 64
    config.mamba_conv_kernel = 4
    config.mamba2_faithful = True
    config.mamba_expand = 2
    config.mamba_head_dim = 64
    config.mamba_n_groups = 1
    config.tie_word_embeddings = True
    config.use_exact_attention_training = False
    config.use_flash_attn = False
    config.use_gradient_checkpointing = bool(
        preset.get("use_gradient_checkpointing", False)
    )
    config.dropout = float(preset["dropout"])
    config.max_context_tokens = max(1024, int(preset["seq_len"]))
    config.default_batch_size = 1
    config.use_cuda = device == nsos.Device.GPU

    # --- Sobrescritas de arquitetura vindas da CLI --------------------------
    #
    # Sem isto a arquitetura é fixa no código e as perguntas "attention
    # melhora Mamba?" e "TTT melhora?" só podem ser respondidas editando o
    # script entre corridas — o que quebra a comparabilidade, porque
    # `script_sha256` entra na identidade de execução.
    if args is not None:
        def override(name: str, attr: str | None = None) -> None:
            value = getattr(args, name, None)
            if value is not None:
                setattr(config, attr or name, value)

        override("attention_period")
        override("attention_slot")
        override("ttt_period")
        override("ttt_slot")
        override("moe_period")
        override("moe_slot")
        for flag in (
            "use_ttt", "use_moe", "use_chrass", "use_kan",
            "use_slender_embedding", "force_mamba_last_layer",
            "mamba2_faithful",
        ):
            value = getattr(args, flag, None)
            if value is not None:
                setattr(config, flag, bool(value))

        # CHRASS e slender embedding não têm caminho GPU. `ChrassLayer::forward`
        # chama `.data()` (ponteiro de host) e roda um laço CSR sob OpenMP;
        # num tensor cudaMalloc isso lança lá no fundo do forward, com uma
        # mensagem que não diz qual feature causou. Slender já falha com texto
        # próprio ("Phase 7 (not yet implemented)"). Falhar aqui, antes de
        # alocar modelo e corpus, troca um erro obscuro em runtime por um
        # diagnóstico acionável.
        if device == nsos.Device.GPU:
            unsupported = [
                name for name, on in (
                    ("--use-chrass", config.use_chrass),
                    ("--use-slender-embedding", config.use_slender_embedding),
                ) if on
            ]
            if unsupported:
                raise RuntimeError(
                    f"{' e '.join(unsupported)} não têm implementação GPU "
                    f"(forward roda no host via OpenMP). Use --device cpu, "
                    f"que é ordens de magnitude mais lento, ou rode sem essas "
                    f"flags. KAN tem caminho GPU e funciona."
                )

        # A camada final é o gatilho documentado de travamento: com
        # `mamba2_faithful`, uma camada de attention no fim da pilha prende a
        # loss em nível de acaso (isolado em 3 rodadas de auditoria). O guarda
        # existe no C++, mas silenciosamente converte a camada em Mamba — o
        # que faz `attention_period=16, slot=15` em 16 camadas produzir ZERO
        # attention enquanto o operador acredita ter pedido uma. Aqui isso
        # vira erro, não surpresa.
        layers = int(preset["layers"])
        matched = [
            i for i in range(1, layers + 1)
            if config.attention_period > 0
            and ((i - 1) % config.attention_period)
            == max(0, min(config.attention_slot, config.attention_period - 1))
        ]
        effective = [i for i in matched if not (
            config.mamba2_faithful
            and config.force_mamba_last_layer
            and i == layers
        )]
        if matched and not effective:
            raise RuntimeError(
                f"attention_period={config.attention_period} "
                f"slot={config.attention_slot} casa apenas a camada "
                f"{matched} de {layers}, que force_mamba_last_layer converte "
                f"em Mamba: a corrida seria Mamba puro. Para 1 attention em "
                f"{layers} camadas use period=8 slot=7."
            )
        print(
            f"[arquitetura] camadas={layers} attention={effective or 'nenhuma'} "
            f"ttt={'on' if config.use_ttt else 'off'} "
            f"chrass={'on' if config.use_chrass else 'off'} "
            f"kan={'on' if config.use_kan else 'off'} "
            f"moe={'on' if config.use_moe else 'off'} "
            f"slender={'on' if config.use_slender_embedding else 'off'} "
            f"faithful={'on' if config.mamba2_faithful else 'off'}",
            flush=True,
        )
    return config


def model_config_dict(config) -> Dict[str, Any]:
    names = (
        "num_layers",
        "d_model",
        "vocab_size",
        "n_heads",
        "n_kv_heads",
        "sliding_window",
        "attention_period",
        "attention_slot",
        "force_mamba_last_layer",
        "use_moe",
        "use_ttt",
        "use_chrass",
        "use_kan",
        "use_slender_embedding",
        "mamba_proper_ssm",
        "mamba_state_expansion",
        "mamba_d_state",
        "mamba_conv_kernel",
        "mamba2_faithful",
        "mamba_expand",
        "mamba_head_dim",
        "mamba_n_groups",
        "tie_word_embeddings",
        "use_gradient_checkpointing",
        "dropout",
        "max_context_tokens",
        "use_cuda",
    )
    payload = {name: getattr(config, name) for name in names}
    # `ttt_period`/`ttt_slot`/`moe_period`/`moe_slot` decidem EM QUAIS camadas
    # o bloco entra, mas ficavam fora da identidade: dois runs com TTT em
    # camadas diferentes tinham o mesmo digest e podiam retomar um do
    # checkpoint do outro. Emitidos só quando o bloco está ligado, para não
    # alterar o digest das corridas Mamba-only já certificadas.
    if getattr(config, "use_ttt", False):
        payload["ttt_period"] = config.ttt_period
        payload["ttt_slot"] = config.ttt_slot
    if getattr(config, "use_moe", False):
        payload["moe_period"] = config.moe_period
        payload["moe_slot"] = config.moe_slot
    if getattr(config, "use_chrass", False):
        payload["chrass_density"] = config.chrass_density
        payload["chrass_seed"] = config.chrass_seed
    return payload


def parameter_inventory(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    tensors = 0
    for parameter in model.parameters():
        size = int(parameter.data.size)
        total += size
        tensors += 1
        if bool(parameter.trainable):
            trainable += size
    return {
        "registry_parameters": total,
        "trainable_parameters": trainable,
        "parameter_tensors": tensors,
    }


def phase_step_counts(
    pack_root: Path, manifest: Dict[str, Any], preset: Dict[str, Any]
) -> Dict[str, int]:
    seq_len = int(preset["seq_len"])
    causal_batch = int(preset["causal_batch"])
    base = sum(
        max((int(shard["tokens"]) - 1) // (seq_len * causal_batch), 0)
        for shard in manifest["phases"]["base_train"]["shards"]
    )
    continuation = sum(
        max((int(shard["tokens"]) - 1) // (seq_len * causal_batch), 0)
        for shard in manifest["phases"]["continuation_train"]["shards"]
    )
    sft_records = sum(
        int(shard["records"])
        for shard in manifest["phases"]["sft_train"]["shards"]
    )
    sft = math.ceil(sft_records / max(int(preset["sft_batch"]), 1)) * int(
        preset["sft_epochs"]
    )
    return {"base": base, "continuation": continuation, "sft": sft}


class CheckpointManager:
    def __init__(
        self,
        run_dir: Path,
        model,
        trainer,
        tokenizer_path: Path,
        immutable_identity: Dict[str, Any],
        keep_periodic: int,
        allow_legacy_runtime_identity: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.root = run_dir / "checkpoints"
        self.generations = self.root / "generations"
        self.generations.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.trainer = trainer
        self.tokenizer_path = tokenizer_path
        self.identity = immutable_identity
        self.identity_sha256 = stable_hash_text(stable_json(immutable_identity))
        self.keep_periodic = max(int(keep_periodic), 1)
        self.allow_legacy_runtime_identity = bool(
            allow_legacy_runtime_identity
        )
        self._checkpoint_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nsos-checkpoint",
        )
        self._pending_checkpoint: Tuple[Future[Path], Path] | None = None
        # A publisher failure poisons the manager. Capturing another full host
        # snapshot after durable I/O has already failed can hide the original
        # fault and doubles memory pressure during exception handling.
        self._checkpoint_failure: BaseException | None = None
        self._timing_lock = threading.Lock()
        self._checkpoint_timings: List[Dict[str, Any]] = []
        self._closed = False

    def _legacy_identity_compatible(self, stored: Any) -> bool:
        if not self.allow_legacy_runtime_identity or not isinstance(stored, dict):
            return False
        stored_script_version = stored.get("script_version")
        if not isinstance(stored_script_version, int) or not (
            1 <= stored_script_version <= SCRIPT_VERSION
        ):
            return False
        required = {
            "script_version",
            "preset",
            "preset_config",
            "seed",
            "model_config",
            "pack_manifest_sha256",
            "tokenizer_sha256",
            "build_dir",
            "deterministic_reductions",
        }
        if not required.issubset(stored):
            return False
        for key, value in stored.items():
            if key == "script_version":
                continue
            if self.identity.get(key) != value:
                return False
        return True

    def _manifest_valid(self, generation: Path) -> Tuple[bool, Dict[str, Any] | None]:
        try:
            committed_path = generation / "COMMITTED"
            manifest_path = generation / "checkpoint_manifest.json"
            if not committed_path.is_file() or not manifest_path.is_file():
                return False, None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            format_version = int(manifest.get("format_version", 0))
            if format_version <= 0 or format_version > CHECKPOINT_FORMAT_VERSION:
                return False, None
            if format_version >= 3:
                committed = json.loads(
                    committed_path.read_text(encoding="ascii")
                )
                if (
                    int(committed.get("format_version", 0)) != 1
                    or committed.get("manifest_sha256")
                    != sha256_file(manifest_path)
                    or committed.get("created_at") != manifest.get("created_at")
                ):
                    return False, None
            if manifest.get("identity_sha256") != self.identity_sha256:
                if not self._legacy_identity_compatible(
                    manifest.get("identity")
                ):
                    return False, None
                manifest["legacy_runtime_identity_migration"] = True
            for name, expected in manifest["files"].items():
                path = generation / name
                if not path.is_file() or sha256_file(path) != expected:
                    return False, None
            return True, manifest
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False, None

    def scan(self) -> List[Tuple[Path, Dict[str, Any]]]:
        valid: List[Tuple[Path, Dict[str, Any]]] = []
        for generation in self.generations.iterdir():
            if not generation.is_dir() or generation.name.startswith(".staging-"):
                continue
            ok, manifest = self._manifest_valid(generation)
            if ok and manifest is not None:
                valid.append((generation, manifest))
        valid.sort(
            key=lambda item: (
                int(item[1].get("global_step", -1)),
                str(item[1].get("created_at", "")),
            )
        )
        return valid

    def latest(self) -> Tuple[Path, Dict[str, Any]] | None:
        pointer = self.root / "LATEST.json"
        if pointer.is_file():
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                generation = self.generations / str(payload["generation"])
                ok, manifest = self._manifest_valid(generation)
                if ok and manifest is not None:
                    return generation, manifest
            except (OSError, KeyError, json.JSONDecodeError) as pointer_error:
                print(
                    "[checkpoint] aviso: LATEST.json inválido; "
                    "verificando gerações comprometidas: "
                    f"{pointer_error}",
                    file=sys.stderr,
                    flush=True,
                )
        valid = self.scan()
        return valid[-1] if valid else None

    def _publish_snapshot(
        self,
        snapshot,
        final: Path,
        staging: Path,
        progress_payload: Dict[str, Any],
        kind: str,
        label: str,
        step: int,
        capture_ms: float,
    ) -> Path:
        publish_started = time.perf_counter()
        staging.mkdir()
        model_path = staging / "model.bin"
        state_path = staging / "trainer.state"
        tokenizer_copy = staging / "tokenizer.nsos"
        write_started = time.perf_counter()
        snapshot.write(str(model_path), str(state_path))
        # A checkpoint copy is content-addressed below; inherited read-only
        # metadata would prevent the required Windows durability flush.
        shutil.copyfile(self.tokenizer_path, tokenizer_copy)
        atomic_write_json(staging / "progress.json", progress_payload)
        for durable_path in (
            model_path,
            state_path,
            tokenizer_copy,
            staging / "progress.json",
        ):
            fsync_file(durable_path)
        snapshot_write_ms = (time.perf_counter() - write_started) * 1_000.0
        hash_started = time.perf_counter()
        files = {
            name: sha256_file(staging / name)
            for name in ("model.bin", "trainer.state", "tokenizer.nsos", "progress.json")
        }
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "created_at": utc_now(),
            "global_step": step,
            "kind": kind,
            "label": label,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity,
            "files": files,
            "checkpoint_timing": {
                "capture_ms": float(capture_ms),
                "snapshot_write_and_copy_ms": snapshot_write_ms,
                "pre_manifest_hash_ms": (
                    time.perf_counter() - hash_started
                )
                * 1_000.0,
            },
        }
        manifest_path = staging / "checkpoint_manifest.json"
        atomic_write_json(manifest_path, manifest)
        fsync_file(manifest_path)
        manifest_sha256 = sha256_file(manifest_path)
        with (staging / "COMMITTED").open("w", encoding="ascii") as handle:
            handle.write(
                json.dumps(
                    {
                        "format_version": 1,
                        "created_at": manifest["created_at"],
                        "manifest_sha256": manifest_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(staging)
        rename_error: OSError | None = None
        for attempt in range(6):
            try:
                os.rename(staging, final)
                fsync_directory(self.generations)
                rename_error = None
                break
            except OSError as exc:
                rename_error = exc
                time.sleep(0.05 * (2**attempt))
        if rename_error is not None:
            # OneDrive/antivírus pode manter um handle transitório no diretório e
            # negar a renomeação. O marcador COMMITTED continua fornecendo a
            # transação: copiamos todos os componentes, verificamos os hashes e
            # publicamos COMMITTED por último. Sem o marcador, o scanner ignora
            # a geração após qualquer queda intermediária.
            final.mkdir()
            for child in staging.iterdir():
                if child.name == "COMMITTED":
                    continue
                shutil.copy2(child, final / child.name)
                fsync_file(final / child.name)
            for name, expected in files.items():
                if sha256_file(final / name) != expected:
                    raise RuntimeError(
                        f"Fallback de commit corrompeu {final / name}"
                    )
            if sha256_file(final / "checkpoint_manifest.json") != manifest_sha256:
                raise RuntimeError(
                    "Fallback de commit corrompeu o manifesto do checkpoint"
                )
            committed_tmp = final / "COMMITTED.tmp"
            shutil.copy2(staging / "COMMITTED", committed_tmp)
            fsync_file(committed_tmp)
            os.replace(committed_tmp, final / "COMMITTED")
            fsync_file(final / "COMMITTED")
            fsync_directory(final)
            fsync_directory(self.generations)
            try:
                shutil.rmtree(staging)
            except OSError as cleanup_error:
                # The committed generation is already durable and the scanner
                # deliberately ignores .staging-* directories. Preserve that
                # valid commit, but never hide the residual-directory failure.
                print(
                    "[checkpoint] aviso: não foi possível remover staging "
                    f"residual {staging}: {cleanup_error}",
                    file=sys.stderr,
                    flush=True,
                )
        atomic_write_json(
            self.root / "LATEST.json",
            {
                "format_version": 1,
                "generation": final.name,
                "global_step": step,
                "manifest_sha256": manifest_sha256,
                "updated_at": utc_now(),
            },
        )
        fsync_directory(self.root)
        self._apply_retention()
        total_publish_ms = (time.perf_counter() - publish_started) * 1_000.0
        with self._timing_lock:
            self._checkpoint_timings.append(
                {
                    "global_step": int(step),
                    "kind": str(kind),
                    "label": str(label),
                    "capture_ms": float(capture_ms),
                    "snapshot_write_and_copy_ms": snapshot_write_ms,
                    "publish_total_ms": total_publish_ms,
                }
            )
        print(f"[checkpoint] committed {final}", flush=True)
        return final

    def _drain_pending(self, *, wait: bool) -> Path | None:
        pending = self._pending_checkpoint
        if pending is None:
            return None
        future, expected = pending
        if not wait and not future.done():
            return None
        try:
            result = future.result()
            if result != expected:
                raise RuntimeError(
                    "Worker de checkpoint publicou uma geração inesperada"
                )
            return result
        except BaseException as exc:
            self._checkpoint_failure = exc
            raise
        finally:
            self._pending_checkpoint = None

    def _raise_if_checkpoint_failed(self) -> None:
        if self._checkpoint_failure is not None:
            raise RuntimeError(
                "CheckpointManager recusou nova operação após falha "
                "assíncrona anterior"
            ) from self._checkpoint_failure

    def poll(self) -> None:
        """Surface worker failures at the next completed-step safe point."""
        self._raise_if_checkpoint_failed()
        self._drain_pending(wait=False)

    def save(
        self,
        progress: Dict[str, Any],
        *,
        kind: str,
        label: str,
        latest_loss: float | None,
    ) -> Path:
        if self._closed:
            raise RuntimeError("CheckpointManager já foi encerrado")
        self._raise_if_checkpoint_failed()
        # At most one host snapshot is resident. This bounds RAM and makes a
        # previous I/O failure visible before a newer generation is captured.
        self._drain_pending(wait=True)
        capture_started = time.perf_counter()
        snapshot = self.trainer.capture_checkpoint_snapshot()
        capture_ms = (time.perf_counter() - capture_started) * 1_000.0
        step = int(snapshot.global_step)
        if step != int(self.trainer.global_step_count):
            raise RuntimeError(
                "Checkpoint snapshot step diverged from the trainer safe point"
            )
        stem = f"step-{step:012d}-{safe_component(label)}"
        final = self.generations / stem
        staging = self.generations / f".staging-{final.name}"
        suffix = 1
        while final.exists() or staging.exists():
            final = self.generations / f"{stem}-{suffix}"
            staging = self.generations / f".staging-{final.name}"
            suffix += 1
        # JSON round-trip produces a detached, worker-owned graph made only of
        # the manifest's already-supported scalar/list/dict types.
        progress_snapshot = json.loads(stable_json(progress))
        progress_payload = {
            **progress_snapshot,
            "global_step": step,
            "latest_loss": latest_loss,
            "saved_at": utc_now(),
        }
        future = self._checkpoint_executor.submit(
            self._publish_snapshot,
            snapshot,
            final,
            staging,
            progress_payload,
            str(kind),
            str(label),
            step,
            capture_ms,
        )
        self._pending_checkpoint = (future, final)
        print(f"[checkpoint] snapshot queued {final}", flush=True)
        return final

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._drain_pending(wait=True)
        finally:
            self._closed = True
            self._checkpoint_executor.shutdown(wait=True, cancel_futures=False)

    def load_latest(self) -> Dict[str, Any] | None:
        latest = self.latest()
        if latest is None:
            existing = [
                path
                for path in self.generations.iterdir()
                if path.is_dir() and not path.name.startswith(".staging-")
            ]
            if existing:
                raise RuntimeError(
                    "Há checkpoints no run-dir, mas nenhum satisfaz hashes "
                    "e identidade de execução atuais. A retomada foi recusada "
                    "para evitar reinício silencioso. Para uma geração v1 "
                    "já auditada, use --allow-legacy-runtime-identity uma única vez."
                )
            return None
        generation, manifest = latest
        self.model.load(str(generation / "model.bin"), True)
        self.trainer.load_training_state(
            str(generation / "trainer.state"),
            str(generation / "model.bin"),
            self.allow_legacy_runtime_identity,
        )
        progress = json.loads(
            (generation / "progress.json").read_text(encoding="utf-8")
        )
        print(
            f"[resume] geração={generation.name} "
            f"global_step={self.trainer.global_step_count}",
            flush=True,
        )
        return progress

    def _apply_retention(self) -> None:
        periodic = [
            item
            for item in self.scan()
            if str(item[1].get("kind")) == "periodic"
        ]
        for generation, _manifest in periodic[: -self.keep_periodic]:
            resolved = generation.resolve()
            root = self.generations.resolve()
            if resolved.parent != root or not resolved.name.startswith("step-"):
                raise RuntimeError(f"Retenção recusou caminho inesperado: {resolved}")
            shutil.rmtree(resolved)

    def verify_all(self) -> Dict[str, Any]:
        self._raise_if_checkpoint_failed()
        self._drain_pending(wait=True)
        valid = self.scan()
        all_dirs = [
            path
            for path in self.generations.iterdir()
            if path.is_dir() and not path.name.startswith(".staging-")
        ]
        latest_generation, latest_manifest = (
            valid[-1] if valid else (None, None)
        )
        return {
            "valid": len(valid),
            "invalid": len(all_dirs) - len(valid),
            "latest": (
                latest_generation.name
                if latest_generation is not None
                else None
            ),
            "global_step": (
                int(latest_manifest["global_step"])
                if latest_manifest is not None
                else 0
            ),
            "latest_manifest_sha256": (
                sha256_file(
                    latest_generation / "checkpoint_manifest.json"
                )
                if latest_generation is not None
                else None
            ),
            "latest_identity_sha256": (
                str(latest_manifest["identity_sha256"])
                if latest_manifest is not None
                else None
            ),
            # These content hashes intentionally exclude timestamps and I/O
            # timing. Two fresh deterministic benchmark runs must reproduce
            # model.bin and trainer.state byte for byte.
            "latest_file_sha256": (
                dict(latest_manifest["files"])
                if latest_manifest is not None
                else {}
            ),
        }

    def timing_summary(self) -> Dict[str, Any]:
        with self._timing_lock:
            timings = [dict(item) for item in self._checkpoint_timings]
        return {
            "completed": len(timings),
            "pending": self._pending_checkpoint is not None,
            "capture_ms_total": sum(item["capture_ms"] for item in timings),
            "publish_ms_total": sum(
                item["publish_total_ms"] for item in timings
            ),
            "last": timings[-1] if timings else None,
            "records": timings,
        }


def causal_validation_loss(
    nsos,
    model,
    shard_path: Path,
    seq_len: int,
    max_windows: int,
) -> float:
    tokens = read_u16_tokens(shard_path)
    losses: List[float] = []
    previous_mode = bool(model.training_mode())
    model.set_training_mode(False)
    try:
        # A window is valid while start + seq_len + 1 <= len(tokens).  The
        # exclusive range bound is therefore len(tokens) - seq_len; subtracting
        # one again skipped an exact single-window validation shard and the last
        # aligned window of larger shards.
        for start in range(0, max(0, len(tokens) - seq_len), seq_len):
            if len(losses) >= max_windows:
                break
            window = tokens[start : start + seq_len + 1]
            inputs = window[:-1]
            targets = np.asarray(window[1:], dtype=np.int64)
            model.reset_session()
            logits_tensor = model.forward_ids(inputs, None)
            if logits_tensor.device == nsos.Device.GPU:
                logits_tensor = logits_tensor.cpu()
            logits = np.asarray(logits_tensor.numpy(), dtype=np.float64).reshape(
                len(inputs), -1
            )
            maxima = logits.max(axis=1, keepdims=True)
            logsumexp = maxima[:, 0] + np.log(
                np.exp(logits - maxima).sum(axis=1)
            )
            losses.append(
                float(np.mean(logsumexp - logits[np.arange(len(targets)), targets]))
            )
    finally:
        model.reset_session()
        model.set_training_mode(previous_mode)
        tokens.close()
    return float(np.mean(losses)) if losses else float("nan")


def sft_validation_loss(
    nsos,
    model,
    shard_path: Path,
    max_records: int,
) -> float:
    records_shard = read_sft_records(shard_path)
    records = records_shard[:max_records]
    losses: List[float] = []
    previous_mode = bool(model.training_mode())
    model.set_training_mode(False)
    try:
        for prompt, answer in records:
            inputs = prompt + answer[:-1]
            model.reset_session()
            logits_tensor = model.forward_ids(inputs, None)
            if logits_tensor.device == nsos.Device.GPU:
                logits_tensor = logits_tensor.cpu()
            logits = np.asarray(logits_tensor.numpy(), dtype=np.float64).reshape(
                len(inputs), -1
            )
            start = len(prompt) - 1
            selected = logits[start : start + len(answer)]
            targets = np.asarray(answer, dtype=np.int64)
            maxima = selected.max(axis=1, keepdims=True)
            logsumexp = maxima[:, 0] + np.log(
                np.exp(selected - maxima).sum(axis=1)
            )
            losses.append(
                float(
                    np.mean(
                        logsumexp - selected[np.arange(len(targets)), targets]
                    )
                )
            )
    finally:
        model.reset_session()
        model.set_training_mode(previous_mode)
        records_shard.close()
    return float(np.mean(losses)) if losses else float("nan")


def phase_learning_rate(base_lr: float, phase: str) -> float:
    return base_lr * {"base": 1.0, "continuation": 0.5, "sft": 0.3}[phase]


TOKEN_RATE_KEYS = (
    "raw_tokens",
    "non_padding_tokens",
    "supervised_tokens",
    "assistant_tokens",
)
NATIVE_TIMING_KEYS = (
    "wall_ms",
    "preparation_ms",
    "inter_bucket_ms",
    "forward_ms",
    "loss_ms",
    "backward_ms",
    "optimizer_ms",
    "unaccounted_ms",
)


def trainer_timing_snapshot(trainer) -> Dict[str, Any]:
    timing = trainer.last_step_telemetry
    return {
        "enabled": bool(timing.enabled),
        "global_step": int(timing.global_step),
        "bucket_count": int(timing.bucket_count),
        **{
            key: float(getattr(timing, key))
            for key in NATIVE_TIMING_KEYS
        },
    }


def sft_step_token_counts(
    prompts: Sequence[Sequence[int]],
    answers: Sequence[Sequence[int]],
) -> Dict[str, int]:
    """Mirror the native deterministic length buckets for exact padding counts."""
    if not prompts or len(prompts) != len(answers):
        raise ValueError("SFT token telemetry requires aligned non-empty batches")
    order = sorted(
        range(len(prompts)),
        key=lambda index: (
            -(len(prompts[index]) + len(answers[index])),
            -len(prompts[index]),
            -len(answers[index]),
        ),
    )
    raw_tokens = 0
    non_padding_tokens = 0
    assistant_tokens = 0
    order_index = 0
    while order_index < len(order):
        first = order[order_index]
        max_input = len(prompts[first]) + len(answers[first]) - 1
        min_input = max_input
        max_answer = len(answers[first])
        min_answer = max_answer
        bucket: List[int] = []
        while order_index < len(order):
            index = order[order_index]
            input_len = len(prompts[index]) + len(answers[index]) - 1
            answer_len = len(answers[index])
            candidate_max_input = max(max_input, input_len)
            candidate_min_input = min(min_input, input_len)
            candidate_max_answer = max(max_answer, answer_len)
            candidate_min_answer = min(min_answer, answer_len)
            compatible = not bucket or (
                candidate_max_input - candidate_min_input
                <= max(12, candidate_max_input // 3)
                and candidate_max_answer - candidate_min_answer
                <= max(8, candidate_max_answer // 2)
            )
            if not compatible:
                break
            max_input = candidate_max_input
            min_input = candidate_min_input
            max_answer = candidate_max_answer
            min_answer = candidate_min_answer
            bucket.append(index)
            order_index += 1
        raw_tokens += len(bucket) * max_input
        non_padding_tokens += sum(
            len(prompts[index]) + len(answers[index]) - 1
            for index in bucket
        )
        assistant_tokens += sum(len(answers[index]) for index in bucket)
    return {
        "raw_tokens": raw_tokens,
        "non_padding_tokens": non_padding_tokens,
        "supervised_tokens": assistant_tokens,
        "assistant_tokens": assistant_tokens,
    }


class RollingTrainingTelemetry:
    """Bounded rolling rates plus per-phase and percentile-ready aggregates."""

    def __init__(
        self, window_steps: int = 100, benchmark_steps: int = 1_000
    ) -> None:
        if window_steps <= 0:
            raise ValueError("telemetry window_steps must be positive")
        if benchmark_steps <= 0:
            raise ValueError("telemetry benchmark_steps must be positive")
        self.window_steps = int(window_steps)
        self.benchmark_steps = int(benchmark_steps)
        self._window = deque(maxlen=self.window_steps)
        # Only the pre-registered benchmark prefix needs individual samples for
        # percentiles. Run/phase summaries use constant-memory online totals;
        # logging therefore stays O(window + phases), not O(global_step).
        self._benchmark_samples: List[Dict[str, Any]] = []
        self._run_aggregate = self._new_accumulator()
        self._phase_aggregates: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _new_accumulator() -> Dict[str, Any]:
        return {
            "steps": 0,
            "elapsed_seconds": 0.0,
            "loss_first": None,
            "loss_last": None,
            "loss_sum": 0.0,
            "tokens": {key: 0 for key in TOKEN_RATE_KEYS},
            "native_timing_samples": 0,
            "native_timing_sums_ms": {
                key: 0.0 for key in NATIVE_TIMING_KEYS
            },
        }

    @staticmethod
    def _update_accumulator(
        accumulator: Dict[str, Any], sample: Dict[str, Any]
    ) -> None:
        if accumulator["steps"] == 0:
            accumulator["loss_first"] = float(sample["loss"])
        accumulator["steps"] += 1
        accumulator["elapsed_seconds"] += float(sample["elapsed_seconds"])
        accumulator["loss_last"] = float(sample["loss"])
        accumulator["loss_sum"] += float(sample["loss"])
        for key in TOKEN_RATE_KEYS:
            accumulator["tokens"][key] += int(sample[key])
        timing = sample["native_timing"]
        if bool(timing.get("enabled", False)):
            accumulator["native_timing_samples"] += 1
            for key in NATIVE_TIMING_KEYS:
                accumulator["native_timing_sums_ms"][key] += float(
                    timing.get(key, 0.0)
                )

    @staticmethod
    def _finalize_accumulator(accumulator: Dict[str, Any]) -> Dict[str, Any]:
        steps = int(accumulator["steps"])
        elapsed = float(accumulator["elapsed_seconds"])
        timed = int(accumulator["native_timing_samples"])
        result: Dict[str, Any] = {
            "steps": steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / elapsed if elapsed > 0.0 else 0.0,
            "loss_first": accumulator["loss_first"],
            "loss_last": accumulator["loss_last"],
            "loss_mean": (
                float(accumulator["loss_sum"]) / steps if steps else None
            ),
        }
        for key in TOKEN_RATE_KEYS:
            total = int(accumulator["tokens"][key])
            result[key] = total
            result[f"{key}_per_second"] = (
                total / elapsed if elapsed > 0.0 else 0.0
            )
        result["native_timing_samples"] = timed
        result["native_timing_mean_ms"] = {
            key: (
                float(accumulator["native_timing_sums_ms"][key]) / timed
                if timed
                else None
            )
            for key in NATIVE_TIMING_KEYS
        }
        return result

    def observe(
        self,
        *,
        phase: str,
        global_step: int,
        elapsed_seconds: float,
        loss: float,
        token_counts: Dict[str, int],
        native_timing: Dict[str, Any],
    ) -> None:
        if elapsed_seconds <= 0.0 or not math.isfinite(elapsed_seconds):
            raise RuntimeError("Non-positive/non-finite training step duration")
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite loss in training telemetry")
        counts: Dict[str, int] = {}
        for key in TOKEN_RATE_KEYS:
            value = int(token_counts.get(key, 0))
            if value < 0:
                raise RuntimeError(f"Negative token telemetry: {key}={value}")
            counts[key] = value
        if counts["non_padding_tokens"] > counts["raw_tokens"]:
            raise RuntimeError("Non-padding token count exceeds raw token count")
        if counts["assistant_tokens"] > counts["supervised_tokens"]:
            raise RuntimeError("Assistant token count exceeds supervised count")
        sample = {
            "phase": str(phase),
            "global_step": int(global_step),
            "elapsed_seconds": float(elapsed_seconds),
            "loss": float(loss),
            **counts,
            "native_timing": dict(native_timing),
        }
        self._window.append(sample)
        if len(self._benchmark_samples) < self.benchmark_steps:
            self._benchmark_samples.append(sample)
        self._update_accumulator(self._run_aggregate, sample)
        phase_key = str(phase)
        phase_accumulator = self._phase_aggregates.get(phase_key)
        if phase_accumulator is None:
            phase_accumulator = self._new_accumulator()
            self._phase_aggregates[phase_key] = phase_accumulator
        self._update_accumulator(phase_accumulator, sample)

    @staticmethod
    def _aggregate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        elapsed = sum(float(sample["elapsed_seconds"]) for sample in samples)
        result: Dict[str, Any] = {
            "steps": len(samples),
            "elapsed_seconds": elapsed,
            "steps_per_second": len(samples) / elapsed if elapsed > 0.0 else 0.0,
            "loss_first": float(samples[0]["loss"]) if samples else None,
            "loss_last": float(samples[-1]["loss"]) if samples else None,
            "loss_mean": (
                sum(float(sample["loss"]) for sample in samples) / len(samples)
                if samples
                else None
            ),
        }
        for key in TOKEN_RATE_KEYS:
            total = sum(int(sample[key]) for sample in samples)
            result[key] = total
            result[f"{key}_per_second"] = total / elapsed if elapsed > 0.0 else 0.0
        timed = [
            sample["native_timing"]
            for sample in samples
            if bool(sample["native_timing"].get("enabled", False))
        ]
        result["native_timing_samples"] = len(timed)
        result["native_timing_mean_ms"] = {
            key: (
                sum(float(item.get(key, 0.0)) for item in timed) / len(timed)
                if timed
                else None
            )
            for key in NATIVE_TIMING_KEYS
        }
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "window_steps": self.window_steps,
            "rolling": self._aggregate(list(self._window)),
            "phases": {
                phase: self._finalize_accumulator(accumulator)
                for phase, accumulator in sorted(
                    self._phase_aggregates.items()
                )
            },
            "run": self._finalize_accumulator(self._run_aggregate),
        }

    def benchmark_percentiles(
        self,
        *,
        warmup_steps: int = 100,
        first_steps: int = 1_000,
    ) -> Dict[str, Any]:
        requested = int(first_steps)
        warmup = int(warmup_steps)
        if requested < 0 or warmup < 0:
            raise ValueError("benchmark step counts must be non-negative")
        if requested > self.benchmark_steps:
            raise ValueError(
                "requested benchmark prefix exceeds retained telemetry capacity"
            )
        selected = self._benchmark_samples[:requested]
        measured = selected[warmup:]
        percentiles: Dict[str, Dict[str, float] | None] = {}
        for key in (*TOKEN_RATE_KEYS, "steps"):
            values = []
            for sample in measured:
                numerator = 1 if key == "steps" else int(sample[key])
                values.append(numerator / float(sample["elapsed_seconds"]))
            percentiles[f"{key}_per_second"] = (
                {
                    "p10": float(np.percentile(values, 10)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                }
                if values
                else None
            )
        return {
            "requested_first_steps": requested,
            "warmup_steps": warmup,
            "observed_steps": len(selected),
            "measured_steps": len(measured),
            "percentiles": percentiles,
        }

    def determinism_trace(self) -> List[Dict[str, Any]]:
        """Return the bounded, timing-free prefix used for repeat-run parity."""
        return [
            {
                "run_step": index + 1,
                "global_step": int(sample["global_step"]),
                "phase": str(sample["phase"]),
                "loss_hex": float(sample["loss"]).hex(),
                **{
                    key: int(sample[key])
                    for key in TOKEN_RATE_KEYS
                },
            }
            for index, sample in enumerate(self._benchmark_samples)
        ]


def runtime_resource_snapshot(nsos, model, trainer) -> Dict[str, Any]:
    return {
        "execution_identity": dict(trainer.execution_identity()),
        "execution_identity_sha256": trainer.execution_identity_digest(),
        "model_runtime": audit_deterministic_gpu_runtime(model),
        "pool": dict(nsos.pool_stats()),
        "transfers": dict(nsos.gpu_transfer_stats()),
        "lowp_weight_cache": dict(nsos.lowp_weight_cache_stats()),
    }


def audit_deterministic_gpu_runtime(model) -> Dict[str, Any]:
    """Falha fechado se o treino escapar dos kernels GPU determinísticos."""
    telemetry = dict(model.runtime_telemetry())
    forbidden = {
        "mamba_fast_path_fallbacks": int(
            telemetry.get("mamba_fast_path_fallbacks", 0)
        ),
        "faithful_forward_host_fallbacks": int(
            telemetry.get("faithful_forward_host_fallbacks", 0)
        ),
        "faithful_backward_host_fallbacks": int(
            telemetry.get("faithful_backward_host_fallbacks", 0)
        ),
        "faithful_streaming_host_fallbacks": int(
            telemetry.get("faithful_streaming_host_fallbacks", 0)
        ),
        "stream_priming_host_fallbacks": int(
            telemetry.get("stream_priming_host_fallbacks", 0)
        ),
        "faithful_scalar_atomic_backward_calls": int(
            telemetry.get("faithful_scalar_atomic_backward_calls", 0)
        ),
        "faithful_generic_atomic_conv_backward_calls": int(
            telemetry.get("faithful_generic_atomic_conv_backward_calls", 0)
        ),
    }
    active_forbidden = {name: value for name, value in forbidden.items() if value}
    if active_forbidden:
        raise RuntimeError(
            "Runtime AMD saiu do caminho determinístico: "
            + stable_json(active_forbidden)
        )
    if int(telemetry.get("faithful_backward_gpu_calls", 0)) > 0 and int(
        telemetry.get("faithful_deterministic_backward_calls", 0)
    ) <= 0:
        raise RuntimeError(
            "Backward Mamba GPU ocorreu sem registrar o kernel determinístico"
        )
    return telemetry


def should_checkpoint(
    global_step: int,
    last_checkpoint_time: float,
    every_steps: int,
    every_minutes: float,
) -> bool:
    by_step = every_steps > 0 and global_step % every_steps == 0
    by_time = (
        every_minutes > 0
        and time.monotonic() - last_checkpoint_time >= every_minutes * 60.0
    )
    return by_step or by_time


def train(
    args: argparse.Namespace,
    preset_name: str,
    preset: Dict[str, Any],
    workspace: Path,
    run_dir: Path,
    nsos,
    *,
    on_progress=None,
) -> int:
    manifest_path = validate_training_workspace(
        workspace,
        preset_name,
        preset,
        args.seed,
        args.fixture,
        args.license_policy,
    )
    manifest = _read_json_object(manifest_path, "Pack manifest")
    tokenizer_path = workspace / manifest["tokenizer"]["path"]
    tokenizer = nsos.Tokenizer()
    tokenizer.load(str(tokenizer_path))
    vocab_size = int(tokenizer.vocab_size)
    eos_token_id = int(manifest["tokenizer"]["eos_token_id"])

    if args.device == "gpu":
        device = nsos.Device.GPU
    elif args.device == "cpu":
        device = nsos.Device.CPU
    else:
        device = nsos.Device.GPU
        if hasattr(nsos, "fast_gpu_supported") and not nsos.fast_gpu_supported():
            device = nsos.Device.CPU
    if device == nsos.Device.GPU and hasattr(nsos, "fast_gpu_supported"):
        if not nsos.fast_gpu_supported():
            raise RuntimeError("Backend GPU rápido indisponível para --device gpu")

    nsos.set_seed(args.seed)
    if hasattr(nsos, "set_deterministic_reductions"):
        nsos.set_deterministic_reductions(True)
    if device == nsos.Device.GPU:
        if not hasattr(nsos, "set_strict_gpu_execution"):
            raise RuntimeError(
                "Runtime GPU não expõe o contrato fail-closed obrigatório"
            )
        nsos.set_strict_gpu_execution(True)
        if not bool(nsos.strict_gpu_execution()):
            raise RuntimeError(
                "Runtime recusou o modo strict_gpu_execution obrigatório"
            )
    # Compute precision of the GEMMs only. Master weights, gradients and
    # accumulators stay FP32; the runtime records the selected mode in its
    # execution identity, so a reduced-precision checkpoint can never resume
    # as FP32.
    matmul_precision = getattr(args, "matmul_precision", "fp32")
    if matmul_precision != "fp32":
        if device != nsos.Device.GPU:
            raise RuntimeError(
                "--matmul-precision reduzida exige --device gpu"
            )
        if not hasattr(nsos, "set_matmul_precision"):
            raise RuntimeError(
                "Runtime não expõe set_matmul_precision; reconstrua nsos_ext"
            )
    nsos.set_matmul_precision(matmul_precision)

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = build_model_config(nsos, preset, vocab_size, device, args)
    native_wave = 32
    if device == nsos.Device.GPU and getattr(args, "gpu_training_profile", "inherit") == "redesign-v1":
        selected = int(nsos.selected_gpu_device())
        devices = nsos.gpu_devices()
        native_wave = int(next(item["warp_size"] for item in devices if int(item["index"]) == selected))
    gpu_training_policy = configure_gpu_training_profile(args, config, device == nsos.Device.GPU, native_wave)
    config_dict = model_config_dict(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "effective_model_config.json", config_dict)
    atomic_write_json(run_dir / "effective_gpu_training_policy.json", gpu_training_policy)
    model = nsos.JambaModel(config, device)
    model.to(device)
    inventory = parameter_inventory(model)
    print(
        f"[model] Mamba-only layers={preset['layers']} d_model={preset['d_model']} "
        f"vocab={vocab_size} trainable={inventory['trainable_parameters']:,} "
        f"device={'GPU' if device == nsos.Device.GPU else 'CPU'}",
        flush=True,
    )

    steps_by_phase = phase_step_counts(workspace / "packs", manifest, preset)
    total_steps = sum(steps_by_phase.values())
    trainer = nsos.Trainer(model, float(preset["lr"]))
    trainer.weight_decay = float(preset["weight_decay"])
    # Teto da norma do gradiente. A instrumentação de 2026-08-09 mediu
    # clipped_fraction=1.0 com o valor 1.0 fixo: TODO update era cortado,
    # norma pré-clip média ~30. Com isso o clipping deixa de ser rede de
    # segurança e vira normalização permanente — o modelo usa a direção do
    # gradiente e descarta a magnitude. Configurável para poder medir.
    trainer.max_grad_norm = float(getattr(args, 'max_grad_norm', None) or 1.0)
    trainer.warmup_steps = int(preset["warmup_steps"])
    trainer.min_learning_rate_scale = 0.10
    trainer.first_token_loss_scale = 1.0
    trainer.eos_loss_scale = 0.5
    trainer.repetition_unlikelihood_scale = 0.0
    trainer.eos_token_id = eos_token_id
    trainer.total_training_steps = max(total_steps, 1)
    accumulation = getattr(args, "gradient_accumulation", None)
    if accumulation is not None and accumulation < 1:
        raise RuntimeError("--gradient-accumulation exige valor >= 1")
    if accumulation is not None and not hasattr(
        trainer, "accumulate_microbatch"
    ):
        raise RuntimeError(
            "Runtime não expõe accumulate_microbatch; reconstrua nsos_ext"
        )

    # Scheduler por tokens: warmup e duração deixam de contar optimizer steps
    # e passam a contar tokens commitados, de modo que A não altere a fração
    # do experimento gasta em warmup. Os campos legados permanecem gravados.
    scheduler_unit = getattr(args, "scheduler_unit", "steps")
    scheduler_token_budget: Dict[str, int] | None = None
    if scheduler_unit == "tokens":
        if not hasattr(trainer, "scheduler_unit"):
            raise RuntimeError(
                "Runtime não expõe scheduler_unit; reconstrua nsos_ext"
            )
        tokens_per_sequence = int(preset["seq_len"])
        # Orçamentos derivados de `warmup_steps`/`total_steps` reintroduzem
        # exatamente o confundidor que o scheduler por tokens existe para
        # remover: com acumulação A, `total_steps` cai por A, então dois
        # braços com o MESMO volume de texto ficariam em pontos diferentes
        # da curva de LR. Quando o orçamento vem da CLI, ele é absoluto e
        # idêntico entre braços; a derivação por passos fica só como
        # compatibilidade para quem não informar nada.
        explicit_warmup = getattr(args, "warmup_tokens", None)
        explicit_training = getattr(args, "training_tokens", None)
        explicit_decay = getattr(args, "decay_tokens", None)
        derived = explicit_training is None
        warmup = (
            int(explicit_warmup) if explicit_warmup is not None
            else int(preset["warmup_steps"]) * tokens_per_sequence
        )
        training = (
            int(explicit_training) if explicit_training is not None
            else max(total_steps, 1) * tokens_per_sequence
        )
        decay = int(explicit_decay) if explicit_decay is not None else 0
        if training <= 0:
            raise RuntimeError("--training-tokens deve ser positivo")
        if warmup < 0 or decay < 0:
            raise RuntimeError(
                "--warmup-tokens e --decay-tokens não podem ser negativos"
            )
        if warmup + decay > training:
            raise RuntimeError(
                f"warmup ({warmup:,}) + decay ({decay:,}) excede training "
                f"({training:,}): a curva de LR não teria platô"
            )
        trainer.warmup_tokens = warmup
        trainer.training_tokens = training
        trainer.decay_tokens = decay
        trainer.scheduler_unit = "tokens"
        # Derivação do limite de optimizer steps a partir do orçamento.
        #
        # Passar o limite à mão por braço é onde o experimento se perde: com
        # A=2 o valor certo é 9.764 e não 9.765, e um erro de um passo é
        # invisível no log e fatal na comparação. O orçamento em tokens já é
        # contrato; o número de passos é consequência aritmética dele.
        tokens_per_optimizer_step = tokens_per_sequence * (accumulation or 1)
        if training % tokens_per_optimizer_step != 0:
            lower = (
                training // tokens_per_optimizer_step
            ) * tokens_per_optimizer_step
            raise RuntimeError(
                f"--training-tokens ({training:,}) não é múltiplo de "
                f"seq_len x A ({tokens_per_optimizer_step:,}). Braços com A "
                f"diferentes veriam volumes diferentes de texto. Use "
                f"{lower:,} ou {lower + tokens_per_optimizer_step:,}."
            )
        derived_max_optimizer_steps = training // tokens_per_optimizer_step
        # `--max-train-steps` MENOR que o derivado é parada antecipada, não
        # conflito: a curva de LR continua sendo a que o orçamento define, e
        # a corrida apenas para antes do fim. É assim que se sonda
        # determinismo em 300 passos sobre a configuração real da ablação.
        # MAIOR é incoerente — pediria passos que o orçamento não financia.
        explicit_steps = int(getattr(args, "max_train_steps", 0) or 0)
        if explicit_steps > derived_max_optimizer_steps:
            raise RuntimeError(
                f"--max-train-steps ({explicit_steps:,}) excede o orçamento: "
                f"{training:,} tokens / {tokens_per_optimizer_step:,} por "
                f"step = {derived_max_optimizer_steps:,}."
            )
        early_stop = 0 < explicit_steps < derived_max_optimizer_steps
        if not explicit_steps:
            args.max_train_steps = derived_max_optimizer_steps
        scheduler_token_budget = {
            "warmup_tokens": warmup,
            "training_tokens": training,
            "decay_tokens": decay,
            "tokens_per_optimizer_step": tokens_per_optimizer_step,
            "derived_max_optimizer_steps": derived_max_optimizer_steps,
        }
        print(
            f"[scheduler] unidade=tokens warmup={warmup:,} "
            f"total={training:,} decay={decay:,} "
            f"origem={'preset' if derived else 'cli'}\n"
            f"[scheduler] tokens_per_optimizer_step={tokens_per_optimizer_step:,} "
            f"(seq_len={tokens_per_sequence} x A={accumulation or 1}) "
            f"derived_max_optimizer_steps={derived_max_optimizer_steps:,}"
            + (
                f"\n[scheduler] PARADA ANTECIPADA em {explicit_steps:,} steps "
                f"({explicit_steps * tokens_per_optimizer_step:,} tokens); "
                f"curva de LR permanece a do orçamento completo"
                if early_stop else ""
            ),
            flush=True,
        )
    elif any(
        getattr(args, name, None) is not None
        for name in ("warmup_tokens", "training_tokens", "decay_tokens")
    ):
        # Falha fechada em vez de aceitar em silêncio um orçamento que o
        # scheduler em `steps` jamais consultaria.
        raise RuntimeError(
            "--warmup-tokens/--training-tokens/--decay-tokens exigem "
            "--scheduler-unit tokens"
        )
    if accumulation is not None:
        # Declarado ANTES do primeiro passo: a identidade de runtime grava este
        # contrato, então ele não pode mudar durante a corrida.
        trainer.gradient_accumulation_steps = accumulation
        print(
            f"[accum] microbatches por optimizer step={accumulation} "
            f"(normalizacao mean_before_clip)",
            flush=True,
        )

    scheduler = trainer.phase_scheduler
    scheduler.progressive_qat_enabled = bool(preset["progressive_qat"])
    if not scheduler.progressive_qat_enabled:
        scheduler.ternary_regularization = 0.0
    trainer.phase_scheduler = scheduler

    runtime_execution_identity = dict(trainer.execution_identity())
    validate_native_training_policy(gpu_training_policy, runtime_execution_identity)
    atomic_write_json(run_dir / "effective_gpu_training_policy.json", gpu_training_policy)
    runtime_module_path = Path(nsos.__file__).resolve()
    if not runtime_module_path.is_file():
        raise RuntimeError(
            f"Módulo nativo carregado não é um arquivo auditável: "
            f"{runtime_module_path}"
        )
    immutable_identity = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        # Contrato de treino, não detalhe de implementação: A=1 e A=64 podem
        # partir do mesmo estado e ainda percorrer trajetórias distintas, então
        # um checkpoint de um não pode retomar como o outro.
        "gradient_accumulation": accumulation,
        "scheduler_unit": scheduler_unit,
        # Emitido só quando a unidade é `tokens`, para não alterar o digest
        # das corridas em `steps` já certificadas. Dois braços com o mesmo
        # A mas orçamentos de LR diferentes percorrem curvas diferentes:
        # retomar um do checkpoint do outro produziria um híbrido silencioso.
        **(
            {"scheduler_token_budget": scheduler_token_budget}
            if scheduler_token_budget is not None
            else {}
        ),
        # Identidade do CORPUS LÓGICO: quais documentos foram escolhidos.
        # Independente de formatação, timestamp e caminho.
        "dataset_manifest_content_sha256": (
            manifest.get("dataset", {}).get("manifest_content_sha256")
        ),
        "dataset_license_policy": args.license_policy,
        "dataset_filter_version": (
            manifest.get("dataset", {}).get("filter_version")
        ),
        "dataset_dedup_version": (
            manifest.get("dataset", {}).get("dedup_version")
        ),
        "dataset_quota_policy_version": (
            manifest.get("dataset", {}).get("quota_policy_version")
        ),
        # Identidade do ARTEFATO EMPACOTADO: quais tokens o Trainer consumiu.
        # Os mesmos documentos repackados com outro tokenizer produzem tokens
        # diferentes e não são o mesmo dataset de treino.
        "packing_version": manifest.get("packing", {}).get("version"),
        # Hash dos artefatos consumidos (sha256 de cada shard + contagens +
        # tokenizer + versão do empacotador), não do JSON que os descreve.
        "packing_content_sha256": (
            manifest.get("packing", {}).get("content_sha256")
        ),
        "preset": preset_name,
        "preset_config": preset,
        "seed": args.seed,
        "model_config": config_dict,
        "pack_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "build_dir": str(args.resolved_build_dir),
        "runtime_module_file": runtime_module_path.name,
        "runtime_module_sha256": sha256_file(runtime_module_path),
        "runtime_module_size_bytes": runtime_module_path.stat().st_size,
        "deterministic_reductions": True,
        "runtime_execution_identity": runtime_execution_identity,
        "runtime_execution_identity_sha256": (
            trainer.execution_identity_digest()
        ),
    }
    manager = CheckpointManager(
        run_dir,
        model,
        trainer,
        tokenizer_path,
        immutable_identity,
        args.keep_checkpoints,
        args.allow_legacy_runtime_identity,
    )
    progress: Dict[str, Any] = {
        "phase_index": 0,
        "phase": "base",
        "shard_index": 0,
        "epoch": 0,
        "offset": 0,
    }
    if not args.no_resume:
        restored = manager.load_latest()
        if restored is not None:
            progress.update(restored)

    phases = ["base", "continuation", "sft"]
    latest_loss: float | None = None
    last_checkpoint_time = time.monotonic()
    run_started = time.monotonic()
    initial_global_step = int(trainer.global_step_count)
    telemetry = RollingTrainingTelemetry(window_steps=100)
    metrics: Dict[str, Any] = {
        "format_version": 2,
        "started_at": utc_now(),
        "preset": preset_name,
        "device": "gpu" if device == nsos.Device.GPU else "cpu",
        "model": inventory,
        "steps_by_phase": steps_by_phase,
        "initial_global_step": initial_global_step,
        "runtime_execution_identity": runtime_execution_identity,
        "telemetry_policy": {
            "rolling_window_steps": telemetry.window_steps,
            "benchmark_first_steps": telemetry.benchmark_steps,
            "benchmark_warmup_steps": 100,
            "sample_retention": "rolling_window_plus_benchmark_prefix_v1",
            "primary_gate": "raw_tokens_per_second.p50 >= 1500",
            "native_stage_timing_opt_in": (
                os.environ.get("NSOS_TRAIN_TIMING") == "1"
            ),
        },
        "phase_results": [],
    }

    def evaluate_phase(phase: str) -> float:
        eval_spec = manifest["phases"][f"{phase}_eval"]
        if not eval_spec["shards"]:
            return float("nan")
        eval_path = (
            workspace
            / "packs"
            / eval_spec["directory"]
            / eval_spec["shards"][0]["file"]
        )
        if phase == "sft":
            return sft_validation_loss(
                nsos,
                model,
                eval_path,
                int(preset["validation_windows"]),
            )
        return causal_validation_loss(
            nsos,
            model,
            eval_path,
            int(preset["seq_len"]),
            int(preset["validation_windows"]),
        )

    def refresh_metrics(*, include_runtime: bool) -> None:
        metrics["training_telemetry"] = telemetry.snapshot()
        metrics["determinism_trace"] = telemetry.determinism_trace()
        benchmark = telemetry.benchmark_percentiles()
        metrics["first_1000_step_percentiles"] = benchmark
        raw_percentiles = benchmark["percentiles"][
            "raw_tokens_per_second"
        ]
        measured_steps = int(benchmark["measured_steps"])
        p50 = (
            float(raw_percentiles["p50"])
            if raw_percentiles is not None
            else None
        )
        metrics["performance_gate"] = {
            "metric": "raw_tokens_per_second.p50",
            "threshold": 1_500.0,
            "required_measured_steps": 900,
            "observed_measured_steps": measured_steps,
            "value": p50,
            "status": (
                "incomplete"
                if measured_steps < 900
                else "pass"
                if p50 is not None and p50 >= 1_500.0
                else "fail_optimize_again"
            ),
        }
        metrics["checkpoint_timing"] = manager.timing_summary()
        if include_runtime:
            metrics["runtime_resources"] = runtime_resource_snapshot(
                nsos, model, trainer
            )

    grad_stats = GradientHealth()

    def checkpoint(kind: str, label: str) -> Path:
        nonlocal last_checkpoint_time
        progress["telemetry_snapshot"] = telemetry.snapshot()
        progress["gradient_health"] = grad_stats.snapshot()
        generation = manager.save(
            progress,
            kind=kind,
            label=label,
            latest_loss=latest_loss,
        )
        last_checkpoint_time = time.monotonic()
        return generation

    limited_start_phase = phases[
        min(max(int(progress.get("phase_index", 0)), 0), len(phases) - 1)
    ]
    limited_initial_eval_loss: float | None = None
    def finish_limited_run(
        phase: str,
        phase_start_step: int,
        phase_started: float,
    ) -> int:
        # Compare like with like even if a deliberately small fixture crosses
        # a curriculum boundary within the bounded run. The production pilot
        # has more than 1,000 base steps, but this invariant keeps the quality
        # gate semantically valid for every preset and resume offset.
        quality_eval_phase = limited_start_phase
        final_eval_loss = evaluate_phase(quality_eval_phase)
        if not math.isfinite(final_eval_loss):
            raise RuntimeError(
                "A corrida limitada exige loss held-out final finita"
            )
        initial_eval_loss = limited_initial_eval_loss
        if initial_eval_loss is None:
            raise RuntimeError("Baseline de qualidade da corrida limitada ausente")
        improvement = initial_eval_loss - final_eval_loss
        relative_improvement = (
            improvement / initial_eval_loss
            if initial_eval_loss != 0.0
            else None
        )
        phase_result = {
            "phase": phase,
            "start_global_step": phase_start_step,
            "end_global_step": int(trainer.global_step_count),
            "last_train_loss": latest_loss,
            "eval_loss": final_eval_loss,
            "elapsed_seconds": time.monotonic() - phase_started,
            "limited": True,
        }
        metrics["phase_results"].append(phase_result)
        metrics["limited_run_quality"] = {
            "start_phase": limited_start_phase,
            "end_phase": phase,
            "evaluated_phase": quality_eval_phase,
            "initial_eval_loss": initial_eval_loss,
            "final_eval_loss": final_eval_loss,
            "absolute_improvement": improvement,
            "relative_improvement": relative_improvement,
            "status": (
                "pass_learning_signal"
                if final_eval_loss < initial_eval_loss
                else "fail_no_learning_signal"
            ),
        }
        final_generation = checkpoint("milestone", f"{phase}-max-steps")
        checkpoint_verification = manager.verify_all()
        if checkpoint_verification["latest"] != final_generation.name:
            raise RuntimeError(
                "A verificacao final nao selecionou o checkpoint limitado"
            )
        metrics["checkpoint_verification"] = checkpoint_verification
        metrics["determinism_evidence"] = {
            "comparison_policy": "bitwise_two_fresh_runs_v1",
            "trace_sha256": stable_hash_text(
                stable_json(telemetry.determinism_trace())
            ),
            "model_sha256": checkpoint_verification[
                "latest_file_sha256"
            ].get("model.bin"),
            "trainer_state_sha256": checkpoint_verification[
                "latest_file_sha256"
            ].get("trainer.state"),
            "runtime_identity_sha256": checkpoint_verification[
                "latest_identity_sha256"
            ],
            "status": "requires_independent_repeat",
        }
        refresh_metrics(include_runtime=True)
        atomic_write_json(run_dir / "run_metrics.json", metrics)
        print(
            f"[eval:{phase}:limited] initial={initial_eval_loss:.5f} "
            f"final={final_eval_loss:.5f} delta={improvement:.5f}",
            flush=True,
        )
        return 0

    try:
        if args.max_train_steps > 0:
            limited_initial_eval_loss = evaluate_phase(limited_start_phase)
            if not math.isfinite(limited_initial_eval_loss):
                raise RuntimeError(
                    "A corrida limitada exige loss held-out inicial finita"
                )
            metrics["limited_run_quality"] = {
                "start_phase": limited_start_phase,
                "initial_eval_loss": limited_initial_eval_loss,
                "status": "running",
            }
        for phase_index, phase in enumerate(phases):
            if phase_index < int(progress.get("phase_index", 0)):
                continue
            progress["phase_index"] = phase_index
            progress["phase"] = phase
            trainer.learning_rate = phase_learning_rate(float(preset["lr"]), phase)
            trainer.eos_loss_scale = 0.5 if phase == "sft" else 1.0
            phase_started = time.monotonic()
            phase_start_step = int(trainer.global_step_count)
            if phase in ("base", "continuation"):
                key = f"{phase}_train"
                spec = manifest["phases"][key]
                phase_dir = workspace / "packs" / spec["directory"]
                start_shard = int(progress.get("shard_index", 0))
                for shard_index, shard in enumerate(spec["shards"]):
                    if shard_index < start_shard:
                        continue
                    path = phase_dir / shard["file"]
                    if sha256_file(path) != shard["sha256"]:
                        raise RuntimeError(f"Shard mudou durante o treino: {path}")
                    tokens = read_u16_tokens(path)
                    batch = int(preset["causal_batch"])
                    seq_len = int(preset["seq_len"])
                    shard_steps = max(
                        (len(tokens) - 1) // (seq_len * batch),
                        0,
                    )
                    start_step = (
                        int(progress.get("offset", 0))
                        if shard_index == start_shard
                        else 0
                    )
                    effective_shard_steps = shard_steps
                    if args.max_train_steps > 0:
                        # --max-train-steps conta OPTIMIZER STEPS, como sempre
                        # contou. O laço itera microbatches, então o limite
                        # precisa ser multiplicado por A; sem isso a corrida
                        # pararia no meio de um grupo e o checkpoint final
                        # sairia fora de boundary, com gradientes pendentes.
                        remaining_run_steps = (
                            args.max_train_steps
                            - (
                                int(trainer.global_step_count)
                                - initial_global_step
                            )
                        ) * (accumulation or 1)
                        effective_shard_steps = min(
                            shard_steps,
                            start_step + max(remaining_run_steps, 0),
                        )
                    progress.update(
                        {
                            "shard_index": shard_index,
                            "epoch": 0,
                            "offset": start_step,
                        }
                    )

                    def callback(
                        local_step: int,
                        loss: float,
                        elapsed_seconds: float,
                        token_counts: Dict[str, int],
                    ) -> None:
                        nonlocal latest_loss
                        manager.poll()
                        latest_loss = float(loss)
                        grad_stats.observe(trainer)
                        progress["offset"] = int(local_step)
                        global_step = int(trainer.global_step_count)
                        telemetry.observe(
                            phase=phase,
                            global_step=global_step,
                            elapsed_seconds=elapsed_seconds,
                            loss=float(loss),
                            token_counts=token_counts,
                            native_timing=trainer_timing_snapshot(trainer),
                        )
                        if device == nsos.Device.GPU and (
                            global_step == initial_global_step + 1
                            or global_step % max(args.log_every_steps, 1) == 0
                        ):
                            refresh_metrics(include_runtime=True)
                        if (
                            local_step == 1
                            or global_step % max(args.log_every_steps, 1) == 0
                        ):
                            rolling = telemetry.snapshot()["rolling"]
                            print(
                                f"[train:{phase}] global={global_step} "
                                f"shard={shard_index} local={local_step}/{shard_steps} "
                                f"loss={loss:.5f} "
                                f"raw_tok/s={rolling['raw_tokens_per_second']:.1f} "
                                f"nonpad_tok/s={rolling['non_padding_tokens_per_second']:.1f} "
                                f"steps/s={rolling['steps_per_second']:.4f}",
                                flush=True,
                            )
                            if on_progress is not None:
                                on_progress({"global_step":global_step,"phase":phase,
                                             "latest_loss":float(loss),"rolling":rolling})
                        at_run_limit = (
                            args.max_train_steps > 0
                            and global_step - initial_global_step
                            >= args.max_train_steps
                        )
                        if not at_run_limit and should_checkpoint(
                            global_step,
                            last_checkpoint_time,
                            args.checkpoint_every_steps,
                            args.checkpoint_every_minutes,
                        ):
                            checkpoint("periodic", f"{phase}-s{shard_index}")

                    # Uma chamada train_step por sequência usa a mesma fronteira
                    # transacional coberta pelo gate nativo bit-exact de
                    # checkpoint/continuação. O piloto prioriza retomada
                    # matematicamente idêntica; batch maior só volta após ganhar
                    # um gate equivalente no runtime.
                    tokens_per_step = seq_len * batch
                    for zero_based_step in range(
                        start_step, effective_shard_steps
                    ):
                        data_start = zero_based_step * tokens_per_step
                        available_sequences = (
                            len(tokens) - 1 - data_start
                        ) // seq_len
                        actual_batch = min(batch, available_sequences)
                        if actual_batch <= 0:
                            raise RuntimeError(
                                f"Sem sequência causal em {path}: "
                                f"step={zero_based_step}"
                            )
                        if actual_batch != 1:
                            raise RuntimeError(
                                "O caminho determinístico causal exige batch=1"
                            )
                        sequence = tokens[
                            data_start : data_start + seq_len + 1
                        ]
                        if len(sequence) != seq_len + 1:
                            raise RuntimeError(
                                f"Janela causal truncada em {path}: "
                                f"step={zero_based_step}"
                            )
                        # O relógio cobre o GRUPO inteiro, não o último
                        # microbatch: medir só o commit inflaria tok/s por um
                        # fator ~A e tornaria incomparáveis os braços de um
                        # A/B de batch efetivo.
                        if trainer.pending_accumulation_microbatches == 0:
                            group_started = time.perf_counter()
                        if accumulation is None:
                            # Caminho legado: uma sequência por optimizer step.
                            loss = trainer.train_step(
                                sequence[:-1], sequence[1:]
                            )
                            committed = True
                        else:
                            # Acumulação explícita. O grupo só vira trajetória
                            # no commit; qualquer exceção no meio precisa
                            # descartá-lo, senão os gradientes dos microbatches
                            # já computados contaminam a próxima tentativa.
                            try:
                                loss = trainer.accumulate_microbatch(
                                    sequence[:-1], sequence[1:]
                                )
                                pending = (
                                    trainer.pending_accumulation_microbatches
                                )
                                committed = pending >= accumulation
                                if committed:
                                    loss = trainer.commit_optimizer_step(
                                        accumulation
                                    )
                            except BaseException:
                                trainer.abort_gradient_accumulation()
                                raise
                        step_elapsed = time.perf_counter() - group_started
                        # O callback publica progresso e pode disparar
                        # checkpoint; chamá-lo apenas em boundary garante que
                        # pesos, otimizador e cursor de dados sejam persistidos
                        # no mesmo ponto da trajetória.
                        if committed:
                            callback(
                                zero_based_step + 1,
                                float(loss),
                                step_elapsed,
                                {
                                    "raw_tokens": (len(sequence) - 1)
                                    * (accumulation or 1),
                                    "non_padding_tokens": (len(sequence) - 1)
                                    * (accumulation or 1),
                                    "supervised_tokens": (len(sequence) - 1)
                                    * (accumulation or 1),
                                    "assistant_tokens": 0,
                                },
                            )
                    tokens.close()
                    if effective_shard_steps < shard_steps:
                        progress["shard_index"] = shard_index
                        progress["offset"] = effective_shard_steps
                        return finish_limited_run(
                            phase, phase_start_step, phase_started
                        )
                    progress["shard_index"] = shard_index + 1
                    progress["offset"] = 0
                    if (
                        args.max_train_steps > 0
                        and int(trainer.global_step_count) - initial_global_step
                        >= args.max_train_steps
                    ):
                        return finish_limited_run(
                            phase, phase_start_step, phase_started
                        )
                progress["shard_index"] = 0
                progress["offset"] = 0
                eval_loss = evaluate_phase(phase)
            else:
                spec = manifest["phases"]["sft_train"]
                phase_dir = workspace / "packs" / spec["directory"]
                start_epoch = int(progress.get("epoch", 0))
                start_shard = int(progress.get("shard_index", 0))
                for epoch in range(start_epoch, int(preset["sft_epochs"])):
                    shard_order = list(range(len(spec["shards"])))
                    random.Random(args.seed + 10_000 + epoch).shuffle(shard_order)
                    logical_start = start_shard if epoch == start_epoch else 0
                    for logical_index, physical_index in enumerate(shard_order):
                        if logical_index < logical_start:
                            continue
                        shard = spec["shards"][physical_index]
                        path = phase_dir / shard["file"]
                        if sha256_file(path) != shard["sha256"]:
                            raise RuntimeError(f"Shard mudou durante o treino: {path}")
                        records = read_sft_records(path)
                        record_order = list(range(len(records)))
                        random.Random(
                            args.seed + epoch * 100_003 + physical_index
                        ).shuffle(record_order)
                        batch_size = int(preset["sft_batch"])
                        offset = (
                            int(progress.get("offset", 0))
                            if epoch == start_epoch and logical_index == logical_start
                            else 0
                        )
                        progress.update(
                            {
                                "epoch": epoch,
                                "shard_index": logical_index,
                                "physical_shard_index": physical_index,
                                "offset": offset,
                            }
                        )
                        while offset < len(record_order):
                            batch_indices = record_order[
                                offset : offset + batch_size
                            ]
                            batch_records = [
                                records[index] for index in batch_indices
                            ]
                            prompts = [record[0] for record in batch_records]
                            answers = [record[1] for record in batch_records]
                            token_counts = sft_step_token_counts(
                                prompts, answers
                            )
                            step_started = time.perf_counter()
                            latest_loss = float(
                                trainer.train_supervised_batch(prompts, answers)
                            )
                            step_elapsed = time.perf_counter() - step_started
                            manager.poll()
                            offset += len(batch_records)
                            progress["offset"] = offset
                            global_step = int(trainer.global_step_count)
                            telemetry.observe(
                                phase="sft",
                                global_step=global_step,
                                elapsed_seconds=step_elapsed,
                                loss=latest_loss,
                                token_counts=token_counts,
                                native_timing=trainer_timing_snapshot(trainer),
                            )
                            if (
                                global_step == initial_global_step + 1
                                or global_step % max(args.log_every_steps, 1) == 0
                            ):
                                refresh_metrics(
                                    include_runtime=(
                                        device == nsos.Device.GPU
                                    )
                                )
                                rolling = telemetry.snapshot()["rolling"]
                                print(
                                    f"[train:sft] global={global_step} "
                                    f"epoch={epoch + 1}/{preset['sft_epochs']} "
                                    f"shard={logical_index} offset={offset}/{len(record_order)} "
                                    f"loss={latest_loss:.5f} "
                                    f"raw_tok/s={rolling['raw_tokens_per_second']:.1f} "
                                    f"nonpad_tok/s={rolling['non_padding_tokens_per_second']:.1f} "
                                    f"supervised_tok/s={rolling['supervised_tokens_per_second']:.1f} "
                                    f"assistant_tok/s={rolling['assistant_tokens_per_second']:.1f} "
                                    f"steps/s={rolling['steps_per_second']:.4f}",
                                    flush=True,
                                )
                                if on_progress is not None:
                                    on_progress({"global_step":global_step,"phase":"sft",
                                                 "latest_loss":latest_loss,"rolling":rolling})
                            at_run_limit = (
                                args.max_train_steps > 0
                                and global_step - initial_global_step
                                >= args.max_train_steps
                            )
                            if not at_run_limit and should_checkpoint(
                                global_step,
                                last_checkpoint_time,
                                args.checkpoint_every_steps,
                                args.checkpoint_every_minutes,
                            ):
                                checkpoint("periodic", f"sft-e{epoch}-s{logical_index}")
                            if (
                                args.max_train_steps > 0
                                and global_step - initial_global_step
                                >= args.max_train_steps
                            ):
                                records.close()
                                return finish_limited_run(
                                    phase, phase_start_step, phase_started
                                )
                        records.close()
                        progress["shard_index"] = logical_index + 1
                        progress["offset"] = 0
                    progress["epoch"] = epoch + 1
                    progress["shard_index"] = 0
                    progress["offset"] = 0
                eval_loss = evaluate_phase(phase)

            progress.update(
                {
                    "phase_index": phase_index + 1,
                    "phase": phases[phase_index + 1]
                    if phase_index + 1 < len(phases)
                    else "complete",
                    "shard_index": 0,
                    "epoch": 0,
                    "offset": 0,
                }
            )
            phase_result = {
                "phase": phase,
                "start_global_step": phase_start_step,
                "end_global_step": int(trainer.global_step_count),
                "last_train_loss": latest_loss,
                "eval_loss": eval_loss,
                "elapsed_seconds": time.monotonic() - phase_started,
            }
            metrics["phase_results"].append(phase_result)
            refresh_metrics(include_runtime=True)
            atomic_write_json(run_dir / "run_metrics.json", metrics)
            print(
                f"[eval:{phase}] loss={eval_loss:.5f} "
                f"steps={phase_result['end_global_step'] - phase_start_step}",
                flush=True,
            )
            checkpoint("milestone", f"{phase}-complete")

        final_dir = manager.save(
            progress,
            kind="final",
            label="complete",
            latest_loss=latest_loss,
        )
        edge_path = run_dir / "final_edge_linear.nsos"
        model.save_edge_linear_pack(str(edge_path))
        metrics.update(
            {
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - run_started,
                "final_global_step": int(trainer.global_step_count),
                "final_checkpoint": str(final_dir),
                "final_edge_pack": str(edge_path),
                "checkpoint_verification": manager.verify_all(),
                "determinism_report": nsos.get_determinism_report()
                if hasattr(nsos, "get_determinism_report")
                else {},
            }
        )
        refresh_metrics(include_runtime=True)
        atomic_write_json(run_dir / "run_metrics.json", metrics)
        print(f"[done] checkpoint={final_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[interrupt] solicitado; gravando checkpoint de emergência", flush=True)
        if not trainer.optimizer_state_poisoned():
            checkpoint("milestone", "interrupted")
        raise
    except Exception:
        if not trainer.optimizer_state_poisoned():
            try:
                checkpoint("milestone", "exception-safe-point")
            except Exception as checkpoint_error:
                print(
                    f"[checkpoint] falha ao salvar emergência: {checkpoint_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        manager.close()


def default_workspace(preset_name: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "ptbr_conversational"
        / preset_name
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline autoral NSOS Mamba-only conversacional PT-BR."
    )
    parser.add_argument(
        "action",
        choices=["prepare", "train", "run", "status", "verify"],
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="pilot")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="gpu")
    parser.add_argument(
        "--dataset-revisions",
        type=Path,
        default=None,
        help=(
            "Caminho para um dataset_revisions.json de uma preparação "
            "anterior. Fixa as revisões de origem em vez de resolvê-las de "
            "novo — obrigatório na reconstrução B de um teste A/B, já que os "
            "workspaces são diferentes e o dataset upstream pode ter mudado "
            "entre as duas."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help=(
            "Microbatches por optimizer step. Omitido usa o caminho legado "
            "(train_step, um optimizer step por sequência). Com valor, usa "
            "accumulate_microbatch + commit_optimizer_step, que normaliza o "
            "gradiente por A ANTES do clipping. A=1 existe para provar "
            "equivalência com o legado; A>1 muda a trajetória de treino e "
            "entra na identidade de runtime."
        ),
    )
    parser.add_argument(
        "--scheduler-unit",
        choices=["steps", "tokens"],
        default="steps",
        help=(
            "Unidade do scheduler de LR. 'steps' é o contrato legado. Com "
            "acumulação, o número de optimizer steps para o mesmo volume de "
            "dados varia com A, então 'tokens' é o único modo em que braços "
            "de A diferentes recebem a mesma quantidade de dados antes do LR "
            "de pico."
        ),
    )
    # Orçamentos ABSOLUTOS da curva de LR, em tokens commitados. Sem eles os
    # valores são derivados de `warmup_steps`/`total_steps`, que dependem de
    # A — e o braço com A=8 chegaria ao pico com 8x menos texto visto que o
    # braço com A=1. Só válidos com --scheduler-unit tokens.
    parser.add_argument(
        "--warmup-tokens", type=int, default=None,
        help="Tokens commitados até o LR de pico.",
    )
    parser.add_argument(
        "--training-tokens", type=int, default=None,
        help="Orçamento total de tokens da curva de LR.",
    )
    parser.add_argument(
        "--decay-tokens", type=int, default=None,
        help="Tokens finais em decaimento (cauda do WSD).",
    )
    # --- Arquitetura -------------------------------------------------------
    # Default None em todos: quando não informados, a configuração é
    # exatamente a hardcoded de sempre, então o baseline já certificado
    # continua reproduzível bit a bit.
    parser.add_argument("--attention-period", type=int, default=None,
                        help="Periodo do agendamento de attention. 8=1 camada, 6=2, 5=3 (em 16).")
    parser.add_argument("--attention-slot", type=int, default=None)
    parser.add_argument("--use-ttt", action="store_const", const=True, default=None)
    parser.add_argument("--ttt-period", type=int, default=None)
    parser.add_argument("--ttt-slot", type=int, default=None)
    parser.add_argument("--use-moe", action="store_const", const=True, default=None)
    parser.add_argument("--moe-period", type=int, default=None)
    parser.add_argument("--moe-slot", type=int, default=None)
    parser.add_argument("--use-chrass", action="store_const", const=True, default=None)
    parser.add_argument("--use-kan", action="store_const", const=True, default=None)
    parser.add_argument("--use-slender-embedding", action="store_const", const=True, default=None)
    parser.add_argument("--no-force-mamba-last-layer", dest="force_mamba_last_layer",
                        action="store_const", const=False, default=None)
    parser.add_argument("--no-mamba2-faithful", dest="mamba2_faithful",
                        action="store_const", const=False, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="Teto da norma do gradiente (padrão 1.0).")
    parser.add_argument("--gpu-training-profile", choices=("inherit", "legacy", "redesign-v1"),
                        default="inherit", help="Candidato opt-in GPU com histórico compacto, clipping, attention, MoE e TTT no dispositivo; TTT mantém derivada truncada.")
    parser.add_argument("--kan-compute-policy", choices=("inherit", "legacy", "tiled-v1", "wmma-v1"), default="inherit",
                        help="Opt-in implicit RBF/device QAT KAN; versioned native policy required")
    parser.add_argument("--moe-compute-policy", choices=("inherit", "legacy", "grouped-v1", "wmma-v1"), default="inherit",
                        help="Opt-in grouped MoE; wmma-v1 requires compiled RDNA3 wave32 support, lowp eligible shapes use WMMA")
    parser.add_argument("--ttt-gradient-policy", choices=("inherit", "truncated", "full-sequence-v1"),
                        default="inherit", help="Contrato de derivada TTT separado da otimização: BPTT dentro da sequência, batch isolado e boundary history de 32 tokens; estados entre chamadas são detached.")
    parser.add_argument(
        "--matmul-precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help=(
            "Precisão de compute das GEMMs. Master weights, gradientes e "
            "acumuladores permanecem FP32 em qualquer modo. bf16 mantém o "
            "expoente de 8 bits do fp32; fp16 tem faixa menor e depende do "
            "loss scaling dinâmico. O modo escolhido entra na identidade de "
            "runtime, portanto checkpoints de precisões diferentes não se "
            "misturam."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--license-policy",
        choices=["commercial-strict", "research-reviewed"],
        default="commercial-strict",
    )
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=None)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Gate de validação; 0 executa toda a receita.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora checkpoints existentes e inicia pesos novos.",
    )
    parser.add_argument(
        "--allow-legacy-runtime-identity",
        action="store_true",
        help=(
            "Migra explicitamente um checkpoint anterior ao contrato de "
            "identidade v2. Use somente após confirmar manualmente precisão, "
            "backend, GPU, determinismo e política do otimizador originais."
        ),
    )
    return parser.parse_args()


def print_status(
    workspace: Path,
    run_dir: Path,
    preset_name: str,
    preset: Dict[str, Any],
    seed: int,
    fixture: bool,
    license_policy: str,
) -> None:
    payload: Dict[str, Any] = {
        "workspace": str(workspace),
        "run_dir": str(run_dir),
        "corpus_manifest": None,
        "pack_manifest": None,
        "latest_checkpoint": None,
    }
    corpus_manifest_path = workspace / "corpus" / "corpus_manifest.json"
    pack_manifest_path = workspace / "packs" / "pack_manifest.json"
    if corpus_manifest_path.is_file() or pack_manifest_path.is_file():
        validate_training_workspace(
            workspace,
            preset_name,
            preset,
            seed,
            fixture,
            license_policy,
        )
    if corpus_manifest_path.is_file():
        payload["corpus_manifest"] = {
            "path": str(corpus_manifest_path),
            "sha256": sha256_file(corpus_manifest_path),
        }
    if pack_manifest_path.is_file():
        manifest = _read_json_object(pack_manifest_path, "Pack manifest")
        payload["pack_manifest"] = {
            "path": str(pack_manifest_path),
            "sha256": sha256_file(pack_manifest_path),
            "phases": {
                name: {
                    key: value
                    for key, value in spec.items()
                    if key in ("tokens", "target_tokens_packed", "records")
                }
                for name, spec in manifest["phases"].items()
            },
        }
    pointer = run_dir / "checkpoints" / "LATEST.json"
    if pointer.is_file():
        payload["latest_checkpoint"] = json.loads(
            pointer.read_text(encoding="utf-8")
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    preset_name = args.preset
    preset = dict(PRESETS[preset_name])
    workspace = (args.workspace or default_workspace(preset_name)).resolve()
    run_dir = (
        args.run_dir or (workspace / "runs" / "main")
    ).resolve()
    args.checkpoint_every_steps = int(
        preset["checkpoint_every_steps"]
        if args.checkpoint_every_steps is None
        else args.checkpoint_every_steps
    )
    args.checkpoint_every_minutes = float(
        preset["checkpoint_every_minutes"]
        if args.checkpoint_every_minutes is None
        else args.checkpoint_every_minutes
    )

    if args.action == "status":
        print_status(
            workspace,
            run_dir,
            preset_name,
            preset,
            args.seed,
            args.fixture,
            args.license_policy,
        )
        return 0

    os.environ.setdefault("NSOS_DETERMINISTIC", "1")
    # Batch causal >1 é processado pelo Trainer em microchunks de uma amostra:
    # gradientes acumulam, ativações não ocupam VRAM simultaneamente.
    os.environ.setdefault("NSOS_TRAIN_CHUNK_SIZE", "1")
    args.resolved_build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(args.resolved_build_dir)

    if args.action in ("prepare", "run"):
        manifest_path = prepare(
            args, preset_name, preset, workspace, nsos
        )
        print(f"[prepare] pack manifest: {manifest_path}", flush=True)
    if args.action in ("train", "run"):
        return train(
            args,
            preset_name,
            preset,
            workspace,
            run_dir,
            nsos,
        )
    if args.action == "verify":
        # A validação completa da identidade precisa da mesma construção usada
        # em train; status já valida todos os shards de dados.
        print_status(
            workspace,
            run_dir,
            preset_name,
            preset,
            args.seed,
            args.fixture,
            args.license_policy,
        )
        pointer = run_dir / "checkpoints" / "LATEST.json"
        if not pointer.is_file():
            raise RuntimeError("Nenhum checkpoint para verificar")
        generation_name = json.loads(
            pointer.read_text(encoding="utf-8")
        )["generation"]
        generation = run_dir / "checkpoints" / "generations" / generation_name
        checkpoint_manifest_path = generation / "checkpoint_manifest.json"
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        for name, expected in checkpoint_manifest["files"].items():
            if sha256_file(generation / name) != expected:
                raise RuntimeError(f"Checkpoint corrompido: {generation / name}")
        print(f"[verify] checkpoint íntegro: {generation}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
