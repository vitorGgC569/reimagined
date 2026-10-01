"""Deterministic, auditable quality filter for Canarim-Instruct-PTBR.

The source dataset is research-only (CC BY-NC).  This tool does not claim
that the source is clean: it preserves the raw parquet files and emits
accepted, quarantined, and rejected JSONL records with reason codes.

Usage:
    python filter_canarim_sft.py \
      --raw-dir OXN/nsos/artifacts/.../canarim_instruct/raw \
      --out-dir OXN/nsos/artifacts/.../canarim_instruct/filtered_v1

The accepted format is the NSOS SFT format: {kind, prompt, answer, metadata}.
The test split is filtered first and receives priority when preventing prompt
leakage into training.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import pyarrow.parquet as pq


FILTER_VERSION = "canarim-sft-gold-filter-v16"
SOURCE_REPO = "dominguesm/Canarim-Instruct-PTBR-Dataset"
SOURCE_LICENSE = "cc-by-nc-4.0"

PT_STOPWORDS = frozenset(
    "a ao aos as com como da das de do dos e em entre essa esse esta este eu foi "
    "foram há isso isto já mas na nas não no nos o os ou para pela pelas pelo "
    "pelos por que se sem ser sua suas seu seus também tem tendo um uma umas uns "
    "você vocês quando qual quais quem onde porque sobre até são é às".split()
)
EN_STOPWORDS = frozenset(
    "a an and are as at be by for from has have he in is it of on or that the "
    "this to was were what when where which who will with you now world can your "
    "their they them these those one some best how why into over under then all "
    "there about after before first last more most than very not do does did".split()
)

HTML_RE = re.compile(r"<\/?[a-z][^>]*>|&(?:nbsp|amp|quot|lt|gt|apos);", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
WORD_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
NUMBER_RE = re.compile(r"\d")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPLACEMENT_RE = re.compile("\ufffd")

MODEL_LEAK_RE = re.compile(
    r"(?i)\b(?:como\s+(?:modelo|uma\s+ia|intelig[eê]ncia\s+artificial)|"
    r"como\s+assistente\s+de\s+ia|as\s+an\s+ai\s+language\s+model|"
    r"i\s+am\s+an\s+ai)\b"
)
BOILERPLATE_RE = re.compile(
    r"(?i)\b(?:espero\s+que\s+isso\s+ajude|se\s+precisar\s+de\s+mais\s+ajuda|"
    r"claro[,!]?\s+ aqui\s+est[aá]|claro[,!]?\s+posso\s+ajudar|"
    r"leia\s+mais|clique\s+aqui|assine\s+agora|pol[ií]tica\s+de\s+cookies|"
    r"todos\s+os\s+direitos\s+reservados|inscreva-se\s+no\s+canal|"
    r"compartilhe\s+(?:no|via)\s+(?:whatsapp|facebook|twitter|telegram))\b"
)
GENERIC_RE = re.compile(
    r"(?i)^(?:claro[.!]?\s*)?(?:a resposta|isso depende|depende do caso|"
    r"não há uma resposta única|espero que isso ajude)[.!]?\s*$"
)
OUT_OF_SCOPE_RE = re.compile(
    r"(?i)\b(?:leetcode|class\s+solution|def\s*\w*\s*\(|return\b|"
    r"python|javascript|java|c\+\+|programa(?:r|ção)?|programador(?:a)?|c[oó]digos?|algoritmo|"
    r"software|inteiros?|array|fun[cç][aã]o|loop|matriz|matem[aá]tica|"
    r"aritm[eé]tica|[aá]lgebra|equ[aá]ção|fra[cç][aã]o|c[aá]lculo|n[uú]meros?\s+pares?|"
    r"soma\s+de\s+(?:uma\s+)?lista|derivada|integral|"
    r"pent[aâ]metro|sem\s+usar\s+loops|sql|mongodb|html|css|javascript|"
    r"consulta\s+(?:sql|ao\s+banco)|banco\s+de\s+dados|formul[aá]rio\s+html|"
    r"present\s+perfect|present\s+perfect\s+continuous|estrutura\s+sem[aâ]ntica|"
    r"m[eé]todo\s+est[aá]tico|instanciar|vari[aá]veis?\s+est[aá]ticas?|"
    r"heran[cç]a\s+(?:de\s+)?classe|orienta[cç][aã]o\s+a\s+objetos?|"
    r"string|regex|express[aã]o\s+regular|caracteres?\s+que\s+n[aã]o\s+s[aã]o\s+letras|"
    r"entrada\s*:\s*\[|\bage\s*:\s*\d+|\blikes\s*:\s*\[|"
    r"p[aá]gina\s+(?:da\s+)?web|site\s+externo|powerpoint|formul[aá]rio|\burl\b|"
    r"apresenta[cç][aã]o)\b"
)
UNSAFE_OR_SENSITIVE_RE = re.compile(
    r"(?i)\b(?:terroris(?:ta|mo)|g[aá]s\s+cloro|explosivo|arma\s+para\s+matar|"
    r"matar\s+animais|suic[ií]dio|autoagress[aã]o|senha(?:s)?|password|"
    r"rastrear\s+(?:meu|o|a)\s+namorad|contas?\s+de\s+m[ií]dia|"
    r"fraude|golpe|droga|hackear|invadir\s+conta|sonegar|evas[aã]o\s+fiscal|"
    r"burlar|falsificar|lavar\s+dinheiro|subornar|sem\s+(?:declarar|colocar).*imposto|"
    r"roubar|roubo|cometer\s+um\s+crime|\bcrime\b|armas?|arma[s]?\s+qu[ií]mica|"
    r"testemunha|assassin|assalto|ladr[aã]o|tribunal|julgamento)\b"
)
PERSONAL_DATA_RE = re.compile(
    r"(?i)\b(?:endere[cç]o\s+residencial|n[uú]mero\s+de\s+telefone|"
    r"informa[cç][oõ]es?\s+pessoais?|localizar\s+(?:o|a|uma?)\s+pessoa|"
    r"cpf|rg\s+de\s+|data\s+de\s+nascimento)\b"
)
HATE_OR_ABUSE_RE = re.compile(
    r"(?i)\b(?:odeio|n[aã]o\s+gosto\s+d[oa]s?|inferior(?:es)?|"
    r"cheiram\s+mal|burro(?:s|as)?|idiota(?:s)?|ra[cç]a\s+inferior)\b"
)
MEDICAL_RE = re.compile(
    r"(?i)\b(?:s[ií]ntomas?|tratamento|diagn[oó]stico|ataque\s+card[ií]aco|"
    r"antibi[oó]tico|febre|tosse|c[aâ]ncer|medicamento|dose|doen[cç]a|"
    r"hiv|c[eé]lulas?\s+t|infectados?)\b"
)
HEALTH_TOPIC_RE = re.compile(
    r"(?i)\b(?:sa[uú]de|dentes?|dentista|clarear|clareamento|nutri[cç][aã]o|"
    r"calorias?|peso|gravidez|hospital|alergia|depress[aã]o|ansiedade|"
    r"estresse?|estressado|sobrecarregado|relaxamento|mindfulness|yoga|"
    r"exerc[ií]cio\s+f[ií]sico|rem[eé]dio|vitamina|vacina|"
    r"seguro\s+cozinhar|cozinhar.*alum[ií]nio|seguro.*consumir|"
    r"colesterol|press[aã]o\s+arterial|gl[uú]ten|livre\s+de\s+gl[uú]ten)\b"
)
FINANCIAL_TOPIC_RE = re.compile(
    r"(?i)\b(?:ganhar\s+dinheiro(?:\s+(?:rapidamente|r[aá]pido))?|"
    r"economizar\s+dinheiro|hábitos?\s+financeiros?|"
    r"invest(?:ir|imento|imentos?)|criptomoedas?|renda\s+extra|"
    r"or[cç]amento|despesas?|d[ií]vidas?|empr[eé]stimos?|cr[eé]dito|"
    r"cart[aã]o\s+(?:de\s+)?cr[eé]dito|finan[cç]as?|banco|banc[aá]rio|"
    r"pre[cç]os?|desconto|ingressos?\s+com\s+desconto|"
    r"quanto\s+custa|custo\s+(?:de|do|da))\b"
)
ENTITY_ANCHOR_RE = re.compile(r"(?i)\bp[aã]o\s+de\s+a[cç]u[cç]ar\b")
IMPERIAL_RE = re.compile(
    r"(?i)(?:\b(?:fahrenheit|milhas?|libras?|gal[oõ]es?)\b|"
    r"\b\d+(?:[.,]\d+)?\s*[°º]\s*f\b|graus?\s+fahrenheit)"
)
AGENT_COMMAND_RE = re.compile(
    r"(?i)\b(?:agente\s+em\s+seu\s+ambiente|sequ[eê]ncia\s+de\s+a[cç][oõ]es|"
    r"comando\s+correto|navegar\s+por\s+um\s+agente|linguagem\s+natural\s+limitada)\b"
)
MATH_TOPIC_RE = re.compile(
    r"(?i)\b(?:tri[aâ]ngulo|geometr(?:ia|ico)|pol[ií]gono|[aâ]ngulo|"
    r"matem[aá]tica|aritm[eé]tica|[aá]lgebra|equ[aá][cç][aã]o|fra[cç][aã]o|"
    r"porcentagem|probabilidade|estat[ií]stica|raiz\s+quadrada|"
    r"multiplica[cç][aã]o|divis[aã]o|subtra[cç][aã]o|soma\s+de\s+(?:uma\s+)?lista|"
    r"mediana|m[eé]dia|modo|vari[aâ]ncia|desvio\s+padr[aã]o|"
    r"lista\s+de\s+n[uú]meros|conjunto\s+de\s+n[uú]meros|"
    r"quantas\s+maneiras|reorganizar\s+as\s+letras|fatorial)\b"
)
WEB_CONTENT_RE = re.compile(
    r"(?i)\b(?:artigos?\s+de\s+not[ií]cias?|not[ií]cias?\s+separad|"
    r"token\s+especial|mais\s+recente\s+desenvolvimento|veja:|"
    r"blogueir[oa]|calgary|beavertails|donuts?|pesquise|pesquisa|"
    r"situa[cç][aã]o\s+atual|[uú]ltimos\s+anos|t[oó]pico:\s+tecnologia|"
    r"tend[eê]ncias\s+atuais|marketing\s+digital|site|website|"
    r"p[aá]gina\s+(?:na|da|de)|link|fonte\s+(?:online|na\s+internet)|"
    r"atualmente|mais\s+recente|[uú]ltimo\s+lan[cç]amento|"
    r"desastres?\s+naturais|pa[ií]ses\s+ao\s+redor\s+do\s+mundo)\b"
)
PT_PT_RE = re.compile(
    r"(?i)\b(?:guarda-redes|teu|tua|tuas|teus|diz-me|ficheiro|comboio|"
    r"autocarro|ecr[aã]|rapariga|mi[uú]do|bilhete|aluguer|facto|"
    r"partilhar|partilhem|fato\s+de\s+banho|tamp[aã]o\s+de\s+nata[cç][aã]o)\b"
)
OTHER_LANGUAGE_REQUEST_RE = re.compile(
    r"(?i)\b(?:em|para|do|no)\s+(?:ingl[eê]s|espanhol|franc[eê]s|alem[aã]o|"
    r"italiano|ingl[eê]s|english|spanish|french|german)|"
    r"\b(?:in\s+english|in\s+spanish|in\s+french|translate|traduzir|traduza|"
    r"falantes?\s+n[aã]o\s+nativos?\s+de\s+ingl[eê]s|"
    r"sin[oô]nimo[s]?\s+em\s+ingl[eê]s)\b"
)
ENGLISH_CONTENT_RE = re.compile(
    r"(?i)\b(?:laugh\s+out\s+loud|by\s+the\s+way|in\s+real\s+life|"
    r"in\s+my\s+humble\s+opinion|work\s+hard|make\s+something|"
    r"background|present\s+perfect|continuous|please|what\s+is\s+your\s+name|"
    r"now|world|their|they|them|these|those|some|best|first|last|"
    r"more|most|than|very|there|about|after|before|into|over|under|then)\b"
)
TASK_META_RE = re.compile(
    r"(?i)\b(?:tarefa\s*:|nesta\s+tarefa|voc[eê]\s+recebe\s+um\s+texto|"
    r"dada\s+uma\s+frase|"
    r"dada\s+uma\s+frase,?\s+gerar\s+um\s+contexto|"
    r"declara[cç][aã]o\s+anterior|contexto\s+mais\s+prov[aá]vel|"
    r"dada\s+uma\s+premissa|fato:|sa[ií]da:|"
    r"voc[eê]\s+receber[aá]\s+a\s+personalidade|frases?\s+candidatas?|"
    r"responda\s+a\s+pergunta\s+com\s+um\s+n[uú]mero|valor\s+categ[oó]rico|"
    r"identificar\s+e\s+classificar\s+os?\s+diferentes\s+tipos\s+de\s+palavras|"
    r"dada\s+uma\s+passagem|construa\s+uma\s+pergunta|inequ[ií]voca|"
    r"respons[aá]vel\s+pela\s+passagem|pr[oó]xima\s+declara[cç][aã]o|"
    r"mais\s+prov[aá]vel|logicamente\s+correta|voc[eê]\s+recebe\s+um\s+pedido|"
    r"sua\s+tarefa\s+[eé]\s+gerar\s+uma\s+previs[aã]o|"
    r"o\s+solicitante\s+est[aá]\s+realmente\s+tentando|"
    r"conte\s+quantas\s+vezes\s+a\s+palavra|par[aá]grafo\s+de\s+fundo|"
    r"rela[cç][oõ]es\s+causais\s+ou\s+f[ií]sicas|"
    r"a\s+tarefa\s+[eé]\s+ler\s+o\s+contexto|mcqs?|m[uú]ltipla\s+escolha|"
    r"contagem\s+de\s+palavras?|conte\s+quantas?\s+palavras?|"
    r"\bentrada\s*:|sa[ií]da\s*:|falantes?\s+n[aã]o\s+nativos?|"
    r"d[eê]-me\s+um\s+nome\s+para|como\s+você\s+se\s+sente|"
    r"opinião\s+sobre)"
)
GRAMMAR_TASK_RE = re.compile(
    r"(?i)\b(?:erro\s+gramatical|gramaticalmente\s+(?:correta|incorreta)|"
    r"senten[cç]a\s*:|an[aá]lise\s+gramatical)\b"
)
PLACEHOLDER_RE = re.compile(
    r"(?i)\[(?:data|cidade|local|endere[cç]o|nome|empresa|cargo|telefone|"
    r"prazo|inserir|insert|date|city|address|name|company)\]"
)
INCOMPLETE_PROMPT_RE = re.compile(
    r"(?i)(?:\b(?:usando|com|de|para|e|ou|que|em|do|da|sobre|a|o)\s*)$"
)
SENTIMENT_RE = re.compile(
    r"(?i)\b(?:revis[aã]o\s+(?:positiva|negativa)|tom\s+(?:positivo|negativo)|"
    r"sentimento|sentimental|altere\s+o\s+tom|como\s+voc[eê]\s+se\s+sente|"
    r"qual\s+[eé]\s+a\s+sua\s+opini[aã]o)\b"
)
TEMPLATE_RE = re.compile(r"(?i)\[(?:recruiter|nome|name|empresa|company|insira|insert)[^\]]*\]")
PRICE_RE = re.compile(r"(?i)\b(?:pre[cç]o\s+mais\s+baixo|quanto\s+custa|bilhete\s+de\s+avi[aã]o)\b")
EXPLANATION_RE = re.compile(
    r"(?i)\b(?:explique|descreva|compare|por\s+que|como\s+fazer|"
    r"benef[ií]cios|estrat[eé]gias|passos|instru[cç][oõ]es|resuma)\b"
)
TRAVEL_RE = re.compile(
    r"(?i)\b(?:hotel|hospedagem|reservar|viagem|viajar|cidade|visitar|"
    r"restaurante|passagem|voo)\b"
)
PROPER_RE = re.compile(r"\b[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][a-záàãâéêíóôõúç]{2,}\b")
EXTRACTION_RE = re.compile(
    r"(?i)\b(?:escolha|identifique|extraia|liste|encontre)\s+(?:os|as|a|uma|um)?\s*"
    r"(?:substantivos?|palavras?|entidades?|nomes?|itens?)\b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def words(text: str) -> List[str]:
    return [item.lower() for item in WORD_RE.findall(text)]


def language_score(text: str) -> Tuple[int, int, float]:
    tokens = words(text)
    if not tokens:
        return 0, 0, 0.0
    pt_hits = sum(token in PT_STOPWORDS for token in tokens)
    en_hits = sum(token in EN_STOPWORDS for token in tokens)
    accented = sum(ch in "áàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ" for ch in text)
    return pt_hits, en_hits, accented / max(len(text), 1)


def repeated_ngram_count(tokens: Sequence[str], n: int = 4) -> int:
    if len(tokens) < n:
        return 0
    counts = collections.Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return sum(count - 1 for count in counts.values() if count > 1)


def simhash64(tokens: Sequence[str]) -> int:
    """Stable word 3-gram SimHash used only for local near-duplicate buckets."""
    grams = ["\x1f".join(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))]
    if not grams:
        return 0
    weights = [0] * 64
    for gram in set(grams):
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def proper_anchors(text: str) -> List[str]:
    return [token.lower() for token in PROPER_RE.findall(text) if token.lower() not in PT_STOPWORDS]


def iter_parquet_rows(path: Path, split: str) -> Iterator[Dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=["instruction", "input", "output"]):
        table = batch.to_pydict()
        size = len(table["instruction"])
        for offset in range(size):
            yield {
                "split": split,
                "row_index": row_index,
                "instruction": table["instruction"][offset],
                "input": table["input"][offset],
                "output": table["output"][offset],
            }
            row_index += 1


def local_quality(row: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    instruction = normalize_text(row.get("instruction"))
    context = normalize_text(row.get("input"))
    output = normalize_text(row.get("output"))
    combined = "\n\n".join(part for part in (instruction, context, output) if part)
    prompt = "\n\n".join(part for part in (instruction, context) if part)
    iw, ow, cw = words(instruction), words(output), words(context)
    reasons: List[str] = []
    hard_reject = False

    if not instruction or not output:
        reasons.append("missing_instruction_or_output")
        hard_reject = True
    if REPLACEMENT_RE.search(combined):
        reasons.append("invalid_unicode_replacement")
        hard_reject = True
    if HTML_RE.search(combined):
        reasons.append("html_or_entity_residue")
        hard_reject = True
    if not (2 <= len(iw) <= 256):
        reasons.append("instruction_length")
        hard_reject = True
    if not (5 <= len(ow) <= 1024):
        reasons.append("answer_length")
        hard_reject = True
    if len(cw) > 1024:
        reasons.append("context_length")
        hard_reject = True
    if len(words(combined)) > 1536:
        reasons.append("combined_length")
        hard_reject = True

    pt, en, accent_ratio = language_score(combined)
    if (en >= 4 and en > pt + 2) or (en >= 6 and en > pt):
        reasons.append("non_pt_dominant")
        hard_reject = True
    elif pt < 2 and accent_ratio < 0.002:
        reasons.append("language_uncertain")

    url_count = len(URL_RE.findall(combined))
    email_count = len(EMAIL_RE.findall(combined))
    if url_count > 0 or email_count > 0:
        reasons.append("web_or_contact_artifacts")
        hard_reject = True
    if NUMBER_RE.findall(combined) and len(NUMBER_RE.findall(combined)) / max(len(combined), 1) > 0.28:
        reasons.append("numeric_artifact_density")
        hard_reject = True

    if MODEL_LEAK_RE.search(output):
        reasons.append("model_meta_or_leakage")
        hard_reject = True
    if OUT_OF_SCOPE_RE.search(combined):
        reasons.append("out_of_scope_code_math")
        hard_reject = True
    if MATH_TOPIC_RE.search(combined):
        reasons.append("math_topic")
        hard_reject = True
    if UNSAFE_OR_SENSITIVE_RE.search(combined):
        reasons.append("unsafe_or_sensitive_topic")
        hard_reject = True
    if MEDICAL_RE.search(combined):
        reasons.append("medical_advice_topic")
        hard_reject = True
    if HEALTH_TOPIC_RE.search(combined):
        reasons.append("health_topic")
        hard_reject = True
    if FINANCIAL_TOPIC_RE.search(combined):
        reasons.append("financial_topic")
        hard_reject = True
    if AGENT_COMMAND_RE.search(combined):
        reasons.append("agent_command_topic")
        hard_reject = True
    if ENTITY_ANCHOR_RE.search(instruction) and not ENTITY_ANCHOR_RE.search(output):
        reasons.append("entity_anchor_mismatch")
        hard_reject = True
    if WEB_CONTENT_RE.search(combined):
        reasons.append("web_content_or_news_task")
        hard_reject = True
    if ENGLISH_CONTENT_RE.search(combined):
        reasons.append("english_content")
        hard_reject = True
    if TASK_META_RE.search(combined):
        reasons.append("benchmark_task_meta")
        hard_reject = True
    if SENTIMENT_RE.search(combined):
        reasons.append("sentiment_rewrite_task")
        hard_reject = True
    if GRAMMAR_TASK_RE.search(combined):
        reasons.append("grammar_diagnosis_task")
        hard_reject = True
    if PLACEHOLDER_RE.search(combined):
        reasons.append("template_placeholder")
        hard_reject = True
    if len(words(prompt)) >= 5 and INCOMPLETE_PROMPT_RE.search(prompt):
        reasons.append("incomplete_prompt")
        hard_reject = True
    if PT_PT_RE.search(combined):
        reasons.append("pt_pt_or_non_brazilian_variant")
        hard_reject = True
    if OTHER_LANGUAGE_REQUEST_RE.search(combined):
        reasons.append("other_language_request")
        hard_reject = True
    boilerplate_hits = len(BOILERPLATE_RE.findall(output))
    if boilerplate_hits >= 2:
        reasons.append("boilerplate_dominant")
        hard_reject = True
    elif boilerplate_hits == 1:
        reasons.append("boilerplate_marker")

    unique_ratio = len(set(ow)) / max(len(ow), 1)
    max_word_ratio = max(collections.Counter(ow).values(), default=0) / max(len(ow), 1)
    repeated_ngrams = repeated_ngram_count(ow)
    if len(ow) >= 40 and unique_ratio < 0.38:
        reasons.append("low_lexical_diversity")
        hard_reject = True
    if len(ow) >= 40 and max_word_ratio > 0.20:
        reasons.append("dominant_word_repetition")
        hard_reject = True
    if repeated_ngrams >= 3:
        reasons.append("repeated_fourgrams")
        hard_reject = True

    instruction_norm = " ".join(iw)
    output_norm = " ".join(ow)
    if len(iw) >= 8 and instruction_norm in output_norm:
        reasons.append("prompt_leakage")
        hard_reject = True
    if GENERIC_RE.fullmatch(output.strip()):
        reasons.append("generic_answer")
    if EXPLANATION_RE.search(instruction) and len(ow) < 15:
        reasons.append("underdeveloped_explanation")
        hard_reject = True
    if TRAVEL_RE.search(instruction):
        anchors = proper_anchors(instruction)
        answer_lower = output.lower()
        missing = [anchor for anchor in anchors if anchor not in answer_lower]
        answer_places = proper_anchors(output)
        if missing and answer_places:
            reasons.append("named_anchor_mismatch")
            hard_reject = True
    if EXTRACTION_RE.search(instruction):
        source_words = [token for token in iw if len(token) >= 5 and token not in PT_STOPWORDS]
        if source_words and not any(token in output_norm for token in source_words):
            reasons.append("extraction_anchor_mismatch")
            hard_reject = True
    prompt_tokens = set(words(prompt))
    answer_tokens = set(ow)
    copy_overlap = len(prompt_tokens & answer_tokens) / max(len(answer_tokens), 1)
    if len(ow) >= 8 and copy_overlap >= 0.86:
        reasons.append("prompt_copy_dominant")
        hard_reject = True
    if re.search(r"(?i)\b(?:mais\s+espec[ií]fico|n[aã]o\s+entendi\s+o\s+que|"
                 r"poderia\s+especificar)\b", output) and len(ow) < 35:
        reasons.append("unhelpful_generic_clarification")
        hard_reject = True
    if re.search(r"(?i)\b(?:crie|escreva|gere|componha)\s+(?:uma\s+)?situa[cç][aã]o\b", instruction) and output.strip().endswith("?"):
        reasons.append("instruction_answer_type_mismatch")
        hard_reject = True
    if re.search(r"(?i)\b(?:s[eé]rie\s+de\s+perguntas|crie\s+perguntas|fa[cç]a\s+perguntas)\b", instruction) and "?" not in output:
        reasons.append("instruction_answer_type_mismatch")
        hard_reject = True
    if TEMPLATE_RE.search(output) and not re.search(r"(?i)\b(?:e-mail|email|carta|mensagem)\b", instruction):
        reasons.append("template_answer_type_mismatch")
        hard_reject = True
    if PRICE_RE.search(instruction) and re.search(r"\$\s*\d+|\b\d{2,}\b", output):
        reasons.append("unsupported_current_price_claim")
        hard_reject = True
    if re.search(r"(?i)\bdieta\s+vegana\b|\bcomo\s+.+afeta\s+sua\s+sa[uú]de\b", instruction) and not re.search(r"(?i)\b(?:sa[uú]de|benef[ií]cio|risco|nutriente|defici[eê]ncia)\b", output):
        reasons.append("health_question_not_answered")
        hard_reject = True
    if re.search(r"(?i)\bseu\s+pa[ií]s\b", instruction) and re.search(r"(?i)\b(?:americano|estados\s+unidos|usa)\b", output):
        reasons.append("localization_mismatch")
        hard_reject = True
    if IMPERIAL_RE.search(combined):
        reasons.append("imperial_localization")
        hard_reject = True

    quality = "reject" if hard_reject else ("quarantine" if reasons else "accept")
    metrics = {
        "instruction_words": len(iw),
        "context_words": len(cw),
        "answer_words": len(ow),
        "pt_stopwords": pt,
        "en_stopwords": en,
        "accent_ratio": round(accent_ratio, 8),
        "unique_word_ratio": round(unique_ratio, 6),
        "max_word_ratio": round(max_word_ratio, 6),
        "repeated_fourgrams": repeated_ngrams,
        "url_count": url_count,
        "email_count": email_count,
        "prompt_copy_overlap": round(copy_overlap, 6),
    }
    row["instruction"] = instruction
    row["input"] = context
    row["output"] = output
    return quality, reasons, metrics


def prompt_text(row: Dict[str, Any]) -> str:
    return "\n\n".join(part for part in (row["instruction"], row.get("input", "")) if part)


def accepted_record(row: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    prompt = prompt_text(row)
    return {
        "kind": "instruction",
        "prompt": prompt,
        "answer": row["output"],
        "metadata": {
            "source": SOURCE_REPO,
            "source_split": row["split"],
            "source_row": row["row_index"],
            "filter_version": FILTER_VERSION,
            "prompt_sha256": stable_hash(prompt),
            "record_sha256": stable_hash(prompt + "\0" + row["output"]),
            "metrics": metrics,
        },
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def process_split(path: Path, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], collections.Counter]:
    accepted: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reasons = collections.Counter()
    for raw in iter_parquet_rows(path, split):
        verdict, why, metrics = local_quality(raw)
        if verdict == "accept":
            item = accepted_record(raw, metrics)
            item["_prompt_hash"] = stable_hash(prompt_text(raw))
            item["_record_hash"] = stable_hash(prompt_text(raw) + "\0" + raw["output"])
            item["_simhash"] = simhash64(words(prompt_text(raw) + "\n" + raw["output"]))
            accepted.append(item)
        else:
            payload = {
                "split": split,
                "source_row": raw["row_index"],
                "instruction": raw["instruction"],
                "input": raw["input"],
                "output": raw["output"],
                "reasons": why,
                "metrics": metrics,
            }
            for reason in why:
                reasons[f"{verdict}:{reason}"] += 1
            (quarantined if verdict == "quarantine" else rejected).append(payload)
    return accepted, quarantined, rejected, reasons


def deduplicate(rows: List[Dict[str, Any]], split: str, reserved_prompts: set[str], reasons: collections.Counter) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_prompt: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_prompt[row["_prompt_hash"]].append(row)
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for prompt_hash, group in by_prompt.items():
        unique_records = {row["_record_hash"] for row in group}
        if len(unique_records) > 1:
            for row in group:
                removed.append({"record": row, "reasons": ["conflicting_prompt_answers"]})
                reasons["dedup:conflicting_prompt_answers"] += 1
            continue
        kept.append(group[0])
        for duplicate in group[1:]:
            removed.append({"record": duplicate, "reasons": ["duplicate_exact"]})
            reasons["dedup:duplicate_exact"] += 1

    if split == "train":
        for row in list(kept):
            if row["_prompt_hash"] in reserved_prompts:
                kept.remove(row)
                removed.append({"record": row, "reasons": ["eval_prompt_leakage"]})
                reasons["dedup:eval_prompt_leakage"] += 1

    buckets: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    final: List[Dict[str, Any]] = []
    for row in kept:
        bucket = row["_simhash"] >> 48
        near = next((old for old in buckets[bucket] if hamming(old["_simhash"], row["_simhash"]) <= 3), None)
        if near is not None:
            removed.append({"record": row, "reasons": ["duplicate_near_simhash"]})
            reasons["dedup:duplicate_near_simhash"] += 1
            continue
        buckets[bucket].append(row)
        final.append(row)
    return final, removed


def strip_internal(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    for key in ("_prompt_hash", "_record_hash", "_simhash"):
        result.pop(key, None)
    return result


def build_report(manifest: Dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Canarim SFT — filtro rigoroso",
        "",
        f"Filtro: `{FILTER_VERSION}`",
        f"Fonte: `{SOURCE_REPO}`",
        f"Licença da fonte: `{SOURCE_LICENSE}` (pesquisa; não comercial)",
        "",
        "## Resultado",
        "",
        f"- treino aceito: **{counts['train_accepted']:,}**",
        f"- avaliação aceita: **{counts['eval_accepted']:,}**",
        f"- quarentena local: **{counts['train_quarantine'] + counts['eval_quarantine']:,}**",
        f"- rejeitados: **{counts['train_rejected'] + counts['eval_rejected']:,}**",
        f"- removidos por deduplicação/conflito: **{counts['dedup_removed']:,}**",
        f"- vazamento de prompt avaliação→treino: **{counts['eval_prompt_leakage']:,}**",
        f"- sobreposição restante treino/avaliação: **{counts['prompt_overlap_after_filter']:,}**",
        "",
        "## Critério de liberação",
        "",
        f"- pronto para treino: **{manifest['ready_for_training']}**",
        "- o split de avaliação foi processado antes do treino e seus prompts foram reservados;",
        "- cada registro aceito tem `kind`, `prompt`, `answer`, origem e hashes;",
        "- registros duvidosos permanecem em quarentena, não foram misturados ao treino;",
        "- a licença continua restrita a validação de pesquisa (CC BY-NC 4.0).",
        "",
        "## Observação",
        "",
        "Este resultado é um SFT filtrado para experimento. A aprovação automática não substitui revisão humana de uma amostra antes de promover o modelo.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    train_files = sorted((args.raw_dir / "data").glob("train-*.parquet"))
    test_files = sorted((args.raw_dir / "data").glob("test-*.parquet"))
    if len(train_files) != 1 or len(test_files) != 1:
        raise RuntimeError(f"expected exactly one train/test parquet, got train={train_files} test={test_files}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "filter_version": FILTER_VERSION,
        "source": {
            "repo": SOURCE_REPO,
            "license": SOURCE_LICENSE,
            "research_only": True,
            "raw_files": {str(path.relative_to(args.raw_dir)): sha256_file(path) for path in [train_files[0], test_files[0], args.raw_dir / "README.md"] if path.exists()},
        },
        "criteria": {
            "hard_quality": ["schema", "unicode", "html", "length", "language", "web_artifacts_zero_tolerance", "other_language_zero_tolerance", "english_content", "math_topic", "health_topic", "financial_topic", "agent_command_topic", "entity_anchor_mismatch", "imperial_localization", "web_content_or_news_task", "benchmark_task_meta", "sentiment_rewrite_task", "grammar_diagnosis_task", "template_placeholder", "incomplete_prompt", "model_leakage", "boilerplate", "repetition", "prompt_leakage"],
            "dedup": ["exact_record", "conflicting_prompt", "near_simhash", "eval_prompt_leakage"],
            "near_duplicate_hamming_threshold": 3,
        },
    }
    train, train_q, train_r, reasons = process_split(train_files[0], "train")
    eval_rows, eval_q, eval_r, eval_reasons = process_split(test_files[0], "eval")
    reasons.update(eval_reasons)

    eval_clean, eval_dup = deduplicate(eval_rows, "eval", set(), reasons)
    reserved = {row["_prompt_hash"] for row in eval_clean}
    train_clean, train_dup = deduplicate(train, "train", reserved, reasons)
    dedup_removed = eval_dup + train_dup

    clean_train = [strip_internal(row) for row in train_clean]
    clean_eval = [strip_internal(row) for row in eval_clean]
    train_quarantine = train_q
    eval_quarantine = eval_q
    train_rejected = train_r + [item["record"] for item in train_dup]
    eval_rejected = eval_r + [item["record"] for item in eval_dup]

    files: Dict[str, Any] = {}
    for name, rows in (("train.jsonl", clean_train), ("eval.jsonl", clean_eval), ("quarantine_train.jsonl", train_quarantine), ("quarantine_eval.jsonl", eval_quarantine), ("rejected_train.jsonl", train_rejected), ("rejected_eval.jsonl", eval_rejected)):
        count, digest = write_jsonl(args.out_dir / name, rows)
        files[name] = {"rows": count, "sha256": digest}

    counts = {
        "source_train": sum(1 for _ in iter_parquet_rows(train_files[0], "train")),
        "source_eval": sum(1 for _ in iter_parquet_rows(test_files[0], "eval")),
        "train_accepted": len(clean_train),
        "eval_accepted": len(clean_eval),
        "train_quarantine": len(train_quarantine),
        "eval_quarantine": len(eval_quarantine),
        "train_rejected": len(train_rejected),
        "eval_rejected": len(eval_rejected),
        "dedup_removed": len(dedup_removed),
        "eval_prompt_leakage": reasons["dedup:eval_prompt_leakage"],
    }
    final_train_prompts = {row["_prompt_hash"] for row in train_clean}
    final_eval_prompts = {row["_prompt_hash"] for row in eval_clean}
    prompt_overlap_after_filter = len(final_train_prompts & final_eval_prompts)
    counts["prompt_overlap_after_filter"] = prompt_overlap_after_filter
    manifest["counts"] = counts
    manifest["reason_counts"] = dict(sorted(reasons.items()))
    manifest["files"] = files
    manifest["ready_for_training"] = bool(clean_train and clean_eval and prompt_overlap_after_filter == 0 and all(row.get("kind") == "instruction" and row.get("prompt") and row.get("answer") for row in clean_train + clean_eval))
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "FILTER_REPORT.md").write_text(build_report(manifest), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "counts": counts, "ready_for_training": manifest["ready_for_training"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
