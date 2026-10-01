"""Quantified quality report for a prepared PT-BR corpus.

Reads the SQLite corpus produced by ``train_ptbr_conversational.py prepare``
and measures what actually survived ingestion filtering, so a multi-day
training run is not launched over a corpus nobody inspected.

The report is deliberately read-only and offline: it opens the database in
immutable mode and never writes to the workspace. Findings go to stdout and,
optionally, to a JSON file.

Measured dimensions
-------------------
* residual boilerplate: how many surviving documents still match the ingest
  patterns, plus a wider probe set this script owns;
* exact and near-duplicate rate (normalised-text hashing and shingles);
* provenance: token and document share per source/subset, with the license
  decision that admitted each one;
* quality/toxicity score distribution as recorded at ingest time;
* structural health: length distribution, empty/short docs, non-Latin ratio,
  URL/markup residue, repeated-line ratio;
* train/eval leakage: exact overlap of normalised payloads across splits.

Usage
-----
    python scripts/corpus_quality_report.py --preset pilot
    python scripts/corpus_quality_report.py --corpus <path.sqlite3> \
        --sample 20000 --json report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_SCRIPTS = Path(__file__).resolve().parent
if str(REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS))


def load_ingest_patterns() -> List[Tuple[str, "re.Pattern[str]"]]:
    """Reuse the exact patterns the ingest applied, if importable."""
    try:
        import train_ptbr_conversational as pipeline
    except Exception:  # noqa: BLE001 - the report must work standalone
        return []
    return [
        (pattern, compiled)
        for pattern, compiled in zip(
            pipeline.BOILERPLATE_PATTERNS, pipeline.COMPILED_BOILERPLATE
        )
    ]


# Patterns the ingest does NOT check. Residual hits here are the interesting
# ones: they are the boilerplate that survived preparation.
EXTRA_BOILERPLATE = {
    "cookie_consent": r"(?i)\b(aceit\w+|gerenciar)\s+cookies?\b",
    "newsletter": r"(?i)\b(assine|cadastre-se|receba)\s+.{0,30}\bnewsletter\b",
    "paywall": r"(?i)\b(conteúdo exclusivo|assine para continuar|"
               r"faça login para ler)\b",
    "nav_breadcrumb": r"(?i)\b(página inicial|home)\s*[>|»/]\s*\w+",
    "social_cta": r"(?i)\b(siga-nos|curta nossa página|nos siga)\b",
    "byline_date": r"(?i)\bpor\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\w.'-]+\s*[|·-]\s*\d{1,2}"
                   r"[/ ]\w+[/ ]\d{2,4}",
    "comment_count": r"(?i)\b\d+\s+coment[áa]rios?\b",
    "share_widget": r"(?i)\bcompartilh\w+\b.{0,20}\b"
                    r"(whatsapp|facebook|twitter|telegram|linkedin)\b",
    "tag_list": r"(?i)^\s*(tags?|assuntos?|palavras-chave)\s*:",
    "read_time": r"(?i)\b\d+\s*min(utos?)?\s+de\s+leitura\b",
    "copyright_line": r"(?i)©\s*\d{4}|\btodos os direitos reservados\b",
    "html_residue": r"</?(div|span|p|br|img|script|style|a)\b[^>]*>",
    "url_residue": r"https?://\S+",
    "email_residue": r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b",
    "boilerplate_ellipsis": r"\[\s*\.\.\.\s*\]|\(\s*continua\s*\)",
}
EXTRA_COMPILED = {
    name: re.compile(pattern) for name, pattern in EXTRA_BOILERPLATE.items()
}

WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def shingles(text: str, size: int = 8) -> set[int]:
    words = WORD_RE.findall(text)
    if len(words) < size:
        return set()
    return {
        hash(" ".join(words[i:i + size]))
        for i in range(0, len(words) - size + 1, max(1, size // 2))
    }


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(
        1 for c in letters if not ("LATIN" in unicodedata.name(c, ""))
    )
    return non_latin / len(letters)


def repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return 0.0
    counts = Counter(lines)
    repeated = sum(n for n in counts.values() if n > 1)
    return repeated / len(lines)


def percentiles(values: Sequence[float], points=(0.10, 0.50, 0.90, 0.99)):
    if not values:
        return {f"p{int(p * 100)}": None for p in points}
    ordered = sorted(values)
    out = {}
    for p in points:
        idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        out[f"p{int(p * 100)}"] = ordered[idx]
    return out


def open_corpus(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise SystemExit(f"corpus não encontrado: {path}")
    # Immutable: guarantees this tool cannot mutate a training corpus.
    uri = f"file:{path.as_posix()}?immutable=1"
    return sqlite3.connect(uri, uri=True)


def totals(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT phase, split, COUNT(*), COALESCE(SUM(estimated_tokens), 0) "
        "FROM records GROUP BY phase, split ORDER BY phase, split"
    ).fetchall()
    return [
        {"phase": p, "split": s, "documents": n, "tokens": t}
        for p, s, n, t in rows
    ]


def provenance(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT source, subset_name, license_decision, COUNT(*), "
        "       COALESCE(SUM(estimated_tokens), 0), AVG(quality) "
        "FROM records WHERE phase='base' "
        "GROUP BY source, subset_name, license_decision "
        "ORDER BY SUM(estimated_tokens) DESC"
    ).fetchall()
    total_tokens = sum(r[4] for r in rows) or 1
    return [
        {
            "source": src or "(vazio)",
            "subset": sub or "",
            "license_decision": lic,
            "documents": docs,
            "tokens": toks,
            "token_share": toks / total_tokens,
            "mean_quality": round(q, 3) if q is not None else None,
        }
        for src, sub, lic, docs, toks, q in rows
    ]


def sample_payloads(
    conn: sqlite3.Connection, phase: str, split: str, limit: int, seed: int
) -> List[str]:
    total = conn.execute(
        "SELECT COUNT(*) FROM records WHERE phase=? AND split=?",
        (phase, split),
    ).fetchone()[0]
    if total == 0:
        return []
    if total <= limit:
        rows = conn.execute(
            "SELECT payload FROM records WHERE phase=? AND split=?",
            (phase, split),
        ).fetchall()
        return [r[0] for r in rows]
    # Deterministic reservoir over rowids keeps the sample reproducible
    # without loading the whole corpus into memory.
    rng = random.Random(seed)
    ids = [r[0] for r in conn.execute(
        "SELECT rowid FROM records WHERE phase=? AND split=?", (phase, split)
    )]
    chosen = rng.sample(ids, limit)
    marks = ",".join("?" * len(chosen))
    rows = conn.execute(
        f"SELECT payload FROM records WHERE rowid IN ({marks})", chosen
    ).fetchall()
    return [r[0] for r in rows]


def analyse_documents(
    payloads: Sequence[str],
    ingest_patterns: Sequence[Tuple[str, "re.Pattern[str]"]],
) -> Dict[str, Any]:
    n = len(payloads)
    if n == 0:
        return {"documents": 0}

    ingest_hits: Counter[str] = Counter()
    extra_hits: Counter[str] = Counter()
    lengths: List[int] = []
    word_counts: List[int] = []
    non_latin: List[float] = []
    repeated: List[float] = []
    short_docs = 0
    empty_docs = 0
    exact_hashes: Counter[str] = Counter()
    docs_with_any_extra = 0

    for text in payloads:
        if not text or not text.strip():
            empty_docs += 1
            continue
        lengths.append(len(text))
        words = WORD_RE.findall(text)
        word_counts.append(len(words))
        if len(words) < 32:
            short_docs += 1
        non_latin.append(non_latin_ratio(text[:4000]))
        repeated.append(repeated_line_ratio(text))
        exact_hashes[hashlib.blake2b(
            normalise(text).encode("utf-8"), digest_size=16
        ).hexdigest()] += 1

        for pattern, compiled in ingest_patterns:
            if compiled.search(text):
                ingest_hits[pattern] += 1
        hit_extra = False
        for name, compiled in EXTRA_COMPILED.items():
            if compiled.search(text):
                extra_hits[name] += 1
                hit_extra = True
        if hit_extra:
            docs_with_any_extra += 1

    duplicated_docs = sum(c for c in exact_hashes.values() if c > 1)
    return {
        "documents": n,
        "empty_documents": empty_docs,
        "short_documents_lt32_words": short_docs,
        "short_document_rate": short_docs / n,
        "exact_duplicate_documents": duplicated_docs,
        "exact_duplicate_rate": duplicated_docs / n,
        "unique_normalised_documents": len(exact_hashes),
        "char_length": percentiles(lengths),
        "word_count": percentiles(word_counts),
        "non_latin_letter_ratio": percentiles(non_latin),
        "repeated_line_ratio": percentiles(repeated),
        "residual_ingest_pattern_hits": {
            pattern: {"documents": count, "rate": count / n}
            for pattern, count in ingest_hits.most_common()
        },
        "residual_extra_pattern_hits": {
            name: {"documents": count, "rate": count / n}
            for name, count in extra_hits.most_common()
        },
        "documents_with_any_extra_boilerplate": docs_with_any_extra,
        "extra_boilerplate_rate": docs_with_any_extra / n,
    }


def near_duplicate_rate(
    payloads: Sequence[str], budget: int = 4000, threshold: float = 0.5
) -> Dict[str, Any]:
    """Shingle-overlap probe over a bounded sub-sample.

    A document counts as a near-duplicate only when more than ``threshold`` of
    its shingles were already seen. Flagging on a single shared shingle is
    useless here: every SFT record carries the same system prompt, so a
    single-collision rule reports ~90% duplicates purely from that shared
    prefix. Requiring a proportional overlap keeps shared boilerplate from
    dominating while still catching genuinely recycled documents.
    """
    subset = payloads[:budget]
    seen: set[int] = set()
    near_dupes = 0
    scored = 0
    overlaps: List[float] = []
    for text in subset:
        sig = shingles(normalise(text))
        if not sig:
            continue
        scored += 1
        shared = len(sig & seen)
        fraction = shared / len(sig)
        overlaps.append(fraction)
        if fraction > threshold:
            near_dupes += 1
        seen |= sig
    return {
        "probe_documents": len(subset),
        "scored_documents": scored,
        "near_duplicate_documents": near_dupes,
        "near_duplicate_rate": near_dupes / scored if scored else 0.0,
        "shingle_overlap_percentiles": percentiles(overlaps),
        "threshold": threshold,
        "method": (
            "8-word shingles, stride 4; near-duplicate when >"
            f"{threshold:.0%} of a document's shingles were already seen"
        ),
    }


def split_leakage(conn: sqlite3.Connection, phase: str) -> Dict[str, Any]:
    def hashes(split: str) -> set[str]:
        out = set()
        for (payload,) in conn.execute(
            "SELECT payload FROM records WHERE phase=? AND split=?",
            (phase, split),
        ):
            if payload and payload.strip():
                out.add(hashlib.blake2b(
                    normalise(payload).encode("utf-8"), digest_size=16
                ).hexdigest())
        return out

    train = hashes("train")
    evalset = hashes("eval")
    overlap = train & evalset
    return {
        "phase": phase,
        "train_unique": len(train),
        "eval_unique": len(evalset),
        "overlapping_documents": len(overlap),
        "eval_contamination_rate": (
            len(overlap) / len(evalset) if evalset else 0.0
        ),
    }


def quality_distribution(conn: sqlite3.Connection) -> Dict[str, Any]:
    values = [
        r[0] for r in conn.execute(
            "SELECT quality FROM records WHERE phase='base' "
            "AND quality IS NOT NULL"
        )
    ]
    if not values:
        return {"documents": 0}
    buckets = Counter(int(v) for v in values)
    return {
        "documents": len(values),
        "mean": round(statistics.fmean(values), 4),
        "percentiles": percentiles(values),
        "histogram": {str(k): buckets[k] for k in sorted(buckets)},
    }


def render(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("RELATÓRIO DE QUALIDADE DO CORPUS")
    print("=" * 72)
    print(f"corpus : {report['corpus']}")
    print(f"amostra: {report['sample_size']} docs por fase/split "
          f"(seed {report['seed']})")

    print("\n-- volume --")
    print(f"{'fase':14} {'split':6} {'documentos':>12} {'tokens':>14}")
    for row in report["totals"]:
        print(f"{row['phase']:14} {row['split']:6} {row['documents']:12,} "
              f"{row['tokens']:14,}")

    print("\n-- proveniência (fase base, por tokens) --")
    print(f"{'fonte':34} {'licença':12} {'docs':>9} {'tokens':>13} {'share':>7}")
    for row in report["provenance"][:20]:
        label = row["source"]
        if row["subset"]:
            label = f"{label}/{row['subset']}"
        print(f"{label[:34]:34} {row['license_decision'][:12]:12} "
              f"{row['documents']:9,} {row['tokens']:13,} "
              f"{row['token_share'] * 100:6.2f}%")

    for phase, stats in report["documents"].items():
        if not stats.get("documents"):
            continue
        print(f"\n-- estrutura: {phase} --")
        print(f"documentos analisados      : {stats['documents']:,}")
        print(f"vazios                     : {stats['empty_documents']:,}")
        print(f"curtos (<32 palavras)      : "
              f"{stats['short_documents_lt32_words']:,} "
              f"({stats['short_document_rate'] * 100:.2f}%)")
        print(f"duplicatas exatas          : "
              f"{stats['exact_duplicate_documents']:,} "
              f"({stats['exact_duplicate_rate'] * 100:.2f}%)")
        wc = stats["word_count"]
        print(f"palavras p10/p50/p90/p99   : {wc['p10']}/{wc['p50']}/"
              f"{wc['p90']}/{wc['p99']}")
        rl = stats["repeated_line_ratio"]
        print(f"linhas repetidas p50/p90   : {rl['p50']:.3f}/{rl['p90']:.3f}")
        nl = stats["non_latin_letter_ratio"]
        print(f"letras não-latinas p90/p99 : {nl['p90']:.3f}/{nl['p99']:.3f}")

        residual = stats["residual_ingest_pattern_hits"]
        print(f"\n  boilerplate RESIDUAL dos padrões do ingest "
              f"({len(residual)} com ocorrência):")
        if not residual:
            print("    nenhum — os filtros do ingest removeram tudo que casam")
        for pattern, info in list(residual.items())[:10]:
            print(f"    {info['rate'] * 100:6.2f}%  {pattern[:58]}")

        print(f"\n  boilerplate NÃO coberto pelo ingest "
              f"(total {stats['extra_boilerplate_rate'] * 100:.2f}% dos docs):")
        for name, info in list(
            stats["residual_extra_pattern_hits"].items()
        )[:12]:
            print(f"    {info['rate'] * 100:6.2f}%  {name}")

        nd = stats.get("near_duplicates")
        if nd:
            print(f"\n  quase-duplicatas           : "
                  f"{nd['near_duplicate_rate'] * 100:.2f}% "
                  f"({nd['near_duplicate_documents']:,}/"
                  f"{nd['probe_documents']:,})")

    print("\n-- qualidade registrada no ingest (fase base) --")
    q = report["quality"]
    if q.get("documents"):
        print(f"média {q['mean']} | histograma {q['histogram']}")

    print("\n-- vazamento train/eval --")
    for row in report["leakage"]:
        print(f"{row['phase']:14} overlap={row['overlapping_documents']:,} "
              f"contaminação do eval="
              f"{row['eval_contamination_rate'] * 100:.3f}%")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Relatório quantificado de qualidade do corpus PT-BR."
    )
    parser.add_argument("--preset", default="pilot")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    corpus = args.corpus or (
        Path(__file__).resolve().parents[1]
        / "artifacts" / "ptbr_conversational" / args.preset
        / "corpus" / "corpus.sqlite3"
    )
    conn = open_corpus(corpus)
    ingest_patterns = load_ingest_patterns()

    report: Dict[str, Any] = {
        "corpus": str(corpus),
        "sample_size": args.sample,
        "seed": args.seed,
        "ingest_patterns_reused": len(ingest_patterns),
        "totals": totals(conn),
        "provenance": provenance(conn),
        "quality": quality_distribution(conn),
        "documents": {},
        "leakage": [],
    }

    for phase in ("base", "continuation", "sft"):
        payloads = sample_payloads(conn, phase, "train", args.sample, args.seed)
        if not payloads:
            continue
        stats = analyse_documents(payloads, ingest_patterns)
        stats["near_duplicates"] = near_duplicate_rate(payloads)
        report["documents"][phase] = stats
        report["leakage"].append(split_leakage(conn, phase))

    render(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
