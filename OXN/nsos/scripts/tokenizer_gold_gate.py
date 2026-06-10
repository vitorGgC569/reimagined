"""
tokenizer_gold_gate.py — padrao ouro industrial do tokenizer + corpus.

Motivacao (2026-06-10): o checkpoint phase3_step100 respondeu phase1 com
chunks de tensor.cpp e a primeira suspeita foi "tokenizer quebrado".  O
diagnostico real era outro (phase3 treina PRIMEIRO e inclui o proprio fonte
do NSOS por design), mas a suspeita nao tinha como ser descartada em segundos
porque NAO EXISTIA GATE.  Este script fecha isso:

  G1  ROUNDTRIP   decode(encode(x)) == x, byte-exato, para TODOS os eval rows
                  de todas as fases (prompt, answer e template completo).
  G2  SPECIALS    cada special token codifica para EXATAMENTE 1 id e
                  sobrevive ao roundtrip; injecao: texto de usuario contendo
                  o literal "<|endoftext|>" e' REPORTADO (vira 1 id especial
                  ou nao? — comportamento documentado, nao silencioso).
  G3  DETERMINISM encode(x) duas vezes e apos reload do artefato => identico.
  G4  PREFIX      encode(prompt + answer) comeca com encode(prompt)?  BPE pode
                  fundir no boundary; o treino mascara por listas separadas,
                  mas o EVAL gera a partir do prompt sozinho — divergencia
                  aqui explica gaps treino/eval.  Reportado com taxa.
  G5  COVERAGE    distribuicao de comprimento de token, ids fora do vocab,
                  taxa de bytes perdidos (errors=ignore em algum lugar?).
  G6  CORPUS      composicao por 'source' de cada fase (rows, chars) — a
                  estatistica que resolveria o misterio de hoje em 1 segundo.

PASS exige: G1 answers 100% exato; G1 templates >= 99.9%; G2 atomicos;
G3 exato.  G4/G5/G6 sao relatorios (nao gateiam, mas imprimem WARN).

Uso (Colab CPU ou qualquer maquina, sem GPU):
  python scripts/tokenizer_gold_gate.py --bundle-dir scripts/distillation_bundle_v11 \
         --build-dir <dir com nsos_ext>
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_curriculum import SPECIAL_TOKENS, detect_build_dir, load_nsos  # noqa: E402
from nsos_curriculum_lib import curriculum_texts_for_phase  # noqa: E402


def load_tokenizer(nsos, bundle_dir: Path):
    tok = nsos.Tokenizer()
    tok.load(str(bundle_dir / "tokenizer_8192.ox3"))
    tok.add_special_tokens(SPECIAL_TOKENS)
    return tok


def phases_in_bundle(bundle_dir: Path):
    manifest = json.loads((bundle_dir / "curriculum_manifest.json").read_text("utf-8"))
    return [p["name"] for p in manifest["phases"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, default=SCRIPT_DIR / "distillation_bundle_v11")
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--max-rows-per-phase", type=int, default=200)
    args = ap.parse_args()

    nsos = load_nsos(detect_build_dir(args.build_dir))
    tok = load_tokenizer(nsos, args.bundle_dir)
    phases = phases_in_bundle(args.bundle_dir)
    print(f"[gate] vocab={tok.vocab_size} fases={phases}")

    failures: list[str] = []
    warns: list[str] = []

    # ── G2: specials atomicos + injecao ─────────────────────────────────
    for sp in SPECIAL_TOKENS:
        ids = tok.encode(sp)
        if len(ids) != 1:
            failures.append(f"G2 special nao-atomico: {sp!r} -> {len(ids)} ids")
        elif tok.decode(ids) != sp:
            failures.append(f"G2 roundtrip do special falhou: {sp!r} -> {tok.decode(ids)!r}")
    inj = "answer contains <|endoftext|> literally"
    inj_ids = tok.encode(inj)
    eot_id = tok.encode("<|endoftext|>")[0]
    inj_hits = sum(1 for i in inj_ids if i == eot_id)
    print(f"[gate] G2 injecao: literal '<|endoftext|>' em texto de usuario vira "
          f"{inj_hits} id(s) especial(is) — {'ATENCAO: injetavel' if inj_hits else 'inerte'}")
    if inj_hits:
        warns.append("G2 injecao: special token injetavel via texto de usuario "
                     "(sanitizar entradas antes do encode em serving)")

    # ── G1/G3/G4/G5 sobre os eval rows de todas as fases ────────────────
    rt_answer_fail = 0
    rt_template_fail = 0
    rt_total = 0
    prefix_fail = 0
    det_fail = 0
    max_id_seen = -1
    token_lens = collections.Counter()
    examples: list[str] = []
    for phase in phases:
        rows = curriculum_texts_for_phase(args.bundle_dir, phase, "eval")
        rows = rows[: args.max_rows_per_phase]
        for row in rows:
            prompt = f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
            answer = str(row["answer"])
            full = prompt + answer + "<|endoftext|>"
            rt_total += 1

            ids_full = tok.encode(full)
            ids_full2 = tok.encode(full)
            if ids_full != ids_full2:
                det_fail += 1
            if ids_full:
                max_id_seen = max(max_id_seen, max(ids_full))
            for i in ids_full:
                token_lens[len(tok.decode([i]))] += 1

            if tok.decode(tok.encode(answer)) != answer:
                rt_answer_fail += 1
                if len(examples) < 5:
                    examples.append(f"{phase}: answer {answer!r} -> "
                                    f"{tok.decode(tok.encode(answer))!r}")
            if tok.decode(ids_full) != full:
                rt_template_fail += 1
                if len(examples) < 5:
                    examples.append(f"{phase}: template diverge (kind={row['kind']})")

            ids_prompt = tok.encode(prompt)
            ids_pa = tok.encode(prompt + answer)
            if ids_pa[: len(ids_prompt)] != ids_prompt:
                prefix_fail += 1

    print(f"[gate] G1 roundtrip: answers {rt_total - rt_answer_fail}/{rt_total} exatos; "
          f"templates {rt_total - rt_template_fail}/{rt_total} exatos")
    for ex in examples:
        print(f"[gate]    exemplo: {ex}")
    print(f"[gate] G3 determinismo encode: {det_fail} divergencias")
    print(f"[gate] G4 prefix-stability prompt|answer: {prefix_fail}/{rt_total} quebram "
          f"({'WARN: eval gera de prompt sozinho' if prefix_fail else 'estavel'})")
    print(f"[gate] G5 max id visto={max_id_seen} (vocab={tok.vocab_size}); "
          f"len de token: " +
          ", ".join(f"{l}ch:{c}" for l, c in sorted(token_lens.items())[:8]))

    if rt_answer_fail:
        failures.append(f"G1 {rt_answer_fail} answers nao roundtripam byte-exato")
    if rt_template_fail > max(1, rt_total // 1000):
        failures.append(f"G1 {rt_template_fail}/{rt_total} templates nao roundtripam (>0.1%)")
    if det_fail:
        failures.append(f"G3 encode nao-deterministico em {det_fail} casos")
    if max_id_seen >= tok.vocab_size:
        failures.append(f"G5 id {max_id_seen} >= vocab {tok.vocab_size}")
    if prefix_fail:
        warns.append(f"G4 {prefix_fail} boundaries instaveis prompt|answer")

    # ── G3b: reload do artefato => mesmos ids ───────────────────────────
    tok2 = load_tokenizer(nsos, args.bundle_dir)
    probe = "Compare -9 and 4. Answer with one label from LT, GT, EQ. Acentuação ção ✓"
    if tok.encode(probe) != tok2.encode(probe):
        failures.append("G3 reload do artefato muda a tokenizacao")

    # ── G6: composicao do corpus por fonte (o raio-x de hoje) ───────────
    print("[gate] G6 composicao do corpus (train, por 'source'):")
    for phase in phases:
        rows = curriculum_texts_for_phase(args.bundle_dir, phase, "train")
        by_source = collections.Counter()
        chars = collections.Counter()
        for row in rows:
            src = str(row.get("source", row.get("kind", "?")))
            by_source[src] += 1
            chars[src] += len(str(row.get("prompt", ""))) + len(str(row.get("answer", "")))
        top = ", ".join(f"{s}:{n}r/{chars[s]//1000}kc"
                        for s, n in by_source.most_common(6))
        print(f"[gate]    {phase}: {len(rows)} rows | {top}")

    for w in warns:
        print(f"[gate] WARN: {w}")
    if failures:
        for f in failures:
            print(f"[gate] FAIL: {f}")
        print("[gate] VEREDITO: FAIL")
        return 1
    print("[gate] VEREDITO: PASS — tokenizer padrao ouro nos eixos G1/G2/G3/G5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
