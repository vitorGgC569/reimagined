#!/usr/bin/env python3
"""Auditable EDU + Synth + complete-answer SFT recipe for the native HIP trainer.

No teacher API, PyTorch, legacy shards, or legacy model weights are used.
The audit is structural/language/dedup, NOT a certificate of factual correctness.
Use the existing CLI: run/prepare/train/status, --build-dir, --workspace, etc.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

import train_ptbr_conversational as core

VERSION = "ptbr-edu-synth-complete-sft-v2"
PRESET_NAME = "edu-synth-pilot"
REPOS = {
    "base": "Polygl0t/gigaverbo-v2",
    "synth": "Polygl0t/gigaverbo-v2-synth",
    "sft": "Polygl0t/gigaverbo-v2-sft",
}
SFT_MIX = {"general": .60, "math": .15, "retrieval": .15, "structured": .10}
EDU_SHARE = .70
HEADROOM = 1.6
MAX_ROWS_PER_STREAM = 3_000_000
FORBIDDEN = re.compile(
    r"\b(?:leia\s+mais|saiba\s+mais|clique\s+aqui|todos\s+os\s+direitos\s+reservados|"
    r"aceit\w*\s+(?:os\s+)?cookies|assine\s+(?:a\s+)?newsletter|"
    r"compartilhe\s+nas\s+redes|deixe\s+(?:seu|um)\s+coment.rio|"
    r"reportar\s+erro|galeria\s+de\s+imagens|saiba\s+quais|veja\s+tamb.m|"
    r"respostas?\s+relacionadas?|continue\s+lendo|acesse\s+(?:o\s+)?site|"
    r"baixe\s+(?:o\s+)?(?:aplicativo|app))\b", re.I
)
RESTRICTED_NOTICE = re.compile(
    r"CC[ -]?BY[ -]?(?:NC|SA)|n.o[ -]comercial|non[ -]commercial|"
    r"atribui..o\s*[-:]?\s*compartilhada|uso\s+exclusivamente\s+pessoal", re.I)
WORKSHEET = re.compile(r"(?m)^\s*(?:nome|turma|professor\s*/\s*monitor)\s*:", re.I)
SPECIAL = re.compile(r"<\|[^>]*\|>|</?(?:think|im_start|im_end)>", re.I)
HTML_TAG = re.compile(r"</?(?:div|span|p|a|br|html|body|script|style)\b[^>]*>", re.I)
SOURCE_REVIEW = Path(__file__).resolve().parent / "data_recipes" / "ptbr_edu_synth_review.json"


def require_source_quality_review(path=SOURCE_REVIEW):
    review = core._read_json_object(path, "Synthetic source quality review")
    if review.get("dataset") != REPOS["synth"] or review.get("decision") != "approved":
        raise RuntimeError(
            "Synthetic source failed manual quality review. Training is blocked; "
            "see scripts/data_recipes/ptbr_edu_synth_review.json. Structural filters "
            "and a Portuguese language label are not factual/linguistic approval.")
    return review


def make_preset():
    return {
        **core.PRESETS["pilot"],
        "description": "71M native Mamba; 80M EDU/Synth + 12M chat + 2x4M SFT",
        "dataset_repos": REPOS,
        "curated_recipe": {
            "version": VERSION,
            "edu_share": EDU_SHARE,
            "edu_min_score": 4,
            "toxic_max_score": 1,
            "max_edu_source_share": .60,
            "sft_min_score": 4.0,
            "sft_mix": SFT_MIX,
            "language_model": "langid==1.1.6; all languages; p(pt)>=0.90",
            "headroom": HEADROOM,
            "max_rows_per_stream": MAX_ROWS_PER_STREAM,
            "complete_sft_only": True,
            "recipe_code_sha256": core.sha256_file(Path(__file__)),
        },
    }


class PortugueseFilter:
    def __init__(self, dependencies: Path):
        sys.path.insert(0, str(dependencies))
        from langid.langid import LanguageIdentifier, model
        if importlib.metadata.version("langid") != "1.1.6":
            raise RuntimeError("This recipe requires langid==1.1.6")
        self.classifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.model_sha256 = hashlib.sha256(model).hexdigest()

    def __call__(self, text):
        language, probability = self.classifier.classify(text[:4000])
        return language == "pt" and float(probability) >= .90


def clean_text(raw: str, *, conversational=False):
    """Reject contaminated conversations rather than splice incomplete answers."""
    if not isinstance(raw, str) or SPECIAL.search(raw) or "\ufffd" in raw:
        return None, "encoding_or_special_token"
    text = html.unescape(raw).replace("\r\n", "\n").strip()
    if RESTRICTED_NOTICE.search(text):
        return None, "restricted_original_notice"
    if WORKSHEET.search(text):
        return None, "unreviewed_worksheet"
    if text.endswith(("...", "…")):
        return None, "truncated_document"
    if conversational and (FORBIDDEN.search(text) or core.pii_flags(text)):
        return None, "conversation_boilerplate_or_contact"
    if not conversational:
        lines = [line for line in text.splitlines() if not FORBIDDEN.search(line)]
        text = core.normalize_text("\n".join(lines))
    if not text or len(text) < .70 * len(raw.strip()):
        return None, "excessive_removal"
    if FORBIDDEN.search(text) or SPECIAL.search(text) or HTML_TAG.search(text):
        return None, "residual_boilerplate"
    if core.pii_flags(text):
        return None, "residual_contact"
    words = text.split()
    if len(words) > 3000:
        return None, "overlong"
    if len(words) >= 40:
        grams = Counter(tuple(words[i:i+4]) for i in range(len(words)-3))
        repeated = sum(n-1 for n in grams.values()) / max(len(words)-3, 1)
        if repeated > .15:
            return None, "repetition"
    return text, None


def curated_messages(row, is_portuguese):
    score = row.get("instruct_score", 0)
    if not isinstance(score, (float, int)) or not math.isfinite(score) or score < 4:
        return None, "instruction_score"
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, "messages"
    result = []
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None, "messages"
        role = message.get("role")
        if index == 0 and role == "system":
            pass
        elif role != expected:
            return None, "role_order"
        else:
            expected = "assistant" if role == "user" else "user"
        text, reason = clean_text(message.get("content", ""), conversational=True)
        if reason:
            return None, reason
        if role == "assistant" and (len(text.split()) > 110 or not text.strip()):
            return None, "answer_length"
        result.append({"role": role, "content": text})
    if result[-1]["role"] != "assistant":
        return None, "unfinished_conversation"
    # Do not let our default Portuguese system message fool language detection.
    dialogue = "\n".join(m["content"] for m in result if m["role"] != "system")
    if len(dialogue.split()) > 240 or not is_portuguese(dialogue):
        return None, "language_or_context_length"
    if result[0]["role"] != "system":
        result.insert(0, {"role": "system", "content": core.DEFAULT_SYSTEM_PROMPT})
    return result, None


def stream_rows(repo, config, revision, seed):
    from datasets import load_dataset
    # Push score predicates into Arrow before Python/shuffling. The row budget
    # and ingest counters explicitly refer to these prefiltered rows, not the
    # full upstream corpus. Admission checks below still validate every row.
    filters = ([('edu_int_score', '>=', 4), ('toxic_int_score', '<=', 1)]
               if repo == REPOS['base'] else None)
    return load_dataset(repo, config, split="train", streaming=True,
                        revision=revision, filters=filters).shuffle(seed=seed, buffer_size=1000)


def ingest(store, preset, revisions, seed, language):
    store.connection.execute(
        "CREATE TABLE IF NOT EXISTS curated_source_records (fingerprint TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)")
    store.commit()
    streams = [("base", "edu", "default", EDU_SHARE),
               ("synth", "synth", "default", 1-EDU_SHARE)]
    streams += [("sft", category, category, weight) for category, weight in SFT_MIX.items()]
    samples = defaultdict(list)
    for stream_index, (kind, category, config, share) in enumerate(streams):
        if kind == "sft":
            targets = {"continuation": int(preset["continuation_tokens"]*share*HEADROOM),
                       "sft": int(preset["sft_unique_tokens"]*share*HEADROOM)}
        else:
            targets = {"base": int(preset["base_tokens"]*share*HEADROOM)}
        counts = Counter({(p, s): n for p, s, n in store.connection.execute(
            "SELECT phase, split, SUM(estimated_tokens) FROM records WHERE category=? "
            "GROUP BY phase,split", (category,))})
        family_counts = Counter({f: n for f, n in store.connection.execute(
            "SELECT subset_name, SUM(estimated_tokens) FROM records "
            "WHERE category=? AND split='train' GROUP BY subset_name", (category,))})
        def enough():
            return all(counts[(p, "train")] >= n and
                       counts[(p, "eval")] >= max(2000, n//100)
                       for p, n in targets.items())
        key = f"curated_cursor:{kind}:{config}"
        state = json.loads(store.get_meta(key) or '{}')
        seen = int(state.get("seen", 0))
        rejected = Counter(state.get("rejected", {}))
        if enough():
            continue
        print(f"[curated:ingest] {kind}/{config} targets={targets} resume_row={seen}", flush=True)
        dataset = stream_rows(REPOS[kind], config, revisions[kind], seed+stream_index)
        if seen:
            dataset = dataset.skip(seen)
        for row in dataset:
            seen += 1
            reason = None
            quality = None
            if kind == "sft":
                messages, reason = curated_messages(row, language)
                if messages:
                    payload = core.stable_json(messages)
                    content = "\n".join(m["role"]+":"+m["content"] for m in messages
                                        if m["role"] != "system")
                    # All answers to the same user prompt must share the holdout split.
                    split_key = core.content_hash_of("\n".join(
                        m["content"] for m in messages if m["role"] == "user"))
                    phase = core._sft_phase_bucket(core.content_hash_of(content))
                    estimated = (core.estimate_sft_target_tokens(messages, preset["seq_len"])
                                 if phase == "sft" else core.estimate_tokens_from_text(payload))
                    source, subset, decision = REPOS[kind], config, "Apache-2.0"
                    quality = float(row["instruct_score"])
            else:
                raw = row.get("text", "")
                if kind == "base":
                    source = str(row.get("source", ""))
                    original_subset = str(row.get("subset", ""))
                    allowed, decision, _ = core.source_allowed(source, original_subset, "commercial-strict")
                    quality = row.get("edu_int_score", 0)
                    toxic = row.get("toxic_int_score", 5)
                    if not allowed:
                        reason = "source_license"
                    elif not isinstance(quality, (float, int)) or not math.isfinite(quality) or quality < 4:
                        reason = "educational_score"
                    elif not isinstance(toxic, (float, int)) or not math.isfinite(toxic) or toxic > 1:
                        reason = "toxicity_score"
                    subset = core.source_family_of(source, original_subset) or source
                else:
                    source, subset, decision = REPOS[kind], str(row.get("seed", "unknown")), "Apache-2.0"
                if reason is None:
                    payload, reason = clean_text(raw)
                    if payload and (len(payload.split()) < 80 or not language(payload)):
                        reason = "language_or_short_document"
                if reason is None:
                    content = payload
                    split_key = core.content_hash_of(content)
                    phase = "base"
                    estimated = core.estimate_tokens_from_text(payload)
                    if kind == "base" and family_counts[subset]+estimated > targets["base"]*.60:
                        reason = "education_source_cap"
            if reason is None:
                split = core.deterministic_split(split_key, eval_basis_points=100)
                limit = targets[phase] if split == "train" else max(2000, targets[phase]//100)
                if counts[(phase, split)] >= limit:
                    reason = "bucket_full"
            if reason is None:
                digest = core.content_hash_of(content)
                provenance = {
                    "document_id": core.stable_document_id(REPOS[kind], config, revisions[kind],
                                                           str(row.get("id") or digest)),
                    "source_revision": revisions[kind], "source_url_hash": core.stable_hash_text(source),
                    "license_class": decision, "quality_score": quality,
                    "language_score": None, "filter_version": VERSION, "content_hash": digest,
                    "original_char_count": len(str(row.get("text", row.get("messages", "")))),
                    "clean_char_count": len(payload), "clean_utf8_byte_count": len(payload.encode()),
                    "raw_pii_flags": core.pii_flags(str(row.get("text", ""))),
                    "residual_pii_flags": core.pii_flags(payload),
                }
                # A synthetic document has no EDU classifier score. Do not invent one.
                if store.insert(fingerprint=digest, phase=phase, split=split, category=category,
                                source=source, subset=subset, payload=payload, estimated_tokens=estimated,
                                quality=float(quality or 0), license_decision=decision, provenance=provenance):
                    store.connection.execute(
                        "INSERT INTO curated_source_records VALUES (?,?)",
                        (digest, core.stable_json({"repo":REPOS[kind],"revision":revisions[kind],
                         "config":config,"source":row.get("source"),"original_subset":row.get("subset"),
                         "record_id":row.get("id"),"generator":row.get("generator"),"seed":row.get("seed"),
                         "raw_sha256":core.stable_hash_text(core.stable_json(row))})))
                    counts[(phase, split)] += estimated
                    if split == "train":
                        family_counts[subset] += estimated
                    if len(samples[category]) < 8:
                        samples[category].append({"id": digest, "source": source, "split": split,
                                                  "payload": payload, "quality": quality})
                else:
                    reason = "exact_duplicate"
            if reason:
                rejected[reason] += 1
            if seen % 1000 == 0 or enough():
                store.set_meta(key, core.stable_json({"seen": seen, "rejected": dict(rejected)}))
                if samples[category]:
                    core.atomic_write_json(store.path.parent / f"samples_{category}.json", samples[category])
                print(f"[curated:{category}] seen={seen:,} tokens={dict(counts)} rejected={rejected.most_common(4)}", flush=True)
            if enough():
                break
            if seen >= MAX_ROWS_PER_STREAM:
                raise RuntimeError(f"Insufficient eligible {category} at bounded scan limit; refusing relaxed filters")
        if not enough():
            raise RuntimeError(f"Insufficient eligible {category}: {dict(counts)}")


def document_content(phase, payload):
    if phase == "base":
        return payload
    return "\n".join(m["role"]+":"+m["content"] for m in json.loads(payload)
                     if m["role"] != "system")


def deduplicate_and_audit(store, *, exclude_verified=False):
    """Cross-phase exact + bounded lexical near-dedup, preferring holdout anchors."""
    rows = store.connection.execute(
        "SELECT fingerprint,phase,split,category,payload,content_hash FROM records "
        + ("WHERE category!='verified' " if exclude_verified else "") +
        "ORDER BY CASE WHEN split='eval' THEN 0 ELSE 1 END,fingerprint").fetchall()
    anchors, inverted, exact = [], defaultdict(list), {}
    updates, rejected = [], Counter()
    for index, (fingerprint, phase, split, category, payload, digest) in enumerate(rows):
        text = document_content(phase, payload)
        if FORBIDDEN.search(text) or SPECIAL.search(text) or core.pii_flags(text):
            raise RuntimeError(f"Audit contamination in {fingerprint}")
        # A chat role/prompt prefix shifts word offsets. Sampling every fourth
        # window misses verbatim passages after such a prefix; use every window.
        shingles = set(core.dedup_shingles(text, stride=1))
        signature = sorted(shingles)[:32]
        candidates = set(c for sh in signature for c in inverted[sh][:64])
        match = exact.get(digest)
        if match is None:
            for candidate in sorted(candidates):
                other = anchors[candidate][1]
                if len(shingles & other) >= 4 and len(shingles & other)/max(1, min(len(shingles),len(other))) >= .80:
                    match = candidate
                    break
        if match is None:
            match = len(anchors)
            anchors.append((fingerprint, shingles))
            for sh in signature:
                inverted[sh].append(match)
            canonical = 1
        else:
            canonical = 0
            rejected[category] += 1
        exact[digest] = match
        updates.append((anchors[match][0], canonical, canonical if split == "train" else 0,
                        None if canonical else "cross_phase_duplicate", fingerprint))
        if index % 10000 == 0:
            print(f"[curated:dedup] {index:,}/{len(rows):,} duplicates={sum(rejected.values()):,}", flush=True)
    store.connection.executemany(
        "UPDATE records SET dedup_cluster=?,is_canonical=?,is_training_selected=?,rejection_reason=? WHERE fingerprint=?",
        updates)
    store.commit()
    selected = [dict(zip(("phase","split","category","documents","estimated_tokens"), r)) for r in
                store.connection.execute("SELECT phase,split,category,COUNT(*),SUM(estimated_tokens) FROM records "
                                         "WHERE is_canonical=1 GROUP BY phase,split,category")]
    return {"version": VERSION, "documents": len(rows), "duplicates_removed": dict(rejected),
            "selected": selected, "boilerplate_residual_documents": 0,
            "dedup": "cross-phase, holdout-first, 8-word shingles stride 1, bottom-32 candidates, containment>=0.80",
            "ingest_counter_scope": "EDU counts after Arrow score>=4/toxic<=1; other streams count upstream rows",
            "limitations": ["No factual correctness certification", "Bounded lexical dedup is not semantic decontamination",
                            "Language ID distinguishes Portuguese, not reliably PT-BR from PT-PT"]}


def pack_causal(store, tokenizer, phase, split, out_dir, token_target, shard_size, eos_token_id,
                *, groups=None):
    writer = core.TokenShardWriter(out_dir, f"{phase}-{split}", shard_size)
    if groups is None:
        groups = {"edu": EDU_SHARE, "synth": 1-EDU_SHARE} if phase == "base" else SFT_MIX
    iterators = {category: iter(store.connection.execute(
        "SELECT payload,subset_name FROM records WHERE phase=? AND split=? AND category=? AND is_canonical=1 "
        "AND (split='eval' OR is_training_selected=1) ORDER BY fingerprint", (phase,split,category)))
                 for category in groups}
    totals, families, exhausted = Counter(), Counter(), set()
    # Apply the requested mix to ACTUAL tokens, not upstream token_count or character estimates.
    while len(exhausted) < len(groups):
        category = min((c for c in groups if c not in exhausted), key=lambda c: totals[c]/groups[c])
        target = max(1, int(token_target*groups[category]))
        if totals[category] >= target:
            exhausted.add(category)
            continue
        row = next(iterators[category], None)
        if row is None:
            exhausted.add(category)
            continue
        payload, family = row
        text = payload if phase == "base" else core.format_chat(json.loads(payload))
        tokens = list(tokenizer.encode(text))
        if not tokens or tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)
        if len(tokens) > target-totals[category]:
            continue  # Never manufacture an incomplete document to fill the final budget.
        if category == "edu" and families[family]+len(tokens) > target*.60:
            continue
        writer.add(tokens)
        totals[category] += len(tokens)
        if category == "edu":
            families[family] += len(tokens)
    shards = writer.finish()
    for category, share in groups.items():
        if totals[category] < .95*token_target*share:
            raise RuntimeError(f"Insufficient packed {phase}/{split}/{category}: {dict(totals)}")
    core.atomic_write_json(out_dir / "mixture_audit.json", {
        "tokens_by_category": dict(totals), "education_tokens_by_source_family": dict(families),
        "requested_shares": groups, "complete_documents_only": True})
    return {"phase": phase, "split": split, "dtype": "uint16-le",
            "target_tokens": token_target if split == "train" else None,
            "tokens": sum(int(s["tokens"]) for s in shards), "shards": shards}


def pack_sft(store, tokenizer, split, out_dir, target_tokens, records_per_shard, eos_token_id, seq_len):
    writer = core.SFTShardWriter(out_dir, records_per_shard)
    used, skipped = Counter(), Counter()
    # Deterministic global order interleaves categories; each has its own actual-token budget.
    for _, category, payload in store.iter_records("sft", split):
        target = int(target_tokens*SFT_MIX[category])
        for prompt_text, answer_text in core.supervised_examples(json.loads(payload)):
            prompt = list(tokenizer.encode(prompt_text))
            answer = list(tokenizer.encode(answer_text)) + [eos_token_id]
            if len(prompt)+len(answer) > seq_len+1:
                skipped["context_does_not_fit"] += 1
                continue
            if used[category]+len(answer) > target:
                skipped["category_budget"] += 1
                continue
            writer.add(prompt, answer)
            used[category] += len(answer)
    shards = writer.finish()
    if any(used[c] < .95*target_tokens*w for c,w in SFT_MIX.items()):
        raise RuntimeError(f"Insufficient COMPLETE SFT/{split}: {dict(used)}; skipped={dict(skipped)}")
    core.atomic_write_json(out_dir / "mixture_audit.json", {
        "assistant_tokens_by_category": dict(used), "skipped": dict(skipped),
        "truncated_answers": 0, "truncated_prompts": 0})
    return {"phase": "sft", "split": split, "format": "NSOSSFT1",
            "target_tokens": target_tokens if split == "train" else None,
            "target_tokens_packed": sum(s["target_tokens"] for s in shards),
            "records": sum(s["records"] for s in shards), "shards": shards}


def prepare(args, preset, workspace, nsos):
    from huggingface_hub import HfApi
    workspace.mkdir(parents=True, exist_ok=True)
    store = core.CorpusStore(workspace / "corpus" / "corpus.sqlite3")
    identity = core.stable_json(core.corpus_recipe_identity(PRESET_NAME,preset,args.seed,False,args.license_policy))
    try:
        previous = store.get_meta("run_identity")
        if previous not in (None, identity):
            raise RuntimeError("Workspace recipe changed; choose a NEW workspace")
        store.set_meta("run_identity", identity)
        if (workspace / "packs" / "pack_manifest.json").exists():
            return core.validate_prepared_workspace(store, workspace, PRESET_NAME,preset,args.seed,False,args.license_policy)
        revisions_path = workspace / "corpus" / "dataset_revisions.json"
        if revisions_path.exists():
            revisions = core._read_json_object(revisions_path, "Revisions")
        else:
            revisions = {name: HfApi().dataset_info(repo).sha for name,repo in REPOS.items()}
            core.atomic_write_json(revisions_path, revisions)
        if set(revisions) != set(REPOS):
            raise RuntimeError("Incomplete pinned revisions")
        language = PortugueseFilter(workspace / "dependencies")
        ingest(store,preset,revisions,args.seed,language)
        audit = deduplicate_and_audit(store)
        audit["language_model_sha256"] = language.model_sha256
        audit["dataset_revisions"] = revisions
        core.atomic_write_json(workspace / "corpus" / "quality_audit.json", audit)
        manifest = {
            "format_version": core.CORPUS_MANIFEST_FORMAT_VERSION, "created_at": core.utc_now(),
            "preset": PRESET_NAME, "preset_config": preset, "seed": args.seed, "fixture": False,
            "license_policy": args.license_policy, "dataset_revisions": revisions,
            "sources": {k: {"repo":v,"revision":revisions[k]} for k,v in REPOS.items()},
            "source_rules": list(core.BASE_SOURCE_RULES),
            "phases": {p: {"train":store.counts(p,"train").__dict__,"eval":store.counts(p,"eval").__dict__,
                            "train_categories":{k:v.__dict__ for k,v in store.category_counts(p,"train").items()}}
                       for p in ("base","continuation","sft")},
            "legal_notice": "Source licenses are recorded; aggregate metadata is not a legal clearance of original content.",
        }
        corpus_path = workspace / "corpus" / "corpus_manifest.json"
        core.atomic_write_json(corpus_path,manifest)
        ox3 = core.train_tokenizer(store,workspace/"tokenizer",preset["target_vocab"],preset["tokenizer_chars"])
        packed = core.build_packs(store,nsos,ox3,workspace,PRESET_NAME,preset,corpus_path,
            causal_packer=pack_causal,sft_packer=pack_sft,packing_version=VERSION,
            dataset_metadata={"filter_version":VERSION,"dedup_version":VERSION,"quota_policy_version":VERSION,
                              "source_mix_policy":{"edu":EDU_SHARE,"synth":1-EDU_SHARE},
                              "conditional_quality_max_share":0,
                              "quality_audit_sha256":core.sha256_file(workspace/"corpus"/"quality_audit.json")})
        core.validate_prepared_workspace(store,workspace,PRESET_NAME,preset,args.seed,False,args.license_policy)
        return packed
    finally:
        store.close()


def main():
    core.PRESETS[PRESET_NAME] = make_preset()
    if "--preset" not in sys.argv:
        sys.argv += ["--preset",PRESET_NAME]
    args = core.parse_args()
    if args.preset != PRESET_NAME or args.fixture or args.license_policy != "commercial-strict":
        raise RuntimeError("Curated recipe requires its own preset, real data and commercial-strict source policy")
    if args.dataset_revisions is not None:
        raise RuntimeError("Pin revisions in this workspace's corpus/dataset_revisions.json")
    preset = core.PRESETS[PRESET_NAME]
    workspace = (args.workspace or core.SCRIPT_DIR.parent/"artifacts"/"ptbr_edu_synth_20260920_v2").resolve()
    run_dir = (args.run_dir or workspace/"runs"/"main").resolve()
    args.checkpoint_every_steps = args.checkpoint_every_steps or preset["checkpoint_every_steps"]
    args.checkpoint_every_minutes = args.checkpoint_every_minutes or preset["checkpoint_every_minutes"]
    os.environ.setdefault("NSOS_DETERMINISTIC","1")
    os.environ.setdefault("NSOS_TRAIN_CHUNK_SIZE","1")
    if args.action in ("status","verify"):
        core.print_status(workspace,run_dir,PRESET_NAME,preset,args.seed,False,args.license_policy)
        return 0
    # A failed manual sample review must not be bypassed by rerunning the launcher.
    # This gate deliberately precedes GPU loading, downloads and corpus writes.
    require_source_quality_review()
    args.resolved_build_dir = core.detect_build_dir(args.build_dir)
    nsos = core.load_nsos(args.resolved_build_dir)
    workspace.mkdir(parents=True,exist_ok=True)
    state = {"pid":os.getpid(),"started_at":core.utc_now(),"action":args.action,
             "workspace":str(workspace),"run_dir":str(run_dir),"preset":preset,
             "python":sys.executable,"build_dir":str(args.resolved_build_dir)}
    def publish(stage, **values):
        state.update(stage=stage,updated_at=core.utc_now(),**values)
        core.atomic_write_json(workspace/"job_status.json",state)
    try:
        if args.action in ("prepare","run"):
            publish("preparing")
            print(f"[curated] prepared: {prepare(args,preset,workspace,nsos)}",flush=True)
        if args.action in ("train","run"):
            audit_path = workspace/"corpus"/"quality_audit.json"
            pack = core._read_json_object(workspace/"packs"/"pack_manifest.json","Pack")
            if pack["dataset"].get("quality_audit_sha256") != core.sha256_file(audit_path):
                raise RuntimeError("Missing or changed quality audit; training blocked")
            publish("training",packing_content_sha256=pack["packing"]["content_sha256"])
            result = core.train(args,PRESET_NAME,preset,workspace,run_dir,nsos)
            publish("stopped_at_step_limit" if args.max_train_steps else "completed",exit_code=result)
            return result
        publish("prepared",exit_code=0)
        return 0
    except BaseException as exc:
        publish("failed",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[curated:FAILED] {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
