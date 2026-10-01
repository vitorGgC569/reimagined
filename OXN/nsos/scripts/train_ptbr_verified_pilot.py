#!/usr/bin/env python3
"""Native HIP pilot: educational text + 5% exactly verified tasks + complete SFT.

The rejected GigaVerbo Synth source is never read. No previous model or token
pack is reused. Cached educational candidates may be re-filtered with lineage.
"""
from collections import Counter
from types import SimpleNamespace
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys

import train_ptbr_conversational as core
import train_ptbr_edu_synth as curated
import ptbr_verified_tasks as tasks

VERSION = "ptbr-verified-pilot-v2.2"
PRETRAIN_SFT_ADMISSION_FROM = "6a8d2e71bf465b26ff0c94b5d3711d138ec0efe2aad77bc8467375274a3316e2"
PRETRAIN_SFT_HEADROOM_FROM = "946882f750e9d4f254e75271813b211acbd6d3fdac60373311165b01250a151b"
PRETRAIN_PACK_REPAIR_FROM = "30b78b1b16ecb6f833cf67dd076b6671901e7370f98c76d37a4f73cc3b641ce6"
PRETRAIN_PACK_INTERLEAVE_FROM = "8ae92278782ee8c11ca7d2aafa416e832c1f49191035094f298602d06c71212a"
PRESET_NAME = "verified-pilot"
BASE_MIX = {"edu": .95, "verified": .05}
REPOS = {"base": curated.REPOS["base"], "sft": curated.REPOS["sft"],
         "verified": "local:ptbr_verified_tasks"}
REVISIONS = {"base": "b39dfa703102a20dc609ed6e7aaae22e8e3a233f",
             "sft": "845713c9330519809c104fa904481c57ae948789"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "artifacts/ptbr_verified_pilot_20260928"
RECOVERY_WORKSPACE = ROOT / "artifacts/ptbr_verified_pilot_20260920"
RECOVERY_RECIPE_SHA256 = "d213b44a78393d12de84e8bc9625f9d947bc63f2837c68074983b12ee67fc9a1"
EDU_HOLDOUT_FLOOR = 60_000
CANDIDATE_CACHE = ROOT / "artifacts/ptbr_edu_synth_20260920_v2/corpus/corpus.sqlite3"
DEPENDENCIES = ROOT / "artifacts/ptbr_edu_synth_20260920_v2/dependencies"
EXTRA_BOILERPLATE = re.compile(
    r"\b(?:bem[ -]vind[oa]s? ao (?:nosso )?blog|coment.rios abaixo|"
    r"inscreva-se|compartilh[ea] este (?:artigo|post)|liga..es externas)\b", re.I)


def code_identity():
    return {Path(m.__file__).name: core.sha256_file(Path(m.__file__))
            for m in (core, curated, tasks)} | {Path(__file__).name: core.sha256_file(Path(__file__))}


def make_preset():
    return {**core.PRESETS["dryrun"], "description": "~71M native Mamba; small verified PT pilot",
            "base_tokens": 10_000_000, "continuation_tokens": 1_000_000,
            "sft_unique_tokens": 500_000, "sft_epochs": 2,
            "tokenizer_chars": 20_000_000, "warmup_steps": 500,
            "checkpoint_every_steps": 500, "checkpoint_every_minutes": 10.,
            "dataset_repos": REPOS,
            "verified_recipe": {"version": VERSION, "base_mix": BASE_MIX,
                "sft_mix": curated.SFT_MIX, "source_family_cap": .60,
                "edu_score_min": 4, "toxic_score_max": 1, "sft_score_min": 4,
                "revisions": REVISIONS, "code_sha256": code_identity(),
                "language": "langid==1.1.6; all full-text 2000-character windows; p(pt)>=.90",
                "synthetic_verification": tasks.VERSION, "synthetic_eval_percent": 5,
                "edu_holdout_policy": "preserve-existing-plus-source-stratified-hash-reserve-v1",
                "edu_holdout_floor_estimated_tokens_per_family": EDU_HOLDOUT_FLOOR,
                "remote_max_buffer_input_shards": 1,
                "complete_eval_selection": "bounded-subset-sum-v1",
                "sft_admission": "actual-complete-answer-tokenizer-context-v2",
                "headroom": 2.0, "complete_answers_only": True}}


class FullPortugueseFilter(curated.PortugueseFilter):
    def __call__(self, text):
        if not text.strip():
            return False
        # Balanced windows cover the tail, without a tiny final window of only numbers.
        windows = max(1, math.ceil(len(text)/2000))
        for i in range(windows):
            chunk = text[len(text)*i//windows:len(text)*(i+1)//windows]
            language, confidence = self.classifier.classify(chunk)
            if language != "pt" or float(confidence) < .90:
                return False
        return True


def clean(raw, *, conversational=False):
    if isinstance(raw, str) and EXTRA_BOILERPLATE.search(raw):
        return None, "editorial_boilerplate"
    return curated.clean_text(raw, conversational=conversational)


def initialize_tables(store):
    store.connection.execute("CREATE TABLE IF NOT EXISTS curated_source_records "
                             "(fingerprint TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)")
    store.connection.execute("CREATE TABLE IF NOT EXISTS verified_proofs "
                             "(fingerprint TEXT PRIMARY KEY, problem_id TEXT UNIQUE NOT NULL, "
                             "record_json TEXT NOT NULL, proof_json TEXT NOT NULL)")
    store.commit()


def education_family(source, subset):
    for marker in ("hplt", "quati", "crawlpt", "oscar", "legalpt", "blogset"):
        if marker in (source+" "+subset).lower():
            return marker
    return core.source_family_of(source, subset) or source


def recover_candidate_snapshot(store, source_workspace=RECOVERY_WORKSPACE):
    """Reuse a known completed ingest transaction, with no tokenizer/weight reuse."""
    if store.get_meta("candidate_snapshot"):
        return True
    source_path = source_workspace/"corpus/corpus.sqlite3"
    audit_path = source_workspace/"corpus/quality_audit.json"
    if not source_path.is_file() or not audit_path.is_file():
        return False
    if store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]:
        return False  # Never merge two separately advancing candidate stores.
    source_db = sqlite3.connect(source_path.resolve().as_uri()+"?mode=ro", uri=True)
    try:
        source_db.execute("BEGIN")  # Stable source snapshot throughout validation/copy.
        identity = json.loads(source_db.execute("SELECT value FROM metadata WHERE key='run_identity'").fetchone()[0])
        old = identity["preset_config"]["verified_recipe"]
        expected = make_preset()["verified_recipe"]
        if (old["code_sha256"][Path(__file__).name] != RECOVERY_RECIPE_SHA256 or
                identity["license_policy"] != "commercial-strict" or identity["fixture"] or
                any(old[key] != expected[key] for key in (
                    "base_mix", "sft_mix", "source_family_cap", "edu_score_min", "toxic_score_max",
                    "sft_score_min", "revisions", "language", "synthetic_verification", "complete_answers_only"))):
            raise RuntimeError("Candidate snapshot does not match the approved filtering recipe")
        prior_audit = core._read_json_object(audit_path,"Source candidate audit")
        if audit_verified(SimpleNamespace(connection=source_db)) != prior_audit["synthetic"]:
            raise RuntimeError("Source synthetic proofs changed")
        verified_revision = core.sha256_file(Path(tasks.__file__))
        for phase, category, source, revision, payload, fingerprint in source_db.execute(
                "SELECT phase,category,source,source_revision,payload,fingerprint FROM records"):
            expected_revision = (REVISIONS["base"] if category == "edu" else
                                 verified_revision if category == "verified" else REVISIONS["sft"])
            if (category not in {"edu","verified",*curated.SFT_MIX} or revision != expected_revision or
                    source == curated.REPOS["synth"] or
                    core.content_hash_of(curated.document_content(phase,payload)) != fingerprint):
                raise RuntimeError("Candidate content/provenance mismatch")
        with store.connection:
            for table in ("records","curated_source_records","verified_proofs"):
                columns = [r[1] for r in source_db.execute(f"PRAGMA table_info({table})")]
                target_columns = [r[1] for r in store.connection.execute(f"PRAGMA table_info({table})")]
                if columns != target_columns:
                    raise RuntimeError(f"Candidate schema mismatch: {table}")
                store.connection.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})",
                                              source_db.execute(f"SELECT * FROM {table}"))
            for key,value in source_db.execute("SELECT key,value FROM metadata"):
                if key.startswith("remote_cursor:") or key in ("verified_cursor","candidate_import_complete"):
                    store.connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",(key,value))
            counts = store.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            snapshot = {"workspace":str(source_workspace),"records":counts,
                "recipe_sha256":RECOVERY_RECIPE_SHA256,"audit_sha256":core.sha256_file(audit_path),
                "database_sha256":core.sha256_file(source_path),
                "language_model_sha256":prior_audit["language_model_sha256"]}
            store.connection.execute("INSERT INTO metadata VALUES (?,?)",("candidate_snapshot",core.stable_json(snapshot)))
        print(f"[verified:recovery] reused {counts:,} audited candidates; remote ingest skipped",flush=True)
        return True
    finally:
        source_db.close()


def reserve_education_holdout(store, floor=EDU_HOLDOUT_FLOOR):
    """Keep every old holdout; reserve missing families before tokenizer training.

    Selection uses a content-hash ranking, never the answer or a model score.
    Exact/near duplicates are resolved holdout-first by the following audit.
    """
    for fp,source,subset in store.connection.execute(
            "SELECT fingerprint,source,subset_name FROM records WHERE category='edu'").fetchall():
        family = education_family(source,subset)
        if family != subset:
            store.connection.execute("UPDATE records SET subset_name=? WHERE fingerprint=?",(family,fp))
    totals = Counter({f:n for f,n in store.connection.execute(
        "SELECT subset_name,SUM(estimated_tokens) FROM records WHERE category='edu' AND split='eval' GROUP BY subset_name")})
    available = dict(store.connection.execute("SELECT subset_name,SUM(estimated_tokens) FROM records "
                                             "WHERE category='edu' GROUP BY subset_name"))
    if len(available) < 2 or any(n < floor for n in available.values()):
        raise RuntimeError(f"Insufficient stratified educational holdout candidates: {available}")
    promoted = Counter()
    rows = store.connection.execute("SELECT fingerprint,subset_name,estimated_tokens FROM records "
                                    "WHERE category='edu' AND split='train'").fetchall()
    rows.sort(key=lambda r:core.stable_hash_text("edu-holdout-reserve-v1:"+r[0]))
    for fp,family,tokens in rows:
        if totals[family] >= floor:
            continue
        store.connection.execute("UPDATE records SET split='eval',is_training_selected=0 WHERE fingerprint=?",(fp,))
        totals[family] += tokens
        promoted[family] += 1
    store.commit()
    if len(totals) < 2 or any(n < floor for n in totals.values()):
        raise RuntimeError(f"Insufficient stratified educational holdout: {dict(totals)}")
    record = {"policy":"preserve-existing-plus-source-stratified-hash-reserve-v1",
              "estimated_tokens_by_family":dict(totals),"moved_to_holdout_this_pass":dict(promoted)}
    store.set_meta("education_holdout_reserve",core.stable_json(record))
    print(f"[verified:holdout] {record}",flush=True)
    return record


def insert_record(store, *, phase, split, category, payload, source, subset,
                  revision, metadata, quality=None, decision="allow"):
    content = curated.document_content(phase, payload)
    fingerprint = core.content_hash_of(content)
    estimate = (core.estimate_sft_target_tokens(json.loads(payload), 512) if phase == "sft"
                else core.estimate_tokens_from_text(payload))
    provenance = {"document_id": core.stable_document_id(source, subset, revision,
                       str(metadata.get("record_id") or fingerprint)),
        "source_revision": revision, "source_url_hash": core.stable_hash_text(source),
        "license_class": decision, "quality_score": quality, "filter_version": VERSION,
        "content_hash": fingerprint, "clean_char_count": len(payload),
        "clean_utf8_byte_count": len(payload.encode()), "residual_pii_flags": core.pii_flags(payload)}
    inserted = store.insert(fingerprint=fingerprint, phase=phase, split=split, category=category,
        source=source, subset=subset, payload=payload, estimated_tokens=estimate,
        quality=float(quality or 0), license_decision=decision, provenance=provenance)
    if inserted:
        store.connection.execute("INSERT INTO curated_source_records VALUES (?,?)",
                                 (fingerprint, core.stable_json(metadata)))
    return fingerprint, estimate if inserted else 0


def import_education_candidates(store, language):
    """Read-only import of already scored candidates, never shards or model weights.

    Toxicity is an attestation from the original pinned ingest, not a newly run
    classifier; preserve that distinction in the row metadata and audit.
    """
    if store.get_meta("candidate_import_complete") or not CANDIDATE_CACHE.exists():
        return
    source_db = sqlite3.connect(CANDIDATE_CACHE.resolve().as_uri()+"?mode=ro", uri=True)
    rejected = Counter()
    imported = 0
    try:
        identity = json.loads(source_db.execute("SELECT value FROM metadata WHERE key='run_identity'").fetchone()[0])
        old = identity["preset_config"]["curated_recipe"]
        if old["edu_min_score"] != 4 or old["toxic_max_score"] != 1 or identity["license_policy"] != "commercial-strict":
            raise RuntimeError("Untrusted candidate-cache recipe")
        rows = source_db.execute("SELECT r.fingerprint,r.payload,r.source,r.subset_name,r.source_revision,"
            "r.quality,m.metadata_json FROM records r JOIN curated_source_records m USING(fingerprint) "
            "WHERE r.category='edu' AND r.phase='base' ORDER BY r.fingerprint")
        for index, (old_fingerprint, raw, source, subset, revision, quality, lineage) in enumerate(rows, 1):
            if revision != REVISIONS["base"] or not math.isfinite(quality) or quality < 4:
                rejected["revision_or_score"] += 1
                continue
            allowed, decision, _ = core.source_allowed(source, subset, "commercial-strict")
            text, reason = clean(raw)
            if not allowed or reason or not language(text):
                rejected[reason or "language_or_license"] += 1
                continue
            metadata = json.loads(lineage)
            metadata.update(candidate_cache=str(CANDIDATE_CACHE), cache_record=old_fingerprint,
                            original_recipe_sha256=old["recipe_code_sha256"],
                            toxicity_evidence="original pinned ingest required toxic_int_score<=1; raw score not retained")
            split = core.deterministic_split(core.content_hash_of(text), eval_basis_points=100)
            _, added = insert_record(store, phase="base", split=split, category="edu", payload=text,
                source=source, subset=subset, revision=revision, metadata=metadata,
                quality=quality, decision=decision)
            imported += bool(added)
            if index % 1000 == 0:
                store.commit()
                print(f"[verified:cache] scanned={index} imported={imported}", flush=True)
        store.set_meta("candidate_import_complete", core.stable_json({"imported": imported,
            "rejected": dict(rejected), "source": str(CANDIDATE_CACHE)}))
    finally:
        source_db.close()


def complete_sft_target_tokens(payload, tokenizer, eos_token_id, seq_len):
    """The same complete-answer/context contract as the final SFT packer."""
    total = 0
    for prompt_text, answer_text in core.supervised_examples(json.loads(payload)):
        prompt = list(tokenizer.encode(prompt_text))
        answer = list(tokenizer.encode(answer_text)) + [eos_token_id]
        if len(prompt) + len(answer) <= seq_len + 1:
            total += len(answer)
    return total


def sft_inventory(store, tokenizer, eos_token_id, seq_len, *, canonical_only=True):
    counts = Counter()
    predicate = "is_canonical=1" if canonical_only else "(is_canonical=1 OR is_canonical IS NULL)"
    for category, split, payload in store.connection.execute(
            "SELECT category,split,payload FROM records WHERE phase='sft' AND " + predicate):
        counts[(category, split)] += complete_sft_target_tokens(payload, tokenizer, eos_token_id, seq_len)
    return counts


def ingest_remote(store, preset, seed, language, *, tokenizer=None, eos_token_id=None, only_sft=False):
    from datasets import load_dataset
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    streams = [("base", "edu", "default", .95)] + [("sft", c, c, w) for c,w in curated.SFT_MIX.items()]
    actual_sft = (sft_inventory(store, tokenizer, eos_token_id, preset["seq_len"], canonical_only=False)
                  if tokenizer is not None else Counter())
    for stream_index, (kind, category, config, share) in enumerate(streams):
        if only_sft and kind != "sft":
            continue
        targets = ({"base": int(preset["base_tokens"]*share*2)} if kind == "base" else
                   {"continuation": int(preset["continuation_tokens"]*share*2),
                    "sft": int(preset["sft_unique_tokens"]*share*2)})
        counts = Counter({(p,s): n for p,s,n in store.connection.execute(
            "SELECT phase,split,SUM(estimated_tokens) FROM records WHERE category=? GROUP BY phase,split", (category,))})
        if tokenizer is not None and kind == "sft":
            for split in ("train", "eval"):
                counts[("sft", split)] = actual_sft[(category, split)]
        families = Counter({f:n for f,n in store.connection.execute(
            "SELECT subset_name,SUM(estimated_tokens) FROM records WHERE category=? AND split='train' GROUP BY subset_name", (category,))})
        eval_targets = {p: max(6000, int((50000 if p != "sft" else 5000)*share*2)) for p in targets}
        if tokenizer is not None and kind == "sft":
            # Actual answer tokens already incorporate EOS and context rejection.
            # Reserve 2x the unchanged final evaluation quota, with a 2k floor.
            # A 6k estimated-token floor would demand 12x the structured quota
            # after the exact tokenization and exhaust the pinned source.
            eval_targets["sft"] = max(2000, int(5000*share*2))
        def enough():
            return all(counts[(p,"train")] >= t and counts[(p,"eval")] >= eval_targets[p] for p,t in targets.items())
        if enough():
            continue
        cursor = f"remote_cursor:{category}"
        state = json.loads(store.get_meta(cursor) or "{}")
        seen = state.get("seen", 0)
        rejected = Counter(state.get("rejected", {}))
        if kind == "base":
            # Only permitted families, so an anonymous stream does not spend most
            # of its bandwidth delivering FineWeb/mC4 candidates we must reject.
            names = ds.field("source")
            permitted = pc.match_substring(names, "hplt", ignore_case=True)
            for marker in ("quati", "oscar", "crawlpt", "legalpt", "blogset"):
                permitted = permitted | pc.match_substring(names, marker, ignore_case=True)
            filters = (ds.field("edu_int_score") >= 4) & (ds.field("toxic_int_score") <= 1) & permitted
        else:
            filters = [("instruct_score", ">=", 4.)]
        print(f"[verified:ingest] {category} targets={targets} eval={eval_targets} cursor={seen}", flush=True)
        dataset = load_dataset(REPOS[kind], config, split="train", streaming=True,
            revision=REVISIONS[kind], filters=filters).shuffle(
                seed=seed+stream_index, buffer_size=1000, max_buffer_input_shards=1)
        if seen:
            dataset = dataset.skip(seen)
        for row in dataset:
            seen += 1
            reason = None
            if kind == "base":
                source, original_subset = str(row.get("source", "")), str(row.get("subset", ""))
                subset = education_family(source, original_subset)
                allowed, decision, _ = core.source_allowed(source, original_subset, "commercial-strict")
                quality, toxic = row.get("edu_int_score", 0), row.get("toxic_int_score", 5)
                if (not allowed or not isinstance(quality, (int,float)) or not math.isfinite(quality) or quality < 4 or
                        not isinstance(toxic, (int,float)) or not math.isfinite(toxic) or toxic > 1):
                    reason = "source_or_score"
                # A saturated source previously ran cleanup/language ID for
                # 247k rejected rows. Preserve holdout opportunities, but reject
                # fully saturated sources before the expensive text filters.
                if families[subset] >= targets["base"]*.60 and counts[("base","eval")] >= eval_targets["base"]:
                    reason = reason or "source_cap"
                if not reason:
                    payload, reason = clean(row.get("text", ""))
                phase = "base"
                if not reason:
                    split = core.deterministic_split(core.content_hash_of(payload), eval_basis_points=100)
                    if split == "train" and families[subset] >= targets[phase]*.60:
                        reason = "source_cap"
                if not reason and (len(payload.split()) < 80 or not language(payload)):
                    reason = "language_or_short_document"
            else:
                if any(EXTRA_BOILERPLATE.search(str(m.get("content", ""))) for m in row.get("messages", []) if isinstance(m,dict)):
                    reason = "editorial_boilerplate"
                messages, failure = curated.curated_messages(row, language)
                reason = reason or failure
                if not reason:
                    payload = core.stable_json(messages)
                    content = curated.document_content("sft", payload)
                    prompt_key = core.content_hash_of("\n".join(m["content"] for m in messages if m["role"] == "user"))
                    phase = core._sft_phase_bucket(prompt_key)
                    split = core.deterministic_split(prompt_key, eval_basis_points=100)
                    source, subset, decision = REPOS[kind], config, "Apache-2.0"
                    quality = row["instruct_score"]
            if not reason:
                if counts[(phase,split)] >= (targets[phase] if split == "train" else eval_targets[phase]):
                    reason = "bucket_full"
            admissible_tokens = None
            if not reason and tokenizer is not None and phase == "sft":
                admissible_tokens = complete_sft_target_tokens(payload, tokenizer, eos_token_id, preset["seq_len"])
                if admissible_tokens == 0:
                    reason = "complete_answer_context_does_not_fit"
            if not reason:
                _, added = insert_record(store, phase=phase, split=split, category=category,
                    payload=payload, source=source, subset=subset, revision=REVISIONS[kind],
                    quality=quality, decision=decision, metadata={"repo": REPOS[kind],
                        "revision": REVISIONS[kind], "config": config, "record_id": row.get("id"),
                        "source": row.get("source"), "original_subset": row.get("subset"),
                        "edu_int_score": row.get("edu_int_score"), "toxic_int_score": row.get("toxic_int_score"),
                        "instruct_score": row.get("instruct_score"),
                        "raw_sha256": core.stable_hash_text(core.stable_json(row))})
                counts[(phase,split)] += (admissible_tokens if added and admissible_tokens is not None else added)
                if split == "train":
                    families[subset] += added
                if not added:
                    reason = "exact_duplicate"
            if reason:
                rejected[reason] += 1
            if seen % 1000 == 0 or enough():
                store.set_meta(cursor, core.stable_json({"seen":seen,"rejected":dict(rejected)}))
                print(f"[verified:{category}] seen={seen:,} tokens={dict(counts)} rejected={rejected.most_common(4)}", flush=True)
            if enough():
                break
            if seen >= 1_000_000:
                raise RuntimeError(f"Insufficient {category}; bounded scan reached, filters unchanged")
        if not enough():
            raise RuntimeError(f"Insufficient eligible {category}")


def ingest_verified(store, preset, seed):
    revision = core.sha256_file(Path(tasks.__file__))
    counts = Counter({(s,f): n for s,f,n in store.connection.execute(
        "SELECT r.split,r.subset_name,SUM(r.estimated_tokens) FROM records r "
        "WHERE r.category='verified' GROUP BY r.split,r.subset_name")})
    targets = {"train": int(preset["base_tokens"]*.05*2/len(tasks.FAMILIES)), "eval": 3000}
    def enough():
        return all(counts[(s,f)] >= n for s,n in targets.items() for f in tasks.FAMILIES)
    start = int(store.get_meta("verified_cursor") or 0)
    for index in range(start, 1_000_000):
        if enough():
            break
        record = tasks.generate(seed, index)
        proof = tasks.verify(**record)
        split, family = proof["split"], proof["family"]
        if counts[(split,family)] >= targets[split]:
            continue
        if store.connection.execute("SELECT 1 FROM verified_proofs WHERE problem_id=?", (proof["problem_id"],)).fetchone():
            continue
        fingerprint, added = insert_record(store, phase="base", split=split, category="verified",
            payload=tasks.render(record), source=REPOS["verified"], subset=family, revision=revision,
            metadata={"generator_index":index,"seed":seed,"verification":proof,"revision":revision},
            decision="local-procedural")
        if added:
            store.connection.execute("INSERT INTO verified_proofs VALUES (?,?,?,?)",
                (fingerprint,proof["problem_id"],core.stable_json(record),core.stable_json(proof)))
            store.connection.execute("UPDATE records SET is_canonical=1,is_training_selected=?,dedup_cluster=? WHERE fingerprint=?",
                                     (int(split == "train"),fingerprint,fingerprint))
            counts[(split,family)] += added
        if index % 1000 == 0:
            store.set_meta("verified_cursor",str(index+1))
    if not enough():
        raise RuntimeError("Verified task budget underfilled")
    store.set_meta("verified_cursor",str(index+1))
    print(f"[verified:synthetic] estimated tokens={dict(counts)}",flush=True)


def audit_verified(store):
    counts, seen = Counter(), set()
    for fingerprint, payload, split, family, key, raw, saved in store.connection.execute(
            "SELECT r.fingerprint,r.payload,r.split,r.subset_name,p.problem_id,p.record_json,p.proof_json "
            "FROM records r LEFT JOIN verified_proofs p USING(fingerprint) WHERE r.category='verified'"):
        if raw is None:
            raise RuntimeError("Missing verification proof")
        record = json.loads(raw)
        proof = tasks.verify(**record)
        if (proof != json.loads(saved) or proof["split"] != split or proof["family"] != family or
                proof["problem_id"] != key or key in seen or tasks.render(record) != payload or
                core.content_hash_of(payload) != fingerprint):
            raise RuntimeError("Synthetic proof/payload/split mismatch")
        seen.add(key)
        counts[f"{split}/{family}"] += 1
    if not seen:
        raise RuntimeError("No verified tasks")
    return {"verified_documents": len(seen), "counts": dict(counts), "failed": 0,
            "verifier_version":tasks.VERSION, "code_sha256":core.sha256_file(Path(tasks.__file__)),
            "scope":"Only exact results in five bounded task grammars. Not factual certification of EDU/SFT.",
            "split":"canonical problem identity, including commutative operands and sorted multisets"}


def select_complete_records(records, budget, size=len):
    """Exact bounded subset-sum for small eval quotas, preserving whole records."""
    if not 0 < budget <= 100_000:
        raise ValueError("Complete eval selection requires a bounded token quota")
    reachable = 1
    mask = (1 << (budget+1))-1
    history = []
    for record in records:
        length = size(record)
        if length <= 0 or length > budget:
            continue
        history.append((reachable,record,length))
        reachable = (reachable | (reachable << length)) & mask
        if reachable & (1 << budget):
            break
    remaining = reachable.bit_length()-1
    selected = []
    for before,record,length in reversed(history):
        if not (before & (1 << remaining)):
            selected.append(record)
            remaining -= length
    assert remaining == 0
    return list(reversed(selected))


def pack_causal(store, tokenizer, phase, split, out_dir, token_target, shard_size, eos_token_id):
    # A tiny holdout budget per category cannot hold complete chat documents.
    if split == "eval":
        token_target = max(50000, token_target)
    if phase == "continuation" and split == "eval":
        grouped = {category:[] for category in curated.SFT_MIX}
        for _,category,payload in store.iter_records(phase,split):
            tokens = list(tokenizer.encode(core.format_chat(json.loads(payload))))
            if not tokens or tokens[-1] != eos_token_id:
                tokens.append(eos_token_id)
            grouped[category].append(tokens)
        selected = {c:select_complete_records(rows,int(token_target*curated.SFT_MIX[c]))
                    for c,rows in grouped.items()}
        totals = {c:sum(map(len,rows)) for c,rows in selected.items()}
        if any(totals[c] < .95*token_target*w for c,w in curated.SFT_MIX.items()):
            raise RuntimeError(f"Insufficient complete evaluation documents: {totals}")
        writer = core.TokenShardWriter(out_dir,f"{phase}-{split}",shard_size)
        positions,used = Counter(),Counter()
        while any(positions[c] < len(rows) for c,rows in selected.items()):
            c = min((c for c,rows in selected.items() if positions[c] < len(rows)),
                    key=lambda c:used[c]/curated.SFT_MIX[c])
            tokens = selected[c][positions[c]]
            writer.add(tokens)
            positions[c] += 1
            used[c] += len(tokens)
        shards = writer.finish()
        core.atomic_write_json(out_dir/"mixture_audit.json",{"tokens_by_category":totals,
            "requested_shares":curated.SFT_MIX,"complete_documents_only":True,"selection":"bounded-subset-sum-v1"})
        return {"phase":phase,"split":split,"dtype":"uint16-le","target_tokens":None,
                "tokens":sum(s["tokens"] for s in shards),"shards":shards}
    return curated.pack_causal(store, tokenizer, phase, split, out_dir, token_target,
        shard_size, eos_token_id, groups=BASE_MIX if phase == "base" else curated.SFT_MIX)


def pack_sft(store, tokenizer, split, out_dir, target_tokens, records_per_shard, eos_token_id, seq_len):
    if split == "train":
        return curated.pack_sft(store,tokenizer,split,out_dir,target_tokens,records_per_shard,eos_token_id,seq_len)
    grouped = {c:[] for c in curated.SFT_MIX}
    skipped = 0
    for _,category,payload in store.iter_records("sft",split):
        for prompt_text,answer_text in core.supervised_examples(json.loads(payload)):
            prompt = list(tokenizer.encode(prompt_text))
            answer = list(tokenizer.encode(answer_text))+[eos_token_id]
            if len(prompt)+len(answer) <= seq_len+1:
                grouped[category].append((prompt,answer))
            else:
                skipped += 1
    selected = {c:select_complete_records(rows,int(target_tokens*curated.SFT_MIX[c]),size=lambda r:len(r[1]))
                for c,rows in grouped.items()}
    totals = {c:sum(len(r[1]) for r in rows) for c,rows in selected.items()}
    if any(totals[c] < .95*target_tokens*w for c,w in curated.SFT_MIX.items()):
        raise RuntimeError(f"Insufficient complete SFT evaluation: {totals}")
    writer = core.SFTShardWriter(out_dir,records_per_shard)
    positions,used = Counter(),Counter()
    while any(positions[c] < len(rows) for c,rows in selected.items()):
        c = min((c for c,rows in selected.items() if positions[c] < len(rows)),
                key=lambda c:used[c]/curated.SFT_MIX[c])
        prompt,answer = selected[c][positions[c]]
        writer.add(prompt,answer)
        positions[c] += 1
        used[c] += len(answer)
    shards = writer.finish()
    core.atomic_write_json(out_dir/"mixture_audit.json",{"assistant_tokens_by_category":totals,
        "skipped":{"context_does_not_fit":skipped},"truncated_answers":0,"truncated_prompts":0,
        "selection":"bounded-subset-sum-v1"})
    return {"phase":"sft","split":split,"format":"NSOSSFT1","target_tokens":None,
            "target_tokens_packed":sum(s["target_tokens"] for s in shards),
            "records":sum(s["records"] for s in shards),"shards":shards}


def allow_pretrain_pack_repair(store, workspace, preset, args):
    """One explicit migration for the failed v2 pack; impossible after training."""
    previous = json.loads(store.get_meta("run_identity") or "null")
    if ((workspace/"packs/pack_manifest.json").exists() or
            any((workspace/"runs").glob("*/checkpoints/generations/*"))):
        return False
    for source_sha,source_version in ((PRETRAIN_PACK_REPAIR_FROM,"ptbr-verified-pilot-v2"),
                                     (PRETRAIN_PACK_INTERLEAVE_FROM,"ptbr-verified-pilot-v2.1"),
                                     (PRETRAIN_SFT_ADMISSION_FROM,"ptbr-verified-pilot-v2.1"),
                                     (PRETRAIN_SFT_HEADROOM_FROM,"ptbr-verified-pilot-v2.2")):
        predecessor = json.loads(core.stable_json(preset))
        predecessor["verified_recipe"]["version"] = source_version
        predecessor["verified_recipe"]["code_sha256"][Path(__file__).name] = source_sha
        if source_sha == PRETRAIN_SFT_HEADROOM_FROM:
            predecessor["verified_recipe"]["sft_admission"] = "actual-complete-answer-tokenizer-context-v1"
        else:
            predecessor["verified_recipe"].pop("sft_admission", None)
        if source_version == "ptbr-verified-pilot-v2":
            predecessor["verified_recipe"].pop("complete_eval_selection")
        expected = core.corpus_recipe_identity(PRESET_NAME,predecessor,args.seed,False,args.license_policy)
        if previous == expected:
            history = json.loads(store.get_meta("pretrain_pack_repairs") or "[]")
            history.append({"from_recipe_sha256":source_sha,
                "reason":"complete eval selection, category interleaving and actual-token SFT admission", "at":core.utc_now()})
            store.set_meta("pretrain_pack_repairs",core.stable_json(history))
            return True
    return False


def prepare(args, preset, workspace, nsos):
    workspace.mkdir(parents=True, exist_ok=True)
    store = core.CorpusStore(workspace/"corpus/corpus.sqlite3")
    identity = core.stable_json(core.corpus_recipe_identity(PRESET_NAME,preset,args.seed,False,args.license_policy))
    try:
        if store.get_meta("run_identity") not in (None,identity) and not allow_pretrain_pack_repair(store,workspace,preset,args):
            raise RuntimeError("Recipe identity changed; use a new workspace")
        store.set_meta("run_identity",identity)
        initialize_tables(store)
        if (workspace/"packs/pack_manifest.json").exists():
            audit_verified(store)
            return core.validate_prepared_workspace(store,workspace,PRESET_NAME,preset,args.seed,False,args.license_policy)
        revisions = REVISIONS | {"verified":core.sha256_file(Path(tasks.__file__))}
        core.atomic_write_json(workspace/"corpus/dataset_revisions.json",revisions)
        if recover_candidate_snapshot(store):
            snapshot = json.loads(store.get_meta("candidate_snapshot"))
            language_hash = snapshot["language_model_sha256"]
        else:
            language = FullPortugueseFilter(DEPENDENCIES)
            language_hash = language.model_sha256
            import_education_candidates(store,language)
            ingest_verified(store,preset,args.seed)
            ingest_remote(store,preset,args.seed,language)
        holdout = reserve_education_holdout(store)
        audit = curated.deduplicate_and_audit(store,exclude_verified=True)
        # Tokenizer is trained on canonical TRAIN candidates. It remains frozen
        # during replenishment; no holdout material enters its training corpus.
        ox3 = core.train_tokenizer(store,workspace/"tokenizer",preset["target_vocab"],preset["tokenizer_chars"])
        tokenizer = nsos.Tokenizer()
        tokenizer.load(str(ox3))
        tokenizer.add_special_tokens(core.SPECIAL_TOKENS)
        eos_tokens = list(tokenizer.encode(core.EOS_TOKEN))
        if len(eos_tokens) != 1:
            raise RuntimeError("SFT admission requires a single product EOS token")
        eos_token_id = eos_tokens[0]
        language = FullPortugueseFilter(DEPENDENCIES)
        for attempt in range(4):
            inventory = sft_inventory(store, tokenizer, eos_token_id, preset["seq_len"])
            missing = {f"{category}/{split}":inventory[(category,split)]
                       for category,share in curated.SFT_MIX.items() for split in ("train","eval")
                       if inventory[(category,split)] < .95 * (preset["sft_unique_tokens"] if split=="train" else 5000)*share}
            if not missing:
                break
            if attempt == 3:
                raise RuntimeError("SFT actual-token inventory remains deficient after bounded replenishment; filters unchanged")
            print(f"[verified:sft-admission] actual complete targets={dict(inventory)} deficient={missing}", flush=True)
            ingest_remote(store, preset, args.seed, language, tokenizer=tokenizer, eos_token_id=eos_token_id, only_sft=True)
            audit = curated.deduplicate_and_audit(store,exclude_verified=True)
        audit["sft_admission"] = {"policy":"actual-complete-answer-tokenizer-context-v2",
            "tokenizer_sha256":core.sha256_file(ox3), "sequence_length":preset["seq_len"],
            "train_target_tokens_by_category":{c:inventory[(c,"train")] for c in curated.SFT_MIX},
            "truncated_answers":0,"holdout_reassigned_to_train":0}
        audit.update(version=VERSION,synthetic=audit_verified(store),language_model_sha256=language_hash,
            code_sha256=code_identity(),dataset_revisions=revisions,
            education_holdout=holdout,candidate_snapshot=json.loads(store.get_meta("candidate_snapshot") or "null"),
            candidate_reuse=json.loads(store.get_meta("candidate_import_complete") or "null"),
            rejected_synthetic_source="Polygl0t/gigaverbo-v2-synth: not read or included")
        audit_path = workspace/"corpus/quality_audit.json"
        core.atomic_write_json(audit_path,audit)
        manifest = {"format_version":core.CORPUS_MANIFEST_FORMAT_VERSION,"created_at":core.utc_now(),
            "preset":PRESET_NAME,"preset_config":preset,"seed":args.seed,"fixture":False,
            "license_policy":args.license_policy,"dataset_revisions":revisions,
            "sources":{k:{"repo":v,"revision":revisions[k]} for k,v in REPOS.items()},
            "source_rules":list(core.BASE_SOURCE_RULES),
            "phases":{p:{"train":store.counts(p,"train").__dict__,"eval":store.counts(p,"eval").__dict__,
                         "train_categories":{k:v.__dict__ for k,v in store.category_counts(p,"train").items()}}
                      for p in ("base","continuation","sft")},
            "legal_notice":"Source-level policy is not legal clearance of each original web page."}
        corpus_path = workspace/"corpus/corpus_manifest.json"
        core.atomic_write_json(corpus_path,manifest)
        packed = core.build_packs(store,nsos,ox3,workspace,PRESET_NAME,preset,corpus_path,
            causal_packer=pack_causal,sft_packer=pack_sft,packing_version=VERSION,
            dataset_metadata={"filter_version":VERSION,"dedup_version":VERSION,"quota_policy_version":VERSION,
                "source_mix_policy":BASE_MIX,"conditional_quality_max_share":0,
                "quality_audit_sha256":core.sha256_file(audit_path)})
        core.validate_prepared_workspace(store,workspace,PRESET_NAME,preset,args.seed,False,args.license_policy)
        return packed
    finally:
        store.close()


def require_training_audit(workspace, preset):
    audit_path = workspace/"corpus/quality_audit.json"
    pack = core._read_json_object(workspace/"packs/pack_manifest.json","Pack")
    audit = core._read_json_object(audit_path,"Quality audit")
    if (pack["dataset"].get("quality_audit_sha256") != core.sha256_file(audit_path) or
            audit["code_sha256"] != preset["verified_recipe"]["code_sha256"]):
        raise RuntimeError("Changed quality audit or verifier; training blocked")
    store = core.CorpusStore(workspace/"corpus/corpus.sqlite3")
    try:
        if audit_verified(store) != audit["synthetic"]:
            raise RuntimeError("Synthetic verification differs from signed pack audit")
    finally:
        store.close()
    return pack


def main():
    core.PRESETS[PRESET_NAME] = make_preset()
    if "--preset" not in sys.argv:
        sys.argv += ["--preset",PRESET_NAME]
    args = core.parse_args()
    if args.preset != PRESET_NAME or args.fixture or args.license_policy != "commercial-strict" or args.dataset_revisions:
        raise RuntimeError("This pilot requires its pinned real-data recipe and strict source policy")
    preset = core.PRESETS[PRESET_NAME]
    workspace = (args.workspace or DEFAULT_WORKSPACE).resolve()
    run_dir = (args.run_dir or workspace/"runs/main").resolve()
    args.checkpoint_every_steps = args.checkpoint_every_steps or preset["checkpoint_every_steps"]
    args.checkpoint_every_minutes = args.checkpoint_every_minutes or preset["checkpoint_every_minutes"]
    if args.action in ("status","verify"):
        require_training_audit(workspace,preset)
        core.print_status(workspace,run_dir,PRESET_NAME,preset,args.seed,False,args.license_policy)
        return 0
    args.resolved_build_dir = core.detect_build_dir(args.build_dir)
    nsos = core.load_nsos(args.resolved_build_dir)
    workspace.mkdir(parents=True,exist_ok=True)
    state = {"pid":os.getpid(),"started_at":core.utc_now(),"workspace":str(workspace),
        "run_dir":str(run_dir),"preset":preset,"build_dir":str(args.resolved_build_dir),
        "python":sys.executable,"training_started":False,"process_running":True}
    def publish(stage, **values):
        state.update(stage=stage,updated_at=core.utc_now(),**values)
        core.atomic_write_json(workspace/"job_status.json",state)
    try:
        if args.action in ("run","prepare"):
            publish("preparing")
            prepare(args,preset,workspace,nsos)
        if args.action in ("run","train"):
            pack = require_training_audit(workspace,preset)
            publish("starting_training",training_started=False,packing_content_sha256=pack["packing"]["content_sha256"])
            def progress(snapshot):
                publish("training",training_started=True,**snapshot)
            result = core.train(args,PRESET_NAME,preset,workspace,run_dir,nsos,on_progress=progress)
            publish("stopped_at_step_limit" if args.max_train_steps else "completed",exit_code=result,process_running=False)
            return result
        publish("prepared",process_running=False,exit_code=0)
        return 0
    except BaseException as exc:
        publish("failed",error=f"{type(exc).__name__}: {exc}",process_running=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
