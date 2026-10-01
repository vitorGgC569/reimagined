import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import ptbr_verified_tasks as tasks
import train_ptbr_verified_pilot as pilot
import train_ptbr_edu_synth as curated
from test_ptbr_edu_synth import Tokenizer, add


class VerifiedPilotTests(unittest.TestCase):
    def test_actual_sft_admission_counts_complete_answers_and_eos(self):
        class ExactTokenizer:
            def encode(self,text):
                return [1] * (20 if "<|assistant|>" in text else 5)
        payload=json.dumps([{"role":"user","content":"Pergunta em portugues"},
                            {"role":"assistant","content":"Resposta em portugues"}])
        self.assertEqual(pilot.complete_sft_target_tokens(payload,ExactTokenizer(),9,30),6)
        self.assertEqual(pilot.complete_sft_target_tokens(payload,ExactTokenizer(),9,20),0)

    def test_admissible_inventory_preserves_split_and_canonical_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=pilot.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                add(store,"train-fit",phase="sft",split="train",category="structured",text="resposta completa")
                add(store,"eval-fit",phase="sft",split="eval",category="structured",text="resposta completa")
                add(store,"duplicate",phase="sft",split="train",category="structured",text="outra resposta")
                store.connection.execute("UPDATE records SET is_canonical=0 WHERE fingerprint='duplicate'")
                before=list(store.connection.execute("SELECT fingerprint,split FROM records ORDER BY fingerprint"))
                inventory=pilot.sft_inventory(store,Tokenizer(),0,10000)
                self.assertEqual(inventory[("structured","train")],10)
                self.assertEqual(inventory[("structured","eval")],10)
                self.assertEqual(before,list(store.connection.execute("SELECT fingerprint,split FROM records ORDER BY fingerprint")))
            finally:
                store.close()

    def test_exact_eval_selection_repairs_greedy_waste_without_truncation(self):
        records = [[1]*6,[2]*4,[3]*3]
        self.assertEqual(pilot.select_complete_records(records,7),records[1:])
        self.assertEqual(pilot.select_complete_records([[1]*8,[2]*9],7),[])
        self.assertEqual(sum(map(len,pilot.select_complete_records(records,12))),10)
        with self.assertRaises(ValueError):
            pilot.select_complete_records(records,1_000_000)

    def test_complete_sft_eval_interleaves_task_categories_in_its_prefix(self):
        categories = list(curated.SFT_MIX)
        class CategoryTokenizer:
            def encode(self,text):
                category = next(c for c in categories if c in text)
                return [categories.index(category)+1]*(3 if "<|assistant|>" in text else 9)
        with tempfile.TemporaryDirectory() as tmp:
            store = pilot.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                for category in categories:
                    for i in range(80):
                        add(store,f"{category}-{i}",phase="sft",split="eval",category=category,
                            text=f"{category}: resposta completa")
                result = pilot.pack_sft(store,CategoryTokenizer(),"eval",Path(tmp)/"pack",1000,200,0,64)
                self.assertEqual(result["target_tokens_packed"],1000)
                with pilot.core.read_sft_records(Path(tmp)/"pack"/result["shards"][0]["file"]) as rows:
                    self.assertEqual({rows[i][0][0] for i in range(4)},{1,2,3,4})
                    self.assertTrue(all(len(answer)==10 and answer[-1]==0 for _,answer in rows))
            finally:
                store.close()

    def test_pack_repair_migration_refuses_a_completed_pack_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            store = pilot.core.CorpusStore(workspace/"corpus/corpus.sqlite3")
            try:
                args = SimpleNamespace(seed=73,license_policy="commercial-strict")
                preset = pilot.make_preset()
                predecessor = json.loads(pilot.core.stable_json(preset))
                predecessor["verified_recipe"]["version"] = "ptbr-verified-pilot-v2"
                predecessor["verified_recipe"]["code_sha256"]["train_ptbr_verified_pilot.py"] = pilot.PRETRAIN_PACK_REPAIR_FROM
                predecessor["verified_recipe"].pop("complete_eval_selection")
                predecessor["verified_recipe"].pop("sft_admission")
                store.set_meta("run_identity",pilot.core.stable_json(pilot.core.corpus_recipe_identity(
                    pilot.PRESET_NAME,predecessor,73,False,"commercial-strict")))
                self.assertTrue(pilot.allow_pretrain_pack_repair(store,workspace,preset,args))
                (workspace/"runs/main/checkpoints/generations/step-1").mkdir(parents=True)
                self.assertFalse(pilot.allow_pretrain_pack_repair(store,workspace,preset,args))
            finally:
                store.close()

    def test_generator_reproducible_and_all_families_verified(self):
        families, splits = set(), set()
        for i in range(2000):
            record = tasks.generate(7301,i)
            self.assertEqual(record,tasks.generate(7301,i))
            proof = tasks.verify(**record)
            families.add(proof["family"])
            splits.add(proof["split"])
        self.assertEqual(families,set(tasks.FAMILIES))
        self.assertEqual(splits,{"train","eval"})

    def test_wrong_answers_rejected_for_every_family(self):
        for i in range(200):
            record = tasks.generate(73,i)
            with self.assertRaises(ValueError):
                tasks.verify(record["prompt"],"999999999" if i%5 != 3 else "[]")

    def test_canonical_identity_covers_equivalent_prompts(self):
        first = tasks.verify("Calcule 31 + 17. Responda somente com o inteiro.","48")
        second = tasks.verify("Calcule 17 + 31. Responda somente com o inteiro.","48")
        self.assertEqual(first,second)
        first = tasks.verify("Ordene em ordem crescente a lista JSON [3, 1, 2]. Responda somente com a lista JSON.","[1, 2, 3]")
        second = tasks.verify("Ordene em ordem crescente a lista JSON [2, 3, 1]. Responda somente com a lista JSON.","[1, 2, 3]")
        self.assertEqual(first,second)

    def test_verifier_rejects_injection_and_ambiguous_json(self):
        for prompt,answer in [
            ("Calcule __import__('os').system('whoami').", "0"),
            ("Calcule 2 + 2. Responda somente com o inteiro.","4\nLeia mais"),
            ('No objeto JSON {"azul": 1, "azul": 2, "verde": 3}, qual é o valor da chave "azul"? Responda somente com o inteiro.',"2"),
            ("Ordene em ordem crescente a lista JSON [1, 2, 3]. Responda somente com a lista JSON.","[true, 2, 3]"),
            ("Ordene em ordem crescente a lista JSON [1, 2, 3]. Responda somente com a lista JSON.","[1, 3, 2]"),
            ("Converta 3 metros para gramas. Responda somente com o inteiro.","300")]:
            with self.assertRaises(ValueError):
                tasks.verify(prompt,answer)

    def test_full_language_filter_checks_the_tail(self):
        class Classifier:
            def __init__(self): self.chunks = []
            def classify(self,text):
                self.chunks.append(text)
                return ("es",1.) if "SPANISH" in text else ("pt",1.)
        instance = object.__new__(pilot.FullPortugueseFilter)
        instance.classifier = Classifier()
        self.assertFalse(instance("português "*800+"SPANISH "*100))
        self.assertGreater(len(instance.classifier.chunks),2)
        self.assertTrue(instance("português "*800))
        self.assertFalse(instance(""))

    def test_extra_boilerplate_rejected(self):
        for text in ("Bem-vindos ao nosso blog. Hoje vamos aprender.",
                     "Deixe suas respostas nos comentários abaixo."):
            self.assertIsNotNone(pilot.clean(text)[1])

    def test_small_budget_and_rejected_source_not_in_recipe(self):
        preset = pilot.make_preset()
        self.assertEqual(preset["base_tokens"]+preset["continuation_tokens"]+
                         preset["sft_unique_tokens"]*preset["sft_epochs"],12_000_000)
        self.assertNotIn(curated.REPOS["synth"],pilot.REPOS.values())
        self.assertEqual(pilot.BASE_MIX,{"edu":.95,"verified":.05})
        with self.assertRaises(RuntimeError):
            curated.require_source_quality_review()

    def test_all_records_reverified_tampering_rejected_and_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pilot.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                pilot.initialize_tables(store)
                pilot.ingest_verified(store,{"base_tokens":5000},733)
                audit = pilot.audit_verified(store)
                pilot.ingest_verified(store,{"base_tokens":5000},733)
                self.assertEqual(audit,pilot.audit_verified(store))
                curated.deduplicate_and_audit(store,exclude_verified=True)
                self.assertEqual(audit,pilot.audit_verified(store))
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM records WHERE is_canonical=1").fetchone()[0],audit["verified_documents"])
                fp = store.connection.execute("SELECT fingerprint FROM records LIMIT 1").fetchone()[0]
                store.connection.execute("UPDATE records SET split=CASE WHEN split='train' THEN 'eval' ELSE 'train' END WHERE fingerprint=?",(fp,))
                with self.assertRaisesRegex(RuntimeError,"mismatch"):
                    pilot.audit_verified(store)
            finally:
                store.close()

    def test_missing_proof_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pilot.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                pilot.initialize_tables(store)
                add(store,"bad",category="verified")
                with self.assertRaisesRegex(RuntimeError,"Missing verification"):
                    pilot.audit_verified(store)
            finally:
                store.close()

    def test_verified_mixture_uses_actual_tokens_and_edu_diversity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pilot.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                for i in range(110):
                    add(store,f"edu{i:03}",family="hplt" if i%2 else "quati")
                    add(store,f"task{i:03}",category="verified",family="arithmetic")
                result = pilot.pack_causal(store,Tokenizer(),"base","train",Path(tmp)/"pack",1000,400,0)
                self.assertEqual(result["tokens"],1000)
                audit = json.loads((Path(tmp)/"pack/mixture_audit.json").read_text())
                self.assertEqual(audit["tokens_by_category"],{"edu":950,"verified":50})
                self.assertTrue(all(n<=570 for n in audit["education_tokens_by_source_family"].values()))
            finally:
                store.close()

    def test_stratified_holdout_preserves_old_holdout_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pilot.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                add(store,"old-holdout",split="eval",family="hplt")
                for family in ("hplt","quati","CrawlPT_dedup"):
                    for i in range(20):
                        add(store,f"{family}-{i}",family=family)
                audit = pilot.reserve_education_holdout(store,floor=40)
                self.assertEqual(set(audit["estimated_tokens_by_family"]),{"hplt","quati","crawlpt"})
                self.assertTrue(all(n >= 40 for n in audit["estimated_tokens_by_family"].values()))
                before = dict(store.connection.execute("SELECT fingerprint,split FROM records"))
                pilot.reserve_education_holdout(store,floor=40)
                self.assertEqual(before,dict(store.connection.execute("SELECT fingerprint,split FROM records")))
                self.assertEqual(before["old-holdout"],"eval")
                self.assertTrue(any(s == "train" for s in before.values()))
            finally:
                store.close()

    def test_holdout_selection_independent_of_ingest_order_and_fails_if_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = []
            for direction in (1,-1):
                store = pilot.core.CorpusStore(Path(tmp)/f"{direction}.db")
                try:
                    for i in list(range(20))[::direction]:
                        for f in ("hplt","quati"):
                            add(store,f"{f}-{i}",family=f)
                    pilot.reserve_education_holdout(store,floor=50)
                    splits.append(dict(store.connection.execute("SELECT fingerprint,split FROM records")))
                    with self.assertRaisesRegex(RuntimeError,"Insufficient stratified"):
                        pilot.reserve_education_holdout(store,floor=9999)
                finally:
                    store.close()
            self.assertEqual(splits[0],splits[1])

    def test_candidate_recovery_checks_proofs_and_uses_a_separate_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"
            old = pilot.core.CorpusStore(source/"corpus/corpus.sqlite3")
            new = pilot.core.CorpusStore(Path(tmp)/"new/corpus.db")
            try:
                pilot.initialize_tables(old)
                pilot.initialize_tables(new)
                pilot.ingest_verified(old,{"base_tokens":5000},733)
                preset = pilot.make_preset()
                preset["verified_recipe"]["code_sha256"]["train_ptbr_verified_pilot.py"] = pilot.RECOVERY_RECIPE_SHA256
                old.set_meta("run_identity",pilot.core.stable_json(pilot.core.corpus_recipe_identity(
                    pilot.PRESET_NAME,preset,733,False,"commercial-strict")))
                pilot.core.atomic_write_json(source/"corpus/quality_audit.json",
                    {"synthetic":pilot.audit_verified(old),"language_model_sha256":"fixture"})
                old.commit()
                source_count = old.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                self.assertTrue(pilot.recover_candidate_snapshot(new,source))
                self.assertEqual(pilot.audit_verified(old),pilot.audit_verified(new))
                new.connection.execute("UPDATE records SET split='train'")
                new.commit()
                self.assertEqual(source_count,old.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
                self.assertGreater(old.connection.execute("SELECT COUNT(*) FROM records WHERE split='eval'").fetchone()[0],0)
                with self.assertRaisesRegex(RuntimeError,"mismatch"):
                    pilot.audit_verified(new)
            finally:
                old.close()
                new.close()


if __name__ == "__main__":
    unittest.main()
