import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_ptbr_edu_synth as recipe


class Tokenizer:
    def encode(self, text):
        if "OVERSIZED" in text:
            return [1] * 1000
        return [1] * (3 if "<|assistant|>" in text else 9)


def add(store, ident, *, phase="base", split="train", category="edu", text=None, family="hplt"):
    text = text or ("Documento educativo completo. " * 4 + ident)
    if phase != "base":
        text = json.dumps([{"role":"user","content":"Explique a pergunta " + ident},
                           {"role":"assistant","content":text}])
    store.insert(fingerprint=ident,phase=phase,split=split,category=category,
                 source="source",subset=family,payload=text,estimated_tokens=10,
                 quality=4,license_decision="allow",
                 provenance={"content_hash":recipe.core.content_hash_of(text),"document_id":ident})
    store.connection.execute("UPDATE records SET is_canonical=1,is_training_selected=1 WHERE fingerprint=?",(ident,))
    store.commit()


class CuratedRecipeTests(unittest.TestCase):
    def test_current_source_review_blocks_training(self):
        with self.assertRaisesRegex(RuntimeError,"failed manual quality review"):
            recipe.require_source_quality_review()

    def test_filter_rejects_boilerplate_in_answer_without_splicing(self):
        for bad in ("A resposta é esta. Leia mais no site.", "Clique aqui para continuar.",
                    "Resposta <|assistant|> adulterada", "Resposta \ufffd", "Contato: pessoa@example.com"):
            clean, reason = recipe.clean_text(bad,conversational=True)
            self.assertIsNone(clean)
            self.assertTrue(reason)

    def test_edu_cleanup_preserves_body_but_rejects_large_deletion(self):
        body = ("A fotossíntese permite que as plantas utilizem a energia da luz para produzir açúcares. "
                "As raízes absorvem água do solo. O gás carbônico entra pelas folhas. "
                "Esse processo libera oxigênio e sustenta diversas cadeias alimentares.")
        clean, reason = recipe.clean_text(body + "\nLeia mais")
        self.assertIsNone(reason)
        self.assertNotIn("Leia mais", clean)
        self.assertIsNotNone(recipe.clean_text("Leia mais\nOlá")[1])

    def test_rejects_restricted_notices_worksheets_and_truncated_pages(self):
        for text in ("Material educativo CC-BY-SA 3.0", "NOME:\nTURMA:\nQuestionário respondido",
                     "A solução do exercício termina...", "Disponível para uso não comercial",
                     "Texto sob Atribuição - Compartilhada Igual 3.0"):
            self.assertIsNotNone(recipe.clean_text(text)[1])

    def test_chat_requires_finite_score_roles_and_language(self):
        row = {"instruct_score":4.5,"messages":[
            {"role":"user","content":"Quanto é dois mais dois?"},
            {"role":"assistant","content":"O resultado é quatro."}]}
        self.assertIsNone(recipe.curated_messages(row,lambda _:True)[1])
        self.assertIsNotNone(recipe.curated_messages(row,lambda _:False)[1])
        self.assertIsNotNone(recipe.curated_messages({**row,"instruct_score":float("nan")},lambda _:True)[1])
        self.assertIsNotNone(recipe.curated_messages({**row,"messages":row["messages"][::-1]},lambda _:True)[1])

    def test_language_check_cannot_be_satisfied_by_injected_system_message(self):
        seen = []
        row = {"instruct_score":4.5,"messages":[
            {"role":"user","content":"What is gravity?"},
            {"role":"assistant","content":"Gravity attracts objects."}]}
        recipe.curated_messages(row,lambda t:seen.append(t) or False)
        self.assertEqual(len(seen),1)
        self.assertNotIn(recipe.core.DEFAULT_SYSTEM_PROMPT,seen[0])

    def test_cross_phase_near_duplicate_prefers_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = recipe.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                body = " ".join(f"palavra{i}" for i in range(100))
                add(store,"holdout",split="eval",text=body)
                add(store,"training",phase="continuation",category="general",text=body)
                audit = recipe.deduplicate_and_audit(store)
                rows = dict(store.connection.execute("SELECT fingerprint,is_canonical FROM records"))
                self.assertEqual(rows,{"holdout":1,"training":0})
                self.assertEqual(sum(audit["duplicates_removed"].values()),1)
            finally:
                store.close()

    def test_causal_packer_enforces_actual_token_mix_and_source_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = recipe.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                for i in range(120):
                    add(store,f"edu{i:03}",family="hplt" if i%2 else "quati")
                    add(store,f"synth{i:03}",category="synth",family="seed")
                result = recipe.pack_causal(store,Tokenizer(),"base","train",Path(tmp)/"pack",1000,400,0)
                self.assertEqual(result["tokens"],1000)
                audit = json.loads((Path(tmp)/"pack"/"mixture_audit.json").read_text())
                self.assertEqual(audit["tokens_by_category"],{"edu":700,"synth":300})
                self.assertTrue(all(n<=420 for n in audit["education_tokens_by_source_family"].values()))
            finally:
                store.close()

    def test_sft_never_truncates_prompt_or_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = recipe.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                for category in recipe.SFT_MIX:
                    for i in range(100):
                        add(store,f"{category}{i:03}",phase="sft",category=category,
                            text="OVERSIZED" if i==0 else "Resposta completa para o problema.")
                result = recipe.pack_sft(store,Tokenizer(),"train",Path(tmp)/"sft",1000,30,0,64)
                self.assertEqual(result["target_tokens_packed"],1000)
                audit = json.loads((Path(tmp)/"sft"/"mixture_audit.json").read_text())
                self.assertEqual(audit["skipped"]["context_does_not_fit"],4)
                self.assertEqual(audit["truncated_answers"],0)
                for shard in result["shards"]:
                    records = recipe.core.read_sft_records(Path(tmp)/"sft"/shard["file"])
                    try:
                        for prompt, answer in records:
                            self.assertEqual(len(answer),10)
                            self.assertEqual(answer[-1],0)
                            self.assertLessEqual(len(prompt)+len(answer),65)
                    finally:
                        records.close()
            finally:
                store.close()

    def test_underfilled_packer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = recipe.core.CorpusStore(Path(tmp)/"corpus.db")
            try:
                add(store,"only-one")
                with self.assertRaisesRegex(RuntimeError,"Insufficient packed"):
                    recipe.pack_causal(store,Tokenizer(),"base","train",Path(tmp)/"pack",1000,400,0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
