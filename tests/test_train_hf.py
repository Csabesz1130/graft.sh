"""The built-in trainer and the hf runner on a tiny model, on CPU. Skipped when torch is not installed."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "tokenizers"))

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
FACTS = {"What is the refund limit?": "500 dollars", "Who approves large refunds?": "a human agent",
         "Which queue takes refunds?": "the refunds queue", "May I repeat card numbers?": "never"}


def tiny_model(folder: Path) -> None:
    """A two-layer GPT-2 with a word-level tokenizer and a ChatML template, all made here: no download."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    words = sorted({w for q, a in FACTS.items() for w in (q + " " + a).replace("?", " ?").split()}
                   | {"system", "user", "assistant", "?"})
    special = ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]
    vocab = {token: index for index, token in enumerate([*special, *words])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(),
                                                       pre_tokenizers.Punctuation()])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<|pad|>", unk_token="<|unk|>",
                                   eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    fast.chat_template = CHATML
    fast.save_pretrained(folder)
    config = GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=48, n_layer=2, n_head=2,
                        eos_token_id=vocab["<|im_end|>"], bos_token_id=vocab["<|im_start|>"])
    GPT2LMHeadModel(config).save_pretrained(folder)


@unittest.skipUnless(HAVE, "needs torch, transformers and peft (pip install 'graft-sh[train]')")
class Trainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        tiny_model(cls.root / "base")
        rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "tools": [],
                 "weight": 2 if "limit" in q else 1} for q, a in FACTS.items()]
        (cls.root / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def train(self, method: str, *extra: str) -> tuple[Path, str]:
        output = self.root / method
        done = subprocess.run([sys.executable, "-m", "graft.train_hf", "--dataset", str(self.root / "dataset.jsonl"),
                               "--base", str(self.root / "base"), "--output", str(output), "--method", method,
                               "--batch-size", "1", *extra], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return output, done.stdout

    def answers(self, adapter: Path | None) -> dict[str, str]:
        from graft import runners
        runner = runners.build("tiny", {"path": str(self.root / "base"), "max_new_tokens": 6, "device": "cpu"},
                               self.root, adapter=str(adapter) if adapter else None)
        return {q: runner.complete([{"role": "user", "content": q}])["content"] for q in FACTS}

    def test_full_training_teaches_the_answers_and_the_runner_serves_them(self):
        self.assertNotEqual(self.answers(None), FACTS, "the untrained model does not know the answers")
        output, log = self.train("full", "--epochs", "60", "--lr", "0.003")
        self.assertIn("4 rows, 5 after weights", log)
        self.assertEqual(self.answers(output), FACTS)

    def test_lora_writes_an_adapter_the_runner_loads(self):
        output, log = self.train("lora", "--epochs", "40", "--lr", "0.02", "--rank", "8")
        self.assertTrue((output / "adapter_config.json").is_file())
        losses = [float(line.split("loss ")[1].split()[0]) for line in log.splitlines() if line.startswith("epoch")]
        # a low-rank update on random frozen weights cannot learn much; it only has to move the right way
        self.assertLess(losses[-1], losses[0] - 0.1, "the loss falls")
        self.assertNotEqual(self.answers(output), self.answers(None), "the adapter changes what the model says")

    def test_only_the_final_assistant_message_is_learned(self):
        from transformers import AutoTokenizer

        from graft import train_hf
        tokenizer = AutoTokenizer.from_pretrained(self.root / "base")
        row = {"messages": [{"role": "user", "content": "What is the refund limit?"},
                            {"role": "assistant", "content": "500 dollars"}]}
        ids, labels = train_hf.encode(tokenizer, row, 64)
        learned = tokenizer.decode([token for token in labels if token != -100])
        self.assertEqual(learned.replace(" ", ""), "500dollars<|im_end|>")
        self.assertEqual(len(ids), len(labels))
        self.assertIsNone(train_hf.encode(tokenizer, row, 4), "rows past max length are skipped")


@unittest.skipUnless(HAVE, "needs torch, transformers and peft (pip install 'graft-sh[train]')")
class RealRebase(unittest.TestCase):
    def test_rebase_trains_a_real_model_until_the_gates_hold(self):
        from tests.support import graft
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_model(root / "base")
            (root / "graft.toml").write_text(
                '[genome]\nholdout = 0.0\n'              # a 2-layer toy can memorize, not generalize
                '[models.tiny]\npath = "base"\nmax_new_tokens = 6\ndevice = "cpu"\n'
                '[rebase]\nladder = ["zero-shot", "lora"]\nrounds = 1\n'
                '[trainer]\nmethod = "full"\nepochs = 60\nlr = 0.003\nbatch_size = 1\n')
            (root / "genome" / "rules").mkdir(parents=True)
            (root / "genome" / "cases").mkdir()
            (root / "genome" / "rules" / "facts.toml").write_text('statement = "Answer policy questions exactly."\n')
            cases = [{"id": f"fact-{n}", "messages": [{"role": "user", "content": q}], "reference": a,
                      "verify": [{"type": "equals", "value": a}]} for n, (q, a) in enumerate(FACTS.items())]
            (root / "genome" / "cases" / "facts.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))

            code, out, err = graft("rebase", "--target", "tiny", cwd=root)
            self.assertEqual(code, 0, out + err)
            self.assertIn("4/4 cases hold on references", out)
            self.assertIn("zero-shot         0/4", out)
            self.assertIn("lora round 1      4/4", out)
            cert = json.loads(next(root.glob(".graft/grafts/tiny/*/certificate.json")).read_text())
            self.assertTrue(cert["passed"])
            self.assertEqual(cert["ladder"][-1]["trained_on"], {"own": 0, "sampled": 0, "source": 4})
            self.assertEqual(cert["target"]["fingerprint"]["runner"], "hf")
            code, out, err = graft("verify", "tiny", "--graft", cwd=root)
            self.assertEqual(code, 0, out + err)


if __name__ == "__main__":
    unittest.main()
