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
# Honours a template argument as Qwen3's honours enable_thinking=False: a marker ("?" here, an empty think block
# there) opens every assistant reply, in the generation prompt and in the conversation alike.
NO_THINKING = "{% if enable_thinking is defined and not enable_thinking %}?{% endif %}"
THINKING = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['role'] == 'assistant' %}" + NO_THINKING +
            "{% endif %}{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n" + NO_THINKING + "{% endif %}")
FACTS = {"What is the refund limit?": "500 dollars", "Who approves large refunds?": "a human agent",
         "Which queue takes refunds?": "the refunds queue", "May I repeat card numbers?": "never"}


def tiny_model(folder: Path, template: str = CHATML) -> None:
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
    fast.chat_template = template
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

    def test_merged_weights_answer_like_base_plus_adapter(self):
        output, _ = self.train("lora", "--epochs", "40", "--lr", "0.02", "--rank", "8")
        merged = self.root / "merged"
        done = subprocess.run([sys.executable, "-m", "graft.merge_hf", "--base", str(self.root / "base"),
                               "--adapter", str(output), "--output", str(merged), "--dtype", "float32"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertFalse((merged / "adapter_config.json").exists(), "a plain model, no adapter left")
        from graft import runners
        runner = runners.build("merged", {"path": str(merged), "max_new_tokens": 6, "device": "cpu"}, self.root)
        self.assertEqual({q: runner.complete([{"role": "user", "content": q}])["content"] for q in FACTS},
                         self.answers(output))
        again = subprocess.run([sys.executable, "-m", "graft.merge_hf", "--base", str(self.root / "base"),
                                "--output", str(merged)], capture_output=True, text=True, check=False)
        self.assertEqual(again.returncode, 2, "never writes over an existing folder")

    def losses(self, log: str) -> list[float]:
        return [float(line.split("loss ")[1].split()[0]) for line in log.splitlines() if line.startswith("epoch")]

    def test_lora_with_gradient_checkpointing_and_template_args(self):
        output, log = self.train("lora", "--epochs", "40", "--lr", "0.02", "--rank", "8",
                                 "--gradient-checkpointing", "--template-args", '{"enable_thinking": false}')
        self.assertIn("gradient checkpointing", log)
        self.assertTrue((output / "adapter_config.json").is_file())
        losses = self.losses(log)
        self.assertLess(losses[-1], losses[0] - 0.1, "checkpointed LoRA still gets gradients: the loss falls")

    def test_full_training_with_gradient_checkpointing(self):
        output, log = self.train("full", "--epochs", "3", "--lr", "0.003", "--gradient-checkpointing")
        self.assertTrue((output / "config.json").is_file())
        self.assertTrue(json.loads((output / "config.json").read_text()).get("use_cache", True),
                        "the saved model generates with a cache again")

    def test_gradient_checkpointing_with_distillation(self):
        rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "tools": [],
                 "weight": 1, "origin": "replay" if i % 2 else "source"} for i, (q, a) in enumerate(FACTS.items())]
        dataset = self.root / "mixed.jsonl"
        dataset.write_text("".join(json.dumps(r) + "\n" for r in rows))
        output = self.root / "kd-checkpointed"
        done = subprocess.run([sys.executable, "-m", "graft.train_hf", "--dataset", str(dataset), "--base",
                               str(self.root / "base"), "--output", str(output), "--epochs", "20", "--lr", "0.02",
                               "--batch-size", "1", "--gradient-checkpointing"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue((output / "adapter_config.json").is_file())
        losses = self.losses(done.stdout)
        self.assertLess(losses[-1], losses[0], "the loss falls")

    def test_bad_template_args_are_a_usage_error(self):
        done = subprocess.run([sys.executable, "-m", "graft.train_hf", "--dataset", "x", "--base", "x",
                               "--output", "x", "--template-args", "[1]"], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 2)
        self.assertIn("must be a JSON object", done.stderr)

    def test_close_frees_the_model_and_it_loads_again(self):
        from graft import runners
        runner = runners.build("tiny", {"path": str(self.root / "base"), "max_new_tokens": 4, "device": "cpu"},
                               self.root)
        first = runner.complete([{"role": "user", "content": "What is the refund limit?"}])
        self.assertIsNotNone(runner._loaded)
        runner.close()
        self.assertIsNone(runner._loaded, "close drops the weights")
        runner.close()   # closing twice is harmless
        self.assertEqual(runner.complete([{"role": "user", "content": "What is the refund limit?"}]), first)

    def test_replay_rows_are_distilled_from_the_base(self):
        rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "tools": [],
                 "weight": 1, "case": f"replay-{i}", "rule": None, "origin": "replay"}
                for i, (q, a) in enumerate(FACTS.items())]
        dataset = self.root / "replay.jsonl"
        dataset.write_text("".join(json.dumps(r) + "\n" for r in rows))

        def first_loss(*extra):
            done = subprocess.run([sys.executable, "-m", "graft.train_hf", "--dataset", str(dataset), "--base",
                                   str(self.root / "base"), "--output", str(self.root / ("kd" + "".join(extra))),
                                   "--epochs", "1", "--batch-size", "1", "--lr", "0.0", *extra],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            line = next(line for line in done.stdout.splitlines() if line.startswith("epoch"))
            return float(line.split("loss ")[1].split()[0])

        # A new adapter is zero, so the adapted model is the base: kl only sees the student's dropout. ce on the
        # reply's tokens pulls towards one reply instead, and is far larger.
        kl, ce = first_loss(), first_loss("--replay-loss", "ce")
        self.assertGreater(ce, 0.1)
        self.assertLess(kl, ce / 20, "kl against the base starts near zero")

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

    def test_template_args_reach_the_template_in_training_and_inference(self):
        from transformers import AutoTokenizer

        from graft import runners, train_hf
        folder = self.root / "thinking"
        tiny_model(folder, THINKING)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        row = {"messages": [{"role": "user", "content": "What is the refund limit?"},
                            {"role": "assistant", "content": "500 dollars"}]}

        def learned(**kwargs):
            _, labels = train_hf.encode(tokenizer, row, 64, **kwargs)
            return tokenizer.decode([token for token in labels if token != -100]).replace(" ", "")

        self.assertEqual(learned(), "500dollars<|im_end|>")
        # the marker opens the reply in both renderings, so it is context and not learned
        def reply(ids):
            return tokenizer.decode(ids).split("assistant")[-1].replace(" ", "")

        self.assertEqual(reply(train_hf.encode(tokenizer, row, 64)[0]), "500dollars<|im_end|>")
        self.assertEqual(reply(train_hf.encode(tokenizer, row, 64, template_args={"enable_thinking": False})[0]),
                         "?500dollars<|im_end|>")
        self.assertEqual(learned(template_args={"enable_thinking": False}), "500dollars<|im_end|>")

        spec = {"path": str(folder), "max_new_tokens": 1, "device": "cpu", "template_args": {"enable_thinking": False}}
        runner = runners.build("tiny", spec, self.root)
        self.assertEqual(runner.fingerprint()["template_args"], {"enable_thinking": False})
        self.assertNotIn("template_args", runners.build("tiny", {"path": str(folder)}, self.root).fingerprint())
        _, loaded, _ = runner._load()
        render, prompts = loaded.apply_chat_template, []
        loaded.apply_chat_template = lambda *a, **k: prompts.append(render(*a, **k)) or prompts[-1]
        runner.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(prompts[0].endswith("assistant\n?"), prompts[0])


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

            # the trained weights go into a GGUF export (llama.cpp's converter stood in for), merged, not as an adapter
            from tests.test_export import FAKE_CONVERT
            (root / "llama.cpp").mkdir()
            (root / "llama.cpp" / "convert_hf_to_gguf.py").write_text(FAKE_CONVERT)
            code, out, err = graft("export", "--format", "gguf", "--base-model", "base", "--llama-cpp", "llama.cpp",
                                   "--out", "tiny.zip", cwd=root)
            self.assertEqual(code, 0, out + err)
            from zipfile import ZipFile
            with ZipFile(root / "tiny.zip") as z:
                self.assertNotIn("adapter/config.json", z.namelist())
                manifest = json.loads(z.read("manifest.json"))
                converted = json.loads(z.read("model-Q8_0.gguf"))
            self.assertTrue(manifest["gguf"]["adapter_merged"])
            self.assertTrue(converted["from"].endswith("merged"), "converted from the merged weights")


if __name__ == "__main__":
    unittest.main()


class Weights(unittest.TestCase):
    """How a row's weight becomes training copies. Needs no torch."""

    def test_whole_and_fractional_weights(self):
        from graft.train_hf import copies
        self.assertEqual(copies(1), [1.0])
        self.assertEqual(copies(2), [1.0, 1.0], "a whole weight is unchanged: that many full copies")
        self.assertEqual(copies(2.5), [1.0, 1.0, 0.5])
        self.assertEqual(copies(0.5), [0.5], "a weight below one is not rounded up to a full copy")
        self.assertEqual(copies(0), [1.0])

    def test_targets(self):
        from graft.train_hf import targets
        self.assertEqual(targets("all-linear"), "all-linear")
        self.assertEqual(targets(""), "all-linear")
        self.assertEqual(targets("q_proj, k_proj,v_proj"), ["q_proj", "k_proj", "v_proj"])
