from tests.support import ExampleCase


class VerifyAgainst(ExampleCase):
    """graft verify --against: does a deployment still behave as its certificate says?"""

    def add_model(self, name: str, model: str) -> None:
        self.edit(lambda text: text + f'\n[models.{name}]\nendpoint = "{self.endpoint}"\nmodel = "{model}"\n')

    def test_a_served_adapter_passes_and_the_bare_base_does_not(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        self.add_model("served", "new-base-graft")     # the endpoint serving the trained adapter
        code, out, err = self.graft("verify", "served", "--against", "new-base")
        self.assertEqual(code, 0, out + err)
        self.assertIn("The deployment behaves as certified.", out)

        code, out, _ = self.graft("verify", "new-base", "--against", "new-base")
        self.assertEqual(code, 1, "the base without its adapter is not the certified deployment")
        self.assertIn("FAILS", out)
        self.assertIn("does not behave as certified", out)

        # the same check from an export bundle, as a deploy pipeline would run it
        self.assertEqual(self.graft("export", "--out", "release.zip")[0], 0)
        self.assertEqual(self.graft("verify", "served", "--against", "release.zip")[0], 0)
        code, out, _ = self.graft("verify", "new-base", "--against", "release.zip", "--json")
        self.assertEqual(code, 1)
        self.assertIn('"ok": false', out)

        # several passes: every case has to hold in each one, and the failures say how often they failed
        code, out, err = self.graft("verify", "served", "--against", "new-base", "--passes", "2")
        self.assertEqual(code, 0, out + err)
        self.assertRegex(out, r"passes: (\d+)/(\d+), \1/\2\. \1/\2 of the certified cases pass in every one")
        code, out, _ = self.graft("verify", "new-base", "--against", "new-base", "--passes", "2", "--json")
        self.assertEqual(code, 1)
        report = __import__("json").loads(out)
        self.assertEqual(len(report["passes"]), 2)
        self.assertTrue(all(f["failed_in"] == 2 for f in report["failures"] if f["rule"] != "account-closure"))
        self.assertEqual(self.graft("verify", "served", "--against", "new-base", "--passes", "0")[0], 1)

    def test_a_prompt_graft_is_sent_by_the_client(self):
        # without the one rule the stand-in only learns by training, the prompt rung is enough
        for part in ("rules/account-closure.toml", "cases/account-closure.jsonl"):
            (self.root / "genome" / part).unlink()
        code, out, err = self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot,prompt")
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.graft("show", "--json")[0], 0)
        code, out, err = self.graft("verify", "new-base", "--against", "new-base")
        self.assertEqual(code, 0, out + err)
        self.assertIn("prompt graft", out)
        self.assertEqual(self.graft("export", "--out", "p.zip")[0], 0)
        self.assertEqual(self.graft("verify", "new-base", "--against", "p.zip")[0], 0, "system.txt from the bundle")

    def test_refuses_a_changed_genome_or_mixed_options(self):
        self.assertEqual(self.graft("rebase", "--target", "prod")[0], 0)
        self.assertEqual(self.graft("verify", "prod", "--against", "prod")[0], 0)
        self.assertEqual(self.graft("verify", "prod", "--against", "prod", "--graft")[0], 1)
        cases = next((self.root / "genome" / "cases").glob("*.jsonl"))
        cases.write_text(cases.read_text() + cases.read_text().splitlines()[0].replace('"id": "', '"id": "copy-')
                         + "\n")
        code, _, err = self.graft("verify", "prod", "--against", "prod")
        self.assertEqual(code, 1)
        self.assertIn("genome changed", err)
