import json
import tempfile
import unittest
from pathlib import Path

from graft import genome
from tests.support import EXAMPLE


def case(n, **extra):
    return {"id": f"c{n}", "messages": [{"role": "user", "content": f"q{n}"}],
            "verify": [{"type": "contains", "value": "a"}], **extra}


class Folder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "genome"
        genome.write_rule(self.folder, "polite", "Always say please.")

    def cases(self, rows, rule="polite"):
        (self.folder / "cases" / f"{rule}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_load(self):
        self.cases([case(n) for n in range(10)])
        found = genome.load(self.folder, 0.2)
        self.assertEqual(len(found.cases), 10)
        self.assertEqual(found.rules["polite"].statement, "Always say please.")
        self.assertEqual(sum(c.split == "holdout" for c in found.cases), 2)

    def test_splits_are_stable_and_overridable(self):
        self.cases([case(n) for n in range(10)])
        first = {c.id: c.split for c in genome.load(self.folder, 0.3).cases}
        self.assertEqual(first, {c.id: c.split for c in genome.load(self.folder, 0.3).cases})
        self.assertEqual(sum(s == "holdout" for s in first.values()), 3)
        train = next(c for c, s in first.items() if s == "train")
        self.cases([case(n, **({"split": "holdout"} if f"c{n}" == train else {})) for n in range(10)])
        again = {c.id: c.split for c in genome.load(self.folder, 0.3).cases}
        self.assertEqual(again[train], "holdout")

    def test_small_rules_and_zero_holdout(self):
        self.cases([case(n) for n in range(3)])
        self.assertEqual({c.split for c in genome.load(self.folder, 0.1).cases}, {"train"})
        self.cases([case(n) for n in range(4)])
        self.assertEqual(sum(c.split == "holdout" for c in genome.load(self.folder, 0.1).cases), 1)
        self.assertEqual({c.split for c in genome.load(self.folder, 0.0).cases}, {"train"})

    def test_id_covers_every_file(self):
        self.cases([case(1)])
        before = genome.load(self.folder).id
        self.assertEqual(before, genome.load(self.folder).id)
        (self.folder / "verifiers").mkdir()
        (self.folder / "verifiers" / "v.py").write_text("print(1)\n")
        changed = genome.load(self.folder).id
        self.assertNotEqual(before, changed)
        self.cases([case(1), case(2)])
        self.assertNotEqual(changed, genome.load(self.folder).id)

    def test_pairs_and_rule_verifiers(self):
        (self.folder / "rules" / "polite.toml").write_text(
            'statement = "Say please."\ngate = 0.9\n[[verify]]\ntype = "contains"\nvalue = "please"\n')
        side = {"messages": [{"role": "user", "content": "x"}]}
        self.cases([{"id": "p", "pair": [side, {**side, "verify": [{"type": "no_tool_call"}]}]},
                    {"messages": [{"role": "user", "content": "y"}], "reference": "please"}])
        found = genome.load(self.folder)
        pair, plain = found.case("p"), [c for c in found.cases if c.id != "p"][0]
        self.assertTrue(pair.pair)
        self.assertEqual([len(u.verify) for u in pair.units], [1, 2])
        self.assertEqual(plain.units[0].reference, {"content": "please"})
        self.assertEqual(len(plain.id), 12)
        self.assertEqual(found.rules["polite"].gate, 0.9)

    def test_errors_name_the_line(self):
        bad = [("{not json", "not JSON"), (json.dumps({"messages": []}), "non-empty"),
               (json.dumps({"messages": [{"role": "user", "content": "x"}]}), "nothing verifies"),
               (json.dumps({"id": "a b", "messages": [{"role": "user"}], "verify": [{"type": "no_tool_call"}]}),
                "case id"),
               (json.dumps({"pair": [{}]}), "exactly two")]
        for line, why in bad:
            (self.folder / "cases" / "polite.jsonl").write_text(line + "\n")
            with self.assertRaises(genome.GenomeError) as caught:
                genome.load(self.folder)
            self.assertIn(why, str(caught.exception))
            self.assertIn("polite.jsonl:1", str(caught.exception))

    def test_duplicate_ids_and_orphan_cases(self):
        genome.write_rule(self.folder, "other", "Another rule.")
        self.cases([case(1)])
        self.cases([case(1)], rule="other")
        with self.assertRaisesRegex(genome.GenomeError, "used in both"):
            genome.load(self.folder)
        (self.folder / "cases" / "ghost.jsonl").write_text(json.dumps(case(9)) + "\n")
        (self.folder / "cases" / "other.jsonl").unlink()
        with self.assertRaisesRegex(genome.GenomeError, "no rule 'ghost'"):
            genome.load(self.folder)

    def test_write_rule_and_append(self):
        with self.assertRaisesRegex(genome.GenomeError, "already exists"):
            genome.write_rule(self.folder, "polite", "again")
        with self.assertRaises(genome.GenomeError):
            genome.write_rule(self.folder, "bad id", "x")
        genome.write_rule(self.folder, "quoted", 'Say "please" and \\ thanks.', gate=0.5)
        self.assertEqual(genome.load(self.folder).rules["quoted"].statement, 'Say "please" and \\ thanks.')
        self.assertEqual(genome.append_cases(self.folder, "polite", [case(1), case(2)]), 2)
        self.assertEqual(genome.append_cases(self.folder, "polite", [case(2), case(3)]), 1)
        self.assertEqual(len(genome.load(self.folder).cases), 3)


class Example(unittest.TestCase):
    def test_example_genome_loads(self):
        found = genome.load(EXAMPLE / "genome", 0.2)
        self.assertEqual(sorted(found.rules), ["account-closure", "card-numbers", "no-legal-advice",
                                               "refund-limit", "ticket-json"])
        for rule in found.rules.values():
            self.assertGreaterEqual(sum(c.split == "holdout" for c in rule.cases), 1, rule.id)
            self.assertTrue(all(u.reference for c in rule.cases for u in c.units))
        suite = genome.load_suite(EXAMPLE / "genome" / "suites" / "arithmetic.jsonl")
        self.assertEqual({c.split for c in suite}, {"holdout"})


if __name__ == "__main__":
    unittest.main()
