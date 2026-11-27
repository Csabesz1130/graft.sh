"""--runner modal without Modal: what goes up, what comes back, and the refusals. The Modal run itself is
checked on Modal (scripts/ and the README say how)."""
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from graft import config, rebase, remote
from tests.support import ExampleCase


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class Packing(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "proj"
        write(self.root / "graft.toml", "[models.t]\npath = \"weights\"\n")
        write(self.root / "genome" / "rules" / "r.toml", 'statement = "s"\n')
        write(self.root / "weights" / "config.json", "{}")
        write(self.root / ".git" / "HEAD")
        write(self.root / "old-release.zip")
        self.done = self.root / ".graft" / "grafts" / "t" / "0001-aaaa"
        write(self.done / "state.json", json.dumps({"target": "t", "run": "0001-aaaa", "status": "passed"}))
        write(self.done / "lora" / "round-1" / "adapter" / "adapter_model.safetensors", "finished run's weights")
        self.project = config.Project(self.root, {"models": {"t": {"path": "weights"}}})

    def names(self, archive: bytes) -> set[str]:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            return set(tar.getnames())

    def test_what_goes_up(self):
        archive, sent = remote.pack(self.project)
        names = self.names(archive)
        self.assertIn("graft.toml", names)
        self.assertIn("weights/config.json", names, "local weights named by path go with it")
        self.assertIn(".graft/grafts/t/0001-aaaa/state.json", names)
        self.assertNotIn(".graft/grafts/t/0001-aaaa/lora/round-1/adapter/adapter_model.safetensors", names,
                         "a finished run's adapters stay here")
        self.assertFalse({n for n in names if n.startswith(".git/") or n.endswith(".zip")})
        self.assertEqual(set(sent), {n for n in names if n.startswith(".graft/")})

    def test_a_run_in_progress_takes_its_adapters_along(self):
        current = self.root / ".graft" / "grafts" / "t" / "0002-bbbb"
        write(current / "state.json", json.dumps({"target": "t", "run": "0002-bbbb", "status": "conflict"}))
        write(current / "lora" / "round-1" / "adapter" / "w.safetensors")
        write(rebase.head_path(self.project), "t/0002-bbbb")
        names = self.names(remote.pack(self.project)[0])
        self.assertIn(".graft/grafts/t/0002-bbbb/lora/round-1/adapter/w.safetensors", names)

    def test_round_trip_brings_back_changes_the_kept_adapter_and_deletions(self):
        write(rebase.head_path(self.project), "t/0001-aaaa")
        archive, sent = remote.pack(self.project)
        # the remote side: unpack, restore exact times, then a run changes things
        with tempfile.TemporaryDirectory() as far:
            work = Path(far)
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                tar.extractall(work, filter="data")
            remote.restore_times(work, sent)
            new = work / ".graft" / "grafts" / "t" / "0003-cccc"
            write(new / "state.json", json.dumps({"target": "t", "run": "0003-cccc", "status": "passed",
                                                  "graft": {"kind": "lora", "adapter":
                                                            ".graft/grafts/t/0003-cccc/lora/round-2/adapter"}}))
            write(new / "lora" / "round-1" / "adapter" / "w.safetensors", "worse round")
            write(new / "lora" / "round-2" / "adapter" / "w.safetensors", "kept round")
            write(new / "lora" / "round-1" / "dataset.jsonl", "{}")
            (work / ".graft" / rebase.HEAD).unlink()      # the rebase finished
            changes, deleted = remote.collect(work, sent)
        names = self.names(changes)
        self.assertEqual(names, {".graft/grafts/t/0003-cccc/state.json",
                                 ".graft/grafts/t/0003-cccc/lora/round-1/dataset.jsonl",
                                 ".graft/grafts/t/0003-cccc/lora/round-2/adapter/w.safetensors"},
                         "unchanged files stay; of the adapters only the kept one comes back")
        self.assertEqual(deleted, [f".graft/{rebase.HEAD}"])
        self.assertEqual(remote.unpack(self.project, changes, deleted), 3)
        self.assertFalse(rebase.head_path(self.project).exists())
        self.assertEqual((self.root / ".graft/grafts/t/0003-cccc/lora/round-2/adapter/w.safetensors").read_text(),
                         "kept round")

    def test_unpack_refuses_paths_outside_graft(self):
        for name in ("../evil", "/etc/passwd", "genome/rules/r.toml", ".graft/../graft.toml"):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                info = tarfile.TarInfo(name)
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(remote.RemoteError):
                remote.unpack(self.project, buffer.getvalue(), [])
        with self.assertRaises(remote.RemoteError):
            remote.unpack(self.project, remote.collect(self.root, {})[0], ["graft.toml"])

    def test_settings(self):
        self.assertEqual(remote.settings(self.project, None)["gpu"], "L4")
        self.assertEqual(remote.settings(self.project, "none")["gpu"], "none")
        project = config.Project(self.root, {"modal": {"gpu": "A100-80GB", "secrets": ["huggingface"]}})
        self.assertEqual(remote.settings(project, None)["secrets"], ["huggingface"])
        for bad in ({"gpu": "RTX9000"}, {"secrets": "huggingface"}, {"packages": [1]}):
            with self.assertRaises(remote.RemoteError):
                remote.settings(config.Project(self.root, {"modal": bad}), None)


class Fetch(unittest.TestCase):
    """--fetch with a stand-in for Modal's volume API: the result of a run nobody watched comes back."""

    setUp = Packing.setUp

    def fake_modal(self, files: dict[str, bytes]):
        class Volume:
            @staticmethod
            def from_name(name):
                self.assertEqual(name, "graft-runs")
                return Volume()

            def read_file(self, path):
                if path not in files:
                    raise FileNotFoundError(path)
                yield files[path]
        return type(sys)("modal"), Volume

    def run_fetch(self, files, wanted=None):
        module, volume = self.fake_modal(files)
        module.Volume = volume
        saved = sys.modules.get("modal")
        sys.modules["modal"] = module
        try:
            return remote.fetch(self.project, wanted)
        finally:
            if saved is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = saved

    def test_fetch_the_run_started_here(self):
        with self.assertRaises(remote.RemoteError):
            self.run_fetch({})                       # nothing started here, nothing named
        write(self.root / ".graft" / remote.PENDING, json.dumps({"id": "proj-1", "volume": "graft-runs"}))
        with self.assertRaises(remote.RemoteError) as caught:
            self.run_fetch({})                       # not finished yet
        self.assertIn("still running", str(caught.exception))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            data = b'{"target": "t", "run": "0009-dddd", "status": "passed"}'
            info = tarfile.TarInfo(".graft/grafts/t/0009-dddd/state.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        code = self.run_fetch({"proj-1/result.json": b'{"code": 0, "deleted": []}',
                               "proj-1/result.tar.gz": buffer.getvalue()})
        self.assertEqual(code, 0)
        self.assertTrue((self.root / ".graft/grafts/t/0009-dddd/state.json").is_file())
        self.assertFalse((self.root / ".graft" / remote.PENDING).exists(), "fetched: nothing pending any more")
        with self.assertRaises(remote.RemoteError):
            self.run_fetch({}, "../elsewhere")


class Cli(ExampleCase):
    def test_a_localhost_endpoint_is_refused_before_anything_is_uploaded(self):
        code, _, err = self.graft("rebase", "--target", "new-base", "--runner", "modal")
        self.assertEqual(code, 1)
        self.assertIn("cannot reach", err)
        self.assertFalse((self.root / ".graft").exists(), "nothing started")

    def test_without_the_modal_client(self):
        self.edit(lambda text: text.replace("127.0.0.1", "example.invalid"))
        saved = sys.modules.get("modal")
        sys.modules["modal"] = None      # import modal now fails, as without graft-sh[modal]
        try:
            code, _, err = self.graft("rebase", "--target", "new-base", "--runner", "modal")
        finally:
            if saved is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = saved
        self.assertEqual(code, 1)
        self.assertIn("graft-sh[modal]", err)

    def test_gpu_needs_the_modal_runner(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base", "--gpu", "L4")[0], 2)

    def test_abort_stays_local(self):
        self.assertEqual(self.graft("rebase", "--abort", "--runner", "modal")[1].strip(), "no rebase is in progress")


if __name__ == "__main__":
    unittest.main()
