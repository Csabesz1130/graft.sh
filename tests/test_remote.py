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


def result(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeModal:
    """Stands in for the parts of the Modal client --fetch and --cancel use."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.removed, self.cancelled = dict(files), [], []
        fake = self

        class Volume:
            @staticmethod
            def from_name(name):
                assert name == "graft-runs", name
                return Volume()

            def read_file(self, path):
                if path not in fake.files:
                    raise FileNotFoundError(path)
                yield fake.files[path]

            def remove_file(self, path, recursive=False):
                fake.removed.append(path)
                fake.files.pop(path, None)

        class FunctionCall:
            @staticmethod
            def from_id(call_id):
                class Call:
                    def cancel(self, terminate_containers=False):
                        fake.cancelled.append((call_id, terminate_containers))
                return Call()

        self.module = type(sys)("modal")
        self.module.Volume, self.module.FunctionCall = Volume, FunctionCall

    def __enter__(self):
        self.saved = sys.modules.get("modal")
        sys.modules["modal"] = self.module
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = self.saved


class Fetch(unittest.TestCase):
    """--fetch and --cancel against a stand-in for Modal: what comes back, what is refused, what is cleaned up."""

    setUp = Packing.setUp

    def start(self, **extra):
        """As run() leaves it: the pending record, with what was uploaded."""
        _, sent = remote.pack(self.project)
        record = {"id": "proj-1", "volume": "graft-runs", "app": "ap-123", "call": "fc-456", "genome": "g1",
                  "sent": sent, **extra}
        write(self.root / ".graft" / remote.PENDING, json.dumps(record))
        return record

    def bundle(self, files: dict[str, bytes], code: int = 0, deleted=(), **status) -> dict[str, bytes]:
        changes = result(files)
        body = {"code": code, "deleted": list(deleted), "sha256": __import__("hashlib").sha256(changes).hexdigest(),
                "files": remote.manifest(changes), **status}
        return {"proj-1/result.json": json.dumps(body).encode(), "proj-1/result.tar.gz": changes}

    NEW = ".graft/grafts/t/0009-dddd/state.json"

    def test_plain_fetch_uses_the_run_started_here_and_cleans_up(self):
        with FakeModal({}), self.assertRaises(remote.RemoteError):
            remote.fetch(self.project, None)                  # nothing started here, nothing named
        self.start()
        with FakeModal({}) as fake, self.assertRaises(remote.RemoteError) as caught:
            remote.fetch(self.project, None)                  # not finished yet
        self.assertIn("still running", str(caught.exception))
        self.assertIn("ap-123", str(caught.exception))
        with FakeModal(self.bundle({self.NEW: b'{"status": "passed"}'})) as fake:
            self.assertEqual(remote.fetch(self.project, None, "g1"), 0)
        self.assertTrue((self.root / self.NEW).is_file())
        self.assertFalse((self.root / ".graft" / remote.PENDING).exists(), "nothing pending any more")
        self.assertEqual(sorted(fake.removed), ["proj-1/result.json", "proj-1/result.tar.gz"],
                         "the bundle leaves the volume once the files here match")

    def test_keep_remote(self):
        self.start()
        with FakeModal(self.bundle({self.NEW: b"{}"})) as fake:
            remote.fetch(self.project, None, keep_remote=True)
        self.assertEqual(fake.removed, [])

    def test_edits_made_here_while_it_ran_are_never_overwritten(self):
        self.start()
        (self.root / "graft.toml").write_text("# edited while the job ran\n")
        (self.root / "genome" / "rules" / "r.toml").write_text('statement = "edited"\n')
        # a result that tries to write outside .graft is refused whole, and nothing is cleaned up
        with FakeModal(self.bundle({self.NEW: b"{}", "graft.toml": b"from modal"})) as fake:
            with self.assertRaises(remote.RemoteError):
                remote.fetch(self.project, None)
        self.assertEqual((self.root / "graft.toml").read_text(), "# edited while the job ran\n")
        self.assertFalse((self.root / self.NEW).exists(), "nothing was written")
        self.assertEqual(fake.removed, [], "a refused result stays on Modal")
        # a file under .graft changed here since the upload is not overwritten either
        state = self.done / "state.json"
        state.write_text(json.dumps({"target": "t", "run": "0001-aaaa", "status": "passed", "note": "mine"}))
        with FakeModal(self.bundle({".graft/grafts/t/0001-aaaa/state.json": b"{}"})):
            with self.assertRaises(remote.RemoteError) as caught:
                remote.fetch(self.project, None)
        self.assertIn("changed here", str(caught.exception))
        self.assertIn('"mine"', state.read_text())
        # the edited genome is untouched by a good fetch, and the user is told the run checked the old one
        with FakeModal(self.bundle({self.NEW: b"{}"})):
            out = io.StringIO()
            with __import__("contextlib").redirect_stderr(out):
                remote.fetch(self.project, None, "g2")
        self.assertIn("genome changed", out.getvalue())
        self.assertEqual((self.root / "genome" / "rules" / "r.toml").read_text(), 'statement = "edited"\n')

    def test_a_bad_path_anywhere_means_nothing_is_written(self):
        self.start()
        with FakeModal(self.bundle({self.NEW: b"{}", "../evil": b"x"})) as fake:
            with self.assertRaises(remote.RemoteError):
                remote.fetch(self.project, None)
        self.assertFalse((self.root / self.NEW).exists())
        self.assertFalse((self.root.parent / "evil").exists())
        self.assertEqual(fake.removed, [])

    def test_a_damaged_result_is_refused(self):
        self.start()
        files = self.bundle({self.NEW: b"{}"})
        files["proj-1/result.tar.gz"] = result({self.NEW: b"tampered"})
        with FakeModal(files) as fake, self.assertRaises(remote.RemoteError) as caught:
            remote.fetch(self.project, None)
        self.assertIn("checksum", str(caught.exception))
        self.assertFalse((self.root / self.NEW).exists())
        self.assertEqual(fake.removed, [])

    def test_cancel(self):
        with FakeModal({}), self.assertRaises(remote.RemoteError):
            remote.cancel(self.project)                       # nothing running
        self.start()
        with FakeModal({}) as fake:
            self.assertEqual(remote.cancel(self.project), 0)
        self.assertEqual(fake.cancelled, [("fc-456", True)], "the containers stop, so the GPU stops")
        self.assertFalse((self.root / ".graft" / remote.PENDING).exists())

    def test_the_detach_message_says_what_runs_and_how_to_stop_it(self):
        lines = remote.detached_message({"id": "proj-1", "app": "ap-123"}).splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("Still running", lines[0])
        self.assertIn("billed", lines[0])
        self.assertIn("graft rebase --fetch proj-1", lines[1])
        self.assertTrue(all(len(line) <= 92 for line in lines), "fits a terminal without wrapping")
        self.assertIn("graft rebase --cancel", lines[2])
        self.assertIn("modal app stop ap-123", lines[2])

    def test_the_pending_record_never_goes_up(self):
        self.start()
        archive, sent = remote.pack(self.project)
        self.assertNotIn(f".graft/{remote.PENDING}", sent)


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
