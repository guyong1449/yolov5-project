"""Tests for scripts/run_with_log.py."""

import importlib.util
import io
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_with_log.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_with_log", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunWithLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_echo_writes_log_and_markdown(self):
        log_dir = REPO_ROOT / "runs" / "logs" / "_pytest_run_with_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "echo_test.log"
        md_file = log_dir / "echo_test.md"
        if log_file.exists():
            log_file.unlink()
        if md_file.exists():
            md_file.unlink()

        code = self.mod.run_with_log(
            [sys.executable, "-c", "print('hello-run-with-log')"],
            log_file=log_file,
            md_file=md_file,
            realtime=False,
            cwd=REPO_ROOT,
        )
        self.assertEqual(code, 0)
        log_text = log_file.read_text(encoding="utf-8")
        md_text = md_file.read_text(encoding="utf-8")
        self.assertIn("hello-run-with-log", log_text)
        self.assertIn("hello-run-with-log", md_text)
        self.assertIn("```text", md_text)
        self.assertIn("Return code: `0`", md_text)

    def test_realtime_output_tolerates_non_utf8_console(self):
        log_dir = REPO_ROOT / "runs" / "logs" / "_pytest_run_with_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "replacement_char_test.log"
        md_file = log_dir / "replacement_char_test.md"
        if log_file.exists():
            log_file.unlink()
        if md_file.exists():
            md_file.unlink()

        old_stdout = sys.stdout
        fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp936", errors="strict")
        sys.stdout = fake_stdout
        try:
            code = self.mod.run_with_log(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'\\xef\\xbf\\xbd\\n'); sys.stdout.flush()",
                ],
                log_file=log_file,
                md_file=md_file,
                realtime=True,
                cwd=REPO_ROOT,
            )
        finally:
            sys.stdout = old_stdout
            fake_stdout.close()

        self.assertEqual(code, 0)
        self.assertIn("\ufffd", log_file.read_text(encoding="utf-8"))

    def test_train_command_defaults_logs_into_train_run_dir(self):
        log_file, md_file = self.mod._resolve_paths(
            None,
            None,
            "_pytest_train_run_with_log",
            command=[
                sys.executable,
                "train.py",
                "--project",
                "runs/train",
                "--name",
                "_pytest_train_run_with_log",
            ],
        )
        self.assertEqual(log_file, REPO_ROOT / "runs" / "train" / "_pytest_train_run_with_log" / "run.log")
        self.assertEqual(md_file, REPO_ROOT / "runs" / "train" / "_pytest_train_run_with_log" / "run.md")

    def test_explicit_log_and_md_override_train_defaults(self):
        custom_dir = REPO_ROOT / "runs" / "logs" / "_pytest_run_with_log"
        custom_dir.mkdir(parents=True, exist_ok=True)
        log_file = custom_dir / "override_train.log"
        md_file = custom_dir / "override_train.md"
        if log_file.exists():
            log_file.unlink()
        if md_file.exists():
            md_file.unlink()

        code = self.mod.run_with_log(
            [
                sys.executable,
                "-c",
                "print('custom-train-log-target')",
            ],
            log_file=log_file,
            md_file=md_file,
            realtime=False,
            cwd=REPO_ROOT,
        )
        self.assertEqual(code, 0)
        self.assertTrue(log_file.exists())
        self.assertTrue(md_file.exists())


if __name__ == "__main__":
    unittest.main()
