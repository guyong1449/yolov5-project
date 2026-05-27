import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class GuiPanelTaskSpecTests(unittest.TestCase):
    def test_task_definitions_cover_supported_tasks(self):
        from tools.gui_panel.task_specs import TASK_SPECS

        self.assertEqual(set(TASK_SPECS), {"train", "detect", "val", "fiftyone"})
        self.assertEqual(TASK_SPECS["train"].display_name, "Train")
        self.assertEqual(TASK_SPECS["fiftyone"].defaults["mode"], "dataset_root_auto")

    def test_detect_defaults_match_voc_stride10_workflow(self):
        from tools.gui_panel.task_specs import TASK_SPECS

        defaults = TASK_SPECS["detect"].defaults
        self.assertEqual(defaults["name"], "voc_stride10")
        self.assertTrue(defaults["save_img_frames"])
        self.assertEqual(defaults["vid_stride"], 10)
        self.assertTrue(defaults["incremental_mp4"])
        self.assertTrue(defaults["nosave"])
        self.assertEqual(defaults["voc_root"], "F:/1/labelimg/data/test1_stride10")
        self.assertEqual(defaults["conf_thres"], 0.25)
        self.assertEqual(defaults["iou_thres"], 0.45)


class GuiPanelCommandBuilderTests(unittest.TestCase):
    def test_train_command_wraps_run_with_log(self):
        from tools.gui_panel.services.command_builder import build_command_preview

        command = build_command_preview(
            "train",
            {
                "data": "data/dataAirVis.yaml",
                "weights": "checkpoint/yolov5_best.pt",
                "project": "runs/train",
                "name": "panel_train",
                "device": "0",
                "epochs": 10,
                "batch_size": 4,
                "imgsz": 640,
                "workers": 2,
                "seed": 0,
                "patience": 20,
                "optimizer": "SGD",
                "exist_ok": False,
                "amp_mode": "off",
                "resume": False,
                "hyp": "",
                "cfg": "",
                "extra_args": "",
            },
        )

        self.assertEqual(command[0], sys.executable)
        self.assertIn("scripts/run_with_log.py", command)
        self.assertIn("train.py", command)
        self.assertIn("--device", command)
        self.assertIn("--seed", command)
        self.assertIn("--no-amp", command)
        self.assertIn("--hyp", command)
        self.assertIn("runs/train/test1_stride10_sgd_70e3/hyp.yaml", command)

    def test_fiftyone_auto_mode_resolves_dataset_root(self):
        from tools.gui_panel.services.command_builder import build_command_preview

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fiftyone_voc" / "data").mkdir(parents=True)
            (root / "fiftyone_voc" / "labels").mkdir(parents=True)
            command = build_command_preview(
                "fiftyone",
                {
                    "mode": "dataset_root_auto",
                    "dataset_name": "demo",
                    "dataset_root": str(root),
                    "overwrite": True,
                    "label_field": "ground_truth",
                    "launch_app": True,
                },
            )

        self.assertIn("tools/fiftyone/fiftyone_import_voc.py", command)
        self.assertIn("--data-dir", command)
        self.assertIn("--labels-dir", command)
        self.assertNotIn("--no-app", command)
        self.assertIn("--wait", command)


class GuiPanelResolverTests(unittest.TestCase):
    def test_resolve_prefers_fiftyone_voc_layout(self):
        from tools.gui_panel.services.fiftyone_resolver import resolve_dataset_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary_data = root / "fiftyone_voc" / "data"
            primary_labels = root / "fiftyone_voc" / "labels"
            fallback_data = root / "images"
            fallback_labels = root / "annotations"
            primary_data.mkdir(parents=True)
            primary_labels.mkdir(parents=True)
            fallback_data.mkdir()
            fallback_labels.mkdir()

            resolved = resolve_dataset_root(root)

        self.assertEqual(resolved.data_dir, primary_data.resolve())
        self.assertEqual(resolved.labels_dir, primary_labels.resolve())


class GuiPanelRunnerTests(unittest.TestCase):
    def test_single_task_runner_rejects_parallel_start(self):
        from tools.gui_panel.schemas import RunRequest
        from tools.gui_panel.services.process_runner import ProcessRunner

        runner = ProcessRunner()
        request = RunRequest(
            task_type="train",
            values={
                "data": "data/dataAirVis.yaml",
                "weights": "checkpoint/yolov5_best.pt",
                "project": "runs/train",
                "name": "runner_test",
                "device": "cpu",
                "epochs": 1,
                "batch_size": 1,
                "imgsz": 640,
                "workers": 0,
                "seed": 0,
                "patience": 1,
                "optimizer": "SGD",
                "exist_ok": True,
                "amp_mode": "off",
                "resume": False,
                "hyp": "",
                "cfg": "",
                "extra_args": "",
            },
            command_override=[sys.executable, "-c", "import time; print('hello'); time.sleep(1)"],
        )
        try:
            runner.start(request)
            with self.assertRaises(RuntimeError):
                runner.start(request)
        finally:
            runner.stop()

    def test_log_stream_yields_without_condition_lock_error(self):
        from tools.gui_panel.services.process_runner import ProcessRunner

        runner = ProcessRunner()
        runner._append_log("stream-line")  # noqa: SLF001 - regression test for SSE handoff

        stream = runner.log_stream()
        with ThreadPoolExecutor(max_workers=2) as pool:
            event_id, line = pool.submit(next, stream).result()
            heartbeat_event = pool.submit(
                lambda iterator: next(iterator),
                runner.log_stream(since=1, heartbeat_seconds=0.01),
            ).result()

        self.assertEqual(event_id, 1)
        self.assertEqual(line, "stream-line")
        self.assertEqual(heartbeat_event, (1, None))


class GuiPanelApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fastapi_app_exposes_definitions_and_validation(self):
        from fastapi.testclient import TestClient
        from tools.gui_panel.app import create_app

        app = create_app(session_file=self.tmp / "session.json")
        client = TestClient(app)

        response = client.get("/api/task-definitions")
        self.assertEqual(response.status_code, 200)
        self.assertIn("train", response.json()["tasks"])

        validate_response = client.post(
            "/api/tasks/train/validate",
            json={
                "task_type": "train",
                "values": {
                    "data": str(ROOT / "data" / "dataAirVis.yaml"),
                    "weights": str(ROOT / "checkpoint" / "yolov5_best.pt"),
                    "project": "runs/train",
                    "name": "api_validate",
                    "device": "0",
                    "epochs": 1,
                    "batch_size": 1,
                    "imgsz": 640,
                    "workers": 0,
                    "seed": 0,
                    "patience": 1,
                    "optimizer": "SGD",
                    "exist_ok": True,
                    "amp_mode": "off",
                    "resume": False,
                    "hyp": "",
                    "cfg": "",
                    "extra_args": "",
                },
            },
        )
        self.assertEqual(validate_response.status_code, 200)
        self.assertTrue(validate_response.json()["ok"])
        self.assertIn("train.py", validate_response.json()["command"])


class GuiPanelLauncherTests(unittest.TestCase):
    def test_repo_launcher_uses_pwsh7_and_opens_panel_url(self):
        launcher = (ROOT / "tools" / "gui_panel" / "start_gui_panel.cmd").read_text(encoding="utf-8")

        self.assertIn(r"C:\Program Files\PowerShell\7\pwsh.exe", launcher)
        self.assertIn("tools/gui_panel/start_gui_panel.py", launcher)
        self.assertIn("http://127.0.0.1:8752/", launcher)

    def test_path_dialog_prefers_powershell_sta_dialog(self):
        from tools.gui_panel.services.path_dialog import build_dialog_invocation

        command, script = build_dialog_invocation("directory", "选择目录", r"F:\1\labelimg\data\panel_smoke10")

        self.assertEqual(command[0], r"C:\Program Files\PowerShell\7\pwsh.exe")
        self.assertIn("-STA", command)
        self.assertIn("FolderBrowserDialog", script)
        self.assertIn("panel_smoke10", script)

    def test_index_and_frontend_include_copy_actions(self):
        html = (ROOT / "tools" / "gui_panel" / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "tools" / "gui_panel" / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("copy-log-btn", html)
        self.assertIn("copy-command-btn", js)
        self.assertIn("navigator.clipboard.writeText", js)


if __name__ == "__main__":
    unittest.main()
