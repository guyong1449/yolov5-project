from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

PWSH_PATH = Path("pwsh")


def build_dialog_invocation(kind: str, title: str, initial_path: str = "") -> tuple[list[str], str]:
    initial = Path(initial_path).expanduser() if initial_path else None
    initial_dir = ""
    initial_file = ""
    if initial is not None:
        initial_dir = str(initial.parent if initial.suffix else initial)
        initial_file = initial.name if initial.suffix else ""

    def ps_quote(value: str) -> str:
        return value.replace("'", "''")

    script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$result = ''
if ('{ps_quote(kind)}' -eq 'directory') {{
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = '{ps_quote(title)}'
    if ('{ps_quote(initial_path)}' -and (Test-Path '{ps_quote(initial_path)}')) {{
        $dialog.SelectedPath = '{ps_quote(initial_path)}'
    }}
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
        $result = $dialog.SelectedPath
    }}
}} else {{
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = '{ps_quote(title)}'
    if ('{ps_quote(initial_dir)}' -and (Test-Path '{ps_quote(initial_dir)}')) {{
        $dialog.InitialDirectory = '{ps_quote(initial_dir)}'
    }}
    if ('{ps_quote(initial_file)}') {{
        $dialog.FileName = '{ps_quote(initial_file)}'
    }}
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
        $result = $dialog.FileName
    }}
}}
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
Write-Output $result
""".strip()
    command = [str(PWSH_PATH), "-STA", "-NoLogo", "-NoProfile", "-File"]
    return command, script


def select_path(kind: str, title: str, initial_path: str = "") -> str:
    if PWSH_PATH.exists():
        command, script = build_dialog_invocation(kind, title, initial_path)
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", encoding="utf-8", delete=False) as handle:
            handle.write(script)
            script_path = Path(handle.name)
        try:
            completed = subprocess.run(
                [*command, str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or "path dialog failed").strip())
            return completed.stdout.strip()
        finally:
            script_path.unlink(missing_ok=True)

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        initial_dir = ""
        initial_file = ""
        if initial_path:
            path = Path(initial_path)
            initial_dir = str(path.parent if path.suffix else path)
            initial_file = path.name if path.suffix else ""
        if kind == "directory":
            selected = filedialog.askdirectory(title=title, initialdir=initial_dir or None)
        else:
            selected = filedialog.askopenfilename(
                title=title,
                initialdir=initial_dir or None,
                initialfile=initial_file or None,
            )
        return selected or ""
    finally:
        root.destroy()
