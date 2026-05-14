# Incremental video -> VOC frame export bookkeeping (used by detect.py with --voc-root)
import json
import os
import tempfile
from pathlib import Path
from typing import List, Set

from utils.dataloaders import VID_FORMATS

STATE_FILENAME = '.yolov5_mp4_convert_state.json'
STATE_VERSION = 1

# 与 LoadImages / detect 推理一致的后缀集合（小写带点）
VID_FILE_SUFFIXES = frozenset(f'.{ext.lower()}' for ext in VID_FORMATS)


def list_video_files_in_dir(scan_dir: Path) -> List[Path]:
    """递归列出 scan_dir 下所有常规文件，后缀与 utils.dataloaders.VID_FORMATS 一致（大小写不敏感）。"""
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        return []
    out: List[Path] = []
    for f in sorted(scan_dir.rglob('*')):
        if f.is_file() and f.suffix.lower() in VID_FILE_SUFFIXES:
            out.append(f)
    return out


def list_mp4_in_dir(scan_dir: Path) -> List[Path]:
    """兼容旧名：等价于 list_video_files_in_dir。"""
    return list_video_files_in_dir(scan_dir)


def state_path(voc_root: Path) -> Path:
    return Path(voc_root).resolve() / STATE_FILENAME


def load_converted_set(voc_root: Path) -> Set[str]:
    """已在 voc_root 状态中记录为「已完整转换」的视频绝对路径集合。"""
    p = state_path(voc_root)
    if not p.is_file():
        return set()
    try:
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    if not isinstance(data, dict):
        return set()
    raw = data.get('converted', [])
    if not isinstance(raw, list):
        return set()
    return {str(Path(x).resolve()) for x in raw if x}


def save_converted_set(voc_root: Path, converted: Set[str]) -> None:
    """Atomically write state JSON under voc_root."""
    voc_root = Path(voc_root).resolve()
    voc_root.mkdir(parents=True, exist_ok=True)
    payload = {'version': STATE_VERSION, 'converted': sorted(converted)}
    dst = state_path(voc_root)
    fd, tmp = tempfile.mkstemp(prefix='.yolov5_state_', suffix='.tmp', dir=str(voc_root))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def mark_video_converted(voc_root: Path, video_path: Path) -> None:
    s = load_converted_set(voc_root)
    s.add(str(Path(video_path).resolve()))
    save_converted_set(voc_root, s)


def filter_new_videos(scan_dir: Path, voc_root: Path) -> List[Path]:
    """scan_dir 下尚未写入 voc_root 状态的视频文件（VID_FORMATS 后缀）。"""
    done = load_converted_set(voc_root)
    return [p for p in list_video_files_in_dir(scan_dir) if str(p.resolve()) not in done]


def filter_new_mp4s(scan_dir: Path, voc_root: Path) -> List[Path]:
    """兼容旧名：等价于 filter_new_videos。"""
    return filter_new_videos(scan_dir, voc_root)


def write_source_list_txt(paths: List[Path]) -> str:
    """Write one absolute path per line; returns path to temp .txt (caller deletes)."""
    tf = tempfile.NamedTemporaryFile(mode='w', prefix='yolov5_sources_', suffix='.txt', delete=False, encoding='utf-8')
    try:
        for p in paths:
            tf.write(str(Path(p).resolve()) + '\n')
        tf.close()
        return tf.name
    except Exception:
        tf.close()
        try:
            os.unlink(tf.name)
        except OSError:
            pass
        raise
