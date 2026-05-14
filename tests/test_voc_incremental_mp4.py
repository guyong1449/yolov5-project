# Tests for incremental MP4 -> VOC bookkeeping (no torch / no weights required)
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.voc_incremental import (  # noqa: E402
    STATE_FILENAME,
    filter_new_videos,
    list_video_files_in_dir,
    load_converted_set,
    mark_video_converted,
    save_converted_set,
    state_path,
    write_source_list_txt,
)


class TestListVideo(unittest.TestCase):
    def test_recursive_vid_formats_case_insensitive(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        try:
            (d / 'a.MP4').write_text('x')
            (d / 'b.mp4').write_text('x')
            (d / 'c.txt').write_text('n')
            (d / 'sub').mkdir()
            (d / 'sub' / 'd.mp4').write_text('x')
            (d / 'sub' / 'deep').mkdir()
            (d / 'sub' / 'deep' / 'e.Mp4').write_text('x')
            (d / 'sub' / 'clip.MOV').write_text('x')
            (d / 'x.MKV').write_text('x')
            got = list_video_files_in_dir(d)
            names = {p.name for p in got}
            self.assertEqual(names, {'a.MP4', 'b.mp4', 'd.mp4', 'e.Mp4', 'clip.MOV', 'x.MKV'})
            self.assertTrue(any(p.name == 'd.mp4' and 'sub' in p.parts for p in got))
            self.assertTrue(any(p.name == 'e.Mp4' and 'deep' in p.parts for p in got))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_dir(self):
        self.assertEqual(list_video_files_in_dir(Path('/nonexistent/path/xxx')), [])


class TestStateRoundtrip(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_load_empty(self):
        save_converted_set(self.tmp, set())
        self.assertEqual(load_converted_set(self.tmp), set())
        self.assertTrue((self.tmp / STATE_FILENAME).is_file())

    def test_mark_appends(self):
        v = self.tmp / 'sample.mp4'
        v.write_text('x')
        o = self.tmp / 'other.mp4'
        o.write_text('y')
        mark_video_converted(self.tmp, v)
        mark_video_converted(self.tmp, o)
        s = load_converted_set(self.tmp)
        self.assertEqual(len(s), 2)
        self.assertIn(str(v.resolve()), s)

    def test_corrupt_json_returns_empty(self):
        p = state_path(self.tmp)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('not json {{{')
        self.assertEqual(load_converted_set(self.tmp), set())


class TestFilterNew(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.voc = Path(tempfile.mkdtemp())
        self.scan = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.voc, ignore_errors=True)
        shutil.rmtree(self.scan, ignore_errors=True)

    def test_skips_converted(self):
        a = self.scan / 'a.mp4'
        b = self.scan / 'b.mp4'
        a.write_text('1')
        b.write_text('2')
        mark_video_converted(self.voc, a)
        new_list = filter_new_videos(self.scan, self.voc)
        self.assertEqual([p.resolve() for p in new_list], [b.resolve()])


class TestWriteSourceListTxt(unittest.TestCase):
    def test_paths_and_cleanup_manual(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        p = ''
        try:
            f1 = d / 'v.mp4'
            f1.write_text('x')
            p = write_source_list_txt([f1])
            self.assertTrue(Path(p).is_file())
            lines = Path(p).read_text(encoding='utf-8').strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(Path(lines[0]).resolve(), f1.resolve())
        finally:
            if p:
                Path(p).unlink(missing_ok=True)
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
