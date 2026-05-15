import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import detect  # noqa: E402
from utils.dataloaders import LoadImages  # noqa: E402


class FakeVideoCapture:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key, 0)


class TestDetectSourceBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_directory_source_is_recursive_after_normalization(self):
        (self.tmp / 'root_a.mp4').write_text('a')
        (self.tmp / 'root_b.avi').write_text('b')
        (self.tmp / 'nested').mkdir()
        (self.tmp / 'nested' / 'child.mp4').write_text('c')
        (self.tmp / 'nested' / 'note.txt').write_text('ignored')

        normalized = detect.normalize_source_for_detect(str(self.tmp))
        dataset = LoadImages(normalized)
        loaded = {Path(p).name for p in dataset.files}

        self.assertIn('**', normalized)
        self.assertEqual(loaded, {'root_a.mp4', 'root_b.avi', 'child.mp4'})
        self.assertNotIn('note.txt', loaded)

    def test_glob_source_can_be_recursive(self):
        (self.tmp / 'root_a.mp4').write_text('a')
        (self.tmp / 'nested').mkdir()
        (self.tmp / 'nested' / 'child.mp4').write_text('c')

        dataset = LoadImages(str(self.tmp / '**' / '*.mp4'))
        loaded = {Path(p).name for p in dataset.files}

        self.assertEqual(loaded, {'root_a.mp4', 'child.mp4'})

    def test_format_video_start_log_includes_video_metadata(self):
        cap = FakeVideoCapture(
            {
                detect.cv2.CAP_PROP_FRAME_COUNT: 120,
                detect.cv2.CAP_PROP_FPS: 24,
                detect.cv2.CAP_PROP_FRAME_WIDTH: 1920,
                detect.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            }
        )

        message = detect.format_video_start_log(
            Path('F:/1/video/output/demo.mp4'),
            file_index=2,
            file_total=3,
            vid_cap=cap,
        )

        self.assertIn('demo.mp4', message)
        self.assertIn('2/3', message)
        self.assertIn('120', message)
        self.assertIn('24.00', message)
        self.assertIn('1920x1080', message)

    def test_save_voc_frame_and_xml_preserves_raw_frame_pixels(self):
        image_dir = self.tmp / 'images'
        annotations_dir = self.tmp / 'annotations'
        image_dir.mkdir()
        annotations_dir.mkdir()

        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[3:9, 4:12] = (17, 91, 203)
        expected_path = self.tmp / 'expected.jpg'
        detect.cv2.imwrite(str(expected_path), frame)

        detect.save_voc_frame_and_xml(
            image_dir=image_dir,
            annotations_dir=annotations_dir,
            base_name='video4_7_mp4_frame000123',
            frame_image=frame,
            detections=[(4, 3, 11, 8, 'drone')],
        )

        saved_image = image_dir / 'video4_7_mp4_frame000123.jpg'
        saved_xml = annotations_dir / 'video4_7_mp4_frame000123.xml'

        self.assertTrue(saved_image.is_file())
        self.assertTrue(saved_xml.is_file())
        self.assertEqual(saved_image.read_bytes(), expected_path.read_bytes())

        xml_text = saved_xml.read_text(encoding='utf-8')
        self.assertIn('<filename>video4_7_mp4_frame000123.jpg</filename>', xml_text)
        self.assertIn('<name>drone</name>', xml_text)


if __name__ == '__main__':
    unittest.main()
