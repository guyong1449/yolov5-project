"""Sanity checks for 4-class dataAirVis / test1_stride10 training path."""

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.general import check_dataset  # noqa: E402

EXPECTED_NAMES = ["ptarget", "bird", "drone", "fixedWing"]
DATA_YAMLS = [
    ROOT / "data" / "dataAirVis.yaml",
    Path("F:/1/labelimg/data/test1_stride10/data.yaml"),
]


class TestDataAirVisFourClass(unittest.TestCase):
  def test_yaml_nc_and_names(self):
    for yaml_path in DATA_YAMLS:
      self.assertTrue(yaml_path.is_file(), f"missing {yaml_path}")
      raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
      names = list(raw["names"].values()) if isinstance(raw["names"], dict) else list(raw["names"])
      self.assertEqual(int(raw["nc"]), 4)
      self.assertEqual(names, EXPECTED_NAMES)

  def test_check_dataset_resolves_paths(self):
    data = check_dataset(str(ROOT / "data" / "dataAirVis.yaml"), autodownload=False)
    self.assertEqual(data["nc"], 4)
    self.assertEqual(list(data["names"].values()), EXPECTED_NAMES)
    self.assertTrue(Path(data["train"]).is_file())
    self.assertTrue(Path(data["val"]).is_file())

  def test_labels_class_ids_in_range(self):
    data = check_dataset(str(ROOT / "data" / "dataAirVis.yaml"), autodownload=False)
    nc = data["nc"]
    for split_key in ("train", "val"):
      list_file = Path(data[split_key])
      for line in list_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
          continue
        img = Path(line.strip())
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        if not lbl.is_file():
          continue
        for row in lbl.read_text(encoding="utf-8").splitlines():
          if not row.strip():
            continue
          cls_id = int(float(row.split()[0]))
          self.assertGreaterEqual(cls_id, 0)
          self.assertLess(cls_id, nc, f"{lbl}: class {cls_id} >= nc={nc}")


if __name__ == "__main__":
  unittest.main()
