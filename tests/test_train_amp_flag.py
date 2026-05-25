"""Tests for explicit AMP control in train.py."""

import sys
import unittest

import train


class TrainAmpFlagTests(unittest.TestCase):
    def test_parse_opt_defaults_amp_to_false(self):
        old_argv = sys.argv
        sys.argv = ["train.py"]
        try:
            opt = train.parse_opt()
        finally:
            sys.argv = old_argv

        self.assertFalse(opt.amp)

    def test_parse_opt_accepts_amp_flag(self):
        old_argv = sys.argv
        sys.argv = ["train.py", "--amp"]
        try:
            opt = train.parse_opt()
        finally:
            sys.argv = old_argv

        self.assertTrue(opt.amp)

    def test_parse_opt_accepts_legacy_no_amp_flag(self):
        old_argv = sys.argv
        sys.argv = ["train.py", "--no-amp"]
        try:
            opt = train.parse_opt()
        finally:
            sys.argv = old_argv

        self.assertFalse(opt.amp)


if __name__ == "__main__":
    unittest.main()
