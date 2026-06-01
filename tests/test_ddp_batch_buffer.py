import unittest

from utils.ddp_batch_buffer import (expected_batch_size, is_batch_boundary, resolve_batch_slot,
                                    retain_batch_payload, should_rank_process_frame,
                                    valid_payloads_for_batch)


class DdpBatchBufferTests(unittest.TestCase):
    def test_resolve_batch_slot_maps_frame_index(self):
        self.assertEqual(resolve_batch_slot(0, 4).batch_id, 0)
        self.assertEqual(resolve_batch_slot(0, 4).slot_id, 0)
        self.assertEqual(resolve_batch_slot(5, 4).batch_id, 1)
        self.assertEqual(resolve_batch_slot(5, 4).slot_id, 1)

    def test_high_rank_stays_idle_when_buffer_size_smaller_than_world_size(self):
        self.assertFalse(should_rank_process_frame(0, rank=3, buffer_size=3))
        self.assertTrue(should_rank_process_frame(2, rank=2, buffer_size=3))

    def test_tail_batch_boundary_and_size(self):
        self.assertTrue(is_batch_boundary(4, total_frames=5, buffer_size=3))
        self.assertEqual(expected_batch_size(1, total_frames=5, buffer_size=3), 2)

    def test_valid_payloads_sorted_and_trimmed_to_tail_batch(self):
        payloads = [
            {"frame_idx": 4, "rank": 1},
            None,
            {"frame_idx": 3, "rank": 0},
            {"frame_idx": 5, "rank": 2},
        ]
        valid = valid_payloads_for_batch(payloads, batch_id=1, total_frames=5, buffer_size=3)
        self.assertEqual([item["frame_idx"] for item in valid], [3, 4])

    def test_pending_payload_is_retained_until_batch_boundary(self):
        pending_payload = None
        gathered_frame_idxs = []
        buffer_size = 4
        total_frames = 8

        for frame_idx in range(total_frames):
            current_payload = None
            if should_rank_process_frame(frame_idx, rank=0, buffer_size=buffer_size):
                current_payload = {"frame_idx": frame_idx, "rank": 0}
            pending_payload = retain_batch_payload(pending_payload, current_payload)

            if is_batch_boundary(frame_idx, total_frames=total_frames, buffer_size=buffer_size):
                gathered_frame_idxs.append(None if pending_payload is None else pending_payload["frame_idx"])
                pending_payload = None

        self.assertEqual(gathered_frame_idxs, [0, 4])


if __name__ == "__main__":
    unittest.main()
