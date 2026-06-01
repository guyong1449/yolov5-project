"""Helpers for DDP batch-buffer video inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSlot:
    """Resolved batch/slot assignment for a zero-based frame index."""

    batch_id: int
    slot_id: int


def resolve_batch_slot(frame_idx: int, buffer_size: int) -> BatchSlot:
    """Map a zero-based frame index to a batch id and slot id."""
    if frame_idx < 0:
        raise ValueError("frame_idx must be >= 0")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be > 0")
    return BatchSlot(batch_id=frame_idx // buffer_size, slot_id=frame_idx % buffer_size)


def should_rank_process_frame(frame_idx: int, rank: int, buffer_size: int) -> bool:
    """Return True when *rank* owns the frame inside its micro-batch."""
    if rank < 0:
        return False
    slot = resolve_batch_slot(frame_idx, buffer_size)
    return rank < buffer_size and slot.slot_id == rank


def is_batch_boundary(frame_idx: int, total_frames: int, buffer_size: int) -> bool:
    """Return True when the current frame closes a micro-batch."""
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    slot = resolve_batch_slot(frame_idx, buffer_size)
    return slot.slot_id == buffer_size - 1 or frame_idx == total_frames - 1


def expected_batch_size(batch_id: int, total_frames: int, buffer_size: int) -> int:
    """Return the number of valid frames inside the given batch."""
    if batch_id < 0:
        raise ValueError("batch_id must be >= 0")
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be > 0")
    start = batch_id * buffer_size
    remaining = total_frames - start
    if remaining <= 0:
        return 0
    return min(buffer_size, remaining)


def valid_payloads_for_batch(payloads, *, batch_id: int, total_frames: int, buffer_size: int):
    """Return valid gathered payloads sorted by frame index for a batch."""
    expected = expected_batch_size(batch_id, total_frames, buffer_size)
    valid = [payload for payload in payloads if payload is not None]
    valid.sort(key=lambda item: int(item["frame_idx"]))
    return valid[:expected]


def retain_batch_payload(pending_payload, new_payload):
    """Persist the last payload for a rank until the gather boundary."""
    return new_payload if new_payload is not None else pending_payload
