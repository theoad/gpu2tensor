"""Bounded batches of explicitly selected measurements."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    columns: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    records: tuple[dict, ...]

    def tensors(self, device="cpu"):
        """CPU tensors share owned array storage. Other devices receive a copy."""
        import torch
        return {"values": torch.from_numpy(self.values).to(device),
                "valid": torch.from_numpy(self.valid).to(device)}


def batches(results, columns, *, batch_size=32):
    """Read at most batch_size results ahead; missing measurements have a mask."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    columns = tuple(columns)
    if not columns:
        raise ValueError("Select at least one measurement column.")
    rows = []
    masks = []
    records = []
    for result in results:
        row, mask = [], []
        for column in columns:
            if column == "latency_ms":
                value = result.latency_ms
            else:
                value = result.record
                for part in column.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
            present = isinstance(value, (int, float)) and np.isfinite(value)
            row.append(value if present else 0.0)
            mask.append(present)
        rows.append(row)
        masks.append(mask)
        records.append(result.record)
        if len(rows) == batch_size:
            yield Batch(columns, np.array(rows, dtype=np.float32), np.array(masks, dtype=bool), tuple(records))
            rows, masks, records = [], [], []
    if rows:
        yield Batch(columns, np.array(rows, dtype=np.float32), np.array(masks, dtype=bool), tuple(records))
