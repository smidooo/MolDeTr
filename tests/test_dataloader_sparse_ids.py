"""The training DataReader must index samples by the *actual* .npz files present, not by assuming
contiguous integer names ``0..N-1``.

The Zenodo ``custom_spin_systems.zip`` ships a **sparse** seeded subset (ids 3..4980, no ``0.npz``),
so the old ``np.load(dir / f"{idx}.npz")`` raised ``FileNotFoundError`` on a fresh install. Enumerating
the real files (sorted by integer stem) fixes that and is **bit-identical on contiguous data** — the
seed-42 ``random_split`` still partitions positional indices ``0..N-1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moldetr.config import RegParamIndices
from moldetr.dataloader.dataloader import DataReader, _split_lengths, list_sample_files
from moldetr.dataloader.transforms import Normalize

_REG_PARAMS = (
    "center_position_in_points",
    "line_width_in_points",
    "coupling_constant_1_in_points",
    "coupling_constant_2_in_points",
    "coupling_constant_3_in_points",
    "coupling_constant_4_in_points",
    "bounding_box_range_in_points",
)


def _write_sample(path: Path) -> None:
    """A minimal but structurally valid training sample the DataReader can decode."""
    spec = np.linspace(1.0, 2.0, 64).astype(np.float32)  # non-zero (scale_spectrum asserts this)
    labels = np.array(
        [
            {
                "proton_number": 2,
                "center_position_in_points": 32.0,
                "line_width_in_points": 5.0,
                "bounding_box_range_in_points": 20.0,
                "coupling_constants_in_points": [7.0, 7.0],
            }
        ],
        dtype=object,
    )
    np.savez(path, spec=spec, labels=labels)


def _reader(files_dir: Path) -> DataReader:
    reg = RegParamIndices(*range(len(_REG_PARAMS)))  # name -> index 0..6, matching field order
    extrema = {name: [0.0, 100.0] for name in _REG_PARAMS}
    return DataReader(
        _files=list_sample_files(files_dir),
        _num_classes=5,
        data_augmentation=None,
        transformation=Normalize(extrema=extrema),
        reg_param_indices=reg,
    )


def test_list_sample_files_sorts_sparse_ids_numerically(tmp_path):
    """Sparse ids sort by integer value, not lexicographically (else 4980 < 500 etc.)."""
    for stem in ("17", "3", "4980", "500"):
        (tmp_path / f"{stem}.npz").write_bytes(b"")
    assert [p.stem for p in list_sample_files(tmp_path)] == ["3", "17", "500", "4980"]


def test_list_sample_files_contiguous_order_preserved(tmp_path):
    """On contiguous 0..N-1 data, position i still maps to i.npz -> training stays bit-identical."""
    for i in range(5):
        (tmp_path / f"{i}.npz").write_bytes(b"")
    assert [p.stem for p in list_sample_files(tmp_path)] == ["0", "1", "2", "3", "4"]


def test_datareader_reads_sparse_files_by_position(tmp_path):
    """The real symptom: a DataReader over sparse ids (3, 17) loads both by position 0 and 1."""
    for stem in (3, 17):
        _write_sample(tmp_path / f"{stem}.npz")
    reader = _reader(tmp_path)

    assert len(reader) == 2
    for i in range(2):
        features, targets = reader[i]  # would FileNotFoundError on the old 0.npz/1.npz assumption
        assert features.shape[0] == 1  # (1, spectrum_length)
        assert targets.shape[0] == 1  # one multiplet in the sample


@pytest.mark.parametrize("n", [100, 101, 137, 50, 3, 5_000_000])
def test_split_lengths_always_sum_to_n(n):
    """torch.random_split needs lengths summing to exactly n; the naive
    int(0.92n)+int(0.06n)+int(0.02n) is short by 1-2 unless n is a multiple of 50 -> ValueError."""
    lengths = _split_lengths(n)
    assert sum(lengths) == n
    assert all(x >= 0 for x in lengths)


def test_split_lengths_bit_identical_on_shipped_and_training_sizes():
    """The released synthetic subset (N=100) and the training set (N=5,000,000) keep the historical
    92/6/2 % partition exactly, so train/val/test membership -- and reproducibility -- is unchanged."""
    assert _split_lengths(100) == (92, 6, 2)
    assert _split_lengths(5_000_000) == (4_600_000, 300_000, 100_000)
