import torch
import numpy as np
from typing import Union, Tuple, Optional
import pytest

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.series.series import TimeSeries  # type: ignore

class TestTimeSeries:
  """Tests for TimeSeries class."""

  def test_init_basic(self):
    """Test basic initialization with arrays."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float64)[:,None]

    # Create TimeSeries
    ts = TimeSeries(times=times, vals=values)

    # Check that attributes were set correctly
    assert torch.equal(ts.times, times)
    assert torch.equal(ts.vals, values)
    assert torch.equal(ts.mask, torch.ones_like(times, dtype=torch.bool))

  def test_init_with_mask(self):
    """Test initialization with explicit mask."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float64)[:,None]
    mask = torch.tensor([True, True, False, True])

    # Create TimeSeries
    ts = TimeSeries(times=times, vals=values, mask=mask)

    # Check that attributes were set correctly
    assert torch.equal(ts.times, times)
    assert torch.equal(ts.vals, values)
    assert torch.equal(ts.mask, mask)

  def test_init_with_2d_values(self):
    """Test initialization with already 2D values."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float64)

    # Create TimeSeries
    ts = TimeSeries(times=times, vals=values)

    # Check that attributes were set correctly
    assert torch.equal(ts.times, times)
    assert torch.equal(ts.vals, values)
    assert torch.equal(ts.mask, torch.ones_like(times, dtype=torch.bool))

  def test_init_with_multi_feature_values(self):
    """Test initialization with 2D values with multiple features."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[10.0, 20.0, 30.0],
                           [11.0, 21.0, 31.0],
                           [12.0, 22.0, 32.0],
                           [13.0, 23.0, 33.0]], dtype=torch.float64)

    # Create TimeSeries
    ts = TimeSeries(times=times, vals=values)

    # Check that attributes were set correctly
    assert torch.equal(ts.times, times)
    assert torch.equal(ts.vals, values)
    assert torch.equal(ts.mask, torch.ones_like(times, dtype=torch.bool))

  def test_is_fully_uncertain(self):
    """Test is_fully_uncertain method."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float64)
    mask = torch.tensor([True, False, True, False])

    ts = TimeSeries(times=times, vals=values, mask=mask)

    # Check is_fully_uncertain
    expected = torch.tensor([False, True, False, True])
    assert torch.equal(ts.is_fully_uncertain(), expected)

  def test_get_missing_observation_mask(self):
    """Test get_missing_observation_mask method."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float64)
    mask = torch.tensor([True, False, True, False])

    ts = TimeSeries(times=times, vals=values, mask=mask)

    # get_missing_observation_mask should be the same as is_fully_uncertain
    expected = torch.tensor([False, True, False, True])
    assert torch.equal(ts.get_missing_observation_mask(), expected)

  def test_make_windowed_batches(self):
    """Test make_windowed_batches method."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    values = torch.tensor([[10.0, 20.0],
                          [11.0, 21.0],
                          [12.0, 22.0],
                          [13.0, 23.0],
                          [14.0, 24.0]], dtype=torch.float64)

    ts = TimeSeries(times=times, vals=values)

    # Create windowed batches with window size 3
    window_size = 3
    batched_ts = ts.make_windowed_batches(window_size)

    # Expected: 3 windows with a window size of 3
    # Window 1: [0, 1, 2]
    # Window 2: [1, 2, 3]
    # Window 3: [2, 3, 4]
    assert batched_ts.times.shape == (3, 3)

    # Check specific windows
    expected_times = torch.tensor([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0]
    ], dtype=torch.float64)
    assert torch.allclose(batched_ts.times, expected_times)

    # Check that values have the right shape and content
    assert batched_ts.vals.shape == (3, 3, 2)

    expected_values = torch.tensor([
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
        [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
        [[12.0, 22.0], [13.0, 23.0], [14.0, 24.0]]
    ], dtype=torch.float64)
    assert torch.allclose(batched_ts.vals, expected_values)

  def test_getitem(self):
    """Test the __getitem__ method inherited from AbstractBatchableObject."""
    times = torch.tensor([[0.0, 1.0, 2.0],
                          [3.0, 4.0, 5.0]], dtype=torch.float64)
    values = torch.tensor([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
                          [[13.0, 23.0], [14.0, 24.0], [15.0, 25.0]]], dtype=torch.float64)
    mask = torch.ones_like(times, dtype=torch.bool)

    ts = TimeSeries(times=times, vals=values, mask=mask)

    # Get first batch
    ts_0 = ts[0]

    assert torch.equal(ts_0.times, times[0])
    assert torch.equal(ts_0.vals, values[0])
    assert torch.equal(ts_0.mask, mask[0])

    # Get using a slice
    ts_slice = ts[0:1]

    assert ts_slice.times.shape == (1, 3)
    assert ts_slice.vals.shape == (1, 3, 2)
    assert torch.equal(ts_slice.times, times[0:1])
    assert torch.equal(ts_slice.vals, values[0:1])
    assert torch.equal(ts_slice.mask, mask[0:1])
