import torch
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.series.series import TimeSeries
from diffusion_helper.src.series.plot import (
  _calculate_alpha,
  _compute_y_ranges,
  _style_axes,
  _set_y_limits,
  _plot_timeseries,
  plot_series,
  plot_multiple_series,
)


class TestCalculateAlpha:
  """Tests for _calculate_alpha helper function."""

  def test_single_sample_returns_max_alpha(self):
    """Single sample should return max_alpha."""
    assert _calculate_alpha(1) == 1.0
    assert _calculate_alpha(1, max_alpha=0.8) == 0.8

  def test_linear_scaling(self):
    """Test linear alpha scaling."""
    alpha = _calculate_alpha(10, max_alpha=1.0, alpha_scaling='linear')
    assert alpha == pytest.approx(0.1, rel=0.01)

  def test_sqrt_scaling(self):
    """Test sqrt alpha scaling."""
    alpha = _calculate_alpha(4, max_alpha=1.0, alpha_scaling='sqrt')
    assert alpha == pytest.approx(0.5, rel=0.01)

  def test_log_scaling(self):
    """Test log alpha scaling."""
    alpha = _calculate_alpha(10, max_alpha=1.0, alpha_scaling='log')
    expected = 1.0 / (1 + np.log(10))
    assert alpha == pytest.approx(expected, rel=0.01)

  def test_respects_min_alpha(self):
    """Alpha should not go below min_alpha."""
    alpha = _calculate_alpha(1000, min_alpha=0.2, max_alpha=1.0, alpha_scaling='linear')
    assert alpha >= 0.2

  def test_respects_max_alpha(self):
    """Alpha should not exceed max_alpha."""
    alpha = _calculate_alpha(1, min_alpha=0.1, max_alpha=0.5)
    assert alpha <= 0.5


class TestComputeYRanges:
  """Tests for _compute_y_ranges helper function."""

  def test_single_dimension(self):
    """Test y-range computation for single dimension."""
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    mask = np.array([True, True, True, True])

    y_ranges = _compute_y_ranges([values], [mask], num_dims=1)

    assert len(y_ranges) == 1
    y_min, y_max = y_ranges[0]
    assert y_min < 1.0  # Should have buffer
    assert y_max > 4.0  # Should have buffer

  def test_multiple_dimensions(self):
    """Test y-range computation for multiple dimensions."""
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    mask = np.array([True, True, True])

    y_ranges = _compute_y_ranges([values], [mask], num_dims=2)

    assert len(y_ranges) == 2

  def test_handles_nan_values(self):
    """Test that NaN values are filtered out."""
    values = np.array([[1.0], [np.nan], [3.0]])
    mask = np.array([True, True, True])

    y_ranges = _compute_y_ranges([values], [mask], num_dims=1)

    y_min, y_max = y_ranges[0]
    assert np.isfinite(y_min)
    assert np.isfinite(y_max)

  def test_handles_inf_values(self):
    """Test that Inf values are filtered out."""
    values = np.array([[1.0], [np.inf], [3.0]])
    mask = np.array([True, True, True])

    y_ranges = _compute_y_ranges([values], [mask], num_dims=1)

    y_min, y_max = y_ranges[0]
    assert np.isfinite(y_min)
    assert np.isfinite(y_max)

  def test_empty_data_returns_default_range(self):
    """Test that empty data returns default range."""
    values = np.array([]).reshape(0, 1)
    mask = np.array([], dtype=bool)

    y_ranges = _compute_y_ranges([values], [mask], num_dims=1)

    assert y_ranges[0] == (-1, 1)


class TestSetYLimits:
  """Tests for _set_y_limits helper function."""

  def test_sets_limits_correctly(self):
    """Test that y limits are set correctly."""
    fig, ax = plt.subplots()
    y_ranges = [(0.0, 10.0), (5.0, 15.0)]

    _set_y_limits(ax, k=0, y_ranges=y_ranges)

    y_min, y_max = ax.get_ylim()
    assert y_min == 0.0
    assert y_max == 10.0
    plt.close(fig)

  def test_handles_out_of_range_index(self):
    """Test that out of range index uses default."""
    fig, ax = plt.subplots()
    y_ranges = [(0.0, 10.0)]

    _set_y_limits(ax, k=5, y_ranges=y_ranges)

    y_min, y_max = ax.get_ylim()
    assert y_min == -1
    assert y_max == 1
    plt.close(fig)


class TestPlotTimeseries:
  """Tests for _plot_timeseries helper function."""

  def test_plots_single_dimension(self):
    """Test plotting a single dimension."""
    fig, ax = plt.subplots()
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    mask = np.array([True, True, True, True])

    line = _plot_timeseries(ax, times, values, mask, k=0, color='blue',
                            line_width=1.0, alpha=0.7)

    assert len(ax.lines) == 1
    plt.close(fig)

  def test_respects_mask(self):
    """Test that mask is respected."""
    fig, ax = plt.subplots()
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    mask = np.array([True, False, True, False])

    _plot_timeseries(ax, times, values, mask, k=0, color='blue',
                     line_width=1.0, alpha=0.7)

    # Line should only contain observed points
    line = ax.lines[0]
    assert len(line.get_xdata()) == 2  # Only 2 observed points
    plt.close(fig)

  def test_returns_none_when_no_observations(self):
    """Test that None is returned when no observations."""
    fig, ax = plt.subplots()
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0], [2.0], [3.0]])
    mask = np.array([False, False, False])

    result = _plot_timeseries(ax, times, values, mask, k=0, color='blue',
                              line_width=1.0, alpha=0.7)

    assert result is None
    plt.close(fig)

  def test_adds_markers_when_specified(self):
    """Test that markers are added when marker_style is specified."""
    fig, ax = plt.subplots()
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0], [2.0], [3.0]])
    mask = np.array([True, True, True])

    _plot_timeseries(ax, times, values, mask, k=0, color='blue',
                     line_width=1.0, alpha=0.7, marker_style='o')

    # Should have scatter points
    assert len(ax.collections) == 1
    plt.close(fig)


class TestPlotSeries:
  """Tests for plot_series function."""

  def test_plots_single_timeseries(self):
    """Test plotting a single TimeSeries."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    ts = TimeSeries(times=times, vals=values)

    fig, axes = plot_series(ts, show_plot=False)

    assert fig is not None
    assert axes is not None
    plt.close(fig)

  def test_plots_multi_dimensional_timeseries(self):
    """Test plotting a multi-dimensional TimeSeries."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=torch.float64)
    ts = TimeSeries(times=times, vals=values)

    fig, axes = plot_series(ts, show_plot=False)

    assert fig is not None
    assert len(axes) == 2  # Two dimensions
    plt.close(fig)

  def test_plots_batched_timeseries(self):
    """Test plotting a batched TimeSeries."""
    times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    values = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=torch.float64)
    ts = TimeSeries(times=times, vals=values)

    fig, axes = plot_series(ts, index='all', show_plot=False)

    assert fig is not None
    plt.close(fig)

  def test_plots_with_mask(self):
    """Test plotting with observation mask."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    mask = torch.tensor([True, False, True, False])
    ts = TimeSeries(times=times, vals=values, mask=mask)

    fig, axes = plot_series(ts, show_plot=False)

    assert fig is not None
    plt.close(fig)


class TestPlotMultipleSeries:
  """Tests for plot_multiple_series function."""

  def test_plots_multiple_series(self):
    """Test plotting multiple TimeSeries side by side."""
    times = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    values1 = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    values2 = torch.tensor([[2.0], [3.0], [4.0], [5.0]], dtype=torch.float64)

    ts1 = TimeSeries(times=times, vals=values1)
    ts2 = TimeSeries(times=times, vals=values2)

    fig, axes = plot_multiple_series([ts1, ts2], show_plot=False)

    assert fig is not None
    assert axes.shape == (1, 2)  # 1 dimension, 2 series
    plt.close(fig)

  def test_plots_with_titles(self):
    """Test plotting with custom titles."""
    times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)

    ts1 = TimeSeries(times=times, vals=values)
    ts2 = TimeSeries(times=times, vals=values)

    fig, axes = plot_multiple_series([ts1, ts2], titles=['Series A', 'Series B'], show_plot=False)

    assert fig is not None
    plt.close(fig)

  def test_raises_error_for_empty_list(self):
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="No series provided"):
      plot_multiple_series([], show_plot=False)

  def test_plots_batched_series(self):
    """Test plotting multiple batched TimeSeries."""
    times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=torch.float64)
    values = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=torch.float64)

    ts1 = TimeSeries(times=times, vals=values)
    ts2 = TimeSeries(times=times, vals=values)

    fig, axes = plot_multiple_series([ts1, ts2], index='all', show_plot=False)

    assert fig is not None
    plt.close(fig)
