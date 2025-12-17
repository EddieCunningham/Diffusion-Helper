import torch
import pytest
from pathlib import Path
import sys
import cola

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore
from diffusion_helper.src.functional.linear_functional import LinearFunctional  # type: ignore
from diffusion_helper.src.matrix.tags import TAGS  # type: ignore


def make_dense_matrix(mat: torch.Tensor) -> SquareMatrix:
  return SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Dense(mat))


def make_diagonal_matrix(diag: torch.Tensor) -> SquareMatrix:
  return SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(diag))


def create_matrix(dim: int, matrix_type: str, seed: int = 0) -> SquareMatrix:
  g = torch.Generator().manual_seed(seed)
  if matrix_type == "dense":
    M = torch.randn((dim, dim), generator=g, dtype=torch.float64)
    return make_dense_matrix(M)
  elif matrix_type == "diagonal":
    d = torch.randn((dim,), generator=g, dtype=torch.float64)
    return make_diagonal_matrix(d)
  else:
    raise ValueError(f"Unknown matrix type: {matrix_type}")


def create_functional(dim: int, matrix_type: str = "dense", seed: int = 0) -> LinearFunctional:
  A = create_matrix(dim, matrix_type, seed)
  g = torch.Generator().manual_seed(seed + 1)
  b = torch.randn((dim,), generator=g, dtype=torch.float64)
  return LinearFunctional(A, b)


@pytest.mark.parametrize("matrix_type", ["dense", "diagonal"])
class TestLinearFunctional:
  def test_initialization(self, matrix_type):
    dim = 3
    A = create_matrix(dim, matrix_type)
    b = torch.ones(dim, dtype=torch.float64)
    lf = LinearFunctional(A, b)
    assert lf.A is A
    assert torch.allclose(lf.b, b)

  def test_call_method(self, matrix_type):
    dim = 2
    A = create_matrix(dim, matrix_type, seed=1)
    b = torch.tensor([5.0, 6.0], dtype=torch.float64)
    lf = LinearFunctional(A, b)
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = lf(x)
    expected = A @ x + b
    assert torch.allclose(result, expected)

  def test_addition(self, matrix_type):
    dim = 3
    lf1 = create_functional(dim, matrix_type, seed=0)
    lf2 = create_functional(dim, matrix_type, seed=1)

    lf_sum = lf1 + lf2
    assert isinstance(lf_sum, LinearFunctional)
    assert torch.allclose(lf_sum.A.to_dense(), (lf1.A + lf2.A).to_dense())
    assert torch.allclose(lf_sum.b, lf1.b + lf2.b)

    vec = torch.ones(dim, dtype=torch.float64)
    lf_sum_vec = lf1 + vec
    assert torch.allclose(lf_sum_vec.A.to_dense(), lf1.A.to_dense())
    assert torch.allclose(lf_sum_vec.b, lf1.b + vec)

  def test_subtraction(self, matrix_type):
    dim = 2
    lf1 = create_functional(dim, matrix_type, seed=2)
    lf2 = create_functional(dim, matrix_type, seed=3)

    lf_sub = lf1 - lf2
    assert torch.allclose(lf_sub.A.to_dense(), (lf1.A - lf2.A).to_dense())
    assert torch.allclose(lf_sub.b, lf1.b - lf2.b)

  def test_negation(self, matrix_type):
    dim = 4
    lf = create_functional(dim, matrix_type, seed=4)
    neg_lf = -lf
    assert torch.allclose(neg_lf.A.to_dense(), -lf.A.to_dense())
    assert torch.allclose(neg_lf.b, -lf.b)

  def test_scalar_multiplication(self, matrix_type):
    dim = 3
    lf = create_functional(dim, matrix_type, seed=5)
    scalar = 2.5

    scaled_lf = lf * scalar
    assert torch.allclose(scaled_lf.A.to_dense(), (lf.A * scalar).to_dense())
    assert torch.allclose(scaled_lf.b, lf.b * scalar)

    scaled_lf_r = scalar * lf
    assert torch.allclose(scaled_lf_r.A.to_dense(), (lf.A * scalar).to_dense())
    assert torch.allclose(scaled_lf_r.b, lf.b * scalar)

  def test_matrix_multiplication(self, matrix_type):
    dim = 2
    lf = create_functional(dim, matrix_type, seed=6)
    M = create_matrix(dim, matrix_type, seed=7)

    new_lf = M @ lf
    assert torch.allclose(new_lf.A.to_dense(), (M @ lf.A).to_dense())
    assert torch.allclose(new_lf.b, M @ lf.b)

  def test_matrix_solve(self, matrix_type):
    dim = 3
    lf = create_functional(dim, matrix_type, seed=8)
    # Use a PD matrix to guarantee invertibility
    g = torch.Generator().manual_seed(9)
    X = torch.randn((dim, dim), generator=g, dtype=torch.float64)
    M = make_dense_matrix(X @ X.T + dim * torch.eye(dim, dtype=torch.float64))

    # Solve M y = lf(x) for y, by applying M^{-1}
    Minv = M.get_inverse()
    solved_lf = Minv @ lf

    x = torch.randn((dim,), generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    reconstructed_val = M @ solved_lf(x)
    original_val = lf(x)
    assert torch.allclose(reconstructed_val, original_val, atol=1e-6)

  def test_get_inverse(self, matrix_type):
    dim = 2
    if matrix_type == "dense":
      A = create_matrix(dim, "dense", seed=11)
    else:
      d = torch.randn((dim,), generator=torch.Generator().manual_seed(11), dtype=torch.float64)
      d = torch.where(torch.abs(d) < 1e-3, torch.ones_like(d), d)
      A = make_diagonal_matrix(d)

    b = torch.randn((dim,), generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    lf = LinearFunctional(A, b)

    lf_inv = lf.get_inverse()

    x = torch.randn((dim,), generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    y = lf(x)
    x_recovered = lf_inv(y)
    assert torch.allclose(x_recovered, x, atol=1e-6)

    y_test = torch.randn((dim,), generator=torch.Generator().manual_seed(14), dtype=torch.float64)
    x_from_y = lf_inv(y_test)
    y_recovered = lf(x_from_y)
    assert torch.allclose(y_recovered, y_test, atol=1e-6)

    A_inv_expected = A.get_inverse()
    b_inv_expected = -(A_inv_expected @ b)
    assert torch.allclose(lf_inv.A.to_dense(), A_inv_expected.to_dense(), atol=1e-6)
    assert torch.allclose(lf_inv.b, b_inv_expected, atol=1e-6)

  def test_composition(self, matrix_type):
    dim = 3
    lf1 = create_functional(dim, matrix_type, seed=15)
    lf2 = create_functional(dim, matrix_type, seed=16)

    lf_composed = lf1(lf2)
    expected_A = lf1.A @ lf2.A
    expected_b = lf1.A @ lf2.b + lf1.b
    assert torch.allclose(lf_composed.A.to_dense(), expected_A.to_dense())
    assert torch.allclose(lf_composed.b, expected_b)

    x = torch.randn((dim,), generator=torch.Generator().manual_seed(17), dtype=torch.float64)
    composed_eval = lf_composed(x)
    sequential_eval = lf1(lf2(x))
    assert torch.allclose(composed_eval, sequential_eval)


