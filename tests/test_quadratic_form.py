import torch
import pytest
from pathlib import Path
import sys
import cola

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore
from diffusion_helper.src.functional.quadratic_form import QuadraticForm, resolve_quadratic_form  # type: ignore
from diffusion_helper.src.functional.funtional_ops import vdot  # type: ignore
from diffusion_helper.src.functional.linear_functional import LinearFunctional  # type: ignore
from diffusion_helper.src.matrix.tags import TAGS  # type: ignore


def dense_square_matrix(t: torch.Tensor) -> SquareMatrix:
  return SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Dense(t))


def create_quadratic_form(dim: int, seed: int = 0) -> QuadraticForm:
  g = torch.Generator().manual_seed(seed)
  A = dense_square_matrix(torch.randn((dim, dim), generator=g, dtype=torch.float64))
  b = torch.randn((dim,), generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
  c = torch.randn((), generator=torch.Generator().manual_seed(seed + 2), dtype=torch.float64)
  return QuadraticForm(A, b, c)


def create_linear_functional(dim: int, seed: int = 0) -> LinearFunctional:
  g = torch.Generator().manual_seed(seed)
  A = dense_square_matrix(torch.randn((dim, dim), generator=g, dtype=torch.float64))
  b = torch.randn((dim,), generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
  return LinearFunctional(A, b)


class TestQuadraticForm:
  key = 0
  dim = 4

  def test_init(self):
    qf = create_quadratic_form(self.dim, self.key)
    assert isinstance(qf, QuadraticForm)
    assert qf.A.shape == (self.dim, self.dim)
    assert qf.b.shape == (self.dim,)
    assert isinstance(qf.c, torch.Tensor) and qf.c.ndim == 0
    # symmetry
    D = qf.A.to_dense()
    assert torch.allclose(D, D.T)

  def test_repr(self):
    qf = create_quadratic_form(self.dim, self.key)
    s = repr(qf)
    assert isinstance(s, str)
    assert "QuadraticForm" in s
    assert "A=" in s
    assert "b=" in s
    assert "c=" in s

  def test_call(self):
    qf = create_quadratic_form(self.dim, self.key)
    x = torch.randn((self.dim,), generator=torch.Generator().manual_seed(self.key), dtype=torch.float64)
    expected = 0.5 * torch.dot(x, qf.A @ x) + torch.dot(qf.b, x) + qf.c.to(torch.float64)
    result = qf(x)
    assert torch.allclose(result, expected)

  def test_addition(self):
    qf1 = create_quadratic_form(self.dim, 1)
    qf2 = create_quadratic_form(self.dim, 2)
    qf_sum = qf1 + qf2
    assert torch.allclose(qf_sum.A.to_dense(), (qf1.A + qf2.A).to_dense())
    assert torch.allclose(qf_sum.b, qf1.b + qf2.b)
    assert torch.allclose(qf_sum.c, qf1.c + qf2.c)

    scalar = 2.5
    qf_sum_scalar = qf1 + scalar
    assert torch.allclose(qf_sum_scalar.A.to_dense(), qf1.A.to_dense())
    assert torch.allclose(qf_sum_scalar.b, qf1.b)
    assert torch.allclose(qf_sum_scalar.c, qf1.c + torch.tensor(scalar, dtype=qf1.b.dtype))

    qf_sum_scalar_r = scalar + qf1
    assert torch.allclose(qf_sum_scalar_r.A.to_dense(), qf1.A.to_dense())
    assert torch.allclose(qf_sum_scalar_r.b, qf1.b)
    assert torch.allclose(qf_sum_scalar_r.c, qf1.c + torch.tensor(scalar, dtype=qf1.b.dtype))

  def test_subtraction(self):
    qf1 = create_quadratic_form(self.dim, 3)
    qf2 = create_quadratic_form(self.dim, 4)
    qf_sub = qf1 - qf2
    assert torch.allclose(qf_sub.A.to_dense(), (qf1.A - qf2.A).to_dense())
    assert torch.allclose(qf_sub.b, qf1.b - qf2.b)
    assert torch.allclose(qf_sub.c, qf1.c - qf2.c)

    scalar = 2.5
    qf_sub_scalar = qf1 - scalar
    assert torch.allclose(qf_sub_scalar.A.to_dense(), qf1.A.to_dense())
    assert torch.allclose(qf_sub_scalar.b, qf1.b)
    assert torch.allclose(qf_sub_scalar.c, qf1.c - torch.tensor(scalar, dtype=qf1.b.dtype))

    qf_sub_scalar_r = scalar - qf1
    assert torch.allclose(qf_sub_scalar_r.A.to_dense(), (-qf1.A).to_dense())
    assert torch.allclose(qf_sub_scalar_r.b, -qf1.b)
    assert torch.allclose(qf_sub_scalar_r.c, torch.tensor(scalar, dtype=qf1.b.dtype) - qf1.c)

  def test_negation(self):
    qf = create_quadratic_form(self.dim, 5)
    qf_neg = -qf
    assert torch.allclose(qf_neg.A.to_dense(), (-qf.A).to_dense())
    assert torch.allclose(qf_neg.b, -qf.b)
    assert torch.allclose(qf_neg.c, -qf.c)

  def test_multiplication(self):
    qf = create_quadratic_form(self.dim, 6)
    scalar = 2.5
    qf_mul = qf * scalar
    assert torch.allclose(qf_mul.A.to_dense(), (scalar * qf.A).to_dense())
    assert torch.allclose(qf_mul.b, scalar * qf.b)
    assert torch.allclose(qf_mul.c, torch.tensor(scalar, dtype=qf.b.dtype) * qf.c)

    qf_mul_r = scalar * qf
    assert torch.allclose(qf_mul_r.A.to_dense(), (scalar * qf.A).to_dense())
    assert torch.allclose(qf_mul_r.b, scalar * qf.b)
    assert torch.allclose(qf_mul_r.c, torch.tensor(scalar, dtype=qf.b.dtype) * qf.c)


class TestVdot:
  key = 0
  dim = 4

  def test_vdot_lf_lf(self):
    lf1 = create_linear_functional(self.dim, seed=7)
    lf2 = create_linear_functional(self.dim, seed=8)
    x = torch.randn((self.dim,), generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    expected = torch.dot(lf1(x), lf2(x))
    qf = vdot(lf1, lf2)
    actual = qf(x)
    assert isinstance(qf, QuadraticForm)
    assert torch.allclose(actual, expected)

  def test_vdot_lf_vec(self):
    lf = create_linear_functional(self.dim, seed=10)
    vec = torch.randn((self.dim,), generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    x = torch.randn((self.dim,), generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    expected = torch.dot(lf(x), vec)
    qf = vdot(lf, vec)
    actual = qf(x)
    assert isinstance(qf, QuadraticForm)
    assert torch.allclose(actual, expected)


class TestResolve:
  key = 0
  dim = 4

  def test_resolve_quadratic_form(self):
    qf = create_quadratic_form(self.dim, seed=13)
    x = torch.randn((self.dim,), generator=torch.Generator().manual_seed(14), dtype=torch.float64)
    tree = {"a": qf, "b": torch.ones(1, dtype=torch.float64)}
    resolved_tree = resolve_quadratic_form(tree, x)
    assert torch.allclose(resolved_tree["a"], qf(x))
    assert torch.allclose(resolved_tree["b"], torch.ones(1, dtype=torch.float64))


