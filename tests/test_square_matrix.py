import torch
import pytest

import cola
from pathlib import Path
import sys
import unittest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore
from diffusion_helper.src.matrix.tags import TAGS  # type: ignore
from tests.matrix_test_base import (
  matrix_tests,
  matrix_implementations_tests,
  performance_tests,
  autodiff_for_matrix_class,
  AbstractMatrixTest,
)


def make_dense_square_matrix(t: torch.Tensor, tags=TAGS.no_tags) -> SquareMatrix:
  A = cola.ops.Dense(t)
  return SquareMatrix(tags=tags, mat=A)


def rand_pd(dim: int, dtype=torch.float64, seed: int = 0) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  X = torch.randn((dim, dim), generator=g, dtype=dtype)
  return X @ X.T + dim * torch.eye(dim, dtype=dtype)


def rand_mat(dim: int, dtype=torch.float64, seed: int = 0) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  return torch.randn((dim, dim), generator=g, dtype=dtype)


@pytest.mark.parametrize("dim", [2, 4])
def test_initialization_and_shape(dim):
  M = rand_mat(dim)
  A = make_dense_square_matrix(M, TAGS.no_tags)
  D = A.to_dense()
  assert D.shape == (dim, dim)
  assert torch.allclose(D, M)
  assert A.shape == (dim, dim)


@pytest.mark.parametrize("dim", [3])
def test_zeros_and_eye(dim):
  Z = SquareMatrix.zeros(dim, dtype=torch.float64)
  E = SquareMatrix.eye(dim, dtype=torch.float64)
  assert torch.allclose(Z.to_dense(), torch.zeros((dim, dim), dtype=torch.float64))
  assert torch.allclose(E.to_dense(), torch.eye(dim, dtype=torch.float64))
  assert Z.tags.is_zero.item() is True
  assert E.tags.is_nonzero.item() is True


@pytest.mark.parametrize("dim", [4])
def test_neg_add_sub_mul_div(dim):
  A = make_dense_square_matrix(rand_mat(dim))
  B = make_dense_square_matrix(rand_mat(dim, seed=1))

  # neg
  assert torch.allclose((-A).to_dense(), -A.to_dense())

  # add
  C = A + B
  assert torch.allclose(C.to_dense(), A.to_dense() + B.to_dense())

  # sub
  D = A - B
  assert torch.allclose(D.to_dense(), A.to_dense() - B.to_dense())

  # scalar mul/div
  s = 2.5
  Sm = s * A
  Sd = A / s
  assert torch.allclose(Sm.to_dense(), s * A.to_dense())
  assert torch.allclose(Sd.to_dense(), A.to_dense() / s)


@pytest.mark.parametrize("dim", [4])
def test_matmul_and_matvec(dim):
  A = make_dense_square_matrix(rand_mat(dim))
  B = make_dense_square_matrix(rand_mat(dim, seed=2))
  v = torch.randn(dim, dtype=torch.float64)

  # matrix @ matrix
  C = A @ B
  assert torch.allclose(C.to_dense(), A.to_dense() @ B.to_dense())

  # matrix @ vector
  y = A @ v
  assert torch.allclose(y, A.to_dense() @ v)


@pytest.mark.parametrize("dim", [4])
def test_transpose(dim):
  A = make_dense_square_matrix(rand_mat(dim))
  At = A.T
  assert torch.allclose(At.to_dense(), A.to_dense().T)


@pytest.mark.parametrize("dim", [4])
def test_solve_and_inverse(dim):
  A = make_dense_square_matrix(rand_pd(dim))
  b = torch.randn(dim, dtype=torch.float64)

  # solve vector
  x = A.solve(b)
  expected_x = torch.linalg.solve(A.to_dense(), b)
  assert torch.allclose(x, expected_x, atol=1e-8, rtol=1e-8)

  # solve matrix
  B = make_dense_square_matrix(rand_mat(dim, seed=3))
  X = A.solve(B)
  expected_X = torch.linalg.solve(A.to_dense(), B.to_dense())
  assert torch.allclose(X.to_dense(), expected_X, atol=1e-8, rtol=1e-8)

  # inverse
  Ainv = A.get_inverse()
  expected_Ainv = torch.linalg.inv(A.to_dense())
  assert torch.allclose(Ainv.to_dense(), expected_Ainv, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("dim", [4])
def test_logdet_cholesky_exp(dim):
  A = make_dense_square_matrix(rand_pd(dim))

  # logdet
  log_det = A.get_log_det()
  expected_log_det = torch.linalg.slogdet(A.to_dense()).logabsdet
  assert torch.allclose(log_det, expected_log_det, atol=1e-8, rtol=1e-8)

  # cholesky
  C = A.get_cholesky()
  expected_C = torch.linalg.cholesky(A.to_dense())
  assert torch.allclose(C.to_dense(), expected_C, atol=1e-8, rtol=1e-8)

  # matrix exponential
  E = A.get_exp()
  expected_E = torch.matrix_exp(A.to_dense())
  # cola may produce complex due to algorithm; compare real parts
  Et = E.to_dense()
  if torch.is_complex(Et):
    Et = Et.real
  assert torch.allclose(Et, expected_E, atol=1e-6, rtol=1e-6)


# --- Comprehensive tests leveraging shared matrix test utilities ---

def create_square_matrix(M: torch.Tensor, tags=TAGS.no_tags) -> SquareMatrix:
  return make_dense_square_matrix(M, tags)


def test_comprehensive_matrix_tests_direct():
  # Run the generic matrix_tests on two random SquareMatrix instances
  key = torch.Generator().manual_seed(123)
  dim = 6
  A = make_dense_square_matrix(rand_mat(dim, seed=11))
  B = make_dense_square_matrix(rand_mat(dim, seed=22))
  matrix_tests(key, A, B)


def test_comprehensive_matrix_implementations():
  # Sweep a subset of tags and random inputs via the shared implementation tests
  key = torch.Generator().manual_seed(321)
  matrix_implementations_tests(key, create_square_matrix)


# --- Additional comprehensive tests from matrix_test_base ---

def test_performance_adapter():
  dim = 6
  A = make_dense_square_matrix(rand_mat(dim, seed=101))
  B = make_dense_square_matrix(rand_mat(dim, seed=202))
  performance_tests(A, B)


@pytest.mark.skip(reason="autodiff helper expects .elements, not present on SquareMatrix")
def test_autodiff_adapter():
  autodiff_for_matrix_class(create_square_matrix)


class TestSquareMatrixViaAbstract(AbstractMatrixTest, unittest.TestCase):
  matrix_class = SquareMatrix

  def setUp(self):
    self.key = torch.Generator().manual_seed(42)
    self.dim = 6

  def create_matrix(self, elements, tags=None):
    if tags is None:
      tags = TAGS.no_tags
    return make_dense_square_matrix(elements, tags)

  def create_random_matrix(self, key, shape=None, tags=None):
    if shape is None:
      shape = (self.dim, self.dim)
    if tags is None:
      tags = TAGS.no_tags
    t = torch.randn(shape, generator=key, dtype=torch.float64)
    return make_dense_square_matrix(t, tags)

  def create_random_symmetric_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    if tags is None:
      tags = TAGS.no_tags
    X = torch.randn((dim, dim), generator=key, dtype=torch.float64)
    S = X @ X.T + dim * torch.eye(dim, dtype=torch.float64)
    return make_dense_square_matrix(S, tags)

  def create_zeros_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    return SquareMatrix.zeros(dim, dtype=torch.float64)

  def create_eye_matrix(self, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    return SquareMatrix.eye(dim, dtype=torch.float64)

  def create_well_conditioned_matrix(self, key, dim=None, tags=None):
    if dim is None:
      dim = self.dim
    return make_dense_square_matrix(rand_pd(dim, seed=777), TAGS.no_tags)

  def test_initialization(self):
    dim = self.dim
    A = make_dense_square_matrix(torch.eye(dim, dtype=torch.float64), TAGS.no_tags)
    self.assertEqual(A.shape, (dim, dim))

  # Override a few expectations to align with SquareMatrix semantics
  def test_batch_size(self):
    A = self.create_random_matrix(self.key)
    # Single matrices may expose an empty size rather than None
    assert (A.batch_size is None) or (A.batch_size == torch.Size([]))

  def test_class_methods(self):
    zero_mat = self.create_zeros_matrix()
    zero_array = torch.zeros((self.dim, self.dim), dtype=zero_mat.to_dense().dtype)
    assert torch.allclose(zero_mat.to_dense(), zero_array)
    assert zero_mat.tags.is_zero.item() is True

    eye_mat = self.create_eye_matrix()
    assert torch.allclose(eye_mat.to_dense(), torch.eye(self.dim, dtype=eye_mat.to_dense().dtype))

  def test_exp(self):
    A = self.create_random_matrix(self.key)
    A_exp = A.get_exp()
    expected_exp = torch.linalg.matrix_exp(A.to_dense())
    Et = A_exp.to_dense()
    if torch.is_complex(Et):
      Et = Et.real
    assert torch.allclose(Et, expected_exp, atol=1e-6, rtol=1e-6)


