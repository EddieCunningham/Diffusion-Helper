import torch
import pytest
import cola
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore
from diffusion_helper.src.matrix.tags import TAGS  # type: ignore


def make(Aop) -> SquareMatrix:
  # Identity has no parameters; SquareMatrix TensorClass cannot store an empty param list.
  # Wrap as Dense eye for construction while preserving semantics.
  from cola.ops.operators import Identity as _Identity
  if isinstance(Aop, _Identity):
    eye = torch.eye(Aop.shape[0], dtype=Aop.dtype)
    Aop = cola.ops.Dense(eye)
  return SquareMatrix(tags=TAGS.no_tags, mat=Aop)


@pytest.mark.parametrize("dim", [3])
def test_identity_and_diagonal(dim):
  I = cola.ops.Identity((dim, dim), dtype=torch.float64)
  D = cola.ops.Diagonal(torch.arange(1, dim + 1, dtype=torch.float64))
  Ai = make(I)
  Ad = make(D)
  mat = torch.randn((dim, dim), dtype=torch.float64)
  assert torch.allclose(Ai.to_dense(), torch.eye(dim, dtype=torch.float64))
  assert torch.allclose(Ad.to_dense(), torch.diag(torch.arange(1, dim + 1, dtype=torch.float64)))
  assert torch.allclose((Ai @ Ad).to_dense(), Ad.to_dense())
  assert torch.allclose((Ad @ Ai).to_dense(), Ad.to_dense())


@pytest.mark.parametrize("dim", [4])
def test_triangular(dim):
  M = torch.randn((dim, dim), dtype=torch.float64)
  L = torch.tril(M)
  U = torch.triu(M)
  Lt = cola.ops.Triangular(L, lower=True)
  Ut = cola.ops.Triangular(U, lower=False)
  Al = make(Lt)
  Au = make(Ut)
  v = torch.randn(dim, dtype=torch.float64)
  assert torch.allclose(Al.to_dense(), L)
  assert torch.allclose(Au.to_dense(), U)
  assert torch.allclose(Al @ v, L @ v)
  assert torch.allclose(Au @ v, U @ v)


@pytest.mark.parametrize("dim", [2])
def test_product_and_sum(dim):
  A = cola.ops.Dense(torch.randn((dim, dim), dtype=torch.float64))
  B = cola.ops.Dense(torch.randn((dim, dim), dtype=torch.float64))
  P = cola.ops.Product(A, B)
  S = cola.ops.Sum(A, B)
  Ap = make(P)
  As = make(S)
  assert torch.allclose(Ap.to_dense(), A.to_dense() @ B.to_dense())
  assert torch.allclose(As.to_dense(), A.to_dense() + B.to_dense())


@pytest.mark.parametrize("dim", [2])
def test_kronecker_and_kronsum(dim):
  A = cola.ops.Dense(torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64))
  B = cola.ops.Dense(torch.tensor([[0., 5.], [6., 7.]], dtype=torch.float64))
  K = cola.ops.Kronecker(A, B)
  Ks = cola.ops.KronSum(A, B)
  Ak = make(K)
  Aks = make(Ks)
  expect_k = torch.kron(A.to_dense(), B.to_dense())
  expect_ks = torch.kron(A.to_dense(), torch.eye(dim)) + torch.kron(torch.eye(dim), B.to_dense())
  assert torch.allclose(Ak.to_dense(), expect_k)
  assert torch.allclose(Aks.to_dense(), expect_ks)


@pytest.mark.parametrize("dim", [2])
def test_blockdiag(dim):
  A = cola.ops.Dense(torch.randn((dim, dim), dtype=torch.float64))
  B = cola.ops.Dense(torch.randn((dim, dim), dtype=torch.float64))
  BD = cola.ops.BlockDiag(A, B)
  Abd = make(BD)
  expect = torch.block_diag(A.to_dense(), B.to_dense())
  assert torch.allclose(Abd.to_dense(), expect)


@pytest.mark.parametrize("dim", [3])
def test_transpose_and_adjoint(dim):
  A = cola.ops.Dense(torch.randn((dim, dim), dtype=torch.float64))
  T = cola.ops.Transpose(A)
  H = cola.ops.Adjoint(A)
  At = make(T)
  Ah = make(H)
  assert torch.allclose(At.to_dense(), A.to_dense().T)
  assert torch.allclose(Ah.to_dense(), A.to_dense().T)  # real dtype


@pytest.mark.parametrize("n", [4])
def test_tridiagonal(n):
  alpha = torch.randn(n - 1, dtype=torch.float64)
  beta = torch.randn(n, dtype=torch.float64)
  gamma = torch.randn(n - 1, dtype=torch.float64)
  T = cola.ops.Tridiagonal(alpha, beta, gamma)
  At = make(T)
  dense = torch.diag(beta) + torch.diag(alpha, -1) + torch.diag(gamma, 1)
  assert torch.allclose(At.to_dense(), dense)


