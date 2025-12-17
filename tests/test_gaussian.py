import torch
import pytest
from pathlib import Path
import sys
import cola

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore
from diffusion_helper.src.gaussian.gaussian import StandardGaussian, MixedGaussian  # type: ignore
from diffusion_helper.src.matrix.tags import TAGS  # type: ignore


def pd_dense(dim: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  X = torch.randn((dim, dim), generator=g, dtype=dtype)
  return X @ X.T + dim * torch.eye(dim, dtype=dtype)


def make_cov(dim: int, seed: int = 0) -> SquareMatrix:
  return SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Dense(pd_dense(dim, seed)))


def make_prec(dim: int, seed: int = 0) -> SquareMatrix:
  # Use PD then invert to be safe
  cov = cola.ops.Dense(pd_dense(dim, seed))
  J = cola.linalg.inv(cov)
  return SquareMatrix(tags=TAGS.no_tags, mat=J)


@pytest.mark.parametrize("dim", [2, 4])
def test_standard_gaussian_construct_and_score(dim):
  mu = torch.randn(dim, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
  Sigma = make_cov(dim, seed=2)
  gstd = StandardGaussian(mu=mu, Sigma=Sigma)

  assert gstd.dim == dim

  x = torch.randn(dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
  s = gstd.score(x)
  expected = Sigma.solve(mu - x)
  assert torch.allclose(s, expected)


@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_gaussian_construct_and_score(dim):
  mu = torch.randn(dim, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
  J = make_prec(dim, seed=11)
  gmix = MixedGaussian(mu=mu, J=J)

  assert gmix.dim == dim
  x = torch.randn(dim, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
  s = gmix.score(x)
  expected = J @ (mu - x)
  assert torch.allclose(s, expected)


