import unittest
import torch
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.gaussian.transition import GaussianTransition  # type: ignore
from diffusion_helper.src.gaussian.gaussian import StandardGaussian, MixedGaussian  # type: ignore
from diffusion_helper.src.functional.linear_functional import LinearFunctional  # type: ignore
from diffusion_helper.src.functional.funtional_ops import resolve_functional  # type: ignore
from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore


def make_spd(dim: int, seed: int) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  M = torch.randn((dim, dim), generator=g, dtype=torch.float64)
  return M @ M.T + 1e-4 * torch.eye(dim, dtype=torch.float64)


class TestFunctionalGaussianTransition(unittest.TestCase):
  def setUp(self):
    torch.set_default_dtype(torch.float64)
    self.dim_x = 4
    self.dim_y = 4
    self.latent_dim = 4

    # Base transition
    A = SquareMatrix.from_dense(torch.randn((self.dim_y, self.dim_x), generator=torch.Generator().manual_seed(0), dtype=torch.float64))
    u = torch.randn((self.dim_y,), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    Sigma = SquareMatrix.from_dense(make_spd(self.dim_y, 2))
    self.transition = GaussianTransition(A, u, Sigma, logZ=torch.tensor(0.0, dtype=torch.float64))

    # Latent variable
    self.x = torch.randn((self.latent_dim,), generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    # Functional and standard potentials over y
    J = SquareMatrix.from_dense(make_spd(self.dim_y, 4))

    mu_A = SquareMatrix.from_dense(torch.randn((self.dim_y, self.latent_dim), generator=torch.Generator().manual_seed(5), dtype=torch.float64))
    mu_b = torch.randn((self.dim_y,), generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    self.mu_lf = LinearFunctional(mu_A, mu_b)
    self.mu_vec = self.mu_lf(self.x)

    self.sg_functional = StandardGaussian(mu=self.mu_lf, Sigma=J.get_inverse())
    self.sg_standard = StandardGaussian(mu=self.mu_vec, Sigma=J.get_inverse())

    self.mg_functional = MixedGaussian(mu=self.mu_lf, J=J)
    self.mg_standard = MixedGaussian(mu=self.mu_vec, J=J)

  def _resolve(self, obj):
    return resolve_functional(obj, self.x)

  def test_update_y_with_standard_functional(self):
    out_func = self.transition.update_y(self.sg_functional)
    out_std = self.transition.update_y(self.sg_standard)

    # A and Sigma should match exactly
    self.assertTrue(torch.allclose(out_func.A.to_dense(), out_std.A.to_dense()))
    self.assertTrue(torch.allclose(out_func.Sigma.to_dense(), out_std.Sigma.to_dense()))

    # Resolve functional u and compare
    u_func = out_func.u(self.x) if isinstance(out_func.u, LinearFunctional) else out_func.u
    self.assertTrue(torch.allclose(u_func, out_std.u))

    # Additionally check conditioning equivalence
    cond_func = out_func.condition_on_x(self.x)
    cond_std = out_std.condition_on_x(self.x)
    mu_func = cond_func.mu(self.x) if isinstance(cond_func.mu, LinearFunctional) else cond_func.mu
    self.assertTrue(torch.allclose(mu_func, cond_std.mu))
    self.assertTrue(torch.allclose(cond_func.Sigma.to_dense(), cond_std.Sigma.to_dense()))

  def test_update_y_with_mixed_functional(self):
    out_func = self.transition.update_y(self.mg_functional)
    out_std = self.transition.update_y(self.mg_standard)

    # A and Sigma should match exactly
    self.assertTrue(torch.allclose(out_func.A.to_dense(), out_std.A.to_dense()))
    self.assertTrue(torch.allclose(out_func.Sigma.to_dense(), out_std.Sigma.to_dense()))

    # Resolve functional u and compare
    u_func = out_func.u(self.x) if isinstance(out_func.u, LinearFunctional) else out_func.u
    self.assertTrue(torch.allclose(u_func, out_std.u))

    # Additionally check conditioning equivalence
    cond_func = out_func.condition_on_x(self.x)
    cond_std = out_std.condition_on_x(self.x)
    mu_func = cond_func.mu(self.x) if isinstance(cond_func.mu, LinearFunctional) else cond_func.mu
    self.assertTrue(torch.allclose(mu_func, cond_std.mu))
    self.assertTrue(torch.allclose(cond_func.Sigma.to_dense(), cond_std.Sigma.to_dense()))


if __name__ == '__main__':
  unittest.main()


