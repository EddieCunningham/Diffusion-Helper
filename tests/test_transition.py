import torch
import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix  # type: ignore
from diffusion_helper.src.matrix.tags import TAGS  # type: ignore
from diffusion_helper.src.gaussian.gaussian import StandardGaussian, MixedGaussian  # type: ignore
from diffusion_helper.src.gaussian.transition import GaussianTransition  # type: ignore


def make_spd_matrix(dim: int, seed: int = 0) -> SquareMatrix:
  g = torch.Generator().manual_seed(seed)
  M = torch.randn((dim, dim), generator=g, dtype=torch.float64)
  SPD = M @ M.T + dim * torch.eye(dim, dtype=torch.float64)
  return SquareMatrix.from_dense(SPD)


class TestGaussianTransition(unittest.TestCase):
  def setUp(self):
    torch.set_default_dtype(torch.float64)
    self.dim = 4
    self.rng = torch.Generator().manual_seed(42)

  def make_transition(self, seed: int = 0) -> GaussianTransition:
    A_dense = torch.randn((self.dim, self.dim), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    A = SquareMatrix.from_dense(A_dense)
    u = torch.randn((self.dim,), generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
    Sigma = make_spd_matrix(self.dim, seed + 2)
    logZ = torch.tensor(1.3, dtype=torch.float64)
    return GaussianTransition(A, u, Sigma, logZ)

  def make_standard_potential(self, seed: int = 0) -> StandardGaussian:
    mu = torch.randn((self.dim,), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    Sigma = make_spd_matrix(self.dim, seed + 1)
    return StandardGaussian(mu, Sigma)

  def make_mixed_potential(self, seed: int = 0) -> MixedGaussian:
    mu = torch.randn((self.dim,), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    J = make_spd_matrix(self.dim, seed + 1)
    return MixedGaussian(mu, J)

  def test_basic_construction(self):
    t = self.make_transition(0)
    self.assertIsInstance(t.A, SquareMatrix)
    self.assertIsInstance(t.Sigma, SquareMatrix)
    self.assertEqual(t.u.shape, (self.dim,))
    self.assertEqual(t.logZ.shape, ())

  def test_no_op_like(self):
    t = self.make_transition(1)
    no_op = GaussianTransition.no_op_like(t)
    self.assertTrue(torch.allclose(no_op.A.to_dense(), torch.eye(self.dim, dtype=torch.float64)))
    self.assertTrue(torch.allclose(no_op.u, torch.zeros(self.dim, dtype=torch.float64)))
    self.assertTrue(torch.allclose(no_op.Sigma.to_dense(), torch.zeros((self.dim, self.dim), dtype=torch.float64)))

  def test_normalizing_constant(self):
    t = self.make_transition(2)
    nc = t.normalizing_constant()
    self.assertTrue(torch.isfinite(nc))

  def test_condition_on_x(self):
    t = self.make_transition(3)
    x = torch.randn((self.dim,), generator=torch.Generator().manual_seed(99), dtype=torch.float64)
    cond = t.condition_on_x(x)
    self.assertIsInstance(cond, StandardGaussian)
    self.assertEqual(cond.mu.shape, (self.dim,))
    self.assertEqual(cond.Sigma.to_dense().shape, (self.dim, self.dim))
    # __call__ consistency
    y = torch.randn((self.dim,), generator=torch.Generator().manual_seed(100), dtype=torch.float64)
    self.assertTrue(torch.allclose(t(y, x), cond(y)))

  def test_swap_variables(self):
    t = self.make_transition(4)
    swapped = t.swap_variables()
    # swap twice should approximately return original parameters (up to symmetry enforcement)
    unswapped = swapped.swap_variables()
    self.assertTrue(torch.allclose(unswapped.A.to_dense(), t.A.to_dense(), atol=1e-8, rtol=1e-8))
    self.assertTrue(torch.allclose(unswapped.u, t.u, atol=1e-8, rtol=1e-8))
    self.assertTrue(torch.allclose(unswapped.Sigma.to_dense(), t.Sigma.to_dense(), atol=1e-8, rtol=1e-8))

  def test_chain_operation(self):
    t = self.make_transition(5)
    chained = t.chain(t)
    self.assertIsInstance(chained, GaussianTransition)
    self.assertEqual(chained.u.shape, (self.dim,))
    self.assertEqual(chained.Sigma.to_dense().shape, (self.dim, self.dim))

  def test_update_y_standard(self):
    t = self.make_transition(6)
    pot = self.make_standard_potential(7)
    out = t.update_y(pot)
    self.assertIsInstance(out, GaussianTransition)

  def test_update_y_mixed(self):
    t = self.make_transition(8)
    pot = self.make_mixed_potential(9)
    out = t.update_y(pot)
    self.assertIsInstance(out, GaussianTransition)

  def test_marginalize_out_y(self):
    t = self.make_transition(10)
    marg = t.marginalize_out_y()
    self.assertIsInstance(marg, StandardGaussian)
    self.assertTrue(torch.all(torch.isfinite(marg.mu)))
    # Sigma may contain inf or NaN depending on backend handling; only enforce mu validity here

  def test_update_and_marginalize_out_x_equivalence(self):
    t = self.make_transition(11)
    pot = self.make_standard_potential(12)

    out = t.update_and_marginalize_out_x(pot)

    A, u, Sigma = t.A, t.u, t.Sigma
    mu_p, Sigma_p = pot.mu, pot.Sigma
    new_mean = A@mu_p + u
    new_cov = Sigma + A@Sigma_p@A.T
    new_cov = new_cov.set_symmetric()
    expected = StandardGaussian(new_mean, new_cov)
    correction = pot.logZ - pot.normalizing_constant()
    correction += t.logZ - t.normalizing_constant()
    expected = StandardGaussian(new_mean, new_cov, expected.logZ + correction)

    self.assertTrue(torch.allclose(out.mu, expected.mu, atol=1e-10, rtol=1e-10))
    self.assertTrue(torch.allclose(out.Sigma.to_dense(), expected.Sigma.to_dense(), atol=1e-10, rtol=1e-10))
    self.assertTrue(torch.allclose(out.logZ, expected.logZ, atol=1e-10, rtol=1e-10))


if __name__ == '__main__':
  unittest.main()


