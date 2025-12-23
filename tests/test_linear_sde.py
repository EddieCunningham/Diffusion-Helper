import torch
import pytest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.sde.linear_sde import (
  AbstractSDE,
  AbstractLinearSDE,
  AbstractLinearTimeInvariantSDE,
  TimeScaledLinearTimeInvariantSDE,
)
from diffusion_helper.src.sde.ode_sde_simulation import SDESolverParams, sde_sample
from diffusion_helper.src.matrix.matrix import SquareMatrix
import diffusion_helper.src.util as util
from diffusion_helper.src.gaussian.gaussian import StandardGaussian
import cola
from diffusion_helper.src.matrix.tags import TAGS


class _BrownianMotion(AbstractLinearTimeInvariantSDE):
  def __init__(self, sigma: float, dim: int, dtype: torch.dtype = torch.float64):
    self._sigma = float(sigma)
    self.F = SquareMatrix.from_dense(torch.zeros((dim, dim), dtype=dtype))
    self.L = SquareMatrix.from_dense(torch.eye(dim, dtype=dtype) * self._sigma)


class _OrnsteinUhlenbeck(AbstractLinearTimeInvariantSDE):
  def __init__(self, sigma: float, lambda_: float, dim: int, dtype: torch.dtype = torch.float64):
    self._sigma = float(sigma)
    self._lambda = float(lambda_)
    self.F = SquareMatrix.from_dense(-self._lambda * torch.eye(dim, dtype=dtype))
    self.L = SquareMatrix.from_dense(torch.eye(dim, dtype=dtype) * self._sigma)


class _ConstantLinearSDE(AbstractLinearSDE):
  def __init__(self, F: SquareMatrix, u: torch.Tensor, L: SquareMatrix):
    self._F = F
    self._u = u
    self._L = L

  def get_params(self, t):
    return self._F, self._u, self._L


class TestAbstractSDE:
  def test_abstract_sde_cannot_be_instantiated(self):
    with pytest.raises(TypeError):
      AbstractSDE()


class TestAbstractLinearSDE:
  def test_abstract_linear_sde_cannot_be_instantiated(self):
    with pytest.raises(TypeError):
      AbstractLinearSDE()


class TestAbstractLinearTimeInvariantSDE:
  def test_lti_sde_requires_F_and_L_for_use(self):
    sde = AbstractLinearTimeInvariantSDE()
    with pytest.raises(AttributeError):
      _ = sde.dim


class TestTimeScaledLinearTimeInvariantSDE:
  class _MinimalLTI(AbstractLinearTimeInvariantSDE):
    def __init__(self, F: SquareMatrix, L: SquareMatrix, order: int | None = None):
      self.F = F
      self.L = L
      if order is not None:
        self.order = order

  def test_time_scaling_initialization(self):
    base_sde = _BrownianMotion(sigma=1.0, dim=2)
    scaled_sde = TimeScaledLinearTimeInvariantSDE(base_sde, 2.0)
    assert scaled_sde.sde is base_sde
    assert scaled_sde.time_scale == 2.0

  def test_time_scaling_parameters(self):
    base_sde = _OrnsteinUhlenbeck(sigma=1.0, lambda_=0.5, dim=2)
    time_scale = 2.0
    scaled_sde = TimeScaledLinearTimeInvariantSDE(base_sde, time_scale)

    assert torch.allclose(scaled_sde.F.to_dense(), base_sde.F.to_dense() * time_scale)

    expected_L = base_sde.L.to_dense() * torch.sqrt(torch.tensor(time_scale, dtype=base_sde.L.mat.dtype))
    assert torch.allclose(scaled_sde.L.to_dense(), expected_L)

  def test_time_scaling_transitions(self):
    base_sde = _BrownianMotion(sigma=1.0, dim=2)
    time_scale = 2.0
    scaled_sde = TimeScaledLinearTimeInvariantSDE(base_sde, time_scale)

    s, t = 0.0, 1.0
    base_transition = base_sde.get_transition_distribution(s * time_scale, t * time_scale)
    scaled_transition = scaled_sde.get_transition_distribution(s, t)

    assert torch.allclose(base_transition.A.to_dense(), scaled_transition.A.to_dense())
    assert torch.allclose(base_transition.u, scaled_transition.u)
    assert torch.allclose(base_transition.Sigma.to_dense(), scaled_transition.Sigma.to_dense())

  def test_order_attribute_passthrough(self):
    F = SquareMatrix.from_dense(torch.zeros((2, 2), dtype=torch.float64))
    L = SquareMatrix.from_dense(torch.eye(2, dtype=torch.float64))
    base_sde = self._MinimalLTI(F=F, L=L, order=2)
    scaled_sde = TimeScaledLinearTimeInvariantSDE(base_sde, 2.0)
    assert scaled_sde.order == 2

  def test_order_attribute_error(self):
    base_sde = _BrownianMotion(sigma=1.0, dim=2)
    scaled_sde = TimeScaledLinearTimeInvariantSDE(base_sde, 2.0)
    with pytest.raises(AttributeError, match="does not have an order"):
      _ = scaled_sde.order


class TestLinearSDETransitions:
  def test_brownian_motion_transitions_closed_form(self):
    sigma = 1.3
    dim = 4
    sde = _BrownianMotion(sigma=sigma, dim=dim)

    s, t = 0.2, 1.7
    dt = t - s
    transition = sde.get_transition_distribution(s, t)

    expected_A = torch.eye(dim, dtype=torch.float64)
    expected_u = torch.zeros(dim, dtype=torch.float64)
    expected_Sigma = (sigma ** 2) * dt * torch.eye(dim, dtype=torch.float64)

    assert torch.allclose(transition.A.to_dense(), expected_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(transition.u, expected_u, rtol=1e-12, atol=1e-12)
    assert torch.allclose(transition.Sigma.to_dense(), expected_Sigma, rtol=1e-12, atol=1e-12)

  def test_ornstein_uhlenbeck_transitions_closed_form(self):
    sigma = 0.7
    lambda_ = 0.9
    dim = 3
    sde = _OrnsteinUhlenbeck(sigma=sigma, lambda_=lambda_, dim=dim)

    s, t = 1.2, 2.5
    dt = t - s
    transition = sde.get_transition_distribution(s, t)

    exp_term = torch.exp(torch.tensor(-lambda_ * dt, dtype=torch.float64))
    expected_A = exp_term * torch.eye(dim, dtype=torch.float64)
    expected_u = torch.zeros(dim, dtype=torch.float64)
    Sigma_scalar = (sigma ** 2) * (1.0 - torch.exp(torch.tensor(-2.0 * lambda_ * dt, dtype=torch.float64))) / (2.0 * lambda_)
    expected_Sigma = Sigma_scalar * torch.eye(dim, dtype=torch.float64)

    assert torch.allclose(transition.A.to_dense(), expected_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(transition.u, expected_u, rtol=1e-12, atol=1e-12)
    assert torch.allclose(transition.Sigma.to_dense(), expected_Sigma, rtol=1e-12, atol=1e-12)

  def test_ode_based_transition_matches_van_loan_for_constant_coefficients(self):
    dim = 2
    dtype = torch.float64

    F = SquareMatrix.from_dense(torch.tensor([[-0.3, 0.2], [-0.1, -0.5]], dtype=dtype))
    u = torch.tensor([0.1, -0.2], dtype=dtype)
    L = SquareMatrix.from_dense(torch.tensor([[0.4, 0.0], [0.0, 0.2]], dtype=dtype))

    sde_general = _ConstantLinearSDE(F=F, u=u, L=L)

    class _LTI(AbstractLinearTimeInvariantSDE):
      def __init__(self, F: SquareMatrix, L: SquareMatrix):
        self.F = F
        self.L = L

    sde_lti = _LTI(F=F, L=L)

    s, t = 0.0, 0.8
    general = sde_general.get_transition_distribution(s, t)
    lti = sde_lti.get_transition_distribution(s, t)

    assert torch.allclose(general.A.to_dense(), lti.A.to_dense(), rtol=2e-6, atol=2e-8)
    assert torch.allclose(general.Sigma.to_dense(), lti.Sigma.to_dense(), rtol=2e-6, atol=2e-8)

  def test_lti_scalar_matrices_use_scalar_mul_and_closed_form(self):
    class _LTI(AbstractLinearTimeInvariantSDE):
      def __init__(self, F: SquareMatrix, L: SquareMatrix):
        self.F = F
        self.L = L

    dim = 5
    dtype = torch.float64
    f = torch.tensor(-0.7, dtype=dtype)
    l = torch.tensor(0.4, dtype=dtype)
    F = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.ScalarMul(f, shape=(dim, dim), dtype=dtype))
    L = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.ScalarMul(l, shape=(dim, dim), dtype=dtype))
    sde = _LTI(F=F, L=L)

    s, t = 0.0, 1.3
    dt = torch.tensor(t - s, dtype=dtype)
    trans = sde.get_transition_distribution(s, t)

    assert isinstance(trans.A.mat, cola.ops.ScalarMul)
    assert isinstance(trans.Sigma.mat, cola.ops.ScalarMul)

    expected_A = torch.exp(f * dt) * torch.eye(dim, dtype=dtype)
    Sigma_c = (l * l) * (torch.exp(2.0 * f * dt) - 1.0) / (2.0 * f)
    expected_Sigma = Sigma_c * torch.eye(dim, dtype=dtype)

    assert torch.allclose(trans.A.to_dense(), expected_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(trans.Sigma.to_dense(), expected_Sigma, rtol=1e-12, atol=1e-12)

  def test_lti_diagonal_matrices_use_diagonal_and_closed_form(self):
    class _LTI(AbstractLinearTimeInvariantSDE):
      def __init__(self, F: SquareMatrix, L: SquareMatrix):
        self.F = F
        self.L = L

    dim = 4
    dtype = torch.float64
    f = torch.tensor([-0.5, -0.2, 0.0, -1.1], dtype=dtype)
    l = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=dtype)
    F = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(f))
    L = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(l))
    sde = _LTI(F=F, L=L)

    s, t = 0.0, 0.9
    dt = torch.tensor(t - s, dtype=dtype)
    trans = sde.get_transition_distribution(s, t)

    assert isinstance(trans.A.mat, cola.ops.Diagonal)
    assert isinstance(trans.Sigma.mat, cola.ops.Diagonal)

    expected_A_diag = torch.exp(f * dt)
    denom = 2.0 * f
    expected_Sigma_diag = torch.where(
      torch.abs(denom) < 1e-12,
      (l * l) * dt,
      (l * l) * (torch.exp(2.0 * f * dt) - 1.0) / denom,
    )

    assert torch.allclose(torch.diagonal(trans.A.to_dense()), expected_A_diag, rtol=1e-12, atol=1e-12)
    assert torch.allclose(torch.diagonal(trans.Sigma.to_dense()), expected_Sigma_diag, rtol=1e-12, atol=1e-12)

  def test_samples_match_transition_distribution_brownian_diagonal_diffusion(self):
    torch.manual_seed(0)

    dim = 6
    sigma = torch.linspace(0.2, 1.1, dim, dtype=torch.float64)
    F = SquareMatrix.from_dense(torch.zeros((dim, dim), dtype=torch.float64))
    u = torch.zeros((dim,), dtype=torch.float64)
    L = SquareMatrix.from_dense(torch.diag(sigma))
    sde = _ConstantLinearSDE(F=F, u=u, L=L)

    x0 = torch.linspace(-1.0, 1.0, dim, dtype=torch.float64)
    s, t = 0.0, 1.3

    transition = sde.get_transition_distribution(s, t)
    predicted = StandardGaussian(
      transition.A.to_dense() @ x0 + transition.u,
      transition.Sigma,
    )

    n_samples = 12000
    x0_stacked = x0.repeat(n_samples)
    sigma_stacked = sigma.repeat(n_samples)

    def drift(tau, x):
      return torch.zeros_like(x)

    def diffusion(tau, x):
      return sigma_stacked

    save_times = torch.tensor([s, t], dtype=torch.float64)
    params = SDESolverParams(dt=0.01)
    generator = torch.Generator().manual_seed(0)
    ts = sde_sample(drift, diffusion, x0_stacked, save_times, params=params, generator=generator)

    xT = ts.vals[-1].reshape(n_samples, dim)
    empirical = util.empirical_dist(xT)

    w2 = util.w2_distance(empirical, predicted)
    assert w2 < 1.2e-3

  def test_samples_match_transition_distribution_constant_linear_drift(self):
    torch.manual_seed(1)

    dim = 4
    F_dense = torch.tensor(
      [
        [-0.6, 1.2, 0.0, 0.0],
        [-1.1, -0.4, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.8],
        [0.0, 0.0, -0.7, -0.5],
      ],
      dtype=torch.float64,
    )
    F = SquareMatrix.from_dense(F_dense)
    u = torch.tensor([0.2, -0.1, 0.05, -0.15], dtype=torch.float64)
    sigma = torch.tensor([0.4, 0.3, 0.2, 0.25], dtype=torch.float64)
    L = SquareMatrix.from_dense(torch.diag(sigma))
    sde = _ConstantLinearSDE(F=F, u=u, L=L)

    x0 = torch.tensor([0.9, -1.4, 0.2, 0.7], dtype=torch.float64)
    s, t = 0.0, 0.9

    transition = sde.get_transition_distribution(s, t)
    predicted = StandardGaussian(
      transition.A.to_dense() @ x0 + transition.u,
      transition.Sigma,
    )

    n_samples = 14000
    x0_stacked = x0.repeat(n_samples)
    sigma_stacked = sigma.repeat(n_samples)

    def drift(tau, x):
      xr = x.reshape(n_samples, dim)
      dx = xr @ F_dense.T + u
      return dx.reshape(-1)

    def diffusion(tau, x):
      return sigma_stacked

    save_times = torch.tensor([s, t], dtype=torch.float64)
    params = SDESolverParams(dt=0.002)
    generator = torch.Generator().manual_seed(1)
    ts = sde_sample(drift, diffusion, x0_stacked, save_times, params=params, generator=generator)

    xT = ts.vals[-1].reshape(n_samples, dim)
    empirical = util.empirical_dist(xT)

    w2 = util.w2_distance(empirical, predicted)
    assert w2 < 1e-4


if __name__ == "__main__":
  pytest.main([__file__])