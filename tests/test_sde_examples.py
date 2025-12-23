import torch
import pytest

import cola

from diffusion_helper.src.sde.sde_examples import (
  LinearTimeInvariantSDE,
  BrownianMotion,
  OrnsteinUhlenbeck,
  VariancePreserving,
  WienerVelocityModel,
  WienerAccelerationModel,
  CriticallyDampedLangevinDynamics,
  TOLD,
  StochasticHarmonicOscillator,
)
from diffusion_helper.src.sde.ode_sde_simulation import SDESolverParams, sde_sample
from diffusion_helper.src.gaussian.gaussian import StandardGaussian
import diffusion_helper.src.util as util
from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import TAGS

def _sample_lti_endpoints(
  sde,
  x0: torch.Tensor,
  s: float,
  t: float,
  *,
  n_samples: int,
  dt: float,
  seed: int,
) -> torch.Tensor:
  D = int(x0.shape[0])
  x0_stacked = x0.repeat(n_samples)

  F_dense = sde.F.to_dense()
  L_diag = torch.diagonal(sde.L.to_dense()).to(dtype=x0.dtype)
  L_stacked = L_diag.repeat(n_samples)

  def drift(tau, x):
    xr = x.reshape(n_samples, D)
    dx = xr @ F_dense.T
    return dx.reshape(-1)

  def diffusion(tau, x):
    return L_stacked

  save_times = torch.tensor([s, t], dtype=x0.dtype)
  params = SDESolverParams(dt=dt)
  generator = torch.Generator().manual_seed(seed)
  ts = sde_sample(drift, diffusion, x0_stacked, save_times, params=params, generator=generator)
  return ts.vals[-1].reshape(n_samples, D)


class TestSDEExamplesClosedFormTransitions:
  def test_brownian_motion_transition(self):
    sigma = 1.3
    dim = 3
    sde = BrownianMotion(sigma=sigma, dim=dim)

    s, t = 0.4, 1.9
    dt = t - s
    trans = sde.get_transition_distribution(s, t)

    expected_A = torch.eye(dim, dtype=torch.float64)
    expected_u = torch.zeros((dim,), dtype=torch.float64)
    expected_Sigma = (sigma ** 2) * dt * torch.eye(dim, dtype=torch.float64)

    assert torch.allclose(trans.A.to_dense(), expected_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(trans.u, expected_u, rtol=1e-12, atol=1e-12)
    assert torch.allclose(trans.Sigma.to_dense(), expected_Sigma, rtol=1e-12, atol=1e-12)

  def test_ornstein_uhlenbeck_transition(self):
    sigma = 0.7
    lambda_ = 0.9
    dim = 2
    sde = OrnsteinUhlenbeck(sigma=sigma, lambda_=lambda_, dim=dim)

    s, t = 1.2, 2.5
    dt = t - s
    trans = sde.get_transition_distribution(s, t)

    exp_term = torch.exp(torch.tensor(-lambda_ * dt, dtype=torch.float64))
    expected_A = exp_term * torch.eye(dim, dtype=torch.float64)
    expected_u = torch.zeros((dim,), dtype=torch.float64)
    Sigma_scalar = (sigma ** 2) * (1.0 - torch.exp(torch.tensor(-2.0 * lambda_ * dt, dtype=torch.float64))) / (2.0 * lambda_)
    expected_Sigma = Sigma_scalar * torch.eye(dim, dtype=torch.float64)

    assert torch.allclose(trans.A.to_dense(), expected_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(trans.u, expected_u, rtol=1e-12, atol=1e-12)
    assert torch.allclose(trans.Sigma.to_dense(), expected_Sigma, rtol=1e-12, atol=1e-12)

  def test_variance_preserving_transition(self):
    dim = 5
    beta_min = 0.1
    beta_max = 20.0
    sde = VariancePreserving(beta_min=beta_min, beta_max=beta_max, dim=dim)

    s, t = 0.2, 0.9
    trans = sde.get_transition_distribution(s, t)

    def T(x):
      return x * beta_min + 0.5 * x**2 * (beta_max - beta_min)

    dT = T(t) - T(s)
    alpha = torch.exp(torch.tensor(-0.5 * dT, dtype=torch.float64))
    expected_A = alpha * torch.eye(dim, dtype=torch.float64)
    expected_Sigma = (1.0 - torch.exp(torch.tensor(-dT, dtype=torch.float64))) * torch.eye(dim, dtype=torch.float64)

    assert torch.allclose(trans.A.to_dense(), expected_A, rtol=1e-12, atol=1e-12)
    assert torch.allclose(trans.Sigma.to_dense(), expected_Sigma, rtol=1e-12, atol=1e-12)


class TestSDEExamplesStructure:
  def test_linear_time_invariant_sde_is_wrapper(self):
    dim = 2
    F = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(torch.tensor([-0.3, -0.7], dtype=torch.float64)))
    L = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(torch.tensor([0.2, 0.4], dtype=torch.float64)))
    sde = LinearTimeInvariantSDE(F=F, L=L)
    assert sde.F.shape == (dim, dim)
    assert sde.L.shape == (dim, dim)

  def test_brownian_motion_uses_diagonal_operators(self):
    sde = BrownianMotion(sigma=1.0, dim=3)
    assert isinstance(sde.F.mat, (cola.ops.Diagonal, cola.ops.ScalarMul))
    assert isinstance(sde.L.mat, (cola.ops.Diagonal, cola.ops.ScalarMul))

  def test_ornstein_uhlenbeck_uses_diagonal_operators(self):
    sde = OrnsteinUhlenbeck(sigma=1.0, lambda_=0.5, dim=4)
    assert isinstance(sde.F.mat, (cola.ops.Diagonal, cola.ops.ScalarMul))
    assert isinstance(sde.L.mat, (cola.ops.Diagonal, cola.ops.ScalarMul))

  def test_variance_preserving_uses_scalar_mul_matrices(self):
    sde = VariancePreserving(beta_min=0.2, beta_max=5.0, dim=3)
    F, u, L = sde.get_params(0.5)
    assert isinstance(F.mat, cola.ops.ScalarMul)
    assert isinstance(L.mat, cola.ops.ScalarMul)
    assert torch.allclose(u, torch.zeros_like(u))

  def test_wiener_velocity_model_F_is_kron_shift(self):
    position_dim = 2
    order = 3
    sde = WienerVelocityModel(sigma=0.5, position_dim=position_dim, order=order)

    Ipos = torch.eye(position_dim, dtype=torch.float64)
    shift = torch.zeros((order, order), dtype=torch.float64)
    shift = shift + torch.diag(torch.ones(order - 1, dtype=torch.float64), diagonal=1)
    expected = torch.kron(shift, Ipos)

    assert torch.allclose(sde.F.to_dense(), expected, rtol=0.0, atol=0.0)

  def test_wiener_velocity_model_sigma_vector_construction(self):
    position_dim = 3
    order = 4
    sigma = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    sde = WienerVelocityModel(sigma=sigma, position_dim=position_dim, order=order)

    diag = torch.diagonal(sde.L.to_dense())
    expected = torch.repeat_interleave(torch.concatenate([torch.zeros(1, dtype=torch.float64), sigma], dim=0), position_dim)
    assert torch.allclose(diag, expected, rtol=0.0, atol=0.0)

  def test_wiener_acceleration_model_diffusion_only_on_highest_order(self):
    position_dim = 2
    sigma = 0.7
    sde = WienerAccelerationModel(sigma=sigma, position_dim=position_dim)

    diag = torch.diagonal(sde.L.to_dense())
    assert torch.allclose(diag[:-position_dim], torch.zeros_like(diag[:-position_dim]))
    assert torch.allclose(diag[-position_dim:], torch.ones(position_dim, dtype=torch.float64) * sigma)

  def test_critically_damped_shapes(self):
    dim = 3
    sde = CriticallyDampedLangevinDynamics(mass=1.5, beta=0.7, dim=dim)
    assert sde.F.shape == (2 * dim, 2 * dim)
    assert sde.L.shape == (2 * dim, 2 * dim)

  def test_critically_damped_structure(self):
    dim = 4
    sde = CriticallyDampedLangevinDynamics(mass=1.3, beta=0.9, dim=dim)
    assert isinstance(sde.F.mat, cola.ops.BlockDiag)
    assert isinstance(sde.L.mat, cola.ops.Diagonal)

  def test_told_structure(self):
    dim = 3
    sde = TOLD(L=2.0, dim=dim)
    assert isinstance(sde.F.mat, cola.ops.BlockDiag)
    assert isinstance(sde.L.mat, cola.ops.Diagonal)

  def test_stochastic_harmonic_oscillator_structure(self):
    dim = 3
    sde = StochasticHarmonicOscillator(freq=1.2, coeff=0.1, sigma=0.3, observation_dim=dim)
    assert isinstance(sde.F.mat, cola.ops.BlockDiag)
    assert isinstance(sde.L.mat, cola.ops.Diagonal)


class TestSDEExamplesSamplingMatchesTransition:
  def test_wiener_velocity_model_sampling_matches_transition(self):
    torch.manual_seed(0)

    position_dim = 2
    order = 3
    sde = WienerVelocityModel(sigma=0.6, position_dim=position_dim, order=order)
    D = position_dim * order
    x0 = torch.linspace(-1.0, 1.0, D, dtype=torch.float64)

    s, t = 0.0, 0.5
    trans = sde.get_transition_distribution(s, t)
    predicted = StandardGaussian(
      trans.A.to_dense() @ x0 + trans.u,
      trans.Sigma,
      logZ=torch.tensor(0.0, dtype=torch.float64),
    )

    xT = _sample_lti_endpoints(sde, x0, s, t, n_samples=12000, dt=0.002, seed=2)
    empirical = util.empirical_dist(xT)
    w2 = util.w2_distance(empirical, predicted)
    assert w2 < 3e-3

  def test_told_sampling_matches_transition(self):
    torch.manual_seed(0)

    dim = 2
    sde = TOLD(L=1.7, dim=dim)
    D = 3 * dim
    x0 = torch.tensor([0.2, -0.3, 0.5, -0.4, 0.1, -0.2], dtype=torch.float64)

    s, t = 0.0, 0.4
    trans = sde.get_transition_distribution(s, t)
    predicted = StandardGaussian(
      trans.A.to_dense() @ x0 + trans.u,
      trans.Sigma,
      logZ=torch.tensor(0.0, dtype=torch.float64),
    )

    xT = _sample_lti_endpoints(sde, x0, s, t, n_samples=12000, dt=0.001, seed=3)
    empirical = util.empirical_dist(xT)
    w2 = util.w2_distance(empirical, predicted)
    assert w2 < 4e-3

  def test_stochastic_harmonic_oscillator_sampling_matches_transition(self):
    torch.manual_seed(0)

    dim = 2
    sde = StochasticHarmonicOscillator(freq=1.3, coeff=0.2, sigma=0.4, observation_dim=dim)

    x0 = torch.tensor([0.8, -1.1, 0.3, -0.2], dtype=torch.float64)
    s, t = 0.0, 0.6
    trans = sde.get_transition_distribution(s, t)
    predicted = StandardGaussian(
      trans.A.to_dense() @ x0 + trans.u,
      trans.Sigma,
      logZ=torch.tensor(0.0, dtype=torch.float64),
    )

    n_samples = 10000
    x0_stacked = x0.repeat(n_samples)
    D = x0.shape[0]

    F_dense = sde.F.to_dense()
    L_diag = torch.diagonal(sde.L.to_dense())
    L_stacked = L_diag.repeat(n_samples)

    def drift(tau, x):
      xr = x.reshape(n_samples, D)
      dx = xr @ F_dense.T
      return dx.reshape(-1)

    def diffusion(tau, x):
      return L_stacked

    save_times = torch.tensor([s, t], dtype=torch.float64)
    params = SDESolverParams(dt=0.001)
    generator = torch.Generator().manual_seed(0)
    ts = sde_sample(drift, diffusion, x0_stacked, save_times, params=params, generator=generator)

    xT = ts.vals[-1].reshape(n_samples, D)
    empirical = util.empirical_dist(xT)

    w2 = util.w2_distance(empirical, predicted)
    assert w2 < 2e-3


if __name__ == "__main__":
  pytest.main([__file__])

