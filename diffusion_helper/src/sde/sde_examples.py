import torch
from typing import Optional, Tuple, Union

from jaxtyping import Scalar

import cola

from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import TAGS
from diffusion_helper.src.gaussian.transition import GaussianTransition
from diffusion_helper.src.sde.linear_sde import AbstractLinearSDE, AbstractLinearTimeInvariantSDE

__all__ = [
  "LinearTimeInvariantSDE",
  "BrownianMotion",
  "OrnsteinUhlenbeck",
  "VariancePreserving",
  "WienerVelocityModel",
  "WienerAccelerationModel",
  "CriticallyDampedLangevinDynamics",
  "TOLD",
  "StochasticHarmonicOscillator",
]

def _diag_square(diag: torch.Tensor, *, tags=None) -> SquareMatrix:
  if tags is None:
    tags = TAGS.no_tags
  return SquareMatrix(tags=tags, mat=cola.ops.Diagonal(diag))

def _scalar_square(c: torch.Tensor, dim: int, *, tags=None, dtype: torch.dtype) -> SquareMatrix:
  if tags is None:
    tags = TAGS.no_tags
  return SquareMatrix(tags=tags, mat=cola.ops.ScalarMul(c, shape=(dim, dim), dtype=dtype))

def _block_diag_dense(blocks: list[torch.Tensor], *, dtype: torch.dtype) -> SquareMatrix:
  ops = [cola.ops.Dense(b.to(dtype=dtype)) for b in blocks]
  return SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.BlockDiag(*ops))


class LinearTimeInvariantSDE(AbstractLinearTimeInvariantSDE):
  def __init__(self, F: SquareMatrix, L: SquareMatrix):
    self.F = F
    self.L = L


class BrownianMotion(AbstractLinearTimeInvariantSDE):
  def __init__(self, sigma: Scalar, dim: int, dtype: torch.dtype = torch.float64):
    sigma_t = torch.as_tensor(sigma, dtype=dtype)
    self.F = _scalar_square(torch.tensor(0.0, dtype=dtype), dim, tags=TAGS.zero_tags, dtype=dtype)
    self.L = _scalar_square(sigma_t, dim, dtype=dtype)


class OrnsteinUhlenbeck(AbstractLinearTimeInvariantSDE):
  def __init__(self, sigma: Scalar, lambda_: Scalar, dim: int, dtype: torch.dtype = torch.float64):
    sigma_t = torch.as_tensor(sigma, dtype=dtype)
    lambda_t = torch.as_tensor(lambda_, dtype=dtype)
    self.F = _scalar_square(-lambda_t, dim, dtype=dtype)
    self.L = _scalar_square(sigma_t, dim, dtype=dtype)


class VariancePreserving(AbstractLinearSDE):
  def __init__(self, beta_min: Scalar, beta_max: Scalar, dim: int, dtype: torch.dtype = torch.float64):
    self.beta_min = torch.as_tensor(beta_min, dtype=dtype)
    self.beta_max = torch.as_tensor(beta_max, dtype=dtype)
    self.dim = int(dim)

  def beta(self, t: Scalar) -> torch.Tensor:
    t_t = torch.as_tensor(t, dtype=self.beta_min.dtype, device=self.beta_min.device)
    return self.beta_min + t_t * (self.beta_max - self.beta_min)

  def T(self, t: Scalar) -> torch.Tensor:
    t_t = torch.as_tensor(t, dtype=self.beta_min.dtype, device=self.beta_min.device)
    return t_t * self.beta_min + 0.5 * t_t**2 * (self.beta_max - self.beta_min)

  def get_params(self, t: Scalar) -> Tuple[SquareMatrix, torch.Tensor, SquareMatrix]:
    beta_t = self.beta(t)
    F = _scalar_square(-0.5 * beta_t, self.dim, dtype=self.beta_min.dtype)
    u = torch.zeros((self.dim,), dtype=self.beta_min.dtype, device=self.beta_min.device)
    L = _scalar_square(torch.sqrt(beta_t), self.dim, dtype=self.beta_min.dtype)
    return F, u, L

  def get_transition_distribution(self, s: Scalar, t: Scalar) -> GaussianTransition:
    Tt = self.T(t)
    Ts = self.T(s)
    dT = Tt - Ts
    alpha = torch.exp(-0.5 * dT)

    A = _scalar_square(alpha, self.dim, dtype=self.beta_min.dtype)
    u = torch.zeros((self.dim,), dtype=self.beta_min.dtype, device=self.beta_min.device)
    Sigma = _scalar_square((1.0 - torch.exp(-dT)), self.dim, dtype=self.beta_min.dtype)
    return GaussianTransition(A, u, Sigma)


class WienerVelocityModel(AbstractLinearTimeInvariantSDE):
  def __init__(
    self,
    sigma: Union[Scalar, torch.Tensor],
    position_dim: int,
    order: int,
    dtype: torch.dtype = torch.float64,
  ):
    if order <= 1:
      raise ValueError("order must be greater than 1")

    self.position_dim = int(position_dim)
    self.order = int(order)

    shift = torch.zeros((self.order, self.order), dtype=dtype)
    shift = shift + torch.diag(torch.ones(self.order - 1, dtype=dtype), diagonal=1)
    Ipos = torch.eye(self.position_dim, dtype=dtype)
    self.F = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Kronecker(cola.ops.Dense(shift), cola.ops.Dense(Ipos)))

    sigma_t = torch.as_tensor(sigma, dtype=dtype)
    if sigma_t.ndim == 0:
      factor = 3.0
      order_noise = (1.0 / (sigma_t ** (factor - 1.0))) * torch.linspace(
        0.0, float(sigma_t), self.order, dtype=dtype
      ) ** factor
      diag = torch.repeat_interleave(order_noise, self.position_dim)
      self.L = _diag_square(diag)
    else:
      if sigma_t.shape != (self.order - 1,):
        raise ValueError("sigma must be scalar or shape (order - 1,)")
      sigma_pad = torch.concatenate([torch.zeros((1,), dtype=dtype), sigma_t.to(dtype=dtype)], dim=0)
      diag = torch.repeat_interleave(sigma_pad, self.position_dim)
      self.L = _diag_square(diag)


class WienerAccelerationModel(AbstractLinearTimeInvariantSDE):
  def __init__(self, sigma: Scalar, position_dim: int, dtype: torch.dtype = torch.float64):
    self.position_dim = int(position_dim)
    self.order = 3
    D = self.position_dim * self.order

    shift = torch.zeros((self.order, self.order), dtype=dtype)
    shift = shift + torch.diag(torch.ones(self.order - 1, dtype=dtype), diagonal=1)
    Ipos = torch.eye(self.position_dim, dtype=dtype)
    self.F = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Kronecker(cola.ops.Dense(shift), cola.ops.Dense(Ipos)))

    sigma_t = torch.as_tensor(sigma, dtype=dtype)
    diag = torch.zeros((D,), dtype=dtype)
    diag[-self.position_dim:] = sigma_t
    self.L = _diag_square(diag)

class CriticallyDampedLangevinDynamics(AbstractLinearTimeInvariantSDE):
  def __init__(
    self,
    mass: Union[torch.Tensor, Scalar],
    beta: Union[torch.Tensor, Scalar],
    dim: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
  ):
    mass_t = torch.as_tensor(mass, dtype=dtype)
    beta_t = torch.as_tensor(beta, dtype=dtype)

    if mass_t.ndim == 0:
      if dim is None:
        raise ValueError("dim must be provided when mass is scalar")
      mass_t = torch.ones((dim,), dtype=dtype) * mass_t
    else:
      dim = int(mass_t.shape[-1])

    if beta_t.ndim == 0:
      beta_t = torch.ones((dim,), dtype=dtype) * beta_t
    else:
      if beta_t.shape[-1] != dim:
        raise ValueError("beta must be scalar or shape (dim,)")

    gamma = torch.sqrt(4.0 * mass_t)

    blocks = []
    for i in range(dim):
      bi = beta_t[i]
      mi = mass_t[i]
      gi = gamma[i]
      blocks.append(torch.tensor([[0.0, (bi / mi).item()], [(-bi).item(), (-(gi * bi) / mi).item()]], dtype=dtype))
    self.F = _block_diag_dense(blocks, dtype=dtype)

    Ldiag = torch.sqrt(2.0 * gamma * beta_t)
    Ldiag = torch.concatenate([torch.zeros((dim,), dtype=dtype), Ldiag], dim=0)
    self.L = _diag_square(Ldiag)


class TOLD(AbstractLinearTimeInvariantSDE):
  def __init__(self, L: Scalar = 1.0, dim: int = 1, dtype: torch.dtype = torch.float64):
    one = torch.ones((dim,), dtype=dtype)
    zero = torch.zeros((dim,), dtype=dtype)

    s2 = float(torch.sqrt(torch.tensor(2.0, dtype=dtype)))
    s3 = float(torch.sqrt(torch.tensor(3.0, dtype=dtype)))
    blocks = []
    for _ in range(dim):
      blocks.append(
        torch.tensor(
          [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 2.0 * s2],
            [0.0, -2.0 * s2, -3.0 * s3],
          ],
          dtype=dtype,
        )
      )
    self.F = _block_diag_dense(blocks, dtype=dtype)

    L_t = torch.as_tensor(L, dtype=dtype)
    Ldiag = (3.0 ** 0.25) * torch.sqrt(6.0 / L_t) * one
    Ldiag = torch.concatenate([torch.zeros((2 * dim,), dtype=dtype), Ldiag], dim=0)
    self.L = _diag_square(Ldiag)


class StochasticHarmonicOscillator(AbstractLinearTimeInvariantSDE):
  def __init__(
    self,
    freq: Union[torch.Tensor, Scalar],
    coeff: Union[torch.Tensor, Scalar],
    sigma: Union[torch.Tensor, Scalar],
    observation_dim: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
  ):
    freq_t = torch.as_tensor(freq, dtype=dtype)
    coeff_t = torch.as_tensor(coeff, dtype=dtype)
    sigma_t = torch.as_tensor(sigma, dtype=dtype)

    if freq_t.ndim == 0:
      if observation_dim is None:
        raise ValueError("observation_dim must be provided when freq is scalar")
      freq_t = torch.ones((observation_dim,), dtype=dtype) * freq_t
    else:
      observation_dim = int(freq_t.shape[-1])

    if coeff_t.ndim == 0:
      coeff_t = torch.ones((observation_dim,), dtype=dtype) * coeff_t
    else:
      if coeff_t.shape[-1] != observation_dim:
        raise ValueError("coeff must be scalar or shape (observation_dim,)")

    if sigma_t.ndim == 0:
      sigma_t = torch.ones((observation_dim,), dtype=dtype) * sigma_t
    else:
      if sigma_t.shape[-1] != observation_dim:
        raise ValueError("sigma must be scalar or shape (observation_dim,)")

    zero = torch.zeros((observation_dim,), dtype=dtype)
    one = torch.ones((observation_dim,), dtype=dtype)

    blocks = []
    for i in range(observation_dim):
      fi = freq_t[i]
      ci = coeff_t[i]
      blocks.append(torch.tensor([[0.0, 1.0], [(-(fi ** 2)).item(), (-ci).item()]], dtype=dtype))
    self.F = _block_diag_dense(blocks, dtype=dtype)

    Ldiag = torch.concatenate([torch.zeros((observation_dim,), dtype=dtype), sigma_t], dim=0)
    self.L = _diag_square(Ldiag)

