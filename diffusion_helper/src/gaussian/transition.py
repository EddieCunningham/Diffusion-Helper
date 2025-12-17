import torch
from typing import Optional, Tuple, Union, Any
from jaxtyping import Array, Float
from plum import dispatch

import diffusion_helper.src.util as util
from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import TAGS
from diffusion_helper.src.gaussian.gaussian import (
  AbstractGaussianPotential,
  StandardGaussian,
  MixedGaussian,
)
from diffusion_helper.src.functional.funtional_ops import vdot, zeros_like
from diffusion_helper.src.functional.linear_functional import LinearFunctional
from tensordict import TensorClass, TensorDict, MetaData, tensorclass

__all__ = ['GaussianTransition']


class GaussianTransition(TensorClass):

  A: SquareMatrix
  u: Union[torch.Tensor, LinearFunctional]
  Sigma: SquareMatrix
  logZ: torch.Tensor

  def __init__(self,
    A: SquareMatrix,
    u: Union[torch.Tensor, LinearFunctional],
    Sigma: SquareMatrix,
    logZ: Optional[torch.Tensor] = None
  ):
    self.A = A
    self.u = u
    self.Sigma = Sigma.set_symmetric()

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  @classmethod
  @dispatch
  def no_op_like(cls, other: 'GaussianTransition') -> 'GaussianTransition':
    A = SquareMatrix.eye(other.A.shape[0], dtype=other.A.mat.dtype)
    u = other.u*0.0 if isinstance(other.u, torch.Tensor) else zeros_like(other.u)
    Sigma = other.Sigma.set_zero()
    return GaussianTransition(A, u, Sigma, other.logZ)

  @dispatch
  def swap_variables(self) -> 'GaussianTransition':
    Ainv = self.A.get_inverse()
    u = -Ainv@self.u
    Sigma = Ainv@self.Sigma@Ainv.T
    return GaussianTransition(Ainv, u, Sigma, self.logZ)

  def marginalize_out_y(self) -> StandardGaussian:
    zeromu = zeros_like(self.u)
    zerocov = self.Sigma.zeros_like(self.Sigma)
    zerocov = zerocov.set_inf()
    nc = self.normalizing_constant()
    return StandardGaussian(zeromu, zerocov, self.logZ - nc)

  def normalizing_constant(self):
    Sigmainv_u = self.Sigma.solve(self.u)
    dim = self.u.shape[0]

    logZ = 0.5*vdot(Sigmainv_u, self.u)
    logZ = logZ + 0.5*self.Sigma.get_log_det()
    logZ = logZ + 0.5*dim*torch.log(torch.tensor(2*torch.pi, dtype=self.Sigma.mat.dtype, device=self.Sigma.mat.device))

    return util.where(self.Sigma.is_zero, zeros_like(logZ), logZ)

  def __call__(self, y: Float[Array, 'Dy'], x: Float[Array, 'Dx']) -> Float[Array, '']:
    return self.condition_on_x(x)(y)

  def log_prob(self, y: Float[Array, 'Dy'], x: Float[Array, 'Dx']) -> Float[Array, '']:
    return self.condition_on_x(x).log_prob(y)

  @dispatch
  def condition_on_x(self, x: Union[torch.Tensor, LinearFunctional]) -> StandardGaussian:
    Ax = self.A@x

    muy = Ax + self.u
    Sigmay = self.Sigma

    logZ = vdot(Ax, self.Sigma.solve(0.5*Ax + self.u))
    logZ = util.where(self.Sigma.is_zero, zeros_like(logZ), logZ)
    return StandardGaussian(muy, Sigmay, logZ + self.logZ)

  @dispatch
  def update_y(
    self,
    potential: StandardGaussian,
    only_return_transition: bool = False
  ) -> 'GaussianTransition':
    A, u, Sigma = self.A, self.u, self.Sigma
    Sigmax, mux, logZx = potential.Sigma, potential.mu, potential.logZ
    I = SquareMatrix.eye(Sigma.shape[0], dtype=Sigma.mat.dtype)

    Sigma_plus_Sigmax = Sigma + Sigmax
    S = Sigma_plus_Sigmax.T.solve(Sigma).T # Sigma@(Sigma + Sigmax)^{-1}
    S = util.where(Sigmax.tags.is_inf, S.set_zero(), S)

    T = I - S
    T = util.where(Sigmax.tags.is_zero, T.set_zero(), T)

    Sigmabar = T@Sigma
    Sigmabar = Sigmabar.set_symmetric()
    Abar = T@A
    ubar = T@u + S@mux
    new_transition = GaussianTransition(Abar, ubar, Sigmabar, self.logZ)
    return new_transition

  @dispatch
  def update_y(
    self,
    potential: MixedGaussian,
    only_return_transition: bool = False
  ) -> 'GaussianTransition':
    """Incorporate a potential over y into the joint potential"""
    A, u, Sigma = self.A, self.u, self.Sigma
    Jy, muy, logZy = potential.J, potential.mu, potential.logZ
    I = SquareMatrix.eye(Sigma.shape[0], dtype=Sigma.mat.dtype)

    SigmaJ = Sigma@Jy
    I_plus_SigmaJ = I + SigmaJ
    R = I_plus_SigmaJ.T.solve(Jy).T # Jy@(I + Sigma@Jy)^{-1}
    S = Sigma@R                     # Sigma@Jy@(I + Sigma@Jy)^{-1}
    T = I - S
    T, S = util.where(Jy.tags.is_inf, (T.set_zero(), SquareMatrix.eye(S.shape[0], dtype=S.mat.dtype)), (T, S))
    T, S = util.where(Jy.tags.is_zero, (SquareMatrix.eye(T.shape[0], dtype=T.mat.dtype), S.set_zero()), (T, S))

    Sigmabar = T@Sigma
    Sigmabar = Sigmabar.set_symmetric()
    Abar = T@A
    ubar = T@u + S@muy
    new_transition = GaussianTransition(Abar, ubar, Sigmabar, self.logZ)
    return new_transition

  def chain(self, other: 'GaussianTransition') -> 'GaussianTransition':
    Ak, uk, Sigmak = other.A, other.u, other.Sigma    # Ax, ux, Sigmax
    Akm1, ukm1, Sigmakm1 = self.A, self.u, self.Sigma # Az, uz, Sigmaz

    A = Ak@Akm1
    u = Ak@ukm1 + uk
    Sigma = Sigmak + Ak@Sigmakm1@Ak.T
    Sigma = Sigma.set_symmetric()

    dim = Sigmak.shape[0]

    Ax, ux, Sigmax = Ak, uk, Sigmak
    Az, uz, Sigmaz = Akm1, ukm1, Sigmakm1

    Axinv = Ax.get_inverse()
    Sigmazinv = Sigmaz.get_inverse()

    T = (Sigmax + Ax@Sigmaz@Ax.T).get_inverse()

    matxx_inv = T - Sigmax.get_inverse()
    matxz = Axinv@Sigmax + Sigmaz@Ax.T
    matzz_inv = Ax.T@T@Ax - Sigmazinv
    new_term1 = 0.5*vdot(ux, matxx_inv@ux)
    new_term2 = vdot(ux, matxz.solve(uz))
    new_term3 = 0.5*vdot(uz, matzz_inv@uz)

    new_term4 = 0.5*(Sigma.get_log_det() - Sigmaz.get_log_det() - Sigmax.get_log_det())
    new_term5 = -0.5*dim*torch.log(torch.tensor(2*torch.pi, dtype=Sigmak.mat.dtype, device=Sigmak.mat.device))
    new_term6 = self.logZ + other.logZ

    logZ = new_term1 + new_term2 + new_term3 + new_term4 + new_term5 + new_term6
    return GaussianTransition(A, u, Sigma, logZ)

  def update_and_marginalize_out_x(self, potential: AbstractGaussianPotential) -> AbstractGaussianPotential:
    std_potential = potential.to_std()
    mu, Sigma = std_potential.mu, std_potential.Sigma

    new_mean = self.A@mu + self.u
    new_cov = self.Sigma + self.A@Sigma@self.A.T
    new_cov = new_cov.set_symmetric()

    new_dist = StandardGaussian(new_mean, new_cov)

    correction = potential.logZ - potential.normalizing_constant()
    correction += self.logZ - self.normalizing_constant()
    logZ = new_dist.logZ + correction
    out_std = StandardGaussian(new_mean, new_cov, logZ)

    if isinstance(potential, StandardGaussian):
      return out_std
    elif isinstance(potential, MixedGaussian):
      return out_std.to_mixed()
    else:
      raise ValueError(f"Unknown potential type: {type(potential)}")

################################################################################################################
