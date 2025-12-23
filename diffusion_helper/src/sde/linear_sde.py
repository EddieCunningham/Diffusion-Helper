import torch
from typing import Optional, Tuple, Union
from jaxtyping import Scalar
import abc
import cola
from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import TAGS
from diffusion_helper.src.gaussian.transition import GaussianTransition
import diffusion_helper.src.util as util
from diffusion_helper.src.sde.ode_sde_simulation import ode_solve, ODESolverParams

__all__ = [
  "AbstractSDE",
  "AbstractLinearSDE",
  "AbstractLinearTimeInvariantSDE",
  "TimeScaledLinearTimeInvariantSDE",
]

################################################################################################################

class AbstractSDE(abc.ABC):
  """An abstract SDE does NOT support sampling.  We need to incorporate a potential (or initial point)."""

  @abc.abstractmethod
  def get_drift(self, t: Scalar,  xt: torch.Tensor) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def get_diffusion_coefficient(self, t: Scalar, xt: torch.Tensor) -> SquareMatrix:
    pass

  def get_transition_distribution(self, s: Scalar, t: Scalar) -> GaussianTransition:
    raise NotImplementedError

################################################################################################################

class AbstractLinearSDE(AbstractSDE, abc.ABC):

  @abc.abstractmethod
  def get_params(self, t: Scalar) -> Tuple[SquareMatrix,
                                           torch.Tensor,
                                           SquareMatrix]:
    """Get F, u, and L at time t
    """
    pass

  def get_diffusion_coefficient(self, t: Scalar, xt: torch.Tensor) -> SquareMatrix:
    _, _, L = self.get_params(t)
    return L

  def get_drift(self, t: Scalar, xt: torch.Tensor) -> torch.Tensor:
    F, u, _ = self.get_params(t)
    return F@xt + u

  def get_transition_distribution(
    self,
    s: Scalar,
    t: Scalar,
    ode_solver_params: Optional[ODESolverParams] = None,
  ) -> GaussianTransition:
    """Get transition p(x_t | x_s) for linear SDE.

    The SDE is
      dx = (F(t) x + u(t)) dt + L(t) dW

    This returns a Gaussian transition
      x_t | x_s ~ N(A x_s + u_ts, Sigma)

    This implementation follows Särkkä (2019), section 6.1, by solving a
    reverse time ODE for (A, u_ts, Sigma).
    """
    if ode_solver_params is None:
      ode_solver_params = ODESolverParams()

    F0, u0, L0 = self.get_params(s)
    device = u0.device
    dtype = u0.dtype
    D = int(u0.shape[0])

    I = torch.eye(D, device=device, dtype=dtype)
    A_TT = I
    uT = torch.zeros((D,), device=device, dtype=dtype)
    SigmaT = torch.zeros((D, D), device=device, dtype=dtype)

    def pack_state(A: torch.Tensor, b: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
      return torch.concatenate([A.reshape(-1), b.reshape(-1), Sigma.reshape(-1)], dim=0)

    def unpack_state(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      nA = D * D
      nb = D
      A = y[:nA].reshape(D, D)
      b = y[nA:nA + nb]
      Sigma = y[nA + nb:].reshape(D, D)
      return A, b, Sigma

    yT = pack_state(A_TT, uT, SigmaT)

    def reverse_dynamics(tau_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
      tau = tau_batch[0]
      y = y_batch[0]
      A_Ttau, b_Ttau, Sigma_Ttau = unpack_state(y)

      Ftau, utau, Ltau = self.get_params(tau)
      Q = (Ltau @ Ltau.T).to_dense()
      Ftau_dense = Ftau.to_dense()

      dA = -(A_Ttau @ Ftau_dense)
      db = -(A_Ttau @ utau)
      dSigma = -(A_Ttau @ Q @ A_Ttau.T)

      dy = pack_state(dA, db, dSigma)
      return dy.unsqueeze(0)

    save_times = torch.as_tensor([t, s], device=device, dtype=dtype)
    ts = ode_solve(reverse_dynamics, yT, save_times, params=ode_solver_params)
    y0 = ts.vals[-1]
    A_ts_dense, u_ts, Sigma_ts_dense = unpack_state(y0)

    Sigma_ts_dense = 0.5 * (Sigma_ts_dense + Sigma_ts_dense.T)
    Sigma = SquareMatrix.from_dense(Sigma_ts_dense)
    Sigma = util.where(torch.abs(Sigma_ts_dense).max() < 1e-8, Sigma.set_zero(), Sigma)

    A = SquareMatrix.from_dense(A_ts_dense)
    return GaussianTransition(A, u_ts, Sigma)

################################################################################################################

class AbstractLinearTimeInvariantSDE(AbstractLinearSDE, abc.ABC):

  F: SquareMatrix
  L: SquareMatrix

  @property
  def u(self) -> torch.Tensor:
    device = self.F.to_dense().device
    return torch.zeros((self.dim,), dtype=self.F.mat.dtype, device=device)

  @property
  def dim(self) -> int:
    return self.F.shape[0]

  def get_params(self, t: Scalar) -> Tuple[SquareMatrix,
                                           torch.Tensor,
                                           SquareMatrix]:
    return self.F, self.u, self.L

  def get_transition_distribution(self,
                                  s: Scalar,
                                  t: Scalar) -> GaussianTransition:
    """Compute transition distribution using Van Loan matrix fraction method.

    For constant F and L, define Q = L L^T and build the block matrix
      M = [[F, Q],
           [0, -F^T]]
    Then exp(M dt) has blocks
      [[A,  Sigma A^{-T}],
       [0,  A^{-T}]]
    and Sigma = (Sigma A^{-T}) A^T.
    """
    D = self.dim

    Fop = self.F.mat
    Lop = self.L.mat
    dtype = self.F.mat.dtype
    device = self.F.mat.device
    dt = torch.as_tensor(t - s, dtype=dtype, device=device)

    if isinstance(Fop, cola.ops.ScalarMul) and isinstance(Lop, cola.ops.ScalarMul):
      f = torch.as_tensor(Fop.c, dtype=dtype, device=device)
      l = torch.as_tensor(Lop.c, dtype=dtype, device=device)

      A = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.ScalarMul(torch.exp(f * dt), shape=(D, D), dtype=dtype, device=device))

      denom = 2.0 * f
      Sigma_c = torch.where(
        torch.abs(denom) < 1e-12,
        (l * l) * dt,
        (l * l) * (torch.exp(2.0 * f * dt) - 1.0) / denom,
      )
      Sigma = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.ScalarMul(Sigma_c, shape=(D, D), dtype=dtype, device=device))
      Sigma = util.where(torch.abs(Sigma_c) < 1e-12, Sigma.set_zero(), Sigma)
      u = torch.zeros((D,), dtype=dtype, device=device)
      return GaussianTransition(A, u, Sigma)

    if isinstance(Fop, cola.ops.Diagonal) and isinstance(Lop, cola.ops.Diagonal):
      f = Fop.diag.to(dtype=dtype, device=device)
      l = Lop.diag.to(dtype=dtype, device=device)

      A = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(torch.exp(f * dt)))

      denom = 2.0 * f
      Sigma_diag = torch.where(
        torch.abs(denom) < 1e-12,
        (l * l) * dt,
        (l * l) * (torch.exp(2.0 * f * dt) - 1.0) / denom,
      )
      Sigma = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Diagonal(Sigma_diag))
      Sigma = util.where(torch.abs(Sigma_diag).max() < 1e-12, Sigma.set_zero(), Sigma)
      u = torch.zeros((D,), dtype=dtype, device=device)
      return GaussianTransition(A, u, Sigma)

    F = self.F.to_dense()
    L = self.L.to_dense()
    Q = L @ L.T

    zero = torch.zeros((D, D), dtype=dtype, device=device)
    top = torch.concatenate([F, Q], dim=1)
    bottom = torch.concatenate([zero, -F.T], dim=1)
    M = torch.concatenate([top, bottom], dim=0)

    # Use torch expm for this block matrix since it can be non diagonalizable
    Phi = torch.linalg.matrix_exp(M * dt)
    A_dense = Phi[:D, :D]
    Sigma_AinvT = Phi[:D, D:]
    Sigma_dense = Sigma_AinvT @ A_dense.T
    Sigma_dense = 0.5 * (Sigma_dense + Sigma_dense.T)

    A = SquareMatrix.from_dense(A_dense)
    Sigma = SquareMatrix.from_dense(Sigma_dense)
    Sigma = util.where(torch.abs(Sigma_dense).max() < 1e-8, Sigma.set_zero(), Sigma)
    u = torch.zeros((D,), dtype=dtype, device=device)
    return GaussianTransition(A, u, Sigma)

################################################################################################################

class TimeScaledLinearTimeInvariantSDE(AbstractLinearTimeInvariantSDE):
  r"""If sde represents dx_s = Fx_s ds + LdW_s, then this represents reparametrizing time
  as t = \gamma*s and also x_s = \gamma*tilde{x}_s
  """

  sde: AbstractLinearTimeInvariantSDE
  time_scale: Scalar

  def __init__(self,
               sde: AbstractLinearTimeInvariantSDE,
               time_scale: Scalar):
    self.sde = sde
    self.time_scale = time_scale

  @property
  def F(self) -> SquareMatrix:
    return self.sde.F*self.time_scale

  @property
  def L(self) -> SquareMatrix:
    device = self.sde.L.to_dense().device
    scale = torch.sqrt(torch.as_tensor(self.time_scale, dtype=self.sde.L.mat.dtype, device=device))
    return self.sde.L * scale

  @property
  def order(self) -> int:
    """To be compatible with HigherOrderTracking.  There is definitely a better
    way to access member variables of self.sde"""
    try:
      return self.sde.order
    except AttributeError:
      raise AttributeError(f'SDE of type {type(self.sde)} does not have an order')

  def get_transition_distribution(self,
                                  s: Scalar,
                                  t: Scalar) -> GaussianTransition:
    return self.sde.get_transition_distribution(s*self.time_scale, t*self.time_scale)
