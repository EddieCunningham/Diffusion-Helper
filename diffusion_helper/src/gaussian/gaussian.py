import torch
import math
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, TYPE_CHECKING
from jaxtyping import Array, PRNGKeyArray, Float, PyTree, Scalar, Bool
import abc
from plum import dispatch
import cola
from tensordict import TensorClass, tensorclass
import optree
import diffusion_helper.src.util as util
from diffusion_helper.src.matrix.tags import TAGS
from diffusion_helper.src.matrix.matrix import SquareMatrix
if TYPE_CHECKING:
  from diffusion_helper.src.functional.linear_functional import LinearFunctional
  from diffusion_helper.src.functional.quadratic_form import QuadraticForm
from diffusion_helper.src.functional.funtional_ops import vdot, zeros_like

__all__ = [
    'StandardGaussian',
    'MixedGaussian',
]

USE_CHOLESKY_SAMPLING = True

################################################################################################################

class AbstractGaussianPotential(abc.ABC):
  """Abstract base class for Gaussian potentials.

  This class provides a common interface for Gaussian potentials, which are
  used to represent Gaussian distributions in various forms.
  """

  # mean: torch.Tensor
  # mat: cola.LinearOperator
  # logZ: Scalar

  @abc.abstractmethod
  def __call__(self, x: torch.Tensor) -> Scalar:
    pass

  def __add__(self, other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    pass

  @abc.abstractmethod
  def normalizing_constant(self) -> Scalar:
    pass

  @abc.abstractmethod
  def log_prob(self, x: torch.Tensor) -> Scalar:
    pass

  @abc.abstractmethod
  def sample(self, key: PRNGKeyArray) -> torch.Tensor:
    pass

  @classmethod
  @abc.abstractmethod
  def total_certainty_like(cls, x: torch.Tensor, other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    pass

  @classmethod
  @abc.abstractmethod
  def total_uncertainty_like(cls, other: 'AbstractGaussianPotential') -> 'AbstractGaussianPotential':
    pass

  def integrate(self):
    """Compute the value of \int exp{-0.5*x^T J x + x^T h - logZ} dx"""
    return self.normalizing_constant() - self.logZ

  @abc.abstractmethod
  def score(self, x: torch.Tensor) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def get_noise(self, x: torch.Tensor) -> torch.Tensor:
    pass

################################################################################################################

class StandardGaussian(TensorClass, AbstractGaussianPotential):
  """Gaussian distribution in standard (mean and covariance) form.

  Represents a Gaussian distribution N(μ, Σ) with mean vector μ and
  covariance matrix Σ. The density function is:
  exp{-0.5*(x-μ)^T Σ^{-1} (x-μ) - logZ}

  This is the most common and intuitive parametrization of Gaussian distributions,
  particularly useful for sampling and when working with observed data.

  Attributes:
    mu: Mean vector
    Sigma: Covariance matrix
    logZ: Log normalizing constant
  """

  mu: Union[torch.Tensor, 'LinearFunctional']
  Sigma: SquareMatrix
  logZ: Union[Scalar, 'QuadraticForm']

  def __init__(
    self,
    mu: Union[torch.Tensor, 'LinearFunctional'],
    Sigma: SquareMatrix,
    logZ: Optional[Union[Scalar, 'QuadraticForm']] = None
  ):
    """Initialize a Gaussian in standard (mean and covariance) form.

    Args:
      mu: Mean vector
      Sigma: Covariance matrix
      logZ: Log normalizing constant (if None, computed automatically)

    Raises:
      AssertionError: If Sigma is not a matrix or has incorrect dimensions
    """
    assert isinstance(Sigma, SquareMatrix)
    assert Sigma.ndim == 2
    self.Sigma = Sigma.set_symmetric()
    self.mu = mu

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  def __str__(self):
    return f'StandardGaussian(mu={self.mu}, Sigma={self.Sigma}, logZ={self.logZ})'

  def __repr__(self):
    return str(self)

  @property
  def dim(self) -> int:
    return self.mu.shape[-1]

  @classmethod
  def total_certainty_like(self, x: torch.Tensor, other: 'StandardGaussian') -> 'StandardGaussian':
    if util._convert_batch_size(other.batch_size):
      return torch.vmap(lambda _x, o: self.total_certainty_like(_x, o))(x, other)

    out = super().zeros_like(other)
    out.Sigma.tags = TAGS.zero_tags
    out.mu = x
    return out

  @classmethod
  def total_uncertainty_like(cls, other: 'StandardGaussian') -> 'StandardGaussian':
    if util._convert_batch_size(other.batch_size):
      return torch.vmap(lambda o: cls.total_uncertainty_like(o))(other)

    out = super().zeros_like(other)
    out.Sigma.tags = TAGS.inf_tags
    return out

  @classmethod
  def zeros_like(cls, other: 'StandardGaussian') -> 'StandardGaussian':
    return cls.total_uncertainty_like(other)

  def to_std(self):
    """Convert to standard form (identity operation).

    Returns:
      Self, as this is already in standard form
    """
    return self

  def to_mixed(self):
    """Convert to mixed parameter form.

    Transforms from standard parameterization N(μ, Σ) to mixed parameterization
    with mean μ and precision matrix J = Σ^{-1}.

    Returns:
      Equivalent Gaussian in mixed parameter form
    """
    J = self.Sigma.get_inverse()
    return MixedGaussian(self.mu, J, self.logZ)

  def cast(self, other: 'StandardGaussian'):
    mu = self.mu + zeros_like(other.mu) # In case either is a linear functional
    cov = self.Sigma + other.Sigma.zeros_like(other.Sigma) # Correct type for covariance
    logZ = self.logZ + zeros_like(other.logZ) # In case either is a quadratic form
    return StandardGaussian(mu, cov, logZ)

  @dispatch
  def __add__(self, other: 'StandardGaussian') -> 'StandardGaussian':
    """Combine two Gaussian distributions.

    Implements a numerically stable product of Gaussians in standard form.
    This is equivalent to a Kalman filter update step and produces a new
    Gaussian that represents the product of the two input distributions.

    Args:
      other: Another Gaussian in standard form to combine with this one

    Returns:
      A new Gaussian representing the product of the distributions
    """
    mu, Sigma = self.mu, self.Sigma
    mux, Sigmax = other.mu, other.Sigma

    # This determines the output type of the covariance
    Sigma_plus_Sigmax = Sigma + Sigmax
    # Compute the Kalman gain: Sigmax@(Sigma + Sigmax)^{-1}
    S = Sigma_plus_Sigmax.T.solve(Sigmax.T).T
    # Compute the new covariance
    P = S@Sigma
    P = P.set_symmetric()
    # Compute the new mean
    m = S@mu + Sigma@(Sigma_plus_Sigmax.solve(mux))
    logZ = self.logZ + other.logZ
    out = StandardGaussian(m, P, logZ)

    # Handle special cases: infinite covariance (uninformative distribution)
    out = util.where(Sigma.is_inf, other.cast(out), out)
    out = util.where(Sigmax.is_inf, self.cast(out), out)

    # Need to add the logZs together
    out.logZ = self.logZ + other.logZ
    return out

  @dispatch
  def __add__(self, other: 'MixedGaussian') -> 'StandardGaussian':
    """Combine with a Gaussian in mixed form.

    Args:
      other: A Gaussian in mixed form to combine with this one

    Returns:
      A new Gaussian representing the product of the distributions
    """
    return self + other.to_std()

  def normalizing_constant(self):
    """Compute the normalizing constant, which is
    \int exp{-0.5*x^T Sigma^{-1} x + x^T Sigma^{-1}mu} dx. This is different
    than logZ which can be an arbitrary scalar."""
    covinv_mu = self.Sigma.solve(self.mu)
    dim = self.mu.shape[-1]
    logZ = 0.5*vdot(covinv_mu, self.mu)
    logZ = logZ + 0.5*self.Sigma.get_log_det()
    const = torch.tensor(2 * math.pi, dtype=self.Sigma.mat.dtype, device=self.Sigma.mat.device).log()
    logZ = logZ + 0.5 * dim * const

    return util.where(self.Sigma.is_inf|self.Sigma.is_zero, zeros_like(logZ), logZ)

  def __call__(self, x: Array):
    Sigma_inv_x = self.Sigma.solve(x)
    return -0.5*vdot(x, Sigma_inv_x) + vdot(self.mu, Sigma_inv_x) - self.logZ

  def log_prob(self, x: torch.Tensor):
    """Calculate the log probability density of a point.

    Computes the log probability density of x under the N(μ, Σ) distribution.

    Args:
      x: Point to evaluate

    Returns:
      Log probability density at x
    """
    nc = self.normalizing_constant()
    Sigma_inv_x = self.Sigma.solve(x)
    return -0.5*vdot(x, Sigma_inv_x) + vdot(self.mu, Sigma_inv_x) - nc

  def score(self, x: Array) -> Array:
    """Compute the score function (gradient of log density).

    The score function is ∇_x log p(x) = Σ^{-1}(μ - x)

    Args:
      x: Point at which to evaluate the score

    Returns:
      Gradient of the log density at x
    """
    return self.Sigma.solve(self.mu - x)

  def sample(self, key: PRNGKeyArray):
    """Generate a random sample from the Gaussian distribution.

    Uses the reparameterization trick to generate samples efficiently.
    First draws a standard normal sample, then transforms it to have
    the correct mean and covariance.

    Args:
      key: JAX PRNG key for random number generation

    Returns:
      A random sample from the N(μ, Σ) distribution
    """
    eps = torch.randn(self.mu.shape)
    return self._sample(eps)


  if USE_CHOLESKY_SAMPLING:
    def _sample(self, eps: torch.Tensor):
      L = self.Sigma.get_cholesky()
      out = self.mu + L@eps
      out = util.where(self.Sigma.is_zero, self.mu, out)
      return out

    def get_noise(self, x: torch.Tensor):
      L = self.Sigma.get_cholesky()
      eps = L.solve(x - self.mu)
      return eps

  else:

    def _sample(self, eps: torch.Tensor):
      U, Sinv, V = self.Sigma.get_svd()
      out = U@Sinv.get_cholesky()@eps + self.mu
      out = util.where(self.Sigma.is_zero, self.mu, out)
      return out

    def get_noise(self, x: torch.Tensor):
      U, Sinv, V = self.Sigma.get_svd()
      S_chol = Sinv.get_cholesky().get_inverse()
      return S_chol@U.T@(x - self.mu)

  def make_deterministic(self) -> 'StandardGaussian':
    new_Sigma = self.Sigma.zeros_like(self.Sigma)
    return StandardGaussian(self.mu, new_Sigma, self.logZ)

################################################################################################################

class MixedGaussian(TensorClass, AbstractGaussianPotential):
  """Gaussian distribution in mixed parameter form.

  Represents a Gaussian distribution with mean vector μ and precision matrix J
  (inverse covariance). The density function is:
  exp{-0.5*(x-μ)^T J (x-μ) - logZ} = exp{-0.5*x^T J x + x^T Jμ - 0.5*μ^T J μ - logZ}

  This parameterization combines aspects of both standard and natural forms,
  maintaining the intuitive mean parameter while using the precision matrix
  for certain computations that are more efficient in that form.

  Attributes:
    mu: Mean vector
    J: Precision matrix (inverse covariance)
    logZ: Log normalizing constant
  """

  mu: Union[torch.Tensor, 'LinearFunctional']
  J: SquareMatrix
  logZ: Union[Scalar, 'QuadraticForm']

  def __init__(
    self,
    mu: Union[torch.Tensor, 'LinearFunctional'],
    J: SquareMatrix,
    logZ: Optional[Union[Scalar, 'QuadraticForm']] = None
  ):
    """Initialize a Gaussian in mixed parameter form.

    Args:
      mu: Mean vector
      J: Precision matrix (inverse covariance)
      logZ: Log normalizing constant (if None, computed automatically)

    Raises:
      AssertionError: If J is not a matrix or has incorrect dimensions
    """
    assert isinstance(J, SquareMatrix)
    assert J.ndim == 2
    self.J = J.set_symmetric()
    self.mu = mu

    if logZ is None:
      logZ = self.normalizing_constant()
    self.logZ = logZ

  @property
  def dim(self) -> int:
    return self.mu.shape[-1]

  @classmethod
  def construct_deterministic_potential(self, x: torch.Tensor) -> 'MixedGaussian':
    mat = cola.ops.Diagonal(torch.zeros_like(x))
    J = SquareMatrix(tags=TAGS.inf_tags, mat=mat)
    return MixedGaussian(x, J)

  @classmethod
  def total_certainty_like(cls, x: torch.Tensor, other: 'MixedGaussian') -> 'MixedGaussian':
    if util._convert_batch_size(other.batch_size):
      return torch.vmap(lambda _x, o: cls.total_certainty_like(_x, o))(x, other)

    out = super().zeros_like(other)
    out.J.tags = TAGS.inf_tags
    out.mu = x
    return out

  @classmethod
  def total_uncertainty_like(cls, other: 'MixedGaussian') -> 'MixedGaussian':
    if util._convert_batch_size(other.batch_size):
      return torch.vmap(lambda o: cls.total_uncertainty_like(o))(other)

    out = super().zeros_like(other)
    out.J.tags = TAGS.zero_tags
    return out

  @classmethod
  def zeros_like(cls, other: 'MixedGaussian') -> 'MixedGaussian':
    return cls.total_uncertainty_like(other)

  def to_std(self):
    """Convert to standard parameter form.

    Transforms from mixed parameterization (μ, J) to standard parameterization
    with mean μ and covariance matrix Σ = J^{-1}.

    Returns:
      Equivalent Gaussian in standard parameter form
    """
    Sigma = self.J.get_inverse()
    return StandardGaussian(self.mu, Sigma, self.logZ)

  def to_mixed(self):
    """Convert to mixed form (identity operation).

    Returns:
      Self, as this is already in mixed form
    """
    return self

  def cast(self, other: 'MixedGaussian'):
    mu = self.mu + zeros_like(other.mu) # In case either is a linear functional
    J = self.J + other.J.zeros_like(other.J) # Correct type for covariance
    logZ = self.logZ + zeros_like(other.logZ) # In case either is a quadratic form
    return MixedGaussian(mu, J, logZ)

  @dispatch
  def __add__(self, other: 'MixedGaussian') -> 'MixedGaussian':
    """Numerically stable update for standard gaussians.  This is what is used
    in a Kalman filter update."""
    mu, J = self.mu, self.J
    mux, Jx = other.mu, other.J

    Jbar = J + Jx
    mubar = Jbar.solve(J@mu) + Jbar.solve(Jx@mux)
    logZ = self.logZ + other.logZ
    out = MixedGaussian(mubar, Jbar, logZ)

    out = util.where(J.is_zero|Jx.is_inf, other.cast(out), out)
    out = util.where(Jx.is_zero|J.is_inf, self.cast(out), out)

    out.logZ = self.logZ + other.logZ
    return out

  @dispatch
  def __add__(self, other: 'StandardGaussian') -> 'MixedGaussian':
    return self + other.to_mixed()

  def normalizing_constant(self):
    """Compute the normalizing constant, which is
    \int exp{-0.5*x^T Sigma^{-1} x + x^T Sigma^{-1}mu} dx. This is different
    than logZ which can be an arbitrary scalar."""
    Jmu = self.J@self.mu
    dim = self.mu.shape[-1]
    logZ = 0.5*vdot(Jmu, self.mu)
    logZ = logZ - 0.5*self.J.get_log_det()
    const = torch.tensor(2 * math.pi, dtype=self.J.mat.dtype, device=self.J.mat.device).log()
    logZ = logZ + 0.5 * dim * const

    return util.where(self.J.is_inf|self.J.is_zero, zeros_like(logZ), logZ)

  def __call__(self, x: Array):
    Jx = self.J@x
    return -0.5*vdot(x, Jx) + vdot(self.mu, Jx) - self.logZ

  def log_prob(self, x: torch.Tensor):
    nc = self.normalizing_constant()
    Jx = self.J@x
    return -0.5*vdot(x, Jx) + vdot(self.mu, Jx) - nc

  def score(self, x: Array) -> Array:
    """Score function of the Gaussian potential"""
    return self.J@(self.mu - x)

  def sample(self, key: PRNGKeyArray):
    eps = torch.randn(self.mu.shape)
    return self._sample(eps)

  if USE_CHOLESKY_SAMPLING:

    def _sample(self, eps: torch.Tensor):
      J = self.J# + self.J.eye(self.J.shape[0])*1e-6
      L_chol = J.get_cholesky()
      out = J.solve(L_chol@eps) + self.mu
      out = util.where(self.J.is_inf, self.mu, out)
      return out

    def get_noise(self, x: torch.Tensor):
      L_chol = self.J.get_cholesky()
      return L_chol.solve(self.J@x) - L_chol.T@self.mu

  else:

    def _sample(self, eps: torch.Tensor):
      U, S, V = self.J.get_svd()
      out = U@S.get_cholesky().solve(eps) + self.mu
      out = util.where(self.J.is_inf, self.mu, out)
      return out

    def get_noise(self, x: torch.Tensor):
      U, S, V = self.J.get_svd()
      S_chol = S.get_cholesky()
      return S_chol@U.T@(x - self.mu)

  def make_deterministic(self) -> 'MixedGaussian':
    new_J = self.J.set_inf()
    return MixedGaussian(self.mu, new_J, self.logZ)

################################################################################################################

# Ensure our custom repr is used instead of the TensorClass-injected one
def _standard_gaussian_repr(self) -> str:
  return f'StandardGaussian(mu={self.mu}, Sigma={self.Sigma}, logZ={self.logZ})'

StandardGaussian.__repr__ = _standard_gaussian_repr

def _mixed_gaussian_repr(self) -> str:
  return f'MixedGaussian(mu={self.mu}, J={self.J}, logZ={self.logZ})'

MixedGaussian.__repr__ = _mixed_gaussian_repr