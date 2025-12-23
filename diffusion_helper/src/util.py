import torch
from typing import Union, Any, Tuple, TYPE_CHECKING
import optree
if TYPE_CHECKING:
  from diffusion_helper.src.gaussian.gaussian import StandardGaussian
from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import TAGS
import cola
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
import diffusion_helper.src.functional.funtional_ops as functional_ops
from tensordict import TensorClass

################################################################################################################

def _convert_batch_size(batch_size: Any) -> Any:
  """Convert tensordict batch_size to project style.

  - torch.Size([]) -> None
  - torch.Size([n]) or [n] -> int(n)
  - torch.Size([a, b, ...]) -> tuple of ints
  - int -> int
  - None -> None
  """
  if batch_size is None:
    return None
  # torch.Size is a subclass of tuple
  if isinstance(batch_size, torch.Size):
    if len(batch_size) == 0:
      return None
    if len(batch_size) == 1:
      return int(batch_size[0])
    return tuple(int(x) for x in batch_size)
  if isinstance(batch_size, (tuple, list)):
    if len(batch_size) == 0:
      return None
    if len(batch_size) == 1:
      return int(batch_size[0])
    return tuple(int(x) for x in batch_size)
  if isinstance(batch_size, int):
    return batch_size
  return batch_size

################################################################################################################

def matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
  eigvals, eigvecs = torch.linalg.eigh(mat)
  return eigvecs@torch.diag(torch.sqrt(eigvals))@eigvecs.T

def empirical_dist(xts: torch.Tensor) -> 'StandardGaussian':
  from diffusion_helper.src.gaussian.gaussian import StandardGaussian  # local import to avoid circular
  mu = xts.mean(axis=0)
  cov = torch.einsum('bi,bj->ij', xts - mu, xts - mu)/xts.shape[0] + 1e-10*torch.eye(xts.shape[1])
  cov = SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Dense(cov))
  return StandardGaussian(mu, cov)

def w2_distance(gaussian1: 'StandardGaussian',
                gaussian2: 'StandardGaussian') -> Scalar:
  """Compute the Wasserstein-2 distance between two Gaussians"""
  if hasattr(gaussian1, "to_std"):
    gaussian1 = gaussian1.to_std()
  if hasattr(gaussian2, "to_std"):
    gaussian2 = gaussian2.to_std()

  cov1 = gaussian1.Sigma.to_dense()
  cov2 = gaussian2.Sigma.to_dense()
  mu1 = gaussian1.mu
  mu2 = gaussian2.mu

  cov1_sqrt = matrix_sqrt(cov1)

  term = cov1_sqrt@cov2@cov1_sqrt
  term = term + torch.eye(cov1.shape[0])*1e-10
  cov_term_sqrt = matrix_sqrt(term)

  return torch.sum((mu1 - mu2)**2) + torch.trace(cov1 + cov2 - 2*cov_term_sqrt)

################################################################################################################

def where(cond: Bool, true: PyTree, false: PyTree) -> Any:
  # Apply elementwise where on tensor leaves; choose whole objects otherwise.
  def _choose(x: Any, y: Any) -> Any:
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
      return torch.where(cond, x, y)
    # Expect scalar boolean condition; pick branch as a whole.
    return x if bool(cond) else y

  return optree.tree_map(_choose, true, false)

################################################################################################################

def fill_array(buffer: Float[Array, 'T D'], i: Any, value: Union[Float[Array, 'D'], Float[Array, 'K D']]):
  return optree.tree_map(lambda t, elt: t.at[i].set(elt), buffer, value)

################################################################################################################

def get_times_to_interleave_for_upsample(ts: Float[Array, 'N'],
                                         n_points_to_add_inbetween: int) -> Float[Array, 'N * (n_points_to_add_inbetween + 1)']:
  """Get the times to interleave for upsampling a time series"""
  # Construct n_points_to_add_inbetween points in between each point
  assert ts.ndim == 1
  dts = torch.diff(ts)
  dts = torch.concatenate([dts, dts[-1:]])
  offsets = dts[:,None]*torch.arange(1, (n_points_to_add_inbetween + 2))/(n_points_to_add_inbetween + 1)
  new_times = ts[:,None] + dts[:,None]*offsets
  return new_times[...,:-1].ravel()

################################################################################################################

def tree_shapes(tree: PyTree) -> Tuple[Tuple[int]]:
  """Get the shapes of a tree"""
  return optree.tree_map(lambda x: x.shape, tree)

