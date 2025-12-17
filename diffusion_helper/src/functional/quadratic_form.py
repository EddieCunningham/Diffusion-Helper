import torch
from typing import Union, Optional, Any
from plum import dispatch
import optree

from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.functional.linear_functional import LinearFunctional
from tensordict import TensorClass, TensorDict, MetaData, tensorclass

__all__ = ['QuadraticForm', 'resolve_quadratic_form', 'vdot']

NumericScalar = Union[float, int]


class QuadraticForm(TensorClass):
  """Represents f(x) = 0.5 * x^T A x + b^T x + c.

  Attributes:
    A: Symmetric `SquareMatrix`.
    b: 1D torch tensor.
    c: Python float (scalar).
  """
  A: SquareMatrix
  b: torch.Tensor
  c: torch.Tensor  # 0-D tensor

  def __init__(self, A: SquareMatrix, b: torch.Tensor, c: Union[float, int, torch.Tensor]):
    self.A = 0.5 * (A + A.T)
    self.b = b
    if isinstance(c, torch.Tensor):
      self.c = c if c.ndim == 0 else c.squeeze()
    else:
      self.c = torch.as_tensor(c, dtype=b.dtype)

  def __str__(self):
    return f'QuadraticForm(A={self.A}, b={self.b}, c={self.c})'

  def __repr__(self):
    return str(self)

  def _repr(self) -> str:
    return str(self)

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    Ax = self.A @ x
    quad = 0.5 * torch.dot(x, Ax)
    lin = torch.dot(self.b, x)
    return quad + lin + self.c.to(x.dtype)

  @dispatch
  def __add__(self, other: 'QuadraticForm') -> 'QuadraticForm':
    return QuadraticForm(self.A + other.A, self.b + other.b, self.c + other.c)

  @dispatch
  def __add__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(self.A, self.b, self.c + torch.as_tensor(other, dtype=self.b.dtype))

  @dispatch
  def __add__(self, other: torch.Tensor) -> 'QuadraticForm':
    return QuadraticForm(self.A, self.b, self.c + other)

  def __radd__(self, other: NumericScalar) -> 'QuadraticForm':
    return self + other

  @dispatch
  def __sub__(self, other: 'QuadraticForm') -> 'QuadraticForm':
    return QuadraticForm(self.A - other.A, self.b - other.b, self.c - other.c)

  @dispatch
  def __sub__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(self.A, self.b, self.c - torch.as_tensor(other, dtype=self.b.dtype))

  @dispatch
  def __sub__(self, other: torch.Tensor) -> 'QuadraticForm':
    return QuadraticForm(self.A, self.b, self.c - other)

  def __rsub__(self, other: NumericScalar) -> 'QuadraticForm':
    return QuadraticForm(-self.A, -self.b, torch.as_tensor(other, dtype=self.b.dtype) - self.c)

  def __neg__(self) -> 'QuadraticForm':
    return QuadraticForm(-self.A, -self.b, -self.c)

  @dispatch
  def __mul__(self, other: NumericScalar) -> 'QuadraticForm':
    f = torch.as_tensor(other, dtype=self.b.dtype)
    return QuadraticForm((f * self.A), (f * self.b), (f * self.c))

  def __rmul__(self, other: NumericScalar) -> 'QuadraticForm':
    return self * other


def resolve_quadratic_form(pytree: Any, x: torch.Tensor) -> Any:
  """Apply x to all QuadraticForm leaves in a pytree."""
  def is_leaf(obj: Any) -> bool:
    return isinstance(obj, QuadraticForm)

  def resolve(obj: Any) -> Any:
    if isinstance(obj, QuadraticForm):
      return obj(x)
    return obj

  return optree.tree_map(resolve, pytree, is_leaf=is_leaf)

# Ensure our custom repr is used instead of the TensorClass-injected one
def _quadratic_form_repr(self) -> str:
  return f'QuadraticForm(A={self.A}, b={self.b}, c={self.c})'

QuadraticForm.__repr__ = _quadratic_form_repr
