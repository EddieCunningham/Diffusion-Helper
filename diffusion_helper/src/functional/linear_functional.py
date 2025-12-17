import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Mapping, Tuple, Sequence, Union, Any, Callable, Type
from jaxtyping import Array, PRNGKeyArray, Float, Scalar, Bool, PyTree
from plum import dispatch
from diffusion_helper.src.matrix.matrix import SquareMatrix
from tensordict import TensorClass, TensorDict, MetaData, tensorclass
import optree
import torch

__all__ = ['LinearFunctional', 'resolve_linear_functional']

NumericScalar = Union[Scalar, float, int]

################################################################################################################

class LinearFunctional(TensorClass):
  """Represents a linear functional of the form f(x) = Ax + b.

  This class is used to represent vectors that are not known constants, but
  rather depend linearly on some other vector `x`. This is useful for delaying
  computation and for representing conditional distributions where the parameters
  depend on another variable.

  Attributes:
    A: The matrix in the linear functional.
    b: The offset vector in the linear functional.
  """
  A: SquareMatrix
  b: torch.Tensor

  @property
  def shape(self):
    """This is for compatability with code that expects a vector."""
    return self.b.shape

  @dispatch
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return self.A @ x + self.b

  @dispatch
  def __call__(self, other: "LinearFunctional") -> "LinearFunctional":
    """Compose this linear functional with another one.

    If this represents f(x) = Ax + b and other represents g(x) = Cx + d,
    this returns a new linear functional representing f(g(x)).
    f(g(x)) = A(Cx + d) + b = (AC)x + (Ad + b)
    """
    new_A = self.A @ other.A
    new_b = self.A @ other.b + self.b
    return LinearFunctional(new_A, new_b)

  @dispatch
  def __add__(self, other: 'LinearFunctional') -> 'LinearFunctional':
    return LinearFunctional(self.A + other.A, self.b + other.b)

  @dispatch
  def __add__(self, other: torch.Tensor) -> 'LinearFunctional':
    return LinearFunctional(self.A, self.b + other)

  def __radd__(self, other: torch.Tensor) -> 'LinearFunctional':
    return self + other

  @dispatch
  def __sub__(self, other: 'LinearFunctional') -> 'LinearFunctional':
    return LinearFunctional(self.A - other.A, self.b - other.b)

  @dispatch
  def __sub__(self, other: torch.Tensor) -> 'LinearFunctional':
    return LinearFunctional(self.A, self.b - other)

  def __rsub__(self, other: torch.Tensor) -> 'LinearFunctional':
    return LinearFunctional(-self.A, other - self.b)

  def __neg__(self) -> 'LinearFunctional':
    return LinearFunctional(-self.A, -self.b)

  @dispatch
  def __mul__(self, other: NumericScalar) -> 'LinearFunctional':
    return LinearFunctional(other * self.A, other * self.b)

  def __rmul__(self, other: NumericScalar) -> 'LinearFunctional':
    return self * other

  def __rmatmul__(self, other: SquareMatrix) -> 'LinearFunctional':
    return LinearFunctional(other @ self.A, other @ self.b)

  def get_inverse(self) -> 'LinearFunctional':
    """Get the inverse linear functional.

    If this represents f(x) = Ax + b, then the inverse represents
    g(y) = A^{-1}y - A^{-1}b such that g(f(x)) = x.

    Returns:
      LinearFunctional: The inverse linear functional.
    """
    A_inv = self.A.get_inverse()
    b_inv = -A_inv @ self.b
    return LinearFunctional(A_inv, b_inv)


def resolve_linear_functional(pytree: PyTree, x: torch.Tensor) -> PyTree:
  """Apply x to the leaves of pytree that are LinearFunctional objects.

  Args:
    pytree: The pytree to resolve.
    x: The vector to apply to the leaves of the pytree.

  Returns:
    The resolved pytree.
  """
  def is_leaf(obj: Any) -> bool:
    return isinstance(obj, LinearFunctional)

  def resolve(obj: Any) -> Any:
    if isinstance(obj, LinearFunctional):
      return obj(x)
    return obj

  return optree.tree_map(resolve, pytree, is_leaf=is_leaf)

@dispatch
def mat_mul(A: SquareMatrix, B: LinearFunctional) -> LinearFunctional:
  return LinearFunctional(A @ B.A, A @ B.b)

@dispatch
def matrix_solve(A: SquareMatrix, B: LinearFunctional) -> LinearFunctional:
  Ab = A.solve(B.b)
  return LinearFunctional(A.solve(B.A), Ab)

# Ensure our custom repr is used instead of the TensorClass-injected one
def _linear_functional_repr(self) -> str:
  return f'LinearFunctional(A={self.A}, b={self.b})'

LinearFunctional.__repr__ = _linear_functional_repr
