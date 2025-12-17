from plum import dispatch
from typing import Union, Optional, Any
import torch
from jaxtyping import Array, Float, Scalar, PyTree
from diffusion_helper.src.functional.linear_functional import LinearFunctional
from diffusion_helper.src.functional.quadratic_form import QuadraticForm
import optree

__all__ = ['vdot', 'zeros_like', 'resolve_functional']

################################################################################################################

@dispatch
def vdot(a, b):
  """Computes the vector dot product. Falls back to torch.dot by default."""
  return torch.vdot(a, b)

@dispatch
def vdot(a: LinearFunctional, b: LinearFunctional) -> QuadraticForm:
  """Computes the dot product of two LinearFunctionals.

  (A1*x + b1)^T (A2*x + b2) = x^T*A1^T*A2*x + (A1^T*b2 + A2^T*b1)^T*x + b1^T*b2
  This is represented as a QuadraticForm: 0.5*x^T*A*x + b^T*x + c
  """
  A = a.A.T @ b.A + b.A.T @ a.A
  b_vec = a.A.T @ b.b + b.A.T @ a.b
  c_scalar = torch.vdot(a.b, b.b)
  return QuadraticForm(A, b_vec, c_scalar)

@dispatch
def vdot(a: LinearFunctional, b: torch.Tensor) -> QuadraticForm:
  """Computes the dot product of a LinearFunctional and a vector."""
  A = a.A.zeros_like(a.A)
  b_vec = a.A.T @ b
  c_scalar = torch.vdot(a.b, b)
  return QuadraticForm(A, b_vec, c_scalar)

@dispatch
def vdot(a: torch.Tensor, b: LinearFunctional) -> QuadraticForm:
  """Computes the dot product of a vector and a LinearFunctional."""
  return vdot(b, a)

################################################################################################################

@dispatch
def zeros_like(a: torch.Tensor) -> torch.Tensor:
  return torch.zeros_like(a)

@dispatch
def zeros_like(a: LinearFunctional) -> LinearFunctional:
  return LinearFunctional(a.A.zeros_like(a.A), torch.zeros_like(a.b))

@dispatch
def zeros_like(a: QuadraticForm) -> QuadraticForm:
  return QuadraticForm(a.A.zeros_like(a.A), torch.zeros_like(a.b), torch.zeros_like(a.c))

################################################################################################################

def resolve_functional(pytree: PyTree, x: torch.Tensor) -> PyTree:
  """Recursively apply a vector `x` to all LinearFunctional leaves in a PyTree.

  Args:
    pytree: The PyTree to resolve.
    x: The vector to apply to the LinearFunctional leaves.

  Returns:
    The resolved PyTree.
  """
  def is_leaf(obj: Any) -> bool:
    return isinstance(obj, LinearFunctional) or isinstance(obj, QuadraticForm)

  def resolve(obj: Any) -> Any:
    if isinstance(obj, LinearFunctional):
      return obj(x)
    elif isinstance(obj, QuadraticForm):
      return obj(x)
    return obj

  return optree.tree_map(resolve, pytree, is_leaf=is_leaf)