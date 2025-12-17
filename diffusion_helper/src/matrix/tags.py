from typing import Tuple, Union
from jaxtyping import Bool
import numpy as np
from tensordict import tensorclass
import torch

__all__ = ['Tags', 'TAGS']

@tensorclass
class Tags():
  """Contains different properties of a matrix.  Knowing these can facilitate more efficient code"""
  is_nonzero: Bool # Use non-zero so that creating a zero matrix can be done with jtu.tree_map(jnp.zeros_like, ...)
  is_inf: Bool

  def __init__(self, is_nonzero: Bool, is_inf: Bool):
    self.is_nonzero = torch.as_tensor(is_nonzero, dtype=torch.bool)
    self.is_inf = torch.as_tensor(is_inf, dtype=torch.bool)

  @property
  def is_zero(self):
    return ~self.is_nonzero

  def add_update(self, update: 'Tags') -> 'Tags':
    """
    Addition (A + B)
        B
        |     =0        ≠0        =∞        ≠∞
    A  |-------------------------------------------------
    =0 | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    ≠0 |  .         (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    =∞ |  .          .         (≠0, =∞)   (≠0, =∞)
    ≠∞ |  .          .          .         (≠0, ≠∞)
    """
    is_nonzero_after_add = self.is_nonzero | update.is_nonzero
    is_inf_after_add = self.is_inf | update.is_inf
    return Tags(is_nonzero_after_add, is_inf_after_add)

  def mat_mul_update(self, update: 'Tags') -> 'Tags':
    """
    Multiplication (A@B)
      B
      |     =0        ≠0        =∞        ≠∞
    A |-------------------------------------------------
    =0 | (=0, ≠∞)   (=0, ≠∞)    ?         (=0, ≠∞)
    ≠0 | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    =∞ |  ?         (≠0, =∞)   (≠0, =∞)   (≠0, =∞)
    ≠∞ | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    """
    is_nonzero_after_mul = self.is_nonzero & update.is_nonzero
    is_inf_after_mul = self.is_inf | update.is_inf
    return Tags(is_nonzero_after_mul, is_inf_after_mul)

  def scalar_mul_update(self) -> 'Tags':
    return self

  def transpose_update(self) -> 'Tags':
    return self

  def solve_update(self, update: 'Tags') -> 'Tags':
    """Solve (A⁻¹ @ B)
        B
        |     =0        ≠0        =∞        ≠∞
    A  |-------------------------------------------------
    =0 | ?          (≠0, =∞)   (≠0, =∞)   (≠0, =∞)
    ≠0 | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    =∞ | (=0, ≠∞)   (=0, ≠∞)    ?         (=0, ≠∞)
    ≠∞ | (=0, ≠∞)   (≠0, ≠∞)   (≠0, =∞)   (≠0, ≠∞)
    """
    # When A is zero (non-invertible), result should be infinite for nonzero B
    zero_case_inf = self.is_zero & update.is_nonzero

    is_nonzero_after_solve = ~self.is_inf & update.is_nonzero
    is_inf_after_solve = (update.is_inf & ~self.is_inf) | zero_case_inf

    return Tags(is_nonzero_after_solve, is_inf_after_solve)

  def inverse_update(self) -> 'Tags':
    is_nonzero_after_invert = ~self.is_inf
    is_inf_after_invert = self.is_zero
    return Tags(is_nonzero_after_invert, is_inf_after_invert)

  def cholesky_update(self) -> 'Tags':
    is_nonzero_after_cholesky = self.is_nonzero
    is_inf_after_cholesky = self.is_inf
    return Tags(is_nonzero_after_cholesky, is_inf_after_cholesky)

  def exp_update(self) -> 'Tags':
    is_nonzero_after_exp = torch.ones_like(self.is_nonzero, dtype=torch.bool)
    is_inf_after_exp = self.is_inf
    return Tags(is_nonzero_after_exp, is_inf_after_exp)

class TAGS:
  zero_tags = Tags(is_nonzero=np.array(False), is_inf=np.array(False))
  inf_tags = Tags(is_nonzero=np.array(True), is_inf=np.array(True))
  no_tags = Tags(is_nonzero=np.array(True), is_inf=np.array(False))
