import torch
import cola
from cola.ops import LinearOperator
import abc
from diffusion_helper.src.matrix.tags import Tags, TAGS
from jaxtyping import Array, Float, Scalar, Bool
from typing import Any, Union, Tuple, Optional
from plum import dispatch
from tensordict import TensorClass, TensorDict, MetaData, tensorclass
import optree

class SquareMatrix(TensorClass):

  tags: Tags
  mat_tensors: TensorDict
  mat_unflatten: MetaData

  def __init__(self, tags: Tags, mat: LinearOperator):
    mat_tensors, mat_unflatten = mat.flatten()
    self.mat_tensors = mat_tensors
    self.mat_unflatten = mat_unflatten
    self.tags = tags

  def __str__(self):
    return f'{type(self).__name__}(\n{self.to_dense()}\n)'

  def __repr__(self):
    return str(self)

  @property
  def mat(self) -> LinearOperator:
    return self.mat_unflatten(self.mat_tensors)

  # def cast_like(self, other: 'SquareMatrix') -> 'SquareMatrix':
  #   """Cast this matrix to be the same type as another matrix.

  #   This is a simple implementation that uses matrix addition with zeros.
  #   For more sophisticated casting options, use the `cast_matrix` function.

  #   Args:
  #       other: The matrix to cast like

  #   Returns:
  #       This matrix cast to the same type as other
  #   """
  #   return self + self.zeros_like(other)

  @classmethod
  def zeros_like(cls, other: 'SquareMatrix') -> 'SquareMatrix':
    """Sets all of the values of this matrix to zero"""
    dim = other.shape[0]
    return cls.zeros(dim, dtype=other.mat.dtype)

  @classmethod
  def inf_like(cls, other: 'SquareMatrix') -> 'SquareMatrix':
    """Sets all of the values of this matrix to inf"""
    dim = other.shape[0]
    return cls.inf(dim, dtype=other.mat.dtype)

  def set_eye(self) -> 'SquareMatrix':
    return SquareMatrix(tags=TAGS.no_tags, mat=cola.ops.Identity(self.shape[0], dtype=self.mat.dtype))

  def set_symmetric(self) -> 'SquareMatrix':
    return SquareMatrix(tags=self.tags, mat=cola.PSD(0.5*(self.mat + self.mat.T)))

  def set_zero(self) -> 'SquareMatrix':
    return SquareMatrix(tags=TAGS.zero_tags, mat=self.mat*0.0)

  def set_inf(self) -> 'SquareMatrix':
    return SquareMatrix(tags=TAGS.inf_tags, mat=self.mat*torch.inf)

  @property
  def is_zero(self):
    return ~self.tags.is_nonzero

  @property
  def is_inf(self):
    return self.tags.is_inf

  @property
  def is_symmetric(self):
    return self.tags.is_symmetric

  @property
  def is_eye(self):
    return self.tags.is_eye

  @property
  def shape(self):
    return self.mat.shape

  @property
  def ndim(self):
    return len(self.shape)

  @classmethod
  def zeros(cls, dim: int, dtype: torch.dtype) -> 'SquareMatrix':
    matrix = cola.ops.ScalarMul(0.0, shape=(dim, dim), dtype=dtype)
    return SquareMatrix(tags=TAGS.zero_tags, mat=matrix)

  @classmethod
  def eye(cls, dim: int, dtype: torch.dtype) -> 'SquareMatrix':
    matrix = cola.ops.ScalarMul(1.0, shape=(dim, dim), dtype=dtype)
    return SquareMatrix(tags=TAGS.no_tags, mat=matrix)

  @classmethod
  def from_dense(cls, tensor: torch.Tensor) -> "SquareMatrix":
    return cls(tags=TAGS.no_tags, mat=cola.ops.Dense(tensor))

  def to_dense(self) -> Float[Array, "M N"]:
    return self.mat.to_dense()

  def __neg__(self) -> 'SquareMatrix':
    return SquareMatrix(tags=self.tags, mat=-self.mat)

  def __repr__(self):
    return f'{type(self).__name__}(\n{self.to_dense()}\n)'

  def __add__(self, other: 'SquareMatrix') -> 'SquareMatrix':
    new_tags = self.tags.add_update(other.tags)
    new_mat = self.mat + other.mat
    return SquareMatrix(tags=new_tags, mat=new_mat)

  def __sub__(self, other: 'SquareMatrix') -> 'SquareMatrix':
    return self + (-other)

  def __mul__(self, other: Scalar) -> 'SquareMatrix':
    new_tags = self.tags.scalar_mul_update()
    new_mat = self.mat * other
    return SquareMatrix(tags=new_tags, mat=new_mat)

  def __rmul__(self, other: Scalar) -> 'SquareMatrix':
    return self * other

  def __matmul__(self, other: Union['SquareMatrix', Float[Array, 'N']]) -> 'SquareMatrix':
    return mat_mul(self, other)

  def __truediv__(self, other: Scalar) -> 'SquareMatrix':
    return self * (1/other)

  def transpose(self):
    return SquareMatrix(tags=self.tags, mat=self.mat.T)

  @property
  def T(self):
    return self.transpose()

  def solve(self, other: Union['SquareMatrix', Float[Array, 'N']]) -> 'SquareMatrix':
    return matrix_solve(self, other)

  def get_inverse(self) -> 'SquareMatrix':
    new_tags = self.tags.inverse_update()
    new_mat = cola.linalg.inv(self.mat)
    return SquareMatrix(tags=new_tags, mat=new_mat)

  def get_log_det(self) -> Scalar:
    return cola.linalg.logdet(self.mat)

  def get_cholesky(self) -> 'SquareMatrix':
    new_tags = self.tags.cholesky_update()
    new_mat = cola.linalg.decompositions.decompositions.Cholesky()(self.mat)
    return SquareMatrix(tags=new_tags, mat=new_mat)

  def get_exp(self) -> 'SquareMatrix':
    new_tags = self.tags.exp_update()
    new_mat = cola.linalg.exp(self.mat)
    return SquareMatrix(tags=new_tags, mat=new_mat)

  # def get_svd(self) -> Tuple['SquareMatrix', 'SquareMatrix', 'SquareMatrix']:
  #   new_tags = self.tags.svd_update()
  #   Umat, s_elts, Vmat = cola.linalg.svd(self.mat)
  #   return SquareMatrix(tags=new_tags, mat=new_mat)

################################################################################################################

@dispatch
def mat_mul(A: SquareMatrix, B: SquareMatrix) -> SquareMatrix:
  new_tags = A.tags.mat_mul_update(B.tags)
  return SquareMatrix(tags=new_tags, mat=A.mat@B.mat)

@dispatch
def mat_mul(A: SquareMatrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return A.mat@b

@dispatch
def mat_mul(A: SquareMatrix, b: torch.Tensor) -> torch.Tensor:
  return A.mat @ b

@dispatch
def matrix_solve(A: SquareMatrix, B: SquareMatrix) -> SquareMatrix:
  out_elements = cola.linalg.solve(A.mat, B.mat)
  out_tags = A.tags.solve_update(B.tags)
  return SquareMatrix(tags=out_tags, mat=out_elements)

@dispatch
def matrix_solve(A: SquareMatrix, b: Float[Array, 'N']) -> Float[Array, 'M']:
  return cola.linalg.solve(A.mat, b)

@dispatch
def matrix_solve(A: SquareMatrix, b: torch.Tensor) -> torch.Tensor:
  return cola.linalg.solve(A.mat, b)


# Ensure our custom repr is used instead of the TensorClass-injected one
def _square_matrix_repr(self) -> str:
  return f'SquareMatrix(tags={self.tags}, mat={self.mat})'

SquareMatrix.__repr__ = _square_matrix_repr

if __name__ == '__main__':
  from debug import *
  import cola
  A = cola.ops.Dense(torch.Tensor([[1., 2.],
                                   [3., 4.]]))
  tags = TAGS.no_tags
  mat = SquareMatrix(mat=A, tags=tags)
  print(cola.linalg.trace(cola.kron(A, A)))
  import pdb; pdb.set_trace()
