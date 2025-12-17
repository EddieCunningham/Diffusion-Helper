import torch
from typing import Tuple, Union
from jaxtyping import Array, Float, Scalar
from plum import dispatch
from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import TAGS
from tensordict import TensorClass

class MatrixWithInverse(TensorClass):

  matrix: SquareMatrix
  inverse_matrix: SquareMatrix

  def __post_init__(self):
    if self.matrix.shape != self.inverse_matrix.shape:
      raise ValueError(f"Matrix and inverse matrix must have the same shape, got {self.matrix.shape} and {self.inverse_matrix.shape}")

  @property
  def tags(self):
    return self.matrix.tags

  @property
  def mat(self):
    return self.matrix.mat

  def set_eye(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(matrix=self.matrix.set_eye(), inverse_matrix=self.inverse_matrix.set_eye())

  def set_symmetric(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(matrix=self.matrix.set_symmetric(), inverse_matrix=self.inverse_matrix.set_symmetric())

  def set_zero(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(matrix=self.matrix.set_zero(), inverse_matrix=self.inverse_matrix.set_inf())

  def set_inf(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(matrix=self.matrix.set_inf(), inverse_matrix=self.inverse_matrix.set_zero())

  @property
  def batch_size(self) -> Union[None,int,Tuple[int]]:
    return self.matrix.batch_size

  @property
  def shape(self):
    return self.matrix.shape

  def to_dense(self):
    return self.matrix.to_dense()

  @property
  def is_zero(self):
    return self.matrix.is_zero

  @property
  def is_inf(self):
    return self.matrix.is_inf

  @property
  def T(self):
    return MatrixWithInverse(matrix=self.matrix.T, inverse_matrix=self.inverse_matrix.T)

  def transpose(self):
    return self.T

  def __neg__(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(matrix=-self.matrix, inverse_matrix=-self.inverse_matrix)

  def __add__(self, other):
    if isinstance(other, MatrixWithInverse):
      return self.matrix + other.matrix
    return self.matrix + other

  def __radd__(self, other):
    return self.matrix + other

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, scalar):
    return MatrixWithInverse(matrix=scalar * self.matrix, inverse_matrix=(1/scalar) * self.inverse_matrix)

  def __rmul__(self, scalar):
    return self * scalar

  def __truediv__(self, scalar):
    return self * (1/scalar)

  def __matmul__(self, other):
    if isinstance(other, MatrixWithInverse):
      return MatrixWithInverse(matrix=self.matrix @ other.matrix, inverse_matrix=other.inverse_matrix @ self.inverse_matrix)
    elif isinstance(other, SquareMatrix):
      return self.matrix @ other
    else:
      return self.matrix @ other

  def solve(self, other):
    if isinstance(other, MatrixWithInverse):
      sol = self.inverse_matrix @ other.matrix
      sol_inv = other.inverse_matrix @ self.matrix
      return MatrixWithInverse(matrix=sol, inverse_matrix=sol_inv)
    elif isinstance(other, SquareMatrix):
      return self.inverse_matrix @ other
    else:
      return self.inverse_matrix @ other

  def get_inverse(self) -> 'MatrixWithInverse':
    return MatrixWithInverse(matrix=self.inverse_matrix, inverse_matrix=self.matrix)

  def get_log_det(self):
    return self.matrix.get_log_det()

  def get_cholesky(self):
    return self.matrix.get_cholesky()

  def get_exp(self):
    return self.matrix.get_exp()

################################################################################################################

@dispatch
def mat_add(A: MatrixWithInverse, B: MatrixWithInverse) -> SquareMatrix:
  return A.matrix + B.matrix

@dispatch
def mat_add(A: MatrixWithInverse, B: SquareMatrix) -> SquareMatrix:
  return A.matrix + B

@dispatch
def mat_add(A: SquareMatrix, B: MatrixWithInverse) -> SquareMatrix:
  return A + B.matrix

################################################################################################################

@dispatch
def scalar_mul(A: MatrixWithInverse, s: Scalar) -> MatrixWithInverse:
  return MatrixWithInverse(matrix=s*A.matrix, inverse_matrix=(1/s)*A.inverse_matrix)

################################################################################################################

@dispatch
def mat_mul(A: MatrixWithInverse, b: torch.Tensor) -> torch.Tensor:
  return A.matrix @ b

@dispatch
def mat_mul(A: MatrixWithInverse, B: MatrixWithInverse) -> MatrixWithInverse:
  return MatrixWithInverse(matrix=A.matrix @ B.matrix, inverse_matrix=B.inverse_matrix @ A.inverse_matrix)

@dispatch
def mat_mul(A: MatrixWithInverse, B: SquareMatrix) -> SquareMatrix:
  return A.matrix @ B

@dispatch
def mat_mul(A: SquareMatrix, B: MatrixWithInverse) -> SquareMatrix:
  return A @ B.matrix

################################################################################################################

@dispatch
def transpose(A: MatrixWithInverse) -> MatrixWithInverse:
  return MatrixWithInverse(matrix=A.matrix.T, inverse_matrix=A.inverse_matrix.T)

################################################################################################################

@dispatch
def matrix_solve(A: MatrixWithInverse, b: torch.Tensor) -> torch.Tensor:
  return A.inverse_matrix @ b

@dispatch
def matrix_solve(A: MatrixWithInverse, B: MatrixWithInverse) -> MatrixWithInverse:
  sol = A.inverse_matrix @ B.matrix
  sol_inv = B.inverse_matrix @ A.matrix
  return MatrixWithInverse(matrix=sol, inverse_matrix=sol_inv)

@dispatch
def matrix_solve(A: MatrixWithInverse, B: SquareMatrix) -> SquareMatrix:
  return A.inverse_matrix @ B

@dispatch
def matrix_solve(A: SquareMatrix, B: MatrixWithInverse) -> SquareMatrix:
  return A.solve(B.matrix)

################################################################################################################

@dispatch
def get_matrix_inverse(A: MatrixWithInverse) -> MatrixWithInverse:
  return MatrixWithInverse(matrix=A.inverse_matrix, inverse_matrix=A.matrix)

################################################################################################################

@dispatch
def get_log_det(A: MatrixWithInverse) -> Scalar:
  return A.matrix.get_log_det()

@dispatch
def get_cholesky(A: MatrixWithInverse) -> SquareMatrix:
  return A.matrix.get_cholesky()

@dispatch
def get_exp(A: MatrixWithInverse) -> SquareMatrix:
  return A.matrix.get_exp()

# @dispatch
# def get_svd(A: MatrixWithInverse) -> Tuple[SquareMatrix, 'DiagonalMatrix', SquareMatrix]:
#   return A.matrix.get_svd()
