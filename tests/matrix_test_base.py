import torch
from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.matrix.tags import Tags, TAGS
from typing import Callable, Union, Any
import abc

def matrices_equal(A: Union[SquareMatrix, torch.Tensor], B: Union[SquareMatrix, torch.Tensor]):
  # Check if A has to_dense method (covers SquareMatrix and MatrixWithInverse)
  if hasattr(A, 'to_dense'):
    Amat = A.to_dense()
  else:
    Amat = A

  # Check if B has to_dense method (covers SquareMatrix and MatrixWithInverse)
  if hasattr(B, 'to_dense'):
    Bmat = B.to_dense()
  else:
    Bmat = B

  # Compare finite parts and non-finite masks separately
  if Amat.shape != Bmat.shape:
    return False

  finite_mask = torch.isfinite(Amat) & torch.isfinite(Bmat)
  if torch.any(finite_mask):
    if not torch.allclose(Amat[finite_mask], Bmat[finite_mask]):
      return False

  # Both NaN at same positions
  nan_ok = torch.equal(torch.isnan(Amat), torch.isnan(Bmat))
  # Both +inf/-inf at same positions
  inf_ok = torch.equal((Amat == torch.inf), (Bmat == torch.inf)) and \
           torch.equal((Amat == -torch.inf), (Bmat == -torch.inf))

  return nan_ok and inf_ok and (
    torch.all(~finite_mask | torch.isfinite(Amat))  # structure preserved
  )

def matrix_tests(key, A, B):

  A_dense = A.to_dense()
  B_dense = B.to_dense()

  # Check transpose
  if matrices_equal(A.T, A_dense.T) == False:
    raise ValueError(f"Transpose test failed.  Expected {A.T}, got {A_dense.T}")

  # Check addition
  C = A + B
  C_dense = A_dense + B_dense
  if matrices_equal(C, C_dense) == False:
    raise ValueError(f"Addition test failed.  Expected {C}, got {C_dense}")

  # # Check addition with a scalar
  # C = A + 1.0
  # C_dense = A_dense + 1.0
  # if matrices_equal(C, C_dense) == False:
  #   raise ValueError(f"Addition test failed.  Expected {C}, got {C_dense}")

  # Check matrix multiplication
  C = A@B.T
  C_dense = A_dense@B_dense.T
  if matrices_equal(C, C_dense) == False:
    raise ValueError(f"Matrix multiplication test failed.  Expected {C}, got {C_dense}")

  # Check matrix vector products
  x = torch.randn(A.shape[1], dtype=A_dense.dtype, generator=key)
  y = A@x
  y_dense = A_dense@x
  if matrices_equal(y, y_dense) == False:
    raise ValueError(f"Matrix vector product test failed.  Expected {y}, got {y_dense}")

  # Check scalar multiplication
  C = 2.0*A
  C_dense = 2.0*A_dense
  if matrices_equal(C, C_dense) == False:
    raise ValueError(f"Scalar multiplication test failed.  Expected {C}, got {C_dense}")

  if A.shape[0] == A.shape[1]:
    # Check the inverse
    A_inv = A.get_inverse()

    if A.is_inf:
      assert torch.all(A_inv.is_zero)
    elif A.is_zero:
      assert torch.all(A_inv.is_inf)
    else:
      A_inv_dense = torch.linalg.inv(A_dense)
      if matrices_equal(A_inv, A_inv_dense) == False:
        raise ValueError(f"Matrix inverse test failed.  Expected {A_inv}, got {A_inv_dense}")

    # Check solve
    x = torch.randn(A.shape[1], dtype=A_dense.dtype, generator=key)
    y = A.solve(x)
    if A.is_inf:
      pass
    elif A.is_zero:
      pass # Don't check this
    else:
      y_dense = A_inv_dense@x
      if matrices_equal(y, y_dense) == False:
        raise ValueError(f"Matrix solve test failed.  Expected {y}, got {y_dense}")

  # Check the cholesky decomposition
  J = A@A.T
  J_dense = J.to_dense()
  if not J.is_zero:
    J_chol = J.get_cholesky()
    J_chol_dense = torch.linalg.cholesky(J_dense)
    if matrices_equal(J_chol, J_chol_dense) == False:
      raise ValueError(f"Cholesky decomposition test failed.  Expected {J_chol}, got {J_chol_dense}")

  # Check the log determinant
  log_det = J.get_log_det()
  log_det_dense = torch.linalg.slogdet(J_dense)[1]
  if matrices_equal(log_det, log_det_dense) == False:
    raise ValueError(f"Log determinant test failed.  Expected {log_det}, got {log_det_dense}")

  # Check the SVD (skip if not implemented)
  if hasattr(J, 'get_svd') and callable(getattr(J, 'get_svd')):
    (U, s, V) = J.get_svd()
    U_dense, s_dense, Vh_dense = torch.linalg.svd(J_dense, full_matrices=False)
    V_dense = Vh_dense.T

  # SVD is not unique due to potential permutations, sign flips, and ordering
  # Instead of direct comparison, verify reconstruction and orthogonality

  if A.is_inf:
    return

  # Verify singular values match (regardless of order)
  if hasattr(J, 'get_svd') and callable(getattr(J, 'get_svd')):
    s_values = torch.diag(s.to_dense())
    if not torch.allclose(torch.sort(torch.abs(s_values)).values, torch.sort(torch.abs(s_dense)).values, atol=1e-5):
      raise ValueError(f"SVD test failed: singular values don't match")

  # Verify matrices are orthogonal (U*U^T = I, V*V^T = I)
  if hasattr(J, 'get_svd') and callable(getattr(J, 'get_svd')):
    U_mat = U.to_dense()
    V_mat = V.to_dense()
    if not torch.allclose(U_mat @ U_mat.T, torch.eye(U_mat.shape[0], dtype=U_mat.dtype, device=U_mat.device), atol=1e-5):
      raise ValueError(f"SVD test failed: U is not orthogonal")
    if not torch.allclose(V_mat @ V_mat.T, torch.eye(V_mat.shape[0], dtype=V_mat.dtype, device=V_mat.device), atol=1e-5):
      raise ValueError(f"SVD test failed: V is not orthogonal")

  # Verify reconstruction: A = U*S*V^T
  if hasattr(J, 'get_svd') and callable(getattr(J, 'get_svd')):
    reconstruction = U_mat @ torch.diag(s_values) @ V_mat.T
    if not torch.allclose(reconstruction, J_dense, atol=1e-5):
      raise ValueError(f"SVD test failed: reconstruction doesn't match original matrix")

def performance_tests(A, B):
  # Basic operations
  C1 = A + B
  C2 = C1 - B
  C3 = 2.0 * C2
  C4 = C3 / 2.0
  C5 = C4 @ B
  C6 = C5.T

  # Single matrix operations
  # Use an SPD matrix for Cholesky to avoid numerical issues
  SPD = (C6 @ C6.T)
  C7 = SPD.get_inverse()
  C8 = C7.get_cholesky()

  # Matrix-vector operations
  x = torch.ones(A.shape[1], dtype=C8.to_dense().dtype)
  y = C8 @ x
  z = C8.solve(x).reshape(-1, 1) @ y.reshape(1, -1)  # Outer product to get matrix

  # Get log determinant (scalar) and convert back to matrix
  log_det = C8.get_log_det()

class AbstractMatrixTest(abc.ABC):
    """Base test class for any AbstractSquareMatrix implementation.

    Subclasses must override the create_matrix and create_* methods to provide matrix instances
    of the specific type being tested.
    """

    # These properties should be set by subclasses
    matrix_class = None  # The matrix class to test

    # These are factory methods that subclasses must override
    @abc.abstractmethod
    def create_matrix(self, elements, tags=None):
        """Create a matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_matrix")

    @abc.abstractmethod
    def create_random_matrix(self, key, shape=None, tags=None):
        """Create a random matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_random_matrix")

    @abc.abstractmethod
    def create_random_symmetric_matrix(self, key, dim=None, tags=None):
        """Create a random symmetric matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_random_symmetric_matrix")

    @abc.abstractmethod
    def create_zeros_matrix(self, dim=None, tags=None):
        """Create a zeros matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_zeros_matrix")

    @abc.abstractmethod
    def create_eye_matrix(self, dim=None, tags=None):
        """Create an identity matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_eye_matrix")

    @abc.abstractmethod
    def create_well_conditioned_matrix(self, key, dim=None, tags=None):
        """Create a well-conditioned matrix of the specific type being tested."""
        raise NotImplementedError("Subclasses must implement create_well_conditioned_matrix")

    @abc.abstractmethod
    def test_initialization(self):
        raise NotImplementedError("Subclasses must implement test_initialization")

    # Common tests that work with any matrix type
    def test_shape_property(self):
        A = self.create_random_matrix(self.key)
        self.assertEqual(len(A.shape), 2)
        self.assertEqual(A.shape[0], self.dim)
        self.assertEqual(A.shape[1], self.dim)

        zero = self.create_zeros_matrix()
        self.assertEqual(zero.shape, (self.dim, self.dim))

    def test_batch_size(self):
        # Test single matrix (no batch)
        A = self.create_random_matrix(self.key)
        self.assertIsNone(A.batch_size)

    def test_class_methods(self):
        # Test zeros method
        zero_mat = self.create_zeros_matrix()
        zero_array = torch.zeros((self.dim, self.dim), dtype=zero_mat.to_dense().dtype)
        # For diagonal matrices, to_dense() will create a diagonal matrix from elements
        self.assertTrue(matrices_equal(zero_mat.to_dense(), zero_array))
        self.assertEqual(zero_mat.tags, TAGS.zero_tags)

        # Test eye method
        eye_mat = self.create_eye_matrix()
        self.assertTrue(matrices_equal(eye_mat.to_dense(), torch.eye(self.dim, dtype=eye_mat.to_dense().dtype)))

    def test_neg(self):
        A = self.create_random_matrix(self.key)
        neg_A = -A
        self.assertTrue(matrices_equal(neg_A.to_dense(), -A.to_dense()))

    def test_addition(self):
        g = self.key
        s1 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        s2 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        key1 = torch.Generator().manual_seed(s1)
        key2 = torch.Generator().manual_seed(s2)
        A = self.create_random_matrix(key1)
        B = self.create_random_matrix(key2)

        # Test regular addition
        C = A + B
        self.assertTrue(matrices_equal(C.to_dense(), A.to_dense() + B.to_dense()))

        # Test addition with zero
        zero = self.create_zeros_matrix()
        D = A + zero
        self.assertTrue(matrices_equal(D.to_dense(), A.to_dense()))

    def test_subtraction(self):
        g = self.key
        s1 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        s2 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        key1 = torch.Generator().manual_seed(s1)
        key2 = torch.Generator().manual_seed(s2)
        A = self.create_random_matrix(key1)
        B = self.create_random_matrix(key2)

        C = A - B
        self.assertTrue(matrices_equal(C.to_dense(), A.to_dense() - B.to_dense()))

    def test_scalar_multiplication(self):
        A = self.create_random_matrix(self.key)
        scalar = 2.5

        # Test left multiplication
        C = scalar * A
        self.assertTrue(matrices_equal(C.to_dense(), scalar * A.to_dense()))

        # Test right multiplication
        D = A * scalar
        self.assertTrue(matrices_equal(D.to_dense(), A.to_dense() * scalar))

    def test_matrix_multiplication(self):
        g = self.key
        s1 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        s2 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        key1 = torch.Generator().manual_seed(s1)
        key2 = torch.Generator().manual_seed(s2)
        A = self.create_random_matrix(key1)
        B = self.create_random_matrix(key2)

        # Test matrix-matrix multiplication
        C = A @ B
        self.assertTrue(matrices_equal(C.to_dense(), A.to_dense() @ B.to_dense()))

        # Test matrix-vector multiplication
        v = torch.randn(self.dim, dtype=A.to_dense().dtype, generator=self.key)
        result = A @ v
        expected = A.to_dense() @ v
        self.assertTrue(matrices_equal(result, expected))

    def test_division(self):
        A = self.create_random_matrix(self.key)
        scalar = 2.0

        C = A / scalar
        self.assertTrue(matrices_equal(C.to_dense(), A.to_dense() / scalar))

    def test_transpose(self):
        A = self.create_random_matrix(self.key)
        A_t = A.T
        self.assertTrue(matrices_equal(A_t.to_dense(), A.to_dense().T))

    def test_solve(self):
        A = self.create_well_conditioned_matrix(self.key)

        # Test matrix-vector solve
        b = torch.randn(self.dim, dtype=A.to_dense().dtype, generator=self.key)
        x = A.solve(b)
        expected = torch.linalg.solve(A.to_dense(), b)
        self.assertTrue(matrices_equal(x, expected))

        # Test matrix-matrix solve
        g = self.key
        s1 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
        key1 = torch.Generator().manual_seed(s1)
        B = self.create_random_matrix(key1)
        X = A.solve(B)
        expected_mat = torch.linalg.solve(A.to_dense(), B.to_dense())
        self.assertTrue(matrices_equal(X.to_dense(), expected_mat))

    def test_inverse(self):
        A = self.create_well_conditioned_matrix(self.key)

        A_inv = A.get_inverse()
        expected_inv = torch.linalg.inv(A.to_dense())
        self.assertTrue(matrices_equal(A_inv.to_dense(), expected_inv))

    def test_log_det(self):
        A = self.create_well_conditioned_matrix(self.key)

        log_det = A.get_log_det()
        expected_log_det = torch.linalg.slogdet(A.to_dense())[1]
        self.assertTrue(torch.allclose(log_det, expected_log_det))

    def test_cholesky(self):
        A = self.create_random_symmetric_matrix(self.key)

        chol = A.get_cholesky()
        expected_chol = torch.linalg.cholesky(A.to_dense())
        self.assertTrue(matrices_equal(chol.to_dense(), expected_chol))

    def test_exp(self):
        A = self.create_random_matrix(self.key)

        A_exp = A.get_exp()
        expected_exp = torch.linalg.matrix_exp(A.to_dense())
        self.assertTrue(matrices_equal(A_exp.to_dense(), expected_exp))


def autodiff_for_matrix_class(create_diagonal_fn):
    """Test that autodifferentiation works for all matrix operations.

    Args:
        create_diagonal_fn: The matrix class to test
        batch_support: Whether the matrix class supports batched operations
    """
    # Use float64 for better numerical stability

    # Set up test matrices
    gen = torch.Generator().manual_seed(42)
    dim = 6

    # Create well-conditioned matrices for stable testing
    s1 = torch.randint(0, 2**31 - 1, (1,), generator=gen).item()
    s2 = torch.randint(0, 2**31 - 1, (1,), generator=gen).item()
    gen1 = torch.Generator().manual_seed(s1)
    gen2 = torch.Generator().manual_seed(s2)
    A_raw = torch.randn(dim, dim, generator=gen1, dtype=torch.float64)
    A_raw = A_raw @ A_raw.T + dim * torch.eye(dim, dtype=torch.float64)  # Make positive definite

    B_raw = torch.randn(dim, dim, generator=gen2, dtype=torch.float64)
    B_raw = B_raw @ B_raw.T + dim * torch.eye(dim, dtype=torch.float64)

    v = torch.randn(dim, generator=gen, dtype=torch.float64)
    scalar = 2.0

    # Create matrix factory functions
    create_A = lambda x: create_diagonal_fn(x, tags=TAGS.no_tags)
    create_B = lambda: create_diagonal_fn(B_raw, tags=TAGS.no_tags)

    # Cast to the correct structure
    A_raw = create_A(A_raw).to_dense()
    B_raw = create_B().to_dense()

    # Test core operations
    def neg_fn(x):
        A = create_A(x)
        return (-A).elements.sum()

    def direct_neg_fn(x):
        return (-x).sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = neg_fn(x_var)
    grad_neg_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_neg_fn(x_var2)
    expected_grad_neg_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_neg = create_diagonal_fn(grad_neg_raw, TAGS.no_tags)
    expected_grad_neg = create_diagonal_fn(expected_grad_neg_raw, TAGS.no_tags)
    assert torch.allclose(grad_neg.to_dense(), expected_grad_neg.to_dense())

    # Addition
    def add_fn(x):
        A = create_A(x)
        B = create_B()
        return (A + B).elements.sum()

    def direct_add_fn(x):
        return (x + B_raw).sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = add_fn(x_var)
    grad_add_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_add_fn(x_var2)
    expected_grad_add_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_add = create_diagonal_fn(grad_add_raw, TAGS.no_tags)
    expected_grad_add = create_diagonal_fn(expected_grad_add_raw, TAGS.no_tags)
    assert torch.allclose(grad_add.to_dense(), expected_grad_add.to_dense())

    # Subtraction
    def sub_fn(x):
        A = create_A(x)
        B = create_B()
        return (A - B).elements.sum()

    def direct_sub_fn(x):
        return (x - B_raw).sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = sub_fn(x_var)
    grad_sub_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_sub_fn(x_var2)
    expected_grad_sub_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_sub = create_diagonal_fn(grad_sub_raw, TAGS.no_tags)
    expected_grad_sub = create_diagonal_fn(expected_grad_sub_raw, TAGS.no_tags)
    assert torch.allclose(grad_sub.to_dense(), expected_grad_sub.to_dense())

    # Scalar multiplication
    def scalar_mul_fn(x):
        A = create_A(x)
        return (scalar * A).elements.sum()

    def direct_scalar_mul_fn(x):
        return (scalar * x).sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = scalar_mul_fn(x_var)
    grad_scalar_mul_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_scalar_mul_fn(x_var2)
    expected_grad_scalar_mul_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_scalar_mul = create_diagonal_fn(grad_scalar_mul_raw, TAGS.no_tags)
    expected_grad_scalar_mul = create_diagonal_fn(expected_grad_scalar_mul_raw, TAGS.no_tags)
    assert torch.allclose(grad_scalar_mul.to_dense(), expected_grad_scalar_mul.to_dense())

    # Matrix multiplication
    def matmul_fn(x):
        A = create_A(x)
        B = create_B()
        return (A @ B).elements.sum()

    def direct_matmul_fn(x):
        return (x @ B_raw).sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = matmul_fn(x_var)
    grad_matmul_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_matmul_fn(x_var2)
    expected_grad_matmul_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_matmul = create_diagonal_fn(grad_matmul_raw, TAGS.no_tags)
    expected_grad_matmul = create_diagonal_fn(expected_grad_matmul_raw, TAGS.no_tags)
    assert torch.allclose(grad_matmul.to_dense(), expected_grad_matmul.to_dense())

    # Matrix-vector multiplication
    def matvec_fn(x):
        A = create_A(x)
        return torch.sum(A @ v)

    def direct_matvec_fn(x):
        return torch.sum(x @ v)

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = matvec_fn(x_var)
    grad_matvec_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_matvec_fn(x_var2)
    expected_grad_matvec_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_matvec = create_diagonal_fn(grad_matvec_raw, TAGS.no_tags)
    expected_grad_matvec = create_diagonal_fn(expected_grad_matvec_raw, TAGS.no_tags)
    assert torch.allclose(grad_matvec.to_dense(), expected_grad_matvec.to_dense())

    # Division
    def div_fn(x):
        A = create_A(x)
        return (A / scalar).elements.sum()

    def direct_div_fn(x):
        return (x / scalar).sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = div_fn(x_var)
    grad_div_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_div_fn(x_var2)
    expected_grad_div_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_div = create_diagonal_fn(grad_div_raw, TAGS.no_tags)
    expected_grad_div = create_diagonal_fn(expected_grad_div_raw, TAGS.no_tags)
    assert torch.allclose(grad_div.to_dense(), expected_grad_div.to_dense())

    # Transpose
    def transpose_fn(x):
        A = create_A(x)
        return A.T.elements.sum()

    def direct_transpose_fn(x):
        return x.T.sum()

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = transpose_fn(x_var)
    grad_transpose_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_transpose_fn(x_var2)
    expected_grad_transpose_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_transpose = create_diagonal_fn(grad_transpose_raw, TAGS.no_tags)
    expected_grad_transpose = create_diagonal_fn(expected_grad_transpose_raw, TAGS.no_tags)
    assert torch.allclose(grad_transpose.to_dense(), expected_grad_transpose.to_dense())

    # Advanced operations

    # Matrix-vector solve
    def solve_vec_fn(x):
        A = create_A(x)
        return torch.sum(A.solve(v))

    def direct_solve_vec_fn(x):
        return torch.sum(torch.linalg.solve(x, v))

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = solve_vec_fn(x_var)
    grad_solve_vec_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_solve_vec_fn(x_var2)
    expected_grad_solve_vec_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_solve_vec = create_diagonal_fn(grad_solve_vec_raw, TAGS.no_tags)
    expected_grad_solve_vec = create_diagonal_fn(expected_grad_solve_vec_raw, TAGS.no_tags)
    assert torch.allclose(grad_solve_vec.to_dense(), expected_grad_solve_vec.to_dense())

    # Inverse
    def inverse_fn(x):
        A = create_A(x)
        return A.get_inverse().elements.sum()

    def direct_inverse_fn(x):
        return torch.sum(torch.linalg.inv(x))

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = inverse_fn(x_var)
    grad_inverse_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_inverse_fn(x_var2)
    expected_grad_inverse_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_inverse = create_diagonal_fn(grad_inverse_raw, TAGS.no_tags)
    expected_grad_inverse = create_diagonal_fn(expected_grad_inverse_raw, TAGS.no_tags)
    assert torch.allclose(grad_inverse.to_dense(), expected_grad_inverse.to_dense())

    # Log determinant
    def logdet_fn(x):
        A = create_A(x)
        return A.get_log_det()

    def direct_logdet_fn(x):
        return torch.linalg.slogdet(x)[1]

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = logdet_fn(x_var)
    grad_logdet_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_logdet_fn(x_var2)
    expected_grad_logdet_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_logdet = create_diagonal_fn(grad_logdet_raw, TAGS.no_tags)
    expected_grad_logdet = create_diagonal_fn(expected_grad_logdet_raw, TAGS.no_tags)
    assert torch.allclose(grad_logdet.to_dense(), expected_grad_logdet.to_dense())

    # Cholesky
    def cholesky_fn(x):
        A = create_A(x)
        return A.get_cholesky().elements.sum()

    def direct_cholesky_fn(x):
        return torch.sum(torch.linalg.cholesky(x))

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = cholesky_fn(x_var)
    grad_cholesky_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_cholesky_fn(x_var2)
    expected_grad_cholesky_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_cholesky = create_diagonal_fn(grad_cholesky_raw, TAGS.no_tags)
    expected_grad_cholesky = create_diagonal_fn(expected_grad_cholesky_raw, TAGS.no_tags)
    assert torch.allclose(grad_cholesky.to_dense(), expected_grad_cholesky.to_dense())

    # Matrix exponential
    def exp_fn(x):
        A = create_A(x)
        return A.get_exp().elements.sum()

    def direct_exp_fn(x):
        return torch.sum(torch.linalg.matrix_exp(x))

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = exp_fn(x_var)
    grad_exp_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_exp_fn(x_var2)
    expected_grad_exp_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_exp = create_diagonal_fn(grad_exp_raw, TAGS.no_tags)
    expected_grad_exp = create_diagonal_fn(expected_grad_exp_raw, TAGS.no_tags)
    assert torch.allclose(grad_exp.to_dense(), expected_grad_exp.to_dense())

    # SVD (just U component)
    def svd_u_fn(x):
        A = create_A(x)
        U, _, _ = A.get_svd()
        return U.elements.sum()

    def direct_svd_u_fn(x):
        U, _, _ = torch.linalg.svd(x, full_matrices=False)
        return torch.sum(U)

    x_var = A_raw.clone().detach().requires_grad_(True)
    y_val = svd_u_fn(x_var)
    grad_svd_u_raw = torch.autograd.grad(y_val, x_var, retain_graph=False, create_graph=False)[0]
    x_var2 = A_raw.clone().detach().requires_grad_(True)
    y_val2 = direct_svd_u_fn(x_var2)
    expected_grad_svd_u_raw = torch.autograd.grad(y_val2, x_var2, retain_graph=False, create_graph=False)[0]
    grad_svd_u = create_diagonal_fn(grad_svd_u_raw, TAGS.no_tags)
    expected_grad_svd_u = create_diagonal_fn(expected_grad_svd_u_raw, TAGS.no_tags)
    assert torch.allclose(grad_svd_u.to_dense(), expected_grad_svd_u.to_dense())


def matrix_implementations_tests(
    key: torch.Generator,
    create_matrix_fn: Callable[[torch.Tensor, Tags], SquareMatrix]
):
    """Test the correctness of a matrix implementation with different tag combinations.

    Args:
        key: Random key
        matrix_class: The matrix class to test
    """
    # Test a subset of tag combinations
    tag_options = [
        TAGS.zero_tags,
        TAGS.no_tags
    ]

    for tag_A in tag_options:
        for tag_B in tag_options:
            g = key
            s1 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
            s2 = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
            k1 = torch.Generator().manual_seed(s1)
            k2 = torch.Generator().manual_seed(s2)
            # advance key deterministically
            s_next = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
            key = torch.Generator().manual_seed(s_next)

            A_raw = torch.randn(6, 6, generator=k1, dtype=torch.float64)
            B_raw = torch.randn(6, 6, generator=k2, dtype=torch.float64)

            # Modify matrices according to tags
            if tag_A.is_zero:
                A_raw = torch.zeros_like(A_raw)

            if tag_B.is_zero:
                B_raw = torch.zeros_like(B_raw)

            A = create_matrix_fn(A_raw, tag_A)
            B = create_matrix_fn(B_raw, tag_B)

            # Run the matrix tests - should not raise exceptions
            matrix_tests(key, A, B)