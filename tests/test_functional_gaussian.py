import torch
import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))

from diffusion_helper.src.matrix.matrix import SquareMatrix
from diffusion_helper.src.functional.linear_functional import LinearFunctional
from diffusion_helper.src.gaussian.gaussian import StandardGaussian, MixedGaussian
from diffusion_helper.src.functional.quadratic_form import QuadraticForm


class TestFunctionalGaussian(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.key = torch.manual_seed(42)
        self.dim = 4
        self.x_dim = self.dim

        # Create components for the gaussian
        J = SquareMatrix.from_dense(torch.randn(self.dim, self.dim))
        J = J.T@J
        self.J = J.set_symmetric()
        self.logZ = torch.randn(())

        # Create components for the linear functional
        A_dense = torch.randn(self.dim, self.x_dim)
        self.A = SquareMatrix.from_dense(A_dense)
        self.b = torch.randn(self.dim)
        self.h_lf = LinearFunctional(self.A, self.b)

        # Latent variable
        self.x = torch.randn(self.x_dim)

        # The resolved, standard h vector
        self.h_vec = self.h_lf(self.x)

        # Create StandardGaussian instances
        self.sg_functional = StandardGaussian(self.h_lf, self.J)
        self.sg_standard = StandardGaussian(self.h_vec, self.J)

        # Create MixedGaussian instances
        self.mg_functional = MixedGaussian(self.h_lf, self.J, self.logZ)
        self.mg_standard = MixedGaussian(self.h_vec, self.J, self.logZ)

    def test_standard_normalizing_constant(self):
        nc_qf = self.sg_functional.normalizing_constant()
        self.assertIsInstance(nc_qf, QuadraticForm)
        resolved_nc = nc_qf(self.x)
        expected_nc = self.sg_standard.normalizing_constant()
        self.assertTrue(torch.allclose(resolved_nc, expected_nc))

    def test_standard_call(self):
        y = torch.randn(self.dim)
        call_qf = self.sg_functional(y)
        self.assertIsInstance(call_qf, QuadraticForm)
        resolved_call = call_qf(self.x)
        expected_call = self.sg_standard(y)
        self.assertTrue(torch.allclose(resolved_call, expected_call))

    def test_standard_score(self):
        y = torch.randn(self.dim)
        score_lf = self.sg_functional.score(y)
        self.assertIsInstance(score_lf, LinearFunctional)
        resolved_score = score_lf(self.x)
        expected_score = self.sg_standard.score(y)
        self.assertTrue(torch.allclose(resolved_score, expected_score))

    def test_standard_log_prob(self):
        y = torch.randn(self.dim)
        log_prob_qf = self.sg_functional.log_prob(y)
        self.assertIsInstance(log_prob_qf, QuadraticForm)
        resolved_log_prob = log_prob_qf(self.x)
        expected_log_prob = self.sg_standard.log_prob(y)
        self.assertTrue(torch.allclose(resolved_log_prob, expected_log_prob))

    def test_standard_sample(self):
        eps = torch.randn(self.dim)
        sample_lf = self.sg_functional.sample(eps)
        self.assertIsInstance(sample_lf, LinearFunctional)
        resolved_sample = sample_lf(self.x)

        noise = self.sg_standard.get_noise(resolved_sample)

        noise_functional = self.sg_functional.get_noise(sample_lf)
        self.assertIsInstance(noise_functional, LinearFunctional)
        expected_noise = noise_functional(self.x)

        self.assertTrue(torch.allclose(noise, expected_noise))

    def test_mixed_to_std(self):
        sg_from_mg = self.mg_functional.to_std()
        resolved_mu = sg_from_mg.mu(self.x)
        expected_mu = self.mg_standard.to_std().mu
        self.assertTrue(torch.allclose(resolved_mu, expected_mu, atol=1e-5))

    def test_mixed_normalizing_constant(self):
        nc_qf = self.mg_functional.normalizing_constant()
        self.assertIsInstance(nc_qf, QuadraticForm)
        resolved_nc = nc_qf(self.x)
        expected_nc = self.mg_standard.normalizing_constant()
        self.assertTrue(torch.allclose(resolved_nc, expected_nc))

    def test_mixed_call(self):
        y = torch.randn(self.dim)
        call_qf = self.mg_functional(y)
        self.assertIsInstance(call_qf, QuadraticForm)
        resolved_call = call_qf(self.x)
        expected_call = self.mg_standard(y)
        self.assertTrue(torch.allclose(resolved_call, expected_call))

    def test_mixed_log_prob(self):
        y = torch.randn(self.dim)
        log_prob_qf = self.mg_functional.log_prob(y)
        self.assertIsInstance(log_prob_qf, QuadraticForm)
        resolved_log_prob = log_prob_qf(self.x)
        expected_log_prob = self.mg_standard.log_prob(y)
        self.assertTrue(torch.allclose(resolved_log_prob, expected_log_prob))

    def test_mixed_score(self):
        y = torch.randn(self.dim)
        score_lf = self.mg_functional.score(y)
        self.assertIsInstance(score_lf, LinearFunctional)
        resolved_score = score_lf(self.x)
        expected_score = self.mg_standard.score(y)
        self.assertTrue(torch.allclose(resolved_score, expected_score))

    def test_mixed_sample(self):
        eps = torch.randn(self.dim)
        sample_lf = self.mg_functional.sample(eps)
        self.assertIsInstance(sample_lf, LinearFunctional)
        resolved_sample = sample_lf(self.x)

        noise = self.mg_standard.get_noise(resolved_sample)

        expected_noise_functional = self.mg_functional.get_noise(sample_lf)
        self.assertIsInstance(expected_noise_functional, LinearFunctional)
        expected_noise = expected_noise_functional(self.x)
        self.assertTrue(torch.allclose(noise, expected_noise, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
