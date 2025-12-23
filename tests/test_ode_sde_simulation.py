import torch
import pytest
import math

from diffusion_helper.src.sde.ode_sde_simulation import (
  ODESolverParams,
  SDESolverParams,
  ode_solve,
  sde_sample,
)
from diffusion_helper.src.series.series import TimeSeries


class TestODESolverParams:
  """Test ODE solver parameters"""

  def test_default_initialization(self):
    """Test default parameter values"""
    params = ODESolverParams()

    assert params.rtol == 1e-6
    assert params.atol == 1e-6
    assert params.solver == 'dopri5'
    assert params.stepsize_controller == 'integral'
    assert params.max_steps is None

  def test_custom_initialization(self):
    """Test custom parameter values"""
    params = ODESolverParams(
      rtol=1e-8,
      atol=1e-8,
      solver='euler',
      stepsize_controller='fixed',
      max_steps=1000,
    )

    assert params.rtol == 1e-8
    assert params.atol == 1e-8
    assert params.solver == 'euler'
    assert params.stepsize_controller == 'fixed'
    assert params.max_steps == 1000

  def test_to_dict(self):
    """Test conversion to dictionary"""
    params = ODESolverParams()
    param_dict = params.to_dict()

    expected_keys = {'rtol', 'atol', 'solver', 'stepsize_controller', 'max_steps'}
    assert set(param_dict.keys()) == expected_keys

  def test_using_constant_step_size(self):
    """Test constant step size detection"""
    # Integral controller - not constant
    params_integral = ODESolverParams(stepsize_controller='integral')
    assert not params_integral.using_constant_step_size()

    # PID controller - not constant
    params_pid = ODESolverParams(stepsize_controller='pid')
    assert not params_pid.using_constant_step_size()

    # Fixed controller - constant
    params_fixed = ODESolverParams(stepsize_controller='fixed')
    assert params_fixed.using_constant_step_size()

  def test_get_step_method_dopri5(self):
    """Test dopri5 solver instantiation"""
    import torchode as to

    params = ODESolverParams(solver='dopri5')
    term = to.ODETerm(lambda t, y: -y)
    step_method = params.get_step_method(term)
    assert step_method.__class__.__name__ == 'Dopri5'

  def test_get_step_method_tsit5(self):
    """Test tsit5 solver instantiation"""
    import torchode as to

    params = ODESolverParams(solver='tsit5')
    term = to.ODETerm(lambda t, y: -y)
    step_method = params.get_step_method(term)
    assert step_method.__class__.__name__ == 'Tsit5'

  def test_get_step_method_heun(self):
    """Test heun solver instantiation"""
    import torchode as to

    params = ODESolverParams(solver='heun')
    term = to.ODETerm(lambda t, y: -y)
    step_method = params.get_step_method(term)
    assert step_method.__class__.__name__ == 'Heun'

  def test_get_step_method_euler(self):
    """Test euler solver instantiation"""
    import torchode as to

    params = ODESolverParams(solver='euler')
    term = to.ODETerm(lambda t, y: -y)
    step_method = params.get_step_method(term)
    assert step_method.__class__.__name__ == 'Euler'

  def test_get_step_method_invalid(self):
    """Test invalid solver raises error"""
    import torchode as to

    params = ODESolverParams(solver='invalid_solver')
    term = to.ODETerm(lambda t, y: -y)
    with pytest.raises(ValueError, match="Unknown solver"):
      params.get_step_method(term)

  def test_get_stepsize_controller_integral(self):
    """Test integral controller instantiation"""
    import torchode as to

    params = ODESolverParams(stepsize_controller='integral')
    term = to.ODETerm(lambda t, y: -y)
    controller = params.get_stepsize_controller(term)
    assert controller.__class__.__name__ == 'IntegralController'

  def test_get_stepsize_controller_pid(self):
    """Test PID controller instantiation"""
    import torchode as to

    params = ODESolverParams(stepsize_controller='pid')
    term = to.ODETerm(lambda t, y: -y)
    controller = params.get_stepsize_controller(term)
    assert controller.__class__.__name__ == 'PIDController'

  def test_get_stepsize_controller_fixed(self):
    """Test fixed controller instantiation"""
    import torchode as to

    params = ODESolverParams(stepsize_controller='fixed')
    term = to.ODETerm(lambda t, y: -y)
    controller = params.get_stepsize_controller(term)
    assert controller.__class__.__name__ == 'FixedStepController'


class TestSDESolverParams:
  """Test SDE solver parameters"""

  def test_default_initialization(self):
    """Test default parameter values"""
    params = SDESolverParams()

    assert params.dt is None
    assert params.max_steps == 1000

  def test_custom_initialization(self):
    """Test custom parameter values"""
    params = SDESolverParams(dt=0.01, max_steps=500)

    assert params.dt == 0.01
    assert params.max_steps == 500

  def test_to_dict(self):
    """Test conversion to dictionary"""
    params = SDESolverParams()
    param_dict = params.to_dict()

    expected_keys = {'dt', 'max_steps'}
    assert set(param_dict.keys()) == expected_keys


class TestODESolve:
  """Test ODE solving functionality"""

  def test_ode_solve_exponential_decay(self):
    """Test ODE solving with exponential decay: dx/dt = -x"""
    def dynamics(t, x):
      return -x

    x0 = torch.tensor([1.0, 2.0])
    save_times = torch.tensor([0.0, 0.5, 1.0])

    result = ode_solve(dynamics, x0, save_times)

    assert isinstance(result, TimeSeries)
    assert result.times.shape == save_times.shape
    assert result.vals.shape == (len(save_times), len(x0))

    # Check analytical solution at t=1: x(t) = x0 * exp(-t)
    expected_at_1 = x0 * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(result.vals[-1], expected_at_1, rtol=1e-3)

  def test_ode_solve_constant(self):
    """Test ODE solving with constant solution: dx/dt = 0"""
    def dynamics(t, x):
      return torch.zeros_like(x)

    x0 = torch.tensor([3.14, 2.71])
    save_times = torch.tensor([0.0, 0.5, 1.0])

    result = ode_solve(dynamics, x0, save_times)

    # Solution should remain constant
    for i in range(len(save_times)):
      assert torch.allclose(result.vals[i], x0, rtol=1e-5)

  def test_ode_solve_linear_growth(self):
    """Test ODE solving with linear growth: dx/dt = 1"""
    def dynamics(t, x):
      return torch.ones_like(x)

    x0 = torch.tensor([0.0])
    save_times = torch.tensor([0.0, 0.5, 1.0])

    result = ode_solve(dynamics, x0, save_times)

    # Solution should be x(t) = t
    expected = save_times.unsqueeze(-1)
    assert torch.allclose(result.vals, expected, rtol=1e-3)

  def test_ode_solve_returns_timeseries(self):
    """Test that ode_solve returns a TimeSeries object"""
    def dynamics(t, x):
      return -x

    x0 = torch.tensor([1.0])
    save_times = torch.tensor([0.0, 1.0])

    result = ode_solve(dynamics, x0, save_times)

    assert isinstance(result, TimeSeries)
    assert hasattr(result, 'times')
    assert hasattr(result, 'vals')

  def test_ode_solve_with_custom_params(self):
    """Test ODE solving with custom solver parameters"""
    def dynamics(t, x):
      return -x

    x0 = torch.tensor([1.0])
    save_times = torch.tensor([0.0, 1.0])
    params = ODESolverParams(solver='tsit5', rtol=1e-8, atol=1e-8)

    result = ode_solve(dynamics, x0, save_times, params)

    expected_at_1 = x0 * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(result.vals[-1], expected_at_1, rtol=1e-5)


class TestODESolveSolversAndControllers:
  """Test each ODE solver and step size controller end to end"""

  @pytest.mark.parametrize("solver", ["dopri5", "tsit5", "heun", "euler"])
  @pytest.mark.parametrize("stepsize_controller", ["integral", "pid", "fixed"])
  def test_ode_solve_across_solvers_and_controllers(self, solver, stepsize_controller):
    def dynamics(t, x):
      return -x

    x0 = torch.tensor([1.0], dtype=torch.float64)
    save_times = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)

    params = ODESolverParams(
      solver=solver,
      stepsize_controller=stepsize_controller,
      rtol=1e-7,
      atol=1e-9,
      max_steps=10_000,
    )
    result = ode_solve(dynamics, x0, save_times, params)

    assert isinstance(result, TimeSeries)
    assert result.times.shape == save_times.shape
    assert result.vals.shape == (len(save_times), len(x0))

    expected = x0 * torch.exp(-save_times).unsqueeze(-1)

    # Fixed step methods with euler can be noticeably less accurate for coarse grids
    # This stays tight enough to catch wiring bugs while remaining stable
    if solver == "euler" and stepsize_controller == "fixed":
      rtol = 5e-3
      atol = 5e-4
    else:
      rtol = 2e-4
      atol = 2e-6

    assert torch.allclose(result.vals, expected, rtol=rtol, atol=atol)

  def test_pid_matches_integral_on_simple_problem(self):
    def dynamics(t, x):
      return -x

    x0 = torch.tensor([1.0], dtype=torch.float64)
    save_times = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)

    base_kwargs = dict(
      solver="dopri5",
      rtol=1e-8,
      atol=1e-10,
      max_steps=10_000,
    )
    integral = ode_solve(
      dynamics,
      x0,
      save_times,
      ODESolverParams(stepsize_controller="integral", **base_kwargs),
    )
    pid = ode_solve(
      dynamics,
      x0,
      save_times,
      ODESolverParams(stepsize_controller="pid", **base_kwargs),
    )

    assert torch.allclose(pid.vals, integral.vals, rtol=1e-6, atol=1e-8)

  def test_ode_closed_form_affine_linear(self):
    """Test closed form solution for dx dt = a x + b"""
    a = torch.tensor(-2.0, dtype=torch.float64)
    b = torch.tensor(0.3, dtype=torch.float64)

    def dynamics(t, x):
      return a * x + b

    x0 = torch.tensor([1.2, -0.7, 0.05], dtype=torch.float64)
    save_times = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)

    params = ODESolverParams(
      solver="dopri5",
      stepsize_controller="integral",
      rtol=1e-9,
      atol=1e-11,
      max_steps=10_000,
    )
    result = ode_solve(dynamics, x0, save_times, params)

    # Closed form
    # x(t) = exp(a t) x0 + (exp(a t) - 1) b a^{-1}
    exp_at = torch.exp(a * save_times).unsqueeze(-1)
    expected = exp_at * x0 + (exp_at - 1.0) * (b / a)

    assert torch.allclose(result.vals, expected, rtol=5e-7, atol=5e-9)


class TestSDESample:
  """Test SDE sampling functionality"""

  def test_sde_sample_brownian_motion(self):
    """Test basic Brownian motion sampling: dx = sigma * dW"""
    sigma = 0.5

    def drift(t, x):
      return torch.zeros_like(x)

    def diffusion(t, x):
      return torch.ones_like(x) * sigma

    x0 = torch.tensor([0.0])
    save_times = torch.tensor([0.0, 0.5, 1.0])
    params = SDESolverParams(max_steps=1000)

    generator = torch.Generator().manual_seed(42)
    result = sde_sample(drift, diffusion, x0, save_times, params, generator)

    assert isinstance(result, TimeSeries)
    assert result.times.shape == save_times.shape
    assert result.vals.shape == (len(save_times), 1)

    # First time point should be the initial condition
    assert torch.allclose(result.vals[0], x0)

  def test_sde_sample_deterministic(self):
    """Test SDE sampling with zero diffusion (deterministic)"""
    def drift(t, x):
      return -0.5 * x  # OU process drift

    def diffusion(t, x):
      return torch.zeros_like(x)  # No noise

    x0 = torch.tensor([1.0])
    save_times = torch.tensor([0.0, 1.0])
    params = SDESolverParams(max_steps=1000)

    result = sde_sample(drift, diffusion, x0, save_times, params)

    # Should be deterministic: x(t) = x0 * exp(-0.5 * t)
    expected_final = x0 * torch.exp(torch.tensor(-0.5))
    assert torch.allclose(result.vals[-1], expected_final, rtol=1e-2)

  def test_sde_sample_reproducibility(self):
    """Test that same seed gives same result"""
    sigma = 1.0

    def drift(t, x):
      return torch.zeros_like(x)

    def diffusion(t, x):
      return torch.ones_like(x) * sigma

    x0 = torch.tensor([0.0])
    save_times = torch.tensor([0.0, 1.0])
    params = SDESolverParams(max_steps=100)

    generator1 = torch.Generator().manual_seed(123)
    result1 = sde_sample(drift, diffusion, x0, save_times, params, generator1)

    generator2 = torch.Generator().manual_seed(123)
    result2 = sde_sample(drift, diffusion, x0, save_times, params, generator2)

    assert torch.allclose(result1.vals, result2.vals)

  def test_sde_sample_different_seeds(self):
    """Test that different seeds give different results"""
    sigma = 1.0

    def drift(t, x):
      return torch.zeros_like(x)

    def diffusion(t, x):
      return torch.ones_like(x) * sigma

    x0 = torch.tensor([0.0])
    save_times = torch.tensor([0.0, 1.0])
    params = SDESolverParams(max_steps=100)

    generator1 = torch.Generator().manual_seed(1)
    result1 = sde_sample(drift, diffusion, x0, save_times, params, generator1)

    generator2 = torch.Generator().manual_seed(2)
    result2 = sde_sample(drift, diffusion, x0, save_times, params, generator2)

    # Results should be different (with high probability)
    assert not torch.allclose(result1.vals[1:], result2.vals[1:], atol=1e-6)

  def test_sde_sample_multidimensional(self):
    """Test SDE sampling with multiple dimensions"""
    sigma = 0.1

    def drift(t, x):
      return -x

    def diffusion(t, x):
      return torch.ones_like(x) * sigma

    x0 = torch.tensor([1.0, 2.0, 3.0])
    save_times = torch.tensor([0.0, 0.5, 1.0])
    params = SDESolverParams(max_steps=500)

    generator = torch.Generator().manual_seed(42)
    result = sde_sample(drift, diffusion, x0, save_times, params, generator)

    assert result.vals.shape == (len(save_times), len(x0))
    assert torch.allclose(result.vals[0], x0)


class TestConvergence:
  """Test numerical convergence"""

  def test_ode_exponential_decay_convergence(self):
    """Test that ODE solver converges to analytical solution"""
    def dynamics(t, x):
      return -x

    x0 = torch.tensor([1.0])

    # Test at multiple time points
    save_times = torch.linspace(0, 2, 21)
    result = ode_solve(dynamics, x0, save_times)

    # Analytical solution: x(t) = x0 * exp(-t)
    expected = x0 * torch.exp(-save_times).unsqueeze(-1)

    # Should be very close
    assert torch.allclose(result.vals, expected, rtol=1e-4)

  def test_brownian_motion_variance(self):
    """Test that Brownian motion has correct variance scaling"""
    sigma = 1.0
    T = 1.0
    n_samples = 500

    def drift(t, x):
      return torch.zeros_like(x)

    def diffusion(t, x):
      return torch.ones_like(x) * sigma

    x0 = torch.tensor([0.0])
    save_times = torch.tensor([0.0, T])
    params = SDESolverParams(max_steps=100)

    # Generate multiple samples
    final_values = []
    for i in range(n_samples):
      generator = torch.Generator().manual_seed(i)
      result = sde_sample(drift, diffusion, x0, save_times, params, generator)
      final_values.append(result.vals[-1, 0].item())

    final_values = torch.tensor(final_values)

    # Variance should be approximately sigma^2 * T
    empirical_variance = torch.var(final_values).item()
    expected_variance = sigma**2 * T

    # Allow some tolerance for Monte Carlo error
    assert abs(empirical_variance - expected_variance) < 0.2


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
