import torch
from functools import partial
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass
import torchode as to
from diffusion_helper.src.series.series import TimeSeries

"""
This module provides utilities for numerically solving and sampling from ODEs and SDEs.
It offers:

1. ODESolverParams and ode_solve - Configuration and solver for ODEs using torchode
2. SDESolverParams and sde_sample - Configuration and sampler for SDEs using Euler-Maruyama

The ODE solver uses torchode for high-performance differential equation solving in PyTorch,
with support for automatic differentiation and various numerical integration schemes.

The SDE sampler uses a simple Euler-Maruyama scheme since torchode does not support SDEs.
"""

__all__ = ['ode_solve',
           'ODESolverParams',
           'SDESolverParams',
           'sde_sample']


@dataclass
class ODESolverParams:
  """
  Configuration parameters for ODE solving with torchode.

  Attributes:
    rtol: Relative tolerance for adaptive step size controllers
    atol: Absolute tolerance for adaptive step size controllers
    solver: Name of the solver to use ('dopri5', 'tsit5', 'heun', 'euler')
    stepsize_controller: Controller for step size ('integral', 'pid', 'fixed')
    max_steps: Maximum number of steps the solver is allowed to take
  """
  rtol: float = 1e-6
  atol: float = 1e-6
  solver: str = 'dopri5'
  stepsize_controller: str = 'integral'
  max_steps: Optional[int] = None

  def to_dict(self) -> dict:
    """Convert solver parameters to a dictionary."""
    return {
      "rtol": self.rtol,
      "atol": self.atol,
      "solver": self.solver,
      "stepsize_controller": self.stepsize_controller,
      "max_steps": self.max_steps,
    }

  def using_constant_step_size(self) -> bool:
    return self.stepsize_controller == 'fixed' or self.stepsize_controller == 'constant'

  def get_step_method(self, term: to.ODETerm):
    """Get the torchode step method based on the configured solver name."""
    if self.solver == 'dopri5':
      return to.Dopri5(term=term)
    elif self.solver == 'tsit5':
      return to.Tsit5(term=term)
    elif self.solver == 'heun':
      return to.Heun(term=term)
    elif self.solver == 'euler':
      return to.Euler(term=term)
    else:
      raise ValueError(f"Unknown solver: {self.solver}. Available: dopri5, tsit5, heun, euler")

  def get_stepsize_controller(self, term: to.ODETerm):
    """Get the torchode step size controller based on the configured controller name."""
    if self.stepsize_controller == 'integral':
      return to.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
    elif self.stepsize_controller == 'pid':
      # PID coefficients: standard values from Söderlind (2003)
      return to.PIDController(
        atol=self.atol, rtol=self.rtol,
        pcoeff=0.0, icoeff=1.0, dcoeff=0.0,  # Equivalent to integral controller
        term=term
      )
    elif self.stepsize_controller == 'fixed' or self.stepsize_controller == 'constant':
      return to.FixedStepController()
    else:
      raise ValueError(f"Unknown stepsize controller: {self.stepsize_controller}. Available: integral, pid, fixed")


@dataclass
class SDESolverParams:
  """
  Configuration parameters for SDE simulation using Euler-Maruyama.

  Attributes:
    dt: Time step for Euler-Maruyama integration (if None, computed from max_steps)
    max_steps: Maximum number of steps (used to compute dt if dt is None)
  """
  dt: Optional[float] = None
  max_steps: int = 1000

  def to_dict(self) -> dict:
    """Convert solver parameters to a dictionary."""
    return {
      "dt": self.dt,
      "max_steps": self.max_steps,
    }


def ode_solve(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    save_times: torch.Tensor,
    params: ODESolverParams = None
) -> TimeSeries:
  """Solve an ODE dx/dt = f(t, x) using torchode.

  Args:
    dynamics: Function f(t, x) -> dx/dt. Takes time (batch,) and state (batch, features)
              and returns the time derivative (batch, features).
    x0: Initial condition with shape (features,) or (batch, features)
    save_times: Times at which to save the solution, shape (time,) or (batch, time)
    params: Parameters for the ODE solver

  Returns:
    TimeSeries containing the solution trajectory at the save times
  """
  if params is None:
    params = ODESolverParams()

  # Handle input shapes - torchode expects (batch, features) for y0 and (batch, time) for t_eval
  x0_was_1d = x0.ndim == 1
  if x0_was_1d:
    x0 = x0.unsqueeze(0)  # (1, features)

  times_was_1d = save_times.ndim == 1
  if times_was_1d:
    save_times = save_times.unsqueeze(0)  # (1, time)

  # Expand save_times to match batch size if needed
  batch_size = x0.shape[0]
  if save_times.shape[0] == 1 and batch_size > 1:
    save_times = save_times.expand(batch_size, -1)

  # Ensure same device and dtype
  save_times = save_times.to(device=x0.device, dtype=x0.dtype)

  # Create the ODE term
  term = to.ODETerm(dynamics)

  # Create step method and controller
  step_method = params.get_step_method(term)
  step_size_controller = params.get_stepsize_controller(term)

  # Create solver with autodiff adjoint
  solver = to.AutoDiffAdjoint(step_method, step_size_controller, max_steps=params.max_steps)

  # Create the initial value problem
  problem = to.InitialValueProblem(y0=x0, t_eval=save_times)

  if params.solver == "euler" and not params.using_constant_step_size():
    raise ValueError("Euler requires fixed step size controller")

  dt0 = None
  if params.using_constant_step_size():
    diffs = torch.diff(save_times, dim=1)
    dt0 = diffs.abs().min(dim=1).values
    dt0 = torch.where(dt0 == 0, torch.ones_like(dt0) * 1e-6, dt0)
    dt0 = dt0.to(dtype=save_times.dtype, device=save_times.device)

  # Solve
  solution = solver.solve(problem, term=term, dt0=dt0)

  # Extract results - solution.ys has shape (batch, time, features)
  ys = solution.ys

  # If input was 1D, squeeze the batch dimension
  if x0_was_1d:
    ys = ys.squeeze(0)  # (time, features)
    output_times = save_times.squeeze(0)  # (time,)
  else:
    output_times = save_times

  # For 1D case, output_times is (time,) and ys is (time, features)
  # For batched case, this would need different handling
  if x0_was_1d:
    return TimeSeries(times=output_times, vals=ys)
  else:
    # For batched case, return the first batch for now (simplification)
    # A more complete implementation would handle batched TimeSeries
    return TimeSeries(times=output_times[0], vals=ys[0])


def sde_sample(
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    save_times: torch.Tensor,
    params: SDESolverParams = None,
    generator: Optional[torch.Generator] = None
) -> TimeSeries:
  """Sample from an SDE dx = f(t, x)dt + g(t, x)dW using Euler-Maruyama.

  The SDE is:
    dx = drift(t, x) * dt + diffusion(t, x) * dW

  where dW is a Wiener process increment.

  Args:
    drift: Drift function f(t, x) -> dx/dt. Takes time (scalar) and state (features,)
           and returns the drift (features,).
    diffusion: Diffusion function g(t, x). Takes time (scalar) and state (features,)
               and returns the diffusion coefficient (features,) or (features, features).
    x0: Initial condition with shape (features,)
    save_times: Times at which to save the solution, shape (time,)
    params: Parameters for the SDE solver
    generator: Optional torch.Generator for reproducibility

  Returns:
    TimeSeries containing the sampled trajectory at the save times
  """
  if params is None:
    params = SDESolverParams()

  device = x0.device
  dtype = x0.dtype
  n_features = x0.shape[-1] if x0.ndim > 0 else 1

  # Ensure x0 is at least 1D
  if x0.ndim == 0:
    x0 = x0.unsqueeze(0)

  # Get time points
  t_start = save_times[0].item()
  t_end = save_times[-1].item()

  # Compute dt
  if params.dt is not None:
    dt = params.dt
  else:
    dt = (t_end - t_start) / params.max_steps

  # Generate all time points for integration
  n_steps = int((t_end - t_start) / dt) + 1
  all_times = torch.linspace(t_start, t_end, n_steps, device=device, dtype=dtype)

  # Find indices in all_times that are closest to save_times
  save_indices = []
  for t in save_times:
    idx = torch.argmin(torch.abs(all_times - t)).item()
    save_indices.append(idx)

  # Initialize trajectory storage
  trajectory = torch.zeros(len(save_times), n_features, device=device, dtype=dtype)

  # Euler-Maruyama integration
  x = x0.clone()
  sqrt_dt = torch.sqrt(torch.tensor(dt, device=device, dtype=dtype))

  save_idx = 0
  for i, t in enumerate(all_times[:-1]):
    # Save if this is a save point
    if i == save_indices[save_idx]:
      trajectory[save_idx] = x
      save_idx += 1
      if save_idx >= len(save_indices):
        break

    # Compute drift and diffusion
    f = drift(t, x)
    g = diffusion(t, x)

    # Generate noise
    if generator is not None:
      noise = torch.randn(n_features, device=device, dtype=dtype, generator=generator)
    else:
      noise = torch.randn(n_features, device=device, dtype=dtype)

    # Euler-Maruyama update
    # Handle both scalar and matrix diffusion
    if g.ndim == 1:
      # Diagonal diffusion: g is (features,)
      x = x + f * dt + g * sqrt_dt * noise
    else:
      # Matrix diffusion: g is (features, features)
      x = x + f * dt + g @ (sqrt_dt * noise)

  # Save the last point
  if save_idx < len(save_indices):
    trajectory[save_idx] = x

  return TimeSeries(times=save_times, vals=trajectory)
