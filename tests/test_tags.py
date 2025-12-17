import torch
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from diffusion_helper.src.matrix.tags import Tags  # type: ignore


def test_init_casts_to_bool_tensors():
  t = Tags(is_nonzero=False, is_inf=True)
  assert isinstance(t.is_nonzero, torch.Tensor)
  assert isinstance(t.is_inf, torch.Tensor)
  assert t.is_nonzero.dtype == torch.bool
  assert t.is_inf.dtype == torch.bool
  assert t.is_nonzero.item() is False
  assert t.is_inf.item() is True


def test_is_zero_property():
  t = Tags(is_nonzero=False, is_inf=False)
  assert t.is_zero.item() is True
  t2 = Tags(is_nonzero=True, is_inf=False)
  assert t2.is_zero.item() is False


def test_add_update_logic():
  a = Tags(is_nonzero=False, is_inf=False)
  b = Tags(is_nonzero=True, is_inf=True)
  c = a.add_update(b)
  assert c.is_nonzero.item() is True
  assert c.is_inf.item() is True


def test_mat_mul_update_logic():
  a = Tags(is_nonzero=True, is_inf=False)
  b = Tags(is_nonzero=False, is_inf=True)
  c = a.mat_mul_update(b)
  assert c.is_nonzero.item() is False
  assert c.is_inf.item() is True


def test_solve_update_logic_zero_case_inf():
  # When A is zero and B is nonzero -> inf
  a = Tags(is_nonzero=False, is_inf=False)
  b = Tags(is_nonzero=True, is_inf=False)
  c = a.solve_update(b)
  assert c.is_inf.item() is True


def test_inverse_update_logic():
  a = Tags(is_nonzero=True, is_inf=False)
  inv = a.inverse_update()
  assert inv.is_nonzero.item() is True
  assert inv.is_inf.item() is False


def test_cholesky_update_no_change():
  a = Tags(is_nonzero=True, is_inf=False)
  chol = a.cholesky_update()
  assert chol.is_nonzero.item() is True
  assert chol.is_inf.item() is False


def test_exp_update_sets_nonzero_true_preserves_inf():
  a = Tags(is_nonzero=False, is_inf=True)
  e = a.exp_update()
  assert e.is_nonzero.item() is True
  assert e.is_inf.item() is True


