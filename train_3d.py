"""
train_3d.py — Stratos PINN v4
==============================
3D Transient Heat Conduction through a Hollow Elliptical Cone Shell TPS sample.
Implemented with NVIDIA PhysicsNeMo (physicsnemo.sym) for Google Colab T4 GPU.

Physics
-------
PDE:   dT/dt = alpha * (d²T/dx² + d²T/dy² + d²T/dz²)
alpha  = 5e-6 m²/s  (thermal diffusivity of TPS material)
t      in [0, 60] s

Geometry
--------
Hollow Elliptical Cone Shell:
  Base ellipse (z=0): semi-major A_BASE=0.06 m (x), semi-minor B_BASE=0.04 m (y)
  Apex at z = H_CONE = 0.10 m (full cone, not truncated frustum)
  Wall thickness T_WALL = 0.005 m (constant semi-axis reduction)
  Inner ellipse at base: A_INNER=0.055 m, B_INNER=0.035 m

  At height z, outer semi-axes scale as:
      a_out(z) = A_BASE * (H_CONE - z) / H_CONE
      b_out(z) = B_BASE * (H_CONE - z) / H_CONE

  Shell material occupies: F_outer ≤ 0 AND F_inner ≥ 0 AND z ∈ [0, H_CONE]

Boundary Conditions
-------------------
  Outer cone wall, t>0  : T = 4000 K           [Dirichlet — plasma temperature]
  Inner cone wall, t≥0  : dT/dn = 0            [Neumann   — adiabatic hollow cavity]
  Bottom lip  (z=0), t≥0: dT/dn = 0            [Neumann   — adiabatic base edge]
  t=0,  everywhere      : T = 300 K            [Initial Condition]

Key Design Decisions
--------------------
  Normalization : All inputs (x,y,z,t) and output (T) are min-max scaled
                  to [0,1].  Because X_SCALE = 2*A_BASE ≠ Y_SCALE = 2*B_BASE,
                  the normalized PDE uses three distinct alpha coefficients
                  (one per spatial axis) derived from the chain rule.

  Architecture  : FourierFeatureNet — Fourier-feature encoding + SiLU FC backbone.

  Geometry      : Custom HollowEllipticalConeShell class with duck-typed
                  sample_interior / sample_boundary interface compatible with
                  physicsnemo PointwiseInteriorConstraint and
                  PointwiseBoundaryConstraint.

Environment
-----------
  Requires the Colab setup cell to have been run first:
    - nvidia-modulus installed with --no-deps
    - physicsnemo symlinked from the cloned modulus-sym repo
"""

import os
import sys
import glob
import numpy as np
import torch

# ---------------------------------------------------------------------------
# numpy compatibility shim
# ---------------------------------------------------------------------------
def _noop(*args, **kwargs):
    pass

try:
    _marray = np._core.multiarray
except AttributeError:
    _marray = np.core.multiarray

if not hasattr(_marray, "set_legacy_print_mode"):
    _marray.set_legacy_print_mode = _noop

if int(np.__version__.split(".")[0]) < 2:
    try:
        if not hasattr(np.core.multiarray, "set_legacy_print_mode"):
            np.core.multiarray.set_legacy_print_mode = _noop
    except AttributeError:
        pass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, open_dict

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint,
    PointwiseBoundaryConstraint,
)
from physicsnemo.sym.node import Node
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.models.arch import Arch

from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.hydra.training import DefaultTraining, DefaultStopCriterion
from physicsnemo.sym.hydra.optimizer import AdamConf
from physicsnemo.sym.hydra.scheduler import ExponentialLRConf

SimConfig = PhysicsNeMoConfig

from sympy import Symbol, Function, diff


# =============================================================================
# GEOMETRY: HOLLOW ELLIPTICAL CONE SHELL
# =============================================================================
# physicsnemo uses duck-typed geometry objects.  The constraints call
# geometry.sample_interior() and geometry.sample_boundary(); no base-class
# inheritance is required.
#
# IMPORTANT — Parameterization contract:
#   PhysicsNeMo passes a Parameterization object (or a plain dict) to both
#   sample_interior and sample_boundary via the ``parameterization`` keyword.
#   The geometry is responsible for sampling those parameters (e.g. t) and
#   returning them inside the result dict.  If t is absent the constraint
#   graph will raise a silent KeyError during training setup.
#
#   _sample_params() below handles this regardless of whether the caller
#   passes a Parameterization object, a plain {Symbol: (lo,hi)} dict, or None.


def _sample_params(n: int, parameterization=None) -> dict:
    """
    Sample parameterization variables (e.g. t) and return them as a dict
    suitable for merging into a geometry result dict.

    Handles:
      - None                      → empty dict
      - Parameterization object   → call .sample(n) (physicsnemo native)
      - plain dict                → {Symbol/str: (lo, hi) or scalar}
    """
    if parameterization is None:
        return {}

    # PhysicsNeMo Parameterization object with a .sample() method
    if hasattr(parameterization, "sample"):
        try:
            samples = parameterization.sample(n)
            # .sample() returns {str: ndarray}, ensure shape (n, 1)
            return {
                k: (v if v.ndim == 2 else v[:, None]).astype(np.float32)
                for k, v in samples.items()
            }
        except Exception:
            pass  # fall through to dict-style handling

    # Plain dict: {Symbol("t"): (lo, hi)}  or  {Symbol("t"): scalar}
    out: dict = {}
    for key, val in parameterization.items():
        name = key.name if hasattr(key, "name") else str(key)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            lo, hi = float(val[0]), float(val[1])
            out[name] = np.random.uniform(lo, hi, (n, 1)).astype(np.float32)
        else:
            out[name] = np.full((n, 1), float(val), dtype=np.float32)
    return out

class HollowEllipticalConeShell:
    """
    Hollow elliptical cone shell geometry.

    Outer cone at height z: semi-axes
        a_out(z) = A_BASE * (H_CONE - z) / H_CONE
        b_out(z) = B_BASE * (H_CONE - z) / H_CONE
    Inner cone (cavity) at height z: semi-axes reduced by T_WALL on each axis.

    Shell material: inside outer cone AND outside inner cone AND z in [Z_BOTTOM, Z_TOP].
    Apex at z = H_CONE = Z_TOP (full cone, not truncated frustum).
    """

    dims: int = 3   # spatial dimensionality — required by some physicsnemo internals

    def __init__(
        self,
        A_BASE:   float,
        B_BASE:   float,
        H_CONE:   float,
        T_WALL:   float,
        Z_BOTTOM: float,
        Z_TOP:    float,
    ) -> None:
        self.A_BASE   = float(A_BASE)
        self.B_BASE   = float(B_BASE)
        self.H_CONE   = float(H_CONE)
        self.T_WALL   = float(T_WALL)
        self.Z_BOTTOM = float(Z_BOTTOM)
        self.Z_TOP    = float(Z_TOP)
        self.A_INNER  = float(A_BASE - T_WALL)
        self.B_INNER  = float(B_BASE - T_WALL)
        self._h_span  = float(H_CONE - Z_BOTTOM)  # denominator in scale factor

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _scale(self, z: np.ndarray) -> np.ndarray:
        """Taper factor: 1.0 at Z_BOTTOM, 0.0 at H_CONE."""
        return (self.H_CONE - z) / self._h_span

    def _in_shell(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Boolean mask: True if (x,y,z) is inside the shell material."""
        s  = self._scale(z)
        ss = np.maximum(s, 1e-9)               # guard against apex singularity
        a_out = self.A_BASE  * ss
        b_out = self.B_BASE  * ss
        a_in  = self.A_INNER * ss
        b_in  = self.B_INNER * ss
        outer = (x**2 / a_out**2 + y**2 / b_out**2) <= 1.0
        inner = (x**2 / a_in**2  + y**2 / b_in**2)  >= 1.0
        z_ok  = (z >= self.Z_BOTTOM) & (z <= self.Z_TOP)
        return outer & inner & z_ok

    # ------------------------------------------------------------------
    # Normal computation
    # ------------------------------------------------------------------

    def _outer_normal(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple:
        """
        Outward unit normal on the outer cone surface.
        Direction = +∇F_outer / |∇F_outer|  (points away from cone axis, into plasma).
        """
        s  = self._scale(z)
        ss = np.maximum(s, 1e-9)
        a  = self.A_BASE * ss
        b  = self.B_BASE * ss
        # da/dz = -A_BASE / h_span  (semi-axis shrinks as z increases)
        da = -self.A_BASE / self._h_span
        db = -self.B_BASE / self._h_span

        nx = 2.0 * x / a**2
        ny = 2.0 * y / b**2
        nz = -2.0 * x**2 * da / a**3 - 2.0 * y**2 * db / b**3

        mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        return (nx / mag).astype(np.float32), (ny / mag).astype(np.float32), (nz / mag).astype(np.float32)

    def _inner_normal(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple:
        """
        Outward unit normal on the inner cone surface.
        Outward from the SHELL on the inner wall = points INTO the hollow void
        = −∇F_inner / |∇F_inner|.
        """
        s  = self._scale(z)
        ss = np.maximum(s, 1e-9)
        a  = self.A_INNER * ss
        b  = self.B_INNER * ss
        da = -self.A_INNER / self._h_span
        db = -self.B_INNER / self._h_span

        # Negate gradient of F_inner to point toward void
        nx = -(2.0 * x / a**2)
        ny = -(2.0 * y / b**2)
        nz = -(-2.0 * x**2 * da / a**3 - 2.0 * y**2 * db / b**3)

        mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        return (nx / mag).astype(np.float32), (ny / mag).astype(np.float32), (nz / mag).astype(np.float32)

    # ------------------------------------------------------------------
    # Surface samplers (called by sample_boundary and wrapper classes)
    # ------------------------------------------------------------------

    def _sample_outer_wall(self, n: int) -> dict:
        """
        Parametric sampling on the outer cone lateral surface.
        x = a_out(z)*cos(θ),  y = b_out(z)*sin(θ),  z ~ Uniform[Z_BOTTOM, Z_TOP].
        """
        z     = np.random.uniform(self.Z_BOTTOM, self.Z_TOP, n).astype(np.float32)
        theta = np.random.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        s     = self._scale(z)
        x     = (self.A_BASE * s * np.cos(theta)).astype(np.float32)
        y     = (self.B_BASE * s * np.sin(theta)).astype(np.float32)
        nx, ny, nz = self._outer_normal(x, y, z)
        return {
            "x": x[:, None], "y": y[:, None], "z": z[:, None],
            "normal_x": nx[:, None], "normal_y": ny[:, None], "normal_z": nz[:, None],
            "area": np.ones((n, 1), dtype=np.float32),
            "sdf":  np.zeros((n, 1), dtype=np.float32),
        }

    def _sample_inner_wall(self, n: int) -> dict:
        """Parametric sampling on the inner cone lateral surface (cavity wall)."""
        z     = np.random.uniform(self.Z_BOTTOM, self.Z_TOP, n).astype(np.float32)
        theta = np.random.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        s     = self._scale(z)
        x     = (self.A_INNER * s * np.cos(theta)).astype(np.float32)
        y     = (self.B_INNER * s * np.sin(theta)).astype(np.float32)
        nx, ny, nz = self._inner_normal(x, y, z)
        return {
            "x": x[:, None], "y": y[:, None], "z": z[:, None],
            "normal_x": nx[:, None], "normal_y": ny[:, None], "normal_z": nz[:, None],
            "area": np.ones((n, 1), dtype=np.float32),
            "sdf":  np.zeros((n, 1), dtype=np.float32),
        }

    def _sample_bottom_lip(self, n: int) -> dict:
        """
        Rejection sampling on the bottom annular ring (z=Z_BOTTOM).
        Annular region: outside inner ellipse AND inside outer ellipse at z=Z_BOTTOM.
        At z=Z_BOTTOM the scale=1, so a=A_BASE, a_inner=A_INNER (no taper).
        """
        pts_x: list = []
        pts_y: list = []
        collected = 0
        while collected < n:
            n_try = max((n - collected) * 8, 200)
            x = np.random.uniform(-self.A_BASE, self.A_BASE, n_try).astype(np.float32)
            y = np.random.uniform(-self.B_BASE, self.B_BASE, n_try).astype(np.float32)
            outer_ok = (x**2 / self.A_BASE**2  + y**2 / self.B_BASE**2)  <= 1.0
            inner_ok = (x**2 / self.A_INNER**2 + y**2 / self.B_INNER**2) >= 1.0
            mask = outer_ok & inner_ok
            pts_x.append(x[mask])
            pts_y.append(y[mask])
            collected += int(mask.sum())

        x = np.concatenate(pts_x)[:n]
        y = np.concatenate(pts_y)[:n]
        z = np.full(n, self.Z_BOTTOM, dtype=np.float32)
        # Outward normal at bottom face = -z direction (away from shell interior)
        nx = np.zeros(n, dtype=np.float32)
        ny = np.zeros(n, dtype=np.float32)
        nz = -np.ones(n, dtype=np.float32)
        return {
            "x": x[:, None], "y": y[:, None], "z": z[:, None],
            "normal_x": nx[:, None], "normal_y": ny[:, None], "normal_z": nz[:, None],
            "area": np.ones((n, 1), dtype=np.float32),
            "sdf":  np.zeros((n, 1), dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Public interface expected by physicsnemo constraints
    # ------------------------------------------------------------------

    def sample_interior(
        self,
        nr_points: int,
        bounds=None,
        parameterization=None,
        quasirandom: bool = False,
        **kwargs,
    ) -> dict:
        """
        Rejection sampling inside the shell bounding box, filtered to shell material.
        Parameterization variables (e.g. t) are sampled via _sample_params and
        returned in the same dict so the constraint graph can resolve them.
        """
        pts_x: list = []
        pts_y: list = []
        pts_z: list = []
        collected = 0
        # Shell fill fraction ≈ (V_outer - V_inner) / V_bbox ≈ 5 %.
        # OVER=20 → each iteration tries 20× the remaining quota, producing
        # ~1× expected accepts, so we converge in 1–3 iterations on average.
        OVER = 20
        while collected < nr_points:
            n_try = max((nr_points - collected) * OVER, 500)
            x = np.random.uniform(-self.A_BASE, self.A_BASE, n_try).astype(np.float32)
            y = np.random.uniform(-self.B_BASE, self.B_BASE, n_try).astype(np.float32)
            z = np.random.uniform(self.Z_BOTTOM, self.Z_TOP,  n_try).astype(np.float32)
            mask = self._in_shell(x, y, z)
            pts_x.append(x[mask])
            pts_y.append(y[mask])
            pts_z.append(z[mask])
            collected += int(mask.sum())

        x = np.concatenate(pts_x)[:nr_points, None]
        y = np.concatenate(pts_y)[:nr_points, None]
        z = np.concatenate(pts_z)[:nr_points, None]
        result = {
            "x":   x,
            "y":   y,
            "z":   z,
            "sdf": np.zeros_like(x),
        }
        result.update(_sample_params(nr_points, parameterization))
        return result

    def sample_boundary(
        self,
        nr_points: int,
        sdf_fn=None,
        parameterization=None,
        quasirandom: bool = False,
        **kwargs,
    ) -> dict:
        """
        Sample all three boundary surfaces proportionally:
          45 % outer wall, 45 % inner wall, 10 % bottom lip.

        Used only when this object is passed as the geometry for boundary
        constraints that cover all surfaces at once.  For per-surface
        constraints use OuterConeWall / InnerConeWall / BottomAnnularLip.
        """
        n_outer  = int(nr_points * 0.45)
        n_inner  = int(nr_points * 0.45)
        n_bottom = nr_points - n_outer - n_inner

        outer  = self._sample_outer_wall(n_outer)
        inner  = self._sample_inner_wall(n_inner)
        bottom = self._sample_bottom_lip(n_bottom)

        keys = ["x", "y", "z", "normal_x", "normal_y", "normal_z", "area", "sdf"]
        result = {k: np.concatenate([outer[k], inner[k], bottom[k]], axis=0) for k in keys}
        result.update(_sample_params(nr_points, parameterization))
        return result


class OuterConeWall:
    """
    Boundary-only geometry: outer slanted cone surface (plasma-facing Dirichlet wall).
    All batch_size points are drawn from this surface — no criteria lambda needed.
    """

    dims: int = 3

    def __init__(self, cone_shell: HollowEllipticalConeShell) -> None:
        self._cs = cone_shell

    def sample_boundary(self, nr_points: int, parameterization=None, **kwargs) -> dict:
        result = self._cs._sample_outer_wall(nr_points)
        result.update(_sample_params(nr_points, parameterization))
        return result

    def sample_interior(self, nr_points: int, parameterization=None, **kwargs) -> dict:
        # Fallback: return boundary samples (used if framework probes this method)
        result = self._cs._sample_outer_wall(nr_points)
        result.update(_sample_params(nr_points, parameterization))
        return result


class InnerConeWall:
    """
    Boundary-only geometry: inner slanted cone surface (adiabatic Neumann wall).
    """

    dims: int = 3

    def __init__(self, cone_shell: HollowEllipticalConeShell) -> None:
        self._cs = cone_shell

    def sample_boundary(self, nr_points: int, parameterization=None, **kwargs) -> dict:
        result = self._cs._sample_inner_wall(nr_points)
        result.update(_sample_params(nr_points, parameterization))
        return result

    def sample_interior(self, nr_points: int, parameterization=None, **kwargs) -> dict:
        result = self._cs._sample_inner_wall(nr_points)
        result.update(_sample_params(nr_points, parameterization))
        return result


class BottomAnnularLip:
    """
    Boundary-only geometry: bottom annular ring at z=Z_BOTTOM (adiabatic Neumann).
    """

    dims: int = 3

    def __init__(self, cone_shell: HollowEllipticalConeShell) -> None:
        self._cs = cone_shell

    def sample_boundary(self, nr_points: int, parameterization=None, **kwargs) -> dict:
        result = self._cs._sample_bottom_lip(nr_points)
        result.update(_sample_params(nr_points, parameterization))
        return result

    def sample_interior(self, nr_points: int, parameterization=None, **kwargs) -> dict:
        result = self._cs._sample_bottom_lip(nr_points)
        result.update(_sample_params(nr_points, parameterization))
        return result


# =============================================================================
# NETWORK ARCHITECTURE  (pure PyTorch — no physicsnemo.nn dependency)
# =============================================================================

class FourierFeatureNet(Arch):
    """
    Fourier-feature encoding + SiLU fully-connected backbone.

    Inherits from Arch (physicsnemo.sym.models.arch.Arch) so that the
    PhysicsNeMo Solver recognises this module as a trainable network and
    correctly registers its parameters with the Adam optimizer.

    Architecture
    ____________
    1.  Random Fourier projection: x_enc = [sin(Bx), cos(Bx)]  (fixed, not learned)
        B ~ N(0, freq_scale²),  shape (n_inputs, n_frequencies)
    2.  nr_layers × (Linear → SiLU) with weight normalization
    3.  Final linear projection to n_outputs (no activation)
    """

    def __init__(
        self,
        input_keys:    list,
        output_keys:   list,
        layer_size:    int   = 256,
        nr_layers:     int   = 6,
        n_frequencies: int   = 8,
        freq_scale:    float = 1.0,
    ):
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self._in_keys  = [k.name for k in input_keys]
        self._out_keys = [k.name for k in output_keys]
        n_in  = len(input_keys)
        n_out = len(output_keys)

        B = torch.randn(n_in, n_frequencies) * freq_scale
        self.register_buffer("_B", B)

        enc_dim = 2 * n_frequencies
        layers: list[torch.nn.Module] = []
        in_dim = enc_dim
        for _ in range(nr_layers):
            lin = torch.nn.Linear(in_dim, layer_size)
            lin = torch.nn.utils.weight_norm(lin)
            layers += [lin, torch.nn.SiLU()]
            in_dim = layer_size
        layers.append(torch.nn.Linear(layer_size, n_out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, in_vars: dict) -> dict:
        x = torch.cat([in_vars[k] for k in self._in_keys], dim=-1)
        x_proj = x @ self._B
        x_enc  = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        out = self.net(x_enc)
        return {k: out[:, i : i + 1] for i, k in enumerate(self._out_keys)}


# =============================================================================
# BEST-WEIGHT SOLVER  (best-only checkpointing, checked every 100 steps)
# =============================================================================

class BestWeightSolver(Solver):
    """
    Solver subclass that saves network weights only when the loss strictly
    improves, checked every 100 steps (save_network_freq=100).

    At any given moment exactly ONE file exists in outputs/models/:
        best_weights_step_NNNNNN.pth   — the all-time best weights so far

    When a new best is found:
      1. Delete the previous best file (safe no-op if it has already gone).
      2. Write the new file with the current step number.

    Loss capture
    ------------
    compute_gradients() is overridden at the class level so the training loop
    always reaches this method regardless of how PhysicsNeMo caches callables.

    Resume behaviour
    ----------------
    On construction we scan outputs/models/ for any existing best_weights_step_*
    file and restore its loss, so a resumed run will not overwrite a genuinely
    better checkpoint from a previous session.

    File payload
    ------------
    {"states": {"heat_network": <state_dict>}, "step": N, "loss": L}
    """

    _BEST_DIR: str = os.path.join("outputs", "models")

    def __init__(self, cfg: "SimConfig", domain: Domain) -> None:
        os.makedirs(self._BEST_DIR, exist_ok=True)
        self._best_loss: float = float("inf")
        self._step_loss: float = float("inf")
        self._best_file: str   = ""          # path of the currently-saved best file

        # Restore prior best so a resumed run doesn't overwrite a better checkpoint
        existing = sorted(
            glob.glob(os.path.join(self._BEST_DIR, "best_weights_step_*.pth"))
        )
        if existing:
            try:
                ckpt = torch.load(existing[-1], map_location="cpu")
                self._best_loss = float(ckpt.get("loss", float("inf")))
                self._best_file = existing[-1]
                print(
                    f"  ℹ  Restored best loss = {self._best_loss:.4e}"
                    f"  from {os.path.basename(existing[-1])}"
                )
            except Exception:
                pass

        super().__init__(cfg, domain)

    # ------------------------------------------------------------------
    # Loss capture
    # ------------------------------------------------------------------
    def compute_gradients(self):
        # Trainer returns (aggregated_loss_tensor, per_constraint_losses_dict).
        # We unpack defensively so this works if the return type ever changes.
        result = super().compute_gradients()
        try:
            loss_val = result[0] if isinstance(result, (tuple, list)) else result
            if loss_val is not None:
                self._step_loss = float(loss_val)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Conditional checkpoint save — called every 100 steps
    # ------------------------------------------------------------------
    def save_checkpoint(self, step: int) -> None:
        # Always run the native PhysicsNeMo checkpoint (optimizer + scheduler state)
        super().save_checkpoint(step)

        native = "heat_network.0.pth"
        if not os.path.exists(native):
            return

        current_loss = self._step_loss

        # Heartbeat every 500 steps — always printed to stdout so Kaggle
        # shows the run is alive without flooding the cell output.
        if step % 500 == 0:
            print(
                f"[step {step:6d}/{20000}]  loss={current_loss:.4e}"
                f"  best={self._best_loss:.4e}",
                flush=True,
            )

        if current_loss < self._best_loss:
            # --- Delete the previous best file (single-file guarantee) -------
            if self._best_file and os.path.exists(self._best_file):
                try:
                    os.remove(self._best_file)
                except OSError:
                    pass   # already gone — safe to continue

            # --- Save new best ------------------------------------------------
            self._best_loss = current_loss
            new_path = os.path.join(
                self._BEST_DIR, f"best_weights_step_{step:06d}.pth"
            )
            sd = torch.load(native, map_location="cpu")
            torch.save(
                {"states": {"heat_network": sd}, "step": step, "loss": current_loss},
                new_path,
            )
            self._best_file = new_path
            print(
                f"  ✓ New best  step={step:6d}  loss={current_loss:.4e}"
                f"  → {os.path.basename(new_path)}",
                flush=True,
            )


# =============================================================================
# 1.  PHYSICAL & NORMALIZATION CONSTANTS
# =============================================================================

# --- Geometry (Hollow Elliptical Cone Shell) ---------------------------------
A_BASE   = 0.06    # m  — semi-major axis (x) of outer ellipse at z=Z_BOTTOM
B_BASE   = 0.04    # m  — semi-minor axis (y) of outer ellipse at z=Z_BOTTOM
H_CONE   = 0.10    # m  — full cone height; apex at z=H_CONE (= Z_TOP)
T_WALL   = 0.005   # m  — wall thickness (constant semi-axis reduction)
Z_BOTTOM = 0.0     # m  — base of cone (bottom annular lip)
Z_TOP    = 0.10    # m  — apex of cone (single point, no top-lip BC needed)
A_INNER  = A_BASE  - T_WALL   # 0.055 m
B_INNER  = B_BASE  - T_WALL   # 0.035 m

# --- Thermal -----------------------------------------------------------------
ALPHA = 5e-6    # m²/s  — thermal diffusivity

# --- Time domain -------------------------------------------------------------
T_END = 60.0    # s

# --- Temperature range -------------------------------------------------------
T_INITIAL = 300.0
T_PLASMA  = 4000.0
T_RANGE   = T_PLASMA - T_INITIAL   # 3700 K

# --- Min-max normalization to [0, 1] ----------------------------------------
#   x_hat = (x - X_MIN) / X_SCALE
X_MIN,   X_SCALE = -A_BASE,   2.0 * A_BASE          # x ∈ [-0.06,  0.06] → [0,1]
Y_MIN,   Y_SCALE = -B_BASE,   2.0 * B_BASE          # y ∈ [-0.04,  0.04] → [0,1]
Z_MIN,   Z_SCALE =  Z_BOTTOM, Z_TOP - Z_BOTTOM      # z ∈ [ 0.00,  0.10] → [0,1]
T_MIN_T, T_SCALE =  0.0,      T_END                 # t ∈ [ 0.00, 60.00] → [0,1]

# --- Anisotropic normalized diffusivities (chain-rule derivation) -----------
#
# Physical PDE: dT/dt = alpha*(d²T/dx² + d²T/dy² + d²T/dz²)
#
# Substituting T = T_RANGE*T_hat + T_INITIAL  and  x = X_SCALE*x_hat + X_MIN:
#   d²T/dx² = (T_RANGE / X_SCALE²) * d²T_hat/dx_hat²
#
# Because X_SCALE ≠ Y_SCALE ≠ Z_SCALE (ellipse, not sphere), each axis gets
# a distinct coefficient:
#   dT_hat/dt_hat = ALPHA_X_HAT * d²T_hat/dx_hat²
#                 + ALPHA_Y_HAT * d²T_hat/dy_hat²
#                 + ALPHA_Z_HAT * d²T_hat/dz_hat²
#
# Numerical values (ALPHA=5e-6, T_SCALE=60):
#   ALPHA_X_HAT = 5e-6 * 60 / 0.12²  = 0.020833
#   ALPHA_Y_HAT = 5e-6 * 60 / 0.08²  = 0.046875
#   ALPHA_Z_HAT = 5e-6 * 60 / 0.10²  = 0.030000
#
ALPHA_X_HAT = ALPHA * T_SCALE / X_SCALE ** 2   # ≈ 0.020833
ALPHA_Y_HAT = ALPHA * T_SCALE / Y_SCALE ** 2   # ≈ 0.046875
ALPHA_Z_HAT = ALPHA * T_SCALE / Z_SCALE ** 2   # ≈ 0.030000


# =============================================================================
# 2.  NORMALIZED HEAT EQUATION PDE  (anisotropic coefficients)
# =============================================================================

class NormalizedHeatEquation3D(PDE):
    """
    Transient heat equation in normalized [0,1] coordinates with anisotropic
    diffusivity coefficients to handle non-cubic bounding boxes.

    Residual:
        R = dT_hat/dt_hat
            - alpha_x_hat * d²T_hat/dx_hat²
            - alpha_y_hat * d²T_hat/dy_hat²
            - alpha_z_hat * d²T_hat/dz_hat²
          = 0
    """

    name = "NormalizedHeatEquation3D"

    def __init__(
        self,
        alpha_x_hat: float = ALPHA_X_HAT,
        alpha_y_hat: float = ALPHA_Y_HAT,
        alpha_z_hat: float = ALPHA_Z_HAT,
    ):
        x_hat = Symbol("x_hat")
        y_hat = Symbol("y_hat")
        z_hat = Symbol("z_hat")
        t_hat = Symbol("t_hat")

        T_hat = Function("T_hat")(x_hat, y_hat, z_hat, t_hat)

        residual = (
            diff(T_hat, t_hat)
            - alpha_x_hat * diff(T_hat, x_hat, 2)
            - alpha_y_hat * diff(T_hat, y_hat, 2)
            - alpha_z_hat * diff(T_hat, z_hat, 2)
        )

        self.equations = {"heat_equation": residual}


# =============================================================================
# 3.  HYDRA CONFIG STORE
# =============================================================================

cs = ConfigStore.instance()
cs.store(name="config", node=PhysicsNeMoConfig)


# =============================================================================
# 4.  MAIN TRAINING FUNCTION
# =============================================================================

@hydra.main(version_base="1.2", config_path=None, config_name="config")
def run(cfg: SimConfig) -> None:
    import traceback as _tb
    try:
        _run_inner(cfg)
    except Exception as _exc:
        # Hydra captures stdout/stderr; this guarantees the full traceback is
        # visible even when running as a subprocess in Kaggle.
        print(f"\n[FATAL] Training crashed: {_exc}", flush=True)
        _tb.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise


def _run_inner(cfg: SimConfig) -> None:

    print("[Stratos PINN v4] Starting training setup …", flush=True)

    # --- Training hyperparameters --------------------------------------------
    with open_dict(cfg):
        cfg.training = OmegaConf.structured(DefaultTraining(
            max_steps=20_000,
            save_network_freq=100,   # check for best weights every 100 steps
            print_stats_freq=100,
            summary_freq=1_000,
        ))
        cfg.stop_criterion = OmegaConf.structured(DefaultStopCriterion())
        cfg.optimizer      = OmegaConf.structured(AdamConf())
        cfg.scheduler      = OmegaConf.structured(ExponentialLRConf(gamma=0.9999))

    # -------------------------------------------------------------------------
    # 4.1  Geometry
    # -------------------------------------------------------------------------
    # Main geometry object (used for interior PDE and IC constraints).
    # Separate surface objects are used for boundary constraints so that
    # each constraint samples exclusively from its target surface — no
    # criteria lambda needed, giving the full batch_size as effective samples.
    cone_shell = HollowEllipticalConeShell(
        A_BASE=A_BASE, B_BASE=B_BASE, H_CONE=H_CONE,
        T_WALL=T_WALL, Z_BOTTOM=Z_BOTTOM, Z_TOP=Z_TOP,
    )
    outer_wall_geom  = OuterConeWall(cone_shell)
    inner_wall_geom  = InnerConeWall(cone_shell)
    bottom_lip_geom  = BottomAnnularLip(cone_shell)

    # -------------------------------------------------------------------------
    # 4.2  Preprocessing / Postprocessing Nodes
    # -------------------------------------------------------------------------

    # Coordinate normalization: physical (x,y,z,t) → normalized (x_hat,…,t_hat)
    x_hat_node = Node.from_sympy((Symbol("x") - X_MIN)   / X_SCALE, "x_hat")
    y_hat_node = Node.from_sympy((Symbol("y") - Y_MIN)   / Y_SCALE, "y_hat")
    z_hat_node = Node.from_sympy((Symbol("z") - Z_MIN)   / Z_SCALE, "z_hat")
    t_hat_node = Node.from_sympy((Symbol("t") - T_MIN_T) / T_SCALE, "t_hat")

    # Temperature denormalization: T_hat ∈ [0,1] → T in Kelvin
    T_denorm_node = Node.from_sympy(
        Symbol("T_hat") * T_RANGE + T_INITIAL, "T"
    )

    # Neumann cone flux: full 3D condition with anisotropic scale factors.
    #
    # Physical condition on any adiabatic cone surface:
    #   dT/dn = n_x*(dT/dx) + n_y*(dT/dy) + n_z*(dT/dz) = 0
    #
    # Chain-rule in normalized space:
    #   dT/dx = T_RANGE/X_SCALE * dT_hat/dx_hat   (= T_RANGE * T_hat__x_hat / X_SCALE)
    #
    # Dividing through by T_RANGE:
    #   n_x * T_hat__x_hat / X_SCALE
    # + n_y * T_hat__y_hat / Y_SCALE
    # + n_z * T_hat__z_hat / Z_SCALE = 0
    #
    # The cone surface has non-zero normal_z (unlike the cylinder lateral wall),
    # so all three terms are required.  Scale factors are essential because
    # X_SCALE ≠ Y_SCALE ≠ Z_SCALE.
    neumann_cone_node = Node.from_sympy(
        Symbol("normal_x") * Symbol("T_hat__x_hat") / X_SCALE
        + Symbol("normal_y") * Symbol("T_hat__y_hat") / Y_SCALE
        + Symbol("normal_z") * Symbol("T_hat__z_hat") / Z_SCALE,
        "neumann_cone",
    )

    # -------------------------------------------------------------------------
    # 4.3  Network Architecture
    # -------------------------------------------------------------------------
    network = FourierFeatureNet(
        input_keys=[Key("x_hat"), Key("y_hat"), Key("z_hat"), Key("t_hat")],
        output_keys=[Key("T_hat")],
        layer_size=256,
        nr_layers=6,
        n_frequencies=8,
        freq_scale=1.0,
    )
    network_node = network.make_node(name="heat_network", jit=False)

    # -------------------------------------------------------------------------
    # 4.4  PDE Nodes
    # -------------------------------------------------------------------------
    heat_pde  = NormalizedHeatEquation3D(
        alpha_x_hat=ALPHA_X_HAT,
        alpha_y_hat=ALPHA_Y_HAT,
        alpha_z_hat=ALPHA_Z_HAT,
    )
    pde_nodes = heat_pde.make_nodes()

    # -------------------------------------------------------------------------
    # 4.5  Full Node Graph
    # -------------------------------------------------------------------------
    all_nodes = (
        [x_hat_node, y_hat_node, z_hat_node, t_hat_node]
        + [network_node]
        + pde_nodes
        + [T_denorm_node]
        + [neumann_cone_node]
    )

    # -------------------------------------------------------------------------
    # 4.6  Domain & Constraints
    # -------------------------------------------------------------------------
    domain = Domain()

    # --- 1) Interior PDE constraint ------------------------------------------
    # Enforces the normalized heat equation at 2000 collocation points per step
    # throughout the cone shell volume × time domain.
    interior_pde = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=cone_shell,
        outvar={"heat_equation": 0},
        batch_size=2000,
        bounds={
            Symbol("x"): (-A_BASE,  A_BASE),
            Symbol("y"): (-B_BASE,  B_BASE),
            Symbol("z"): (Z_BOTTOM, Z_TOP),
            Symbol("t"): (0.0,      T_END),
        },
        lambda_weighting={"heat_equation": 1.0},
        fixed_dataset=False,
        shuffle=True,
    )
    domain.add_constraint(interior_pde, name="interior_pde")

    # --- 2) Outer cone wall Dirichlet BC: T=4000 K, for t>0 ------------------
    # Plasma-facing exterior slant surface.
    # Weight=10: strong enforcement of the plasma temperature BC.
    # t starts at 0.1 to avoid the t=0 discontinuity (IC=300K vs BC=4000K).
    # OuterConeWall samples exclusively from the outer lateral surface —
    # all batch_size=1000 points are on the plasma face.
    outer_wall_bc = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=outer_wall_geom,
        outvar={"T_hat": 1.0},
        batch_size=1000,
        parameterization={Symbol("t"): (0.1, T_END)},
        lambda_weighting={"T_hat": 10.0},
    )
    domain.add_constraint(outer_wall_bc, name="bc_outer_dirichlet")

    # --- 3) Inner cone wall Neumann BC: dT/dn=0 (adiabatic hollow cavity) ----
    # Interior cavity wall — no heat flux into the void.
    inner_wall_bc = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=inner_wall_geom,
        outvar={"neumann_cone": 0},
        batch_size=1000,
        parameterization={Symbol("t"): (0.0, T_END)},
        lambda_weighting={"neumann_cone": 1.0},
    )
    domain.add_constraint(inner_wall_bc, name="bc_inner_neumann")

    # --- 4) Bottom annular lip Neumann BC: dT/dn=0 (adiabatic base edge) -----
    # Flat annular ring at z=Z_BOTTOM between inner and outer ellipses.
    # Normal = (0,0,-1); condition simplifies to dT_hat/dz_hat = 0.
    bottom_lip_bc = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=bottom_lip_geom,
        outvar={"neumann_cone": 0},
        batch_size=500,
        parameterization={Symbol("t"): (0.0, T_END)},
        lambda_weighting={"neumann_cone": 1.0},
    )
    domain.add_constraint(bottom_lip_bc, name="bc_bottom_neumann")

    # --- 5) Initial condition: T=300 K everywhere at t=0 ---------------------
    # T_hat target = (300 - 300) / 3700 = 0.0
    initial_condition = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=cone_shell,
        outvar={"T_hat": 0.0},
        batch_size=2000,
        bounds={
            Symbol("x"): (-A_BASE,  A_BASE),
            Symbol("y"): (-B_BASE,  B_BASE),
            Symbol("z"): (Z_BOTTOM, Z_TOP),
        },
        parameterization={Symbol("t"): 0.0},   # pins t=0
        lambda_weighting={"T_hat": 5.0},
        fixed_dataset=False,
    )
    domain.add_constraint(initial_condition, name="ic_t0")

    # -------------------------------------------------------------------------
    # 4.7  Inject Loss Config & Launch Solver
    # -------------------------------------------------------------------------
    from physicsnemo.sym.hydra.config import LossConf
    OmegaConf.set_struct(cfg, False)
    loss_cfg = LossConf()
    loss_cfg._target_ = "physicsnemo.sym.loss.aggregator.Sum"
    cfg.loss = loss_cfg

    print("[Stratos PINN v4] All constraints built — launching solver …", flush=True)
    sys.stdout.flush()

    # Expected convergence:
    #   - Outer-wall Dirichlet loss → ~0 within  5 000 steps  (weight = 10)
    #   - IC loss                   → converged by 10 000 steps (weight =  5)
    #   - Interior PDE loss         → below 1e-3 by ~30 000 steps
    #   - Neumann BC losses         → small throughout (soft constraints)
    slv = BestWeightSolver(cfg=cfg, domain=domain)
    slv.solve()


# =============================================================================
# 5.  POST-TRAINING VERIFICATION
# =============================================================================

def plot_temperature_profile(network, checkpoint_path: str = None):
    """
    Sample the trained network along the cone central axis (x=0, y=0) at
    t = {0, 10, 30, 60} seconds and plot T (Kelvin) vs z.

    Physical sanity checks
    ----------------------
    1. t=0            : T ≈ 300 K everywhere           — IC satisfied
    2. z≈0, outer wall: T ≈ 4000 K for t > 0           — Dirichlet BC satisfied
    3. z=Z_TOP        : profile value → some intermediate T (apex is the tip)
    4. Penetration depth at t=60 s: √(alpha·t) ≈ 1.7 cm from the outer wall
    """
    import matplotlib.pyplot as plt

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cuda")
        network.load_state_dict(state)

    network.eval()
    device = next(network.parameters()).device
    z_phys = np.linspace(Z_BOTTOM, Z_TOP, 200, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 5))

    for t_phys in [0.0, 10.0, 30.0, 60.0]:
        x_hat = np.full_like(z_phys, (0.0 - X_MIN) / X_SCALE)
        y_hat = np.full_like(z_phys, (0.0 - Y_MIN) / Y_SCALE)
        z_hat = (z_phys - Z_MIN) / Z_SCALE
        t_hat = np.full_like(z_phys, t_phys / T_SCALE)

        inputs = {
            "x_hat": torch.tensor(x_hat[:, None], device=device),
            "y_hat": torch.tensor(y_hat[:, None], device=device),
            "z_hat": torch.tensor(z_hat[:, None], device=device),
            "t_hat": torch.tensor(t_hat[:, None], device=device),
        }

        with torch.no_grad():
            T_hat_pred = network(inputs)["T_hat"]

        T_pred = T_hat_pred.cpu().numpy().flatten() * T_RANGE + T_INITIAL
        ax.plot(z_phys * 100.0, T_pred, label=f"t = {t_phys:.0f} s")

    ax.axhline(T_INITIAL, color="gray", linestyle="--", linewidth=0.8,
               label=f"T₀ = {T_INITIAL:.0f} K")
    ax.axhline(T_PLASMA,  color="red",  linestyle="--", linewidth=0.8,
               label=f"T_plasma = {T_PLASMA:.0f} K")
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Stratos PINN v4 — Cone Axis Temperature Profile (x=0, y=0)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("temperature_profile.png", dpi=150)
    plt.show()
    print("Plot saved to temperature_profile.png")


# =============================================================================
# 6.  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
