"""
infer_3d.py — Stratos PINN v4  |  Standalone Inference & VTP Export
=====================================================================
Loads trained FourierFeatureNet weights, samples the 3D temperature
field inside the Hollow Elliptical Cone Shell at a set of time snapshots,
and writes one .vtp file per snapshot for visualization in ParaView.

Zero physicsnemo / hydra dependency — runs anywhere PyTorch + pyvista
are installed.

Usage
-----
    python infer_3d.py                    # auto-discovers checkpoint
    python infer_3d.py --ckpt path/to.pth # explicit checkpoint

Checkpoint search order
-----------------------
1. outputs/models/best_weights_*.pth   (BestWeightSolver format)
   payload: {"states": {"heat_network": <state_dict>}, "step": N}
2. outputs/networks/heat_network.0.pth (PhysicsNeMo native, plain state dict)

Output
------
    outputs/inference/T_field_t000.0s.vtp
    outputs/inference/T_field_t005.0s.vtp
    ...  (one file per TIME_SNAPSHOTS entry)

Each .vtp is a pyvista PolyData cloud covering the cone shell interior
with point arrays:
    T_K  — temperature in Kelvin
    T_C  — temperature in Celsius
"""

import os
import sys
import glob
import math
import shutil
import argparse

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# pyvista import (optional — graceful error if not installed)
# ---------------------------------------------------------------------------
try:
    import pyvista as pv
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False
    print(
        "WARNING: pyvista not found.  VTP export will be skipped.\n"
        "         Install with:  pip install pyvista\n"
    )


# =============================================================================
# 1.  PHYSICAL & NORMALIZATION CONSTANTS  (mirror of train_3d.py)
# =============================================================================

# Hollow Elliptical Cone Shell geometry
A_BASE   = 0.06    # m  — semi-major axis (x) at z=Z_BOTTOM
B_BASE   = 0.04    # m  — semi-minor axis (y) at z=Z_BOTTOM
H_CONE   = 0.10    # m  — cone height; apex at z=H_CONE
T_WALL   = 0.005   # m  — wall thickness (semi-axis reduction)
Z_BOTTOM = 0.0     # m  — cone base
Z_TOP    = 0.10    # m  — cone apex
A_INNER  = A_BASE  - T_WALL   # 0.055 m
B_INNER  = B_BASE  - T_WALL   # 0.035 m

ALPHA     = 5e-6   # m²/s — thermal diffusivity
T_END     = 60.0   # s   — simulation end time

T_INITIAL = 300.0
T_PLASMA  = 4000.0
T_RANGE   = T_PLASMA - T_INITIAL   # 3700 K

# Min-max normalization — must match train_3d.py exactly
X_MIN,   X_SCALE = -A_BASE,   2.0 * A_BASE        # [-0.06, 0.06] → [0,1]
Y_MIN,   Y_SCALE = -B_BASE,   2.0 * B_BASE        # [-0.04, 0.04] → [0,1]
Z_MIN,   Z_SCALE =  Z_BOTTOM, Z_TOP - Z_BOTTOM    # [ 0.00, 0.10] → [0,1]
T_MIN_T, T_SCALE =  0.0,      T_END               # [ 0.00, 60.0] → [0,1]

# Time snapshots to export
TIME_SNAPSHOTS = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # seconds


# =============================================================================
# 2.  NETWORK ARCHITECTURE  (plain nn.Module — no physicsnemo dependency)
# =============================================================================

class FourierFeatureNet(nn.Module):
    """
    Random Fourier feature encoding + SiLU fully-connected backbone.
    Intentionally identical to the FourierFeatureNet class in train_3d.py
    except inheriting from nn.Module instead of physicsnemo.sym.Arch.
    The Arch base class contributes zero extra parameters or buffers so the
    state_dict keys are identical and checkpoints load without key remapping.
    """

    _IN_KEYS  = ["x_hat", "y_hat", "z_hat", "t_hat"]
    _OUT_KEYS = ["T_hat"]

    def __init__(
        self,
        layer_size:    int   = 256,
        nr_layers:     int   = 6,
        n_frequencies: int   = 8,
        freq_scale:    float = 1.0,
    ):
        super().__init__()
        n_in  = len(self._IN_KEYS)
        n_out = len(self._OUT_KEYS)

        B = torch.randn(n_in, n_frequencies) * freq_scale
        self.register_buffer("_B", B)

        enc_dim = 2 * n_frequencies
        layers: list[nn.Module] = []
        in_dim = enc_dim
        for _ in range(nr_layers):
            lin = nn.Linear(in_dim, layer_size)
            lin = nn.utils.weight_norm(lin)
            layers += [lin, nn.SiLU()]
            in_dim = layer_size
        layers.append(nn.Linear(layer_size, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, in_vars: dict) -> dict:
        x = torch.cat([in_vars[k] for k in self._IN_KEYS], dim=-1)
        x_proj = x @ self._B
        x_enc  = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        out = self.net(x_enc)
        return {k: out[:, i : i + 1] for i, k in enumerate(self._OUT_KEYS)}


# =============================================================================
# 3.  CHECKPOINT LOADING
# =============================================================================

def load_network(device: torch.device, explicit_ckpt: str = None) -> FourierFeatureNet:
    """
    Build the network and load weights from the best available checkpoint.

    Priority (unless --ckpt is given):
      1. outputs/models/best_weights_*.pth     — BestWeightSolver format
      2. outputs/networks/heat_network.0.pth   — PhysicsNeMo native format
    """
    net = FourierFeatureNet().to(device)

    if explicit_ckpt:
        if not os.path.exists(explicit_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {explicit_ckpt}")
        _load_state_dict(net, explicit_ckpt, label="explicit")
        return net

    # Priority 1: single best-weights file (BestWeightSolver — only one exists at a time)
    step_files = sorted(glob.glob(os.path.join("outputs", "models", "best_weights_step_*.pth")))
    if step_files:
        _load_state_dict(net, step_files[-1], label="best")
        return net

    # Priority 3: PhysicsNeMo native checkpoint
    native = os.path.join("outputs", "networks", "heat_network.0.pth")
    if os.path.exists(native):
        _load_state_dict(net, native, label="native")
        return net

    raise FileNotFoundError(
        "No checkpoint found.  Looked in:\n"
        "  outputs/models/best_weights_step_*.pth\n"
        "  outputs/networks/heat_network.0.pth\n"
        "Run train_3d.py first, or pass --ckpt <path>."
    )


def _load_state_dict(net: FourierFeatureNet, path: str, label: str) -> None:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "states" in raw:
        sd   = raw["states"]["heat_network"]
        step = raw.get("step", "?")
        loss = raw.get("loss", None)
        loss_str = f"  loss={loss:.4e}" if loss is not None else ""
        print(f"[{label}] Loaded weights  step={step}{loss_str}")
        print(f"         File: {os.path.basename(path)}")
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        print(f"[{label}] Loaded wrapped state dict from {os.path.basename(path)}")
    else:
        sd = raw
        print(f"[{label}] Loaded state dict from {os.path.basename(path)}")
    net.load_state_dict(sd)


# =============================================================================
# 4.  3D GRID CONSTRUCTION
# =============================================================================

def build_cone_shell_grid(
    nx: int = 60,
    ny: int = 40,
    nz: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a Cartesian grid inside the hollow elliptical cone shell.

    Grid density:
      nx=60, ny=40, nz=100 → 240 k pre-mask, ~12 k post-mask (shell is thin).
      nz=100 over 10 cm = 1 mm z-resolution, resolving the ~1.7 cm thermal
      penetration depth with ~17 points inside the thermally active zone.

    Shell condition at each (x, y, z):
      inside outer ellipse  AND  outside inner ellipse  AND  z in [Z_BOTTOM, Z_TOP]

    Parameters
    ----------
    nx, ny, nz : grid resolution along each axis.

    Returns
    -------
    x_pts, y_pts, z_pts : float32 arrays of shape (N_pts,)
        Physical coordinates of all shell-interior points.
    """
    xs = np.linspace(-A_BASE,   A_BASE,   nx, dtype=np.float32)
    ys = np.linspace(-B_BASE,   B_BASE,   ny, dtype=np.float32)
    zs = np.linspace(Z_BOTTOM,  Z_TOP,    nz, dtype=np.float32)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")   # (nx, ny, nz)

    # Taper scale at each z: 1.0 at Z_BOTTOM, 0.0 at H_CONE
    s = (H_CONE - ZZ) / (H_CONE - Z_BOTTOM)
    s = np.maximum(s, 1e-9)   # guard near apex

    a_out = A_BASE  * s
    b_out = B_BASE  * s
    a_in  = A_INNER * s
    b_in  = B_INNER * s

    outer_mask = (XX**2 / a_out**2 + YY**2 / b_out**2) <= 1.0
    inner_mask = (XX**2 / a_in**2  + YY**2 / b_in**2)  >= 1.0
    mask = outer_mask & inner_mask

    return XX[mask].ravel(), YY[mask].ravel(), ZZ[mask].ravel()


# =============================================================================
# 5.  BATCHED INFERENCE
# =============================================================================

def infer_temperature(
    net:        FourierFeatureNet,
    x_pts:      np.ndarray,
    y_pts:      np.ndarray,
    z_pts:      np.ndarray,
    t_phys:     float,
    device:     torch.device,
    batch_size: int = 10_000,
) -> np.ndarray:
    """
    Evaluate the network at every (x, y, z) point for a fixed time t_phys.

    Returns
    -------
    T_K : np.ndarray shape (N_pts,), dtype float32  — temperature in Kelvin
    """
    n = len(x_pts)

    x_hat = (x_pts - X_MIN)   / X_SCALE
    y_hat = (y_pts - Y_MIN)   / Y_SCALE
    z_hat = (z_pts - Z_MIN)   / Z_SCALE
    t_hat = np.full(n, t_phys / T_SCALE, dtype=np.float32)

    T_out = np.empty(n, dtype=np.float32)
    net.eval()

    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            inp = {
                key: torch.from_numpy(arr[sl, np.newaxis]).to(device)
                for key, arr in zip(
                    ["x_hat", "y_hat", "z_hat", "t_hat"],
                    [ x_hat,   y_hat,   z_hat,   t_hat],
                )
            }
            T_hat = net(inp)["T_hat"].cpu().numpy().ravel()
            T_out[sl] = T_hat * T_RANGE + T_INITIAL

    return T_out


# =============================================================================
# 6.  CONSOLE SANITY CHECKS
# =============================================================================

def print_sanity_checks(
    net:    FourierFeatureNet,
    x_pts:  np.ndarray,
    y_pts:  np.ndarray,
    z_pts:  np.ndarray,
    device: torch.device,
) -> None:
    """
    Quick physics sanity checks printed before the full export loop.

    Expected outcomes (well-trained network)
    -----------------------------------------
    IC (t=0)     : mean T ≈  300 K ± 20  (initial condition satisfied)
    Outer wall (t=60): T ≈ 4000 K near outer surface (Dirichlet BC satisfied)
    Penetration depth at t=60 s: √(α·t) ≈ 1.73 cm  (analytical reference)
    """
    print("\n── Sanity checks ────────────────────────────────────────────────")

    # 1) Initial condition: t=0, all interior points
    T0 = infer_temperature(net, x_pts, y_pts, z_pts, 0.0, device)
    ic_ok = "✓" if abs(T0.mean() - T_INITIAL) < 20 else "✗"
    print(f"  {ic_ok}  IC   (t=0)       mean T = {T0.mean():.1f} K   (expected ~{T_INITIAL:.0f} K)")

    # 2) Outer wall Dirichlet BC: sample points near the outer cone surface at t=60.
    # Proxy: points close to the outer ellipse (normalized radius ≈ 1) at mid-z.
    s_pts = (H_CONE - z_pts) / (H_CONE - Z_BOTTOM)
    s_pts = np.maximum(s_pts, 1e-9)
    a_out_pts = A_BASE * s_pts
    b_out_pts = B_BASE * s_pts
    ellipse_val = x_pts**2 / a_out_pts**2 + y_pts**2 / b_out_pts**2
    outer_mask = ellipse_val >= 0.90   # within ~5% of outer surface
    if outer_mask.sum() > 0:
        T60w = infer_temperature(
            net, x_pts[outer_mask], y_pts[outer_mask], z_pts[outer_mask], 60.0, device
        )
        bc_ok = "✓" if abs(T60w.mean() - T_PLASMA) < 200 else "✗"
        print(
            f"  {bc_ok}  Outer wall (near-surface, t=60)  mean T = {T60w.mean():.1f} K"
            f"   (expected ~{T_PLASMA:.0f} K)"
        )
    else:
        print("  ?  No near-surface points found — check grid resolution")

    # 3) Analytical penetration depth (reference)
    delta_cm = math.sqrt(ALPHA * T_END) * 100
    print(f"  ℹ  Analytical penetration depth at t=60 s: {delta_cm:.2f} cm (~1.7 cm)")
    print("─────────────────────────────────────────────────────────────────\n")


# =============================================================================
# 7.  VTP EXPORT
# =============================================================================

def export_snapshots(
    net:        FourierFeatureNet,
    x_pts:      np.ndarray,
    y_pts:      np.ndarray,
    z_pts:      np.ndarray,
    device:     torch.device,
    out_dir:    str = os.path.join("outputs", "inference"),
    batch_size: int = 10_000,
) -> None:
    """
    For each snapshot in TIME_SNAPSHOTS:
      1. Run batched inference → T_K
      2. Build pyvista.PolyData with point arrays T_K and T_C
      3. Save as .vtp

    Results can be loaded together in ParaView as a temporal series.
    """
    if not _PYVISTA_OK:
        print("pyvista not available — skipping VTP export.")
        return

    os.makedirs(out_dir, exist_ok=True)
    points = np.column_stack([x_pts, y_pts, z_pts]).astype(np.float32)

    print(f"Exporting {len(TIME_SNAPSHOTS)} snapshots → {out_dir}/")
    for t_s in TIME_SNAPSHOTS:
        T_K = infer_temperature(net, x_pts, y_pts, z_pts, t_s, device,
                                batch_size=batch_size)

        cloud = pv.PolyData(points)
        cloud["T_K"] = T_K
        cloud["T_C"] = T_K - 273.15

        fname = os.path.join(out_dir, f"T_field_t{t_s:06.1f}s.vtp")
        cloud.save(fname)
        print(f"  → {fname}   T ∈ [{T_K.min():.0f}, {T_K.max():.0f}] K")

        if t_s == 60.0:
            alias = os.path.join(out_dir, "cone_shell_heat_t60.vtp")
            shutil.copy2(fname, alias)
            print(f"  → {alias}  (alias)")

    pvd_path = os.path.join(out_dir, "heat_animation.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t_s in TIME_SNAPSHOTS:
            vtp_name = f"T_field_t{t_s:06.1f}s.vtp"
            f.write(f'    <DataSet timestep="{t_s}" group="" part="0" file="{vtp_name}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    print(f"\n  → {pvd_path}  (open this in ParaView for animation)")

    print(f"\nAll {len(TIME_SNAPSHOTS)} snapshots saved.")
    print("Open in ParaView: File → Open → heat_animation.pvd → Apply")
    print("  Colour by: T_K  |  Hit Play to animate through time steps")


# =============================================================================
# 8.  ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratos PINN v4 — Hollow Elliptical Cone Shell inference & VTP export"
    )
    p.add_argument(
        "--model_path",
        default=None,
        metavar="PATH",
        help="Path to checkpoint .pth file (alias for --ckpt).",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        metavar="PATH",
        help="Path to checkpoint .pth file (overrides --model_path default).",
    )
    p.add_argument("--nx", type=int, default=60,  help="Grid points along x (default 60)")
    p.add_argument("--ny", type=int, default=40,  help="Grid points along y (default 40)")
    p.add_argument("--nz", type=int, default=100, help="Grid points along z (default 100)")
    p.add_argument(
        "--out_dir",
        default=os.path.join("outputs", "inference"),
        help="Output directory for .vtp files",
    )
    p.add_argument(
        "--batch_size", type=int, default=10_000,
        help="Inference mini-batch size (default 10000)",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference even if CUDA is available",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")

    explicit_ckpt = args.ckpt or args.model_path
    net = load_network(device, explicit_ckpt=explicit_ckpt)

    x_pts, y_pts, z_pts = build_cone_shell_grid(nx=args.nx, ny=args.ny, nz=args.nz)
    n_pts = len(x_pts)
    print(
        f"Grid   : {args.nx}×{args.ny}×{args.nz} pre-mask → {n_pts:,} points inside cone shell"
        f"  (A={A_BASE}m, B={B_BASE}m, H={H_CONE}m, wall={T_WALL}m)"
    )

    print_sanity_checks(net, x_pts, y_pts, z_pts, device)

    export_snapshots(
        net, x_pts, y_pts, z_pts, device,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
