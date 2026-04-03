# ================================================================
# LBM Ink-on-Paper — Full Physics Kernel + Capillary Wicking
# D2Q9 + Fiber Network + Guo Forcing Scheme + Laplace Evaporation
# ================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ──────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────
Nx, Ny = 512, 512
tau = 0.85
omega = 1.0 / tau
rho0 = 1.0
evap_coeff = 0.0008
drop_radius = 16
steps = 2000
snap_at = [0, 200, 500, 1000, 1500, 1999]

# ── Capillary parameters ──────────────────────────────
# sigma_cap controls capillary strength.
#   Physical interpretation: σ·cos(θ) scaled to lattice units.
#   Safe range: 0.005 – 0.05 for tau=0.85.
#   Above ~0.08 you risk density blow-up.
#   Rule of thumb: sigma_cap < 0.1 * (tau - 0.5)
sigma_cap = 0.02

# Only apply capillary force where ink is present (wet cells).
# This prevents the force field from acting on dry regions.
wet_threshold = 0.005

# ──────────────────────────────────────────────────────
# D2Q9 Lattice
# ──────────────────────────────────────────────────────
c = torch.tensor([
    [0,0],[1,0],[0,1],[-1,0],[0,-1],
    [1,1],[-1,1],[-1,-1],[1,-1]
], device=device, dtype=torch.float32)

w = torch.tensor([
    4/9, 1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
], device=device, dtype=torch.float32)

opposite = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device, dtype=torch.long)

# ──────────────────────────────────────────────────────
# Fiber Deposition Model (Koponen et al.)
# ──────────────────────────────────────────────────────
def generate_fiber_network(nx, ny, n_fibers=4000,
                           fiber_length=35, fiber_width=2,
                           seed=42):
    np.random.seed(seed)
    field = np.zeros((nx, ny), dtype=np.float32)

    for _ in range(n_fibers):
        x0 = np.random.uniform(0, nx)
        y0 = np.random.uniform(0, ny)
        theta = np.random.uniform(0, np.pi)
        length = max(5, np.random.normal(fiber_length, fiber_length * 0.25))

        n_steps = int(length)
        ts = np.linspace(-length / 2, length / 2, n_steps)
        xs = (x0 + ts * np.cos(theta)).astype(int) % nx
        ys = (y0 + ts * np.sin(theta)).astype(int) % ny

        hw = fiber_width // 2
        for dx in range(-hw, hw + 1):
            for dy in range(-hw, hw + 1):
                if dx * dx + dy * dy <= (fiber_width / 2 + 0.5) ** 2:
                    field[(xs + dx) % nx, (ys + dy) % ny] += 1

    field = field / max(field.max(), 1)
    return field

print("Generating fiber network...")
kappa_np = generate_fiber_network(Nx, Ny)
kappa = torch.tensor(kappa_np, dtype=torch.float32, device=device)
kappa = 0.05 + 0.55 * kappa  # map to [0.05, 0.60]
print(f"  κ range: [{kappa.min().item():.3f}, {kappa.max().item():.3f}]")

# ──────────────────────────────────────────────────────
# Precompute capillary force field: F = σ_cap * ∇κ
#   This is static (paper doesn't change), so compute once.
#   Central differences with periodic BCs (via torch.roll).
# ──────────────────────────────────────────────────────
grad_kappa_x = (torch.roll(kappa, -1, 0) - torch.roll(kappa, 1, 0)) / 2.0
grad_kappa_y = (torch.roll(kappa, -1, 1) - torch.roll(kappa, 1, 1)) / 2.0

# F_cap = sigma_cap * grad(kappa)  — points toward higher κ (smaller pores)
F_cap = torch.stack([sigma_cap * grad_kappa_x,
                     sigma_cap * grad_kappa_y], dim=0)  # shape [2, Nx, Ny]

print(f"  Capillary force |F| max: {torch.sqrt(F_cap[0]**2 + F_cap[1]**2).max().item():.6f}")

# ──────────────────────────────────────────────────────
# LBM Core Functions
# ──────────────────────────────────────────────────────
def equilibrium(rho, u):
    """Standard D2Q9 equilibrium with given velocity."""
    cu = torch.einsum('ic,cxy->ixy', c, u)              # c_i · u
    u_sq = 0.5 * (u[0]**2 + u[1]**2)                    # |u|²/2
    return w.view(9,1,1) * rho.unsqueeze(0) * (
        1.0 + 3.0*cu + 4.5*cu**2 - 3.0*u_sq.unsqueeze(0))


def guo_forcing_term(u, F, rho):
    """
    Guo forcing term (Guo, Zheng & Shi, PRE 65, 2002).

    S_i = (1 - 1/(2τ)) · w_i · [ 3(c_i - u) + 9(c_i·u)c_i ] · F

    This enters the collision as:  f_i^* = f_i - ω(f_i - f_i^eq) + S_i

    Parameters
    ----------
    u : [2, Nx, Ny]  macroscopic velocity (already force-corrected)
    F : [2, Nx, Ny]  force field (capillary force, masked to wet cells)
    rho : [Nx, Ny]   density (unused here but available for extensions)

    Returns
    -------
    S : [9, Nx, Ny]  forcing contribution per lattice direction
    """
    prefactor = 1.0 - 0.5 * omega  # (1 - 1/(2τ))

    # c_i · u  and  c_i · F   for all 9 directions
    cu = torch.einsum('ic,cxy->ixy', c, u)   # [9, Nx, Ny]
    cF = torch.einsum('ic,cxy->ixy', c, F)   # [9, Nx, Ny]
    uF = (u[0]*F[0] + u[1]*F[1])             # [Nx, Ny]  — u·F scalar

    # S_i = prefactor * w_i * [ 3*(c_i·F - u·F) + 9*(c_i·u)(c_i·F) ]
    #   Equivalent expansion of the standard Guo formula.
    S = prefactor * w.view(9,1,1) * (
        3.0 * (cF - uF.unsqueeze(0)) +
        9.0 * cu * cF
    )
    return S


def stream(f):
    f_s = torch.zeros_like(f)
    for i in range(9):
        f_s[i] = torch.roll(f[i], shifts=(int(c[i,0]), int(c[i,1])), dims=(0,1))
    return f_s


def bounceback(f, kappa):
    """Partial bounce-back: linear interpolation between free flow and
    full bounce-back, weighted by local solid fraction κ."""
    return (1.0 - kappa).unsqueeze(0) * f + kappa.unsqueeze(0) * f[opposite]


# ──────────────────────────────────────────────────────
# Laplace Vapor Diffusion Evaporation (unchanged)
# ──────────────────────────────────────────────────────
def compute_evaporation_laplace(rho, rho0, kappa, evap_coeff,
                                 c_sat=1.0, c_amb=0.3, iters=60):
    excess = (rho - rho0).clamp(min=0)
    wet = excess > 0.005

    if wet.sum() < 5:
        return torch.zeros_like(rho)

    c_vapor = torch.ones_like(rho) * c_amb
    c_vapor[wet] = c_sat
    wet_float = wet.float()

    for _ in range(iters):
        c_new = 0.25 * (
            torch.roll(c_vapor, 1, 0) + torch.roll(c_vapor, -1, 0) +
            torch.roll(c_vapor, 1, 1) + torch.roll(c_vapor, -1, 1)
        )
        c_new = c_new * (1 - wet_float) + c_sat * wet_float
        c_vapor = c_new

    grad_x = (torch.roll(c_vapor, -1, 0) - torch.roll(c_vapor, 1, 0)) / 2
    grad_y = (torch.roll(c_vapor, -1, 1) - torch.roll(c_vapor, 1, 1)) / 2
    J = torch.sqrt(grad_x**2 + grad_y**2)

    pore_factor = 1.0 - 0.3 * kappa
    J = evap_coeff * J * wet_float * pore_factor
    J = torch.min(J, excess * 0.1)
    return J


# ──────────────────────────────────────────────────────
# Initial Condition
# ──────────────────────────────────────────────────────
Y, X = torch.meshgrid(
    torch.arange(Nx, device=device, dtype=torch.float32),
    torch.arange(Ny, device=device, dtype=torch.float32), indexing='ij')
r = torch.sqrt((X - Nx//2)**2 + (Y - Ny//2)**2)

rho = torch.ones((Nx, Ny), device=device) * rho0
rho[r < drop_radius] = 1.5
u = torch.zeros((2, Nx, Ny), device=device)
f = equilibrium(rho, u)

# ──────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────
#
# Loop structure with Guo forcing:
#
#   1. Macroscopic: ρ = Σ f_i
#                   u* = Σ c_i f_i / ρ + F/(2ρ)    ← half-force correction
#   2. Mask force to wet cells only
#   3. Equilibrium from corrected u*
#   4. Collision: f = f - ω(f - f_eq) + S_i         ← Guo term here
#   5. Partial bounce-back (porous fiber interaction)
#   6. Streaming
#   7. Evaporation (Laplace)
#
# The half-force velocity correction (step 1) is essential:
# it ensures the macroscopic velocity is accurate to O(Δt²).
# The Guo source term S_i (step 4) ensures the force appears
# correctly in the recovered Navier-Stokes momentum equation.
#
snapshots = []

print(f"Running {steps} steps (σ_cap = {sigma_cap})...")
for step in range(steps):

    # ── 1. Macroscopic quantities ──
    rho = f.sum(dim=0)
    momentum = torch.einsum('ic,ixy->cxy', c, f)

    # Mask capillary force to wet cells only.
    # This prevents the static ∇κ field from creating spurious
    # flow in dry regions of the paper.
    excess = (rho - rho0).clamp(min=0)
    wet_mask = (excess > wet_threshold).float()
    F_local = F_cap * wet_mask.unsqueeze(0)   # [2, Nx, Ny]

    # Guo half-force velocity correction:
    #   u = (Σ c_i f_i + F·Δt/2) / ρ
    # In lattice units Δt=1, so:
    u = (momentum + 0.5 * F_local) / (rho.unsqueeze(0) + 1e-10)

    # ── 2. Equilibrium from corrected velocity ──
    feq = equilibrium(rho, u)

    # ── 3. Guo forcing source term ──
    S = guo_forcing_term(u, F_local, rho)

    # ── 4. Collision + forcing + bounce-back + streaming ──
    f_post = f - omega * (f - feq) + S          # BGK + Guo
    f_post = bounceback(f_post, kappa)           # porous medium
    f = stream(f_post)                           # propagation

    # ── 5. Evaporation ──
    rho_post = f.sum(dim=0)
    J = compute_evaporation_laplace(rho_post, rho0, kappa, evap_coeff)
    f -= (J / (rho_post + 1e-10)).unsqueeze(0) * f

    # ── Snapshots ──
    if step in snap_at:
        rho_snap = f.sum(dim=0)
        excess_snap = (rho_snap - rho0).clamp(min=0).cpu().numpy()
        mass = excess_snap.sum()
        snapshots.append((step, excess_snap.copy()))
        print(f"  Step {step:4d} | excess max={excess_snap.max():.4f} | "
              f"total mass={mass:.1f}")

    # ── Stability check ──
    if step % 200 == 0 and step > 0:
        rho_check = f.sum(dim=0)
        if rho_check.max() > 5.0 or torch.isnan(rho_check).any():
            print(f"  ⚠ Instability at step {step}! "
                  f"rho max={rho_check.max().item():.2f}. "
                  f"Try reducing sigma_cap or increasing tau.")
            break

print("Done.")

# ──────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────
ink_cmap = LinearSegmentedColormap.from_list('ink', [
    (1.0, 0.98, 0.94),
    (0.6, 0.62, 0.7),
    (0.25, 0.22, 0.38),
    (0.05, 0.03, 0.1)])

# --- Capillary force field ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(kappa.cpu().numpy(), cmap='bone_r', interpolation='nearest')
axes[0].set_title("Paper Fiber Network (κ)", fontsize=12, fontweight='bold')
axes[0].axis('off')

F_mag = torch.sqrt(F_cap[0]**2 + F_cap[1]**2).cpu().numpy()
axes[1].imshow(F_mag, cmap='inferno', interpolation='bilinear')
axes[1].set_title("|∇κ| — Capillary Force Magnitude", fontsize=12, fontweight='bold')
axes[1].axis('off')

# Quiver plot of force direction (subsample for clarity)
skip = 16
yy, xx = np.mgrid[0:Nx:skip, 0:Ny:skip]
Fx = F_cap[0].cpu().numpy()[::skip, ::skip]
Fy = F_cap[1].cpu().numpy()[::skip, ::skip]
axes[2].imshow(kappa.cpu().numpy(), cmap='bone_r', alpha=0.4)
axes[2].quiver(xx, yy, Fy, Fx, color='crimson', scale=0.5,
               width=0.003, headwidth=3)
axes[2].set_title("Capillary Force Vectors on κ", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('capillary_forces.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Time evolution ---
fig, axes_t = plt.subplots(2, 3, figsize=(15, 10))
for ax, (step_n, excess_s) in zip(axes_t.flat, snapshots):
    vmax = max(excess_s.max() * 0.85, 0.005)
    ax.imshow(excess_s, cmap=ink_cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Step {step_n}", fontsize=11)
    ax.axis('off')
fig.suptitle(f"Ink Spreading with Capillary Wicking (σ_cap={sigma_cap})",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('time_evolution.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Final hero render ---
final_rho = f.sum(dim=0)
final_excess = (final_rho - rho0).clamp(min=0).cpu().numpy()

paper_base = 1.0 - 0.06 * kappa_np
paper_grain = np.random.normal(0, 0.01, (Nx, Ny))
paper = paper_base + paper_grain

ink_opacity = np.clip(final_excess / max(final_excess.max() * 0.7, 0.001), 0, 1)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(paper, cmap='bone', vmin=0.88, vmax=1.05, interpolation='bilinear')
ax.imshow(final_excess, cmap=ink_cmap, vmin=0, vmax=max(final_excess.max()*0.75, 0.001),
          interpolation='bilinear', alpha=ink_opacity * 0.95)
ax.set_title(f"Final — Ink on Paper with Capillary Wicking (σ_cap={sigma_cap})",
             fontsize=15, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('final_render.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figures saved.")
