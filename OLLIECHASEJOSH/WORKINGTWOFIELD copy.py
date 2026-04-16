# -*- coding: utf-8 -*-
"""
Two-field LBM ink-on-paper — simplified coupling version

- rho      : solvent density (LBM, D2Q9)
- phi      : mobile pigment concentration
- deposit  : immobile deposited pigment

Main idea:
- solvent moves using LBM
- pigment is advected by solvent velocity u
- pigment deposits where solvent is drying / thinning
- no extra hard-coded evaporation-flow solve for pigment
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Output folder
# ============================================================
out_dir = "lbm_ring_outputs"
os.makedirs(out_dir, exist_ok=True)
print(f"Saving figures to: {os.path.abspath(out_dir)}")

# ============================================================
# Parameters
# ============================================================
Nx, Ny = 712, 712
steps = 2500
snap_at = [0, 200, 500, 1000, 1800, 2499]

# LBM solvent
tau = 0.88
omega = 1.0 / tau
rho0 = 1.0

# Drop
drop_radius = 52
pin_radius = 63
spread_steps = 100
rho_drop = 1.20
phi_init = 0.009

# Evaporation
evap_coeff = 0.007
laplace_iters = 80
c_sat = 1.0
c_amb = 0.25

# Pigment transport / deposition
D_phi = 0.0015
deposit_rate = 0.85      # drying-to-deposit conversion strength
thin_film_thresh = 0.09
max_phi = 1.0

# Ring support
pin_band = 2.5
pin_strength = 0.0       # set back >0 later if you want to test pinning

# ============================================================
# D2Q9
# ============================================================
c = torch.tensor([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=torch.float32, device=device)

w = torch.tensor([
    4/9, 1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
], dtype=torch.float32, device=device)

opposite = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long, device=device)

# ============================================================
# Figure saving helper
# ============================================================
def savefig(name, dpi=180):
    path = os.path.join(out_dir, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

# ============================================================
# Fiber network
# ============================================================
def generate_fiber_network(nx, ny, n_fibers=90000, fiber_length=15, fiber_width=1, seed=42):
    np.random.seed(seed)
    field = np.zeros((nx, ny), dtype=np.float32)

    for _ in range(n_fibers):
        x0 = np.random.uniform(0, nx)
        y0 = np.random.uniform(0, ny)
        theta = np.random.uniform(0, np.pi)

        length = max(5, np.random.normal(fiber_length, fiber_length * 0.25))
        width = max(1, int(np.random.normal(fiber_width + 0.2, 0.25)))

        n_steps = int(length)
        ts = np.linspace(-length / 2, length / 2, n_steps)

        xs = (x0 + ts * np.cos(theta) + np.random.randn(n_steps) * 0.15).astype(int) % nx
        ys = (y0 + ts * np.sin(theta) + np.random.randn(n_steps) * 0.15).astype(int) % ny

        hw = width // 2
        for dx in range(-hw, hw + 1):
            for dy in range(-hw, hw + 1):
                if dx * dx + dy * dy <= (width / 2 + 0.5) ** 2:
                    field[(xs + dx) % nx, (ys + dy) % ny] += 1.0

    field += 0.01 * np.random.rand(nx, ny).astype(np.float32)
    field = field / max(field.max(), 1e-8)
    return field

print("Generating fiber network...")
kappa_np = generate_fiber_network(Nx, Ny)
kappa = torch.tensor(kappa_np, dtype=torch.float32, device=device)
kappa = 0.03 + 0.22 * kappa
print(f"kappa range: [{kappa.min().item():.3f}, {kappa.max().item():.3f}]")

# ============================================================
# Solvent LBM helpers
# ============================================================
def equilibrium(rho, u):
    cu = torch.einsum('ic,cxy->ixy', c, u)
    u_sq = 0.5 * (u[0]**2 + u[1]**2)
    return w.view(9, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 3.0 * u_sq.unsqueeze(0)
    )

def stream(f):
    out = torch.zeros_like(f)
    for i in range(9):
        out[i] = torch.roll(f[i], shifts=(int(c[i, 0]), int(c[i, 1])), dims=(0, 1))
    return out

def bounceback(f, kappa):
    return (1.0 - kappa).unsqueeze(0) * f + kappa.unsqueeze(0) * f[opposite]

# ============================================================
# Non-periodic helpers for pigment transport
# ============================================================
def pad_rep(field):
    return F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')[0, 0]

def grad(field):
    f = pad_rep(field)
    gx = 0.5 * (f[2:, 1:-1] - f[:-2, 1:-1])
    gy = 0.5 * (f[1:-1, 2:] - f[1:-1, :-2])
    return gx, gy

def laplacian(field):
    f = pad_rep(field)
    return (
        f[2:, 1:-1] + f[:-2, 1:-1] +
        f[1:-1, 2:] + f[1:-1, :-2] -
        4.0 * f[1:-1, 1:-1]
    )

def advect_upwind(m, ux, uy):
    f = pad_rep(m)

    m_xp = f[2:, 1:-1]
    m_xm = f[:-2, 1:-1]
    m_yp = f[1:-1, 2:]
    m_ym = f[1:-1, :-2]

    dm_dx = torch.where(ux > 0, m - m_xm, m_xp - m)
    dm_dy = torch.where(uy > 0, m - m_ym, m_yp - m)

    return m - ux * dm_dx - uy * dm_dy

# ============================================================
# Evaporation
# ============================================================
def compute_evaporation_laplace(rho, rho0, kappa, evap_coeff, pin_mask, c_sat=1.0, c_amb=0.3, iters=60):
    excess = (rho - rho0).clamp(min=0)
    wet = excess > 0.003

    if wet.sum() < 5:
        return torch.zeros_like(rho), wet

    c_vapor = torch.ones_like(rho) * c_amb
    c_vapor[wet] = c_sat
    wet_f = wet.float()

    for _ in range(iters):
        c_new = 0.25 * (
            torch.roll(c_vapor, 1, 0) + torch.roll(c_vapor, -1, 0) +
            torch.roll(c_vapor, 1, 1) + torch.roll(c_vapor, -1, 1)
        )
        c_new = c_new * (1.0 - wet_f) + c_sat * wet_f
        c_vapor = c_new

    gx, gy = grad(c_vapor)
    J = torch.sqrt(gx**2 + gy**2)

    pore_factor = 1.0 - 0.25 * kappa
    rim_boost = 1.0 + 1.2 * pin_mask
    J = evap_coeff * J * wet_f * pore_factor * rim_boost
    J = torch.minimum(J, 0.08 * excess)
    return J, wet

# ============================================================
# Pigment transport
# ============================================================
def advect_diffuse_phi(phi, deposit, rho_old, rho_new, u, J, kappa, pin_active):
    """
    Pigment model:
    - mobile pigment mass m = phi * h_old
    - advect m with solvent velocity u
    - deposit pigment where solvent dried this step
    """
    h_old = (rho_old - rho0).clamp(min=0)
    h_new = (rho_new - rho0).clamp(min=0)

    wet_old = h_old > 1e-8
    wet_new = h_new > 1e-8
    wet_new_f = wet_new.float()

    # mobile pigment mass before this step
    m_old = phi * h_old

    # move with solvent velocity
    ux = torch.clamp(u[0], -0.12, 0.12)
    uy = torch.clamp(u[1], -0.12, 0.12)

    m_adv = advect_upwind(m_old, ux, uy)

    # mild diffusion only
    m_adv = m_adv + D_phi * laplacian(m_old)
    m_adv = torch.clamp(m_adv, min=0.0)

    # pigment cannot remain mobile where there is no solvent now
    m_adv = m_adv * wet_new_f

    # deposition based on drying/thinning
    drying = (h_old - h_new).clamp(min=0.0)
    evap_fraction = drying / (h_old + 1e-10)

    thin_factor = torch.clamp((thin_film_thresh - h_new) / thin_film_thresh, 0.0, 1.0)
    J_norm = J / (J.max() + 1e-10)

    # main deposition trigger = solvent loss
    dep_frac = deposit_rate * (
        0.65 * evap_fraction +
        0.25 * thin_factor**2 +
        0.10 * J_norm
    )
    dep_frac = dep_frac * wet_old.float()
    dep_frac = torch.clamp(dep_frac, 0.0, 1.0)

    dm_dep = dep_frac * m_adv
    dm_dep = torch.minimum(dm_dep, m_adv)

    deposit = deposit + dm_dep
    m_mobile = m_adv - dm_dep

    phi_new = torch.zeros_like(phi)
    phi_new[wet_new] = m_mobile[wet_new] / (h_new[wet_new] + 1e-10)
    phi_new = torch.clamp(phi_new, 0.0, max_phi)

    return phi_new, deposit

# ============================================================
# Diagnostics
# ============================================================
def radial_profile(field, cx, cy, n_bins=120):
    yy, xx = torch.meshgrid(
        torch.arange(field.shape[0], device=field.device),
        torch.arange(field.shape[1], device=field.device),
        indexing='ij'
    )
    rr = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
    rr_i = torch.clamp(rr.long(), 0, n_bins - 1)

    sums = torch.zeros(n_bins, device=field.device)
    counts = torch.zeros(n_bins, device=field.device)

    sums.scatter_add_(0, rr_i.flatten(), field.flatten())
    counts.scatter_add_(0, rr_i.flatten(), torch.ones_like(field).flatten())

    return (sums / (counts + 1e-10)).detach().cpu().numpy()

# ============================================================
# Initial condition
# ============================================================
Y, X = torch.meshgrid(
    torch.arange(Nx, device=device, dtype=torch.float32),
    torch.arange(Ny, device=device, dtype=torch.float32),
    indexing='ij'
)

cx0, cy0 = Nx // 2, Ny // 2
dx = X - cy0
dy = Y - cx0
theta = torch.atan2(dy, dx)
r = torch.sqrt(dx**2 + dy**2)

# uneven blob
base_radius = 52.0
r_blob = (
    base_radius
    + 1.0 * torch.sin(3.0 * theta)
    + 2.0 * torch.sin(7.0 * theta + 0.8)
    + 1.5 * torch.cos(5.0 * theta - 0.4)
)

inside = r < r_blob

rho = torch.ones((Nx, Ny), device=device) * rho0
rho[inside] = rho_drop

u = torch.zeros((2, Nx, Ny), device=device)
f = equilibrium(rho, u)

phi = torch.zeros((Nx, Ny), device=device)
phi[inside] = phi_init

deposit = torch.zeros((Nx, Ny), device=device)
pin_mask = ((r > pin_radius - pin_band) & (r < pin_radius + pin_band)).float()

# ============================================================
# Main loop
# ============================================================
snapshots = []
profiles = []

print(f"Running {steps} steps...")
for step in range(steps):
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = torch.clamp(f, min=0.0)

    # current solvent fields
    rho = f.sum(dim=0).clamp(min=1e-6)
    momentum = torch.einsum('ic,ixy->cxy', c, f)
    u = momentum / (rho.unsqueeze(0) + 1e-10)

    # pin activation
    pin_active = max(0.0, min(1.0, (step - spread_steps) / 50.0))
    active_pin_mask = pin_mask * pin_active

    # gentle pinning force on solvent only
    h = (rho - rho0).clamp(min=0)
    gx_h, gy_h = grad(h)
    rim_force_scale = pin_strength * active_pin_mask * (h > 0.02).float()
    Fx = -rim_force_scale * gx_h
    Fy = -rim_force_scale * gy_h

    u_forced = u.clone()
    u_forced[0] = u_forced[0] + Fx / (rho + 1e-10)
    u_forced[1] = u_forced[1] + Fy / (rho + 1e-10)
    u_forced = torch.clamp(u_forced, -0.1, 0.1)

    # solvent collide / stream
    feq = equilibrium(rho, u_forced)
    f = f - omega * (f - feq)
    f = bounceback(f, kappa)
    f = stream(f)

    # solvent state before evaporation
    rho_before_evap = f.sum(dim=0)

    # evaporation
    evap_scale = 0.2 + 0.8 * pin_active
    J, wet = compute_evaporation_laplace(
        rho_before_evap, rho0, kappa, evap_coeff * evap_scale, active_pin_mask,
        c_sat=c_sat, c_amb=c_amb, iters=laplace_iters
    )

    evap_frac = torch.clamp(J / (rho_before_evap + 1e-10), 0.0, 0.20)
    f = f * (1.0 - evap_frac.unsqueeze(0))

    # solvent state after evaporation
    rho_after_evap = f.sum(dim=0)
    momentum = torch.einsum('ic,ixy->cxy', c, f)
    u = momentum / (rho_after_evap.unsqueeze(0) + 1e-10)

    # pigment follows solvent, deposit where solvent dries
    phi, deposit = advect_diffuse_phi(
        phi, deposit,
        rho_before_evap, rho_after_evap,
        u, J, kappa, pin_active
    )

    if step in snap_at:
        rho_snap = f.sum(dim=0)
        h_snap = (rho_snap - rho0).clamp(min=0)
        mobile_pigment = phi * h_snap

        snapshots.append({
            "step": step,
            "solvent": h_snap.detach().cpu().numpy().copy(),
            "mobile": mobile_pigment.detach().cpu().numpy().copy(),
            "deposit": deposit.detach().cpu().numpy().copy(),
            "J": J.detach().cpu().numpy().copy(),
        })

        profiles.append({
            "step": step,
            "mobile": radial_profile(mobile_pigment, cx0, cy0),
            "deposit": radial_profile(deposit, cx0, cy0),
        })

        total_mobile = mobile_pigment.sum().item()
        total_deposit = deposit.sum().item()

        print(
            f"step {step:4d} | "
            f"solvent={h_snap.sum().item():10.3f} | "
            f"mobile pigment={total_mobile:10.6f} | "
            f"deposit={total_deposit:10.6f} | "
            f"total pigment={total_mobile + total_deposit:10.6f}"
        )

print("Done.")

# ============================================================
# Visualization
# ============================================================
ink_cmap = LinearSegmentedColormap.from_list('ink', [
    (1.0, 0.98, 0.94),
    (0.75, 0.76, 0.82),
    (0.38, 0.33, 0.50),
    (0.08, 0.05, 0.12)
])

dep_cmap = LinearSegmentedColormap.from_list('dep', [
    (0.97, 0.98, 1.00),
    (0.70, 0.80, 0.92),
    (0.25, 0.45, 0.75),
    (0.04, 0.08, 0.25)
])

# --- Fiber field ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(kappa.cpu().numpy(), cmap='bone_r', interpolation='nearest')
ax.set_title("Paper Fiber Network (κ field)", fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("01_fiber_network.png")
plt.show()

# --- Solvent snapshots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, snapshots):
    img = snap["solvent"]
    vmax = max(img.max() * 0.9, 1e-3)
    ax.imshow(img, cmap='Blues', vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Solvent step {snap['step']}")
    ax.axis('off')
fig.suptitle("Solvent film", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("02_solvent_snapshots.png")
plt.show()

# --- Mobile pigment snapshots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, snapshots):
    img = snap["mobile"]
    vmax = max(img.max() * 0.9, 1e-4)
    ax.imshow(img, cmap=ink_cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Mobile pigment step {snap['step']}")
    ax.axis('off')
fig.suptitle("Pigment still in the liquid", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("03_mobile_pigment_snapshots.png")
plt.show()

# --- Deposited pigment snapshots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, snapshots):
    img = snap["deposit"]
    vmax = max(img.max() * 0.9, 1e-4)
    ax.imshow(img, cmap=dep_cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Deposit step {snap['step']}")
    ax.axis('off')
fig.suptitle("Deposited pigment on paper", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("04_deposit_snapshots.png")
plt.show()

# --- Evaporation snapshots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, snapshots):
    img = snap["J"]
    vmax = max(img.max() * 0.95, 1e-6)
    ax.imshow(img, cmap='magma', vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Evaporation flux step {snap['step']}")
    ax.axis('off')
fig.suptitle("Evaporation flux J", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("05_evaporation_flux_snapshots.png")
plt.show()

# --- Final deposit-only render ---
final_solvent = snapshots[-1]["solvent"]
final_mobile = snapshots[-1]["mobile"]
final_deposit = snapshots[-1]["deposit"]

paper_base = 1.0 - 0.0 * kappa_np
paper_grain = np.random.normal(0, 0.000001, (Nx, Ny))
paper = np.clip(paper_base + paper_grain, 0.86, 1.05)

dep_norm = final_deposit / max(final_deposit.max(), 1e-8)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(paper, cmap='bone', vmin=0.88, vmax=1.05, interpolation='bilinear')
ax.imshow(
    final_deposit,
    cmap=dep_cmap,
    vmin=0,
    vmax=max(final_deposit.max() * 0.85, 1e-4),
    alpha=np.clip(dep_norm * 1.1, 0, 1),
    interpolation='bilinear'
)
ax.set_title("Final — deposited pigment only", fontsize=15, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("06_final_deposit_only.png")
plt.show()

# --- Final mobile pigment render ---
mob_norm = final_mobile / max(final_mobile.max(), 1e-8)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(paper, cmap='bone', vmin=0.88, vmax=1.05, interpolation='bilinear')
ax.imshow(
    final_mobile,
    cmap=ink_cmap,
    vmin=0,
    vmax=max(final_mobile.max() * 0.9, 1e-4),
    alpha=np.clip(mob_norm, 0, 1),
    interpolation='bilinear'
)
ax.set_title("Final — remaining mobile pigment", fontsize=15, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("07_final_mobile_pigment.png")
plt.show()

# --- Deposit radial profiles ---
plt.figure(figsize=(9, 5))
for p in profiles:
    plt.plot(p["deposit"], label=f"deposit step {p['step']}")
plt.xlim(0, 80)
plt.xlabel("Radius (pixels)")
plt.ylabel("Azimuthal mean deposited pigment")
plt.title("Deposited pigment radial profile")
plt.legend()
plt.tight_layout()
savefig("08_deposit_radial_profiles.png")
plt.show()

# --- Mobile radial profiles ---
plt.figure(figsize=(9, 5))
for p in profiles:
    plt.plot(p["mobile"], label=f"mobile step {p['step']}")
plt.xlim(0, 80)
plt.xlabel("Radius (pixels)")
plt.ylabel("Azimuthal mean mobile pigment")
plt.title("Mobile pigment radial profile")
plt.legend()
plt.tight_layout()
savefig("09_mobile_radial_profiles.png")
plt.show()

print("\nAll figures saved.")
print("Check 02_solvent_snapshots.png, 03_mobile_pigment_snapshots.png, and 04_deposit_snapshots.png first.")