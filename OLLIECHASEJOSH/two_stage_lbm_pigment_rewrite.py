
"""
Two-stage LBM ink-on-paper
Stage 1:
    solvent only (LBM + evaporation + pinning)
Stage 2:
    pigment only, initialized from the final solvent footprint

Main simplification:
- solvent and pigment no longer fight each other inside the same loop
- pigment starts from the final solvent shape / thickness
- ring formation is driven by a frozen footprint plus a simple drying flow
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
out_dir = "lbm_ring_outputs_two_stage"
os.makedirs(out_dir, exist_ok=True)
print(f"Saving figures to: {os.path.abspath(out_dir)}")

# ============================================================
# Parameters
# ============================================================
Nx, Ny = 712, 712

# Stage lengths
solvent_steps = 1000
pigment_steps = 700

solvent_snap_at = [0, 200, 500, 1000]
pigment_snap_at = [0, 80, 180, 320, 500, 699]

# LBM solvent
tau = 0.88
omega = 1.0 / tau
rho0 = 1.0

# Drop
drop_radius = 52
pin_radius = 63
spread_steps = 100
rho_drop = 1.20

# Pigment loading
phi_init = 0.009
max_phi = 1.0

# Evaporation
evap_coeff = 0.007
laplace_iters = 80
c_sat = 1.0
c_amb = 0.25

# Pigment stage tuning
D_phi = 0.028
deposit_rate = 0.016
thin_film_thresh = 0.055
ring_speed = 0.06
dry_rate = 0.0025
rim_dry_boost = 2.0

# Pinning
pin_band = 2.5
pin_strength = 1.3

# Wetness cutoff
wet_cutoff = 0.003

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
def generate_fiber_network(nx, ny, n_fibers=300000, fiber_length=15, fiber_width=1, seed=42):
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
kappa = 0.05 + 0.55 * kappa
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
# Non-periodic field helpers
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

def blur3(field, rounds=1):
    out = field
    for _ in range(rounds):
        f = pad_rep(out)
        out = (
            4.0 * f[1:-1, 1:-1] +
            2.0 * (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2]) +
            (f[2:, 2:] + f[2:, :-2] + f[:-2, 2:] + f[:-2, :-2])
        ) / 16.0
    return out

# ============================================================
# Evaporation
# ============================================================
def compute_evaporation_laplace(rho, rho0, kappa, evap_coeff, pin_mask, c_sat=1.0, c_amb=0.3, iters=60):
    excess = (rho - rho0).clamp(min=0)
    wet = excess > wet_cutoff

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

    pore_factor = 1.0 - 0.35 * kappa
    rim_boost = 1.0 + 1.2 * pin_mask
    J = evap_coeff * J * wet_f * pore_factor * rim_boost
    J = torch.minimum(J, 0.08 * excess)
    return J, wet

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

pin_mask = ((r > pin_radius - pin_band) & (r < pin_radius + pin_band)).float()

# ============================================================
# STAGE 1 — solvent only
# ============================================================
solvent_snapshots = []
solvent_profiles = []

print(f"Running solvent stage for {solvent_steps} steps...")
for step in range(solvent_steps):
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = torch.clamp(f, min=0.0)

    rho = f.sum(dim=0).clamp(min=1e-6)
    momentum = torch.einsum('ic,ixy->cxy', c, f)
    u = momentum / (rho.unsqueeze(0) + 1e-10)

    pin_active = max(0.0, min(1.0, (step - spread_steps) / 50.0))
    active_pin_mask = pin_mask * pin_active

    # gentle pinning force on solvent thickness
    h = (rho - rho0).clamp(min=0)
    gx_h, gy_h = grad(h)
    rim_force_scale = pin_strength * active_pin_mask * (h > 0.02).float()
    Fx = -rim_force_scale * gx_h
    Fy = -rim_force_scale * gy_h

    u_forced = u.clone()
    u_forced[0] = u_forced[0] + Fx / (rho + 1e-10)
    u_forced[1] = u_forced[1] + Fy / (rho + 1e-10)
    u_forced = torch.clamp(u_forced, -0.1, 0.1)

    feq = equilibrium(rho, u_forced)
    f = f - omega * (f - feq)
    f = bounceback(f, kappa)
    f = stream(f)

    rho_post = f.sum(dim=0)
    evap_scale = 0.2 + 0.8 * pin_active
    J, wet = compute_evaporation_laplace(
        rho_post, rho0, kappa, evap_coeff * evap_scale, active_pin_mask,
        c_sat=c_sat, c_amb=c_amb, iters=laplace_iters
    )

    evap_frac = torch.clamp(J / (rho_post + 1e-10), 0.0, 0.20)
    f = f * (1.0 - evap_frac.unsqueeze(0))

    if step in solvent_snap_at:
        rho_snap = f.sum(dim=0)
        h_snap = (rho_snap - rho0).clamp(min=0)

        solvent_snapshots.append({
            "step": step,
            "solvent": h_snap.detach().cpu().numpy().copy(),
            "J": J.detach().cpu().numpy().copy(),
        })

        solvent_profiles.append({
            "step": step,
            "solvent": radial_profile(h_snap, cx0, cy0),
        })

        print(
            f"[solvent] step {step:4d} | "
            f"solvent mass={h_snap.sum().item():10.3f} | "
            f"wet cells={(h_snap > wet_cutoff).sum().item():8d}"
        )

print("Solvent stage done.")

# final solvent footprint
rho_final = f.sum(dim=0)
h_final = (rho_final - rho0).clamp(min=0)
wet_final = h_final > wet_cutoff

# use one final evaporation field for diagnostics / pigment biasing
J_final, _ = compute_evaporation_laplace(
    rho_final, rho0, kappa, evap_coeff, pin_mask,
    c_sat=c_sat, c_amb=c_amb, iters=laplace_iters
)

# ============================================================
# STAGE 2 — pigment only on frozen solvent footprint
# ============================================================
print(f"Running pigment stage for {pigment_steps} steps...")

# initial mobile pigment mass comes from the final solvent footprint
h_ref = h_final.clone()
wet_ref = wet_final.clone()
h = h_ref.clone()

# mobile pigment mass and deposit
m = phi_init * h_ref * wet_ref.float()
deposit = torch.zeros_like(h_ref)

# build a simple outward drift field from center, masked by final footprint
rr = torch.sqrt(dx**2 + dy**2) + 1e-6
ux_base = ring_speed * (dx / rr)
uy_base = ring_speed * (dy / rr)

# use final solvent shape to bias drying and deposition near the edge
h_norm = h_ref / (h_ref.max() + 1e-8)
edge_indicator = torch.sqrt(grad(blur3(wet_ref.float(), rounds=3))[0]**2 +
                            grad(blur3(wet_ref.float(), rounds=3))[1]**2)
edge_indicator = edge_indicator / (edge_indicator.max() + 1e-8)
rim_weight = torch.clamp(0.35 + 2.2 * edge_indicator + (1.0 - h_norm), 0.2, 2.8)

pigment_snapshots = []
pigment_profiles = []

for step in range(pigment_steps):
    wet = h > wet_cutoff
    wet_f = wet.float()

    # velocity only exists inside the surviving wet region
    fiber_drag = 1.0 - 0.25 * kappa
    ux = ux_base * rim_weight * fiber_drag * wet_f
    uy = uy_base * rim_weight * fiber_drag * wet_f
    ux = torch.clamp(ux, -0.08, 0.08)
    uy = torch.clamp(uy, -0.08, 0.08)

    # transport mobile pigment mass
    m_adv = advect_upwind(m, ux, uy)
    m_adv = m_adv + D_phi * laplacian(m)
    m_adv = torch.clamp(m_adv, min=0.0)
    m_adv = m_adv * wet_f

    # deposition becomes stronger as film thins, especially near rim
    thin_factor = torch.clamp((thin_film_thresh - h) / thin_film_thresh, 0.0, 1.0)
    fiber_factor = 0.85 + 0.15 * kappa
    dep_prob = deposit_rate * (0.20 + 1.6 * thin_factor) * rim_weight * fiber_factor
    dep_prob = torch.clamp(dep_prob, 0.0, 0.45)

    dm_dep = dep_prob * m_adv * wet_f
    m = torch.clamp(m_adv - dm_dep, min=0.0)
    deposit = deposit + dm_dep

    # prescribed drying on the frozen geometry
    local_dry = dry_rate * (0.65 + rim_dry_boost * edge_indicator)
    h = torch.clamp(h - local_dry * wet_f, min=0.0)

    # once dry, remaining mobile mass deposits there
    newly_dry = (h <= wet_cutoff) & (m > 0)
    deposit = deposit + m * newly_dry.float()
    m = m * (h > wet_cutoff).float()

    if step in pigment_snap_at:
        phi = torch.zeros_like(h)
        wet_now = h > wet_cutoff
        phi[wet_now] = m[wet_now] / (h[wet_now] + 1e-10)
        phi = torch.clamp(phi, 0.0, max_phi)

        pigment_snapshots.append({
            "step": step,
            "solvent": h.detach().cpu().numpy().copy(),
            "mobile": m.detach().cpu().numpy().copy(),
            "deposit": deposit.detach().cpu().numpy().copy(),
            "phi": phi.detach().cpu().numpy().copy(),
        })

        pigment_profiles.append({
            "step": step,
            "mobile": radial_profile(m, cx0, cy0),
            "deposit": radial_profile(deposit, cx0, cy0),
        })

        print(
            f"[pigment] step {step:4d} | "
            f"mobile={m.sum().item():10.5f} | "
            f"deposit={deposit.sum().item():10.5f} | "
            f"wet cells={(h > wet_cutoff).sum().item():8d}"
        )

print("Pigment stage done.")

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

# Fiber field
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(kappa.cpu().numpy(), cmap='bone_r', interpolation='nearest')
ax.set_title("Paper Fiber Network (κ field)", fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("01_fiber_network.png")
plt.show()

# Solvent snapshots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, solvent_snapshots):
    img = snap["solvent"]
    vmax = max(img.max() * 0.9, 1e-3)
    ax.imshow(img, cmap='Blues', vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Solvent step {snap['step']}")
    ax.axis('off')
fig.suptitle("Stage 1 — solvent film", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("02_solvent_snapshots.png")
plt.show()

# Evaporation snapshots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, solvent_snapshots):
    img = snap["J"]
    vmax = max(img.max() * 0.95, 1e-6)
    ax.imshow(img, cmap='magma', vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Evaporation step {snap['step']}")
    ax.axis('off')
fig.suptitle("Stage 1 — evaporation flux J", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("03_evaporation_flux_snapshots.png")
plt.show()

# Final solvent footprint used to seed pigment
fig, ax = plt.subplots(figsize=(8, 8))
img = h_ref.detach().cpu().numpy()
ax.imshow(img, cmap='Blues', interpolation='bilinear')
ax.set_title("Stage handoff — final solvent footprint", fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("04_stage_handoff_final_solvent.png")
plt.show()

# Pigment mobile mass snapshots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, pigment_snapshots):
    img = snap["mobile"]
    vmax = max(img.max() * 0.9, 1e-4)
    ax.imshow(img, cmap=ink_cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Pigment step {snap['step']}")
    ax.axis('off')
fig.suptitle("Stage 2 — mobile pigment mass", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("05_mobile_pigment_snapshots.png")
plt.show()

# Deposited pigment snapshots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, snap in zip(axes.flat, pigment_snapshots):
    img = snap["deposit"]
    vmax = max(img.max() * 0.9, 1e-4)
    ax.imshow(img, cmap=dep_cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax.set_title(f"Deposit step {snap['step']}")
    ax.axis('off')
fig.suptitle("Stage 2 — deposited pigment", fontsize=14, fontweight='bold')
plt.tight_layout()
savefig("06_deposit_snapshots.png")
plt.show()

# Final render
final_solvent = pigment_snapshots[-1]["solvent"]
final_mobile = pigment_snapshots[-1]["mobile"]
final_deposit = pigment_snapshots[-1]["deposit"]

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
    alpha=np.clip(dep_norm * 1.15, 0, 1),
    interpolation='bilinear'
)
ax.set_title("Final — two-stage deposited pigment only", fontsize=15, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("07_final_deposit_only.png")
plt.show()

# Deposit radial profiles
plt.figure(figsize=(9, 5))
for p in pigment_profiles:
    plt.plot(p["deposit"], label=f"deposit step {p['step']}")
plt.xlim(0, 90)
plt.xlabel("Radius (pixels)")
plt.ylabel("Azimuthal mean deposited pigment")
plt.title("Stage 2 — deposited pigment radial profile")
plt.legend()
plt.tight_layout()
savefig("08_deposit_radial_profiles.png")
plt.show()

# Mobile radial profiles
plt.figure(figsize=(9, 5))
for p in pigment_profiles:
    plt.plot(p["mobile"], label=f"mobile step {p['step']}")
plt.xlim(0, 90)
plt.xlabel("Radius (pixels)")
plt.ylabel("Azimuthal mean mobile pigment")
plt.title("Stage 2 — mobile pigment radial profile")
plt.legend()
plt.tight_layout()
savefig("09_mobile_radial_profiles.png")
plt.show()

print("\nAll figures saved.")
print("Check 04_stage_handoff_final_solvent.png, 06_deposit_snapshots.png, and 07_final_deposit_only.png first.")
