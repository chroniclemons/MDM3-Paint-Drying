import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

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
steps = 3000
snap_at = [100, 400, 700, 1100, 1600, 2200]

# LBM solvent
tau = 0.8
omega = 1.0 / tau
rho0 = 1.0

# Drop
drop_radius = 52
pin_radius = 72
spread_steps = 100
rho_drop = 1.20
phi_init = 0.04

# Evaporation
evap_coeff = 0.008
laplace_iters = 80
c_sat = 1.0
c_amb = 0.25

# Pigment transport / deposition
D_phi = 0.01
max_phi = 0.64

# Capillary pressure
gamma_cap = 0.1
mu_cap    = 1.0

# Shan-Chen surface tension
G_sc = -5.0
psi_rho0 = 0.0

# Ring support
pin_band = 1.5
pin_strength = 1.5

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
kappa = 0.05 + 0.06 * kappa
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
# Shan-Chen pseudopotential surface tension force
# ============================================================
def shan_chen_force(rho, G):
    excess = (rho - rho0).clamp(min=0)
    psi = 1.0 - torch.exp(-excess)

    grad_psi_x = torch.zeros_like(rho)
    grad_psi_y = torch.zeros_like(rho)
    for i in range(1, 9):
        shifted = torch.roll(psi, shifts=(-int(c[i, 0]), -int(c[i, 1])), dims=(0, 1))
        grad_psi_x += w[i] * c[i, 0] * shifted
        grad_psi_y += w[i] * c[i, 1] * shifted

    Fx = G * psi * grad_psi_x
    Fy = G * psi * grad_psi_y
    return Fx, Fy

# ============================================================
# Non-periodic helpers
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
# Geometry
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

# --- uneven blob radius as a function of angle ---
base_radius = 52.0
r_blob = (
    base_radius
    + 1.0 * torch.sin(3.0 * theta)
    + 2.0 * torch.sin(7.0 * theta + 0.8)
    + 1.5 * torch.cos(5.0 * theta - 0.4)
    + 0.8 * torch.sin(13.0 * theta + 1.2)
)

inside = r < r_blob

rho = torch.ones((Nx, Ny), device=device) * rho0
rho[inside] = rho_drop

u = torch.zeros((2, Nx, Ny), device=device)
f = equilibrium(rho, u)

deposit = torch.zeros((Nx, Ny), device=device)

# --- blob-shaped pin ---
r_pin_blob = r_blob * (pin_radius / base_radius)

pin_mask = ((r > r_pin_blob - pin_band) & (r < r_pin_blob + pin_band)).float()
wet_domain = (r <= r_pin_blob).float()
pin_wall = ((r > r_pin_blob) & (r <= r_pin_blob + 2)).float()

# ============================================================
# Stage 1 — simulation
# ============================================================
snapshots = []
profiles = []
stage1_solvent_snaps = {}

# Animation frames
anim_stride = 25
solvent_frames = []
deposit_frames = []

print(f"Stage 1: running {steps} solvent steps...")
for step in range(steps):
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = torch.clamp(f, min=0.0)
    rho = f.sum(dim=0).clamp(min=1e-6)
    momentum = torch.einsum('ic,ixy->cxy', c, f)
    u = momentum / (rho.unsqueeze(0) + 1e-10)

    pin_active = max(0.0, min(1.0, (step - spread_steps) / 50.0))
    active_pin_mask = pin_mask * pin_active

    if step % 500 == 0:
        u_mag = torch.sqrt(u[0]**2 + u[1]**2)
        h_check = (rho - rho0).clamp(min=0)
        u_radial = (u[0] * dx / (r + 1e-10) + u[1] * dy / (r + 1e-10))
        wet_cells = (h_check > 0.01).float()
        u_rad_mean = (u_radial * wet_cells).sum().item() / (wet_cells.sum().item() + 1e-10)
        print(f"           | u_max={u_mag.max().item():.6f} | u_radial_mean={u_rad_mean:.6f}")

    # hard pin
    kappa_eff = torch.clamp(kappa + pin_active * pin_wall, 0.0, 1.0)

    u_forced = torch.clamp(u, -0.1, 0.1)

    # Coarse-grained density force
    h_force = (rho - rho0).clamp(min=0)
    block_size = 8
    h_coarse = F.avg_pool2d(h_force.unsqueeze(0).unsqueeze(0), block_size, block_size)
    h_smooth = F.interpolate(h_coarse, size=(Nx, Ny), mode='bilinear', align_corners=False)[0, 0]
    gh_x, gh_y = grad(h_smooth)
    force_strength = 0.2
    Fx = -force_strength * gh_x
    Fy = -force_strength * gh_y

    u_forced[0] = u_forced[0] + Fx / (rho + 1e-10) * tau
    u_forced[1] = u_forced[1] + Fy / (rho + 1e-10) * tau
    u_forced = torch.clamp(u_forced, -0.1, 0.1)

    # collide/stream
    feq = equilibrium(rho, u_forced)
    f = f - omega * (f - feq)
    f = bounceback(f, kappa_eff)
    f = stream(f)

    # damp near wall
    if pin_active > 0.5:
        damp_band = ((r > r_pin_blob - 3) & (r <= r_pin_blob)).float()
        rho_local = f.sum(dim=0)
        u_local = torch.einsum('ic,ixy->cxy', c, f) / (rho_local.unsqueeze(0) + 1e-10)
        u_damped = u_local * (1.0 - 0.8 * damp_band).unsqueeze(0)
        f_relax = equilibrium(rho_local, u_damped)
        f = f * (1.0 - damp_band).unsqueeze(0) + f_relax * damp_band.unsqueeze(0)

    # evaporation
    if step > 300:
        rho_post = f.sum(dim=0)
        h_now = (rho_post - rho0).clamp(min=0)
        wet_now_evap = (h_now > 0.003).float()

        r_norm = (r / r_pin_blob).clamp(0, 0.999)
        evap_profile = 1.0 / (1.0 - r_norm**2)**0.2
        evap_profile = evap_profile / evap_profile.max()
        J = evap_coeff * evap_profile * h_now * wet_now_evap

        excess = (rho_post - rho0).clamp(min=1e-10)
        evap_frac = (J / excess).clamp(min=0.0)
        f = f - evap_frac.unsqueeze(0) * (
            f - equilibrium(torch.ones_like(rho_post) * rho0, torch.zeros_like(u))
        )
    else:
        J = torch.zeros_like(rho)
        evap_frac = torch.zeros_like(rho)

    # pigment deposit
    rho_now = f.sum(dim=0)
    h = (rho_now - rho0).clamp(min=0)
    dm_dep = J * phi_init
    deposit = deposit + dm_dep

    # debug
    if step % 500 == 0:
        h_dbg = (f.sum(dim=0) - rho0).clamp(min=0)
        h_center = h[cx0, cy0].item()
        edge_ring = ((r > r_pin_blob - 5) & (r < r_pin_blob)).float()
        h_edge = (h * edge_ring).sum().item() / (edge_ring.sum().item() + 1e-10)
        F_mag = torch.sqrt(Fx**2 + Fy**2).max().item()
        print(f"  step {step:4d} | solvent={h_dbg.sum().item():8.1f} | deposit={deposit.sum().item():8.4f}")
        print(f"           | J_max={J.max().item():.6f} | h_center={h_center:.4f} h_edge={h_edge:.4f} | force_max={F_mag:.6f}")

    # snapshots
    if step in snap_at:
        h_s1 = (f.sum(dim=0) - rho0).clamp(min=0)
        stage1_solvent_snaps[step] = h_s1.detach().cpu().numpy().copy()
        snapshots.append({
            "step": step,
            "solvent": h_s1.detach().cpu().numpy().copy(),
            "deposit": deposit.detach().cpu().numpy().copy(),
        })

    # animation frames
    if step % anim_stride == 0:
        h_anim = (f.sum(dim=0) - rho0).clamp(min=0)
        solvent_frames.append(h_anim.detach().cpu().numpy().copy())
        deposit_frames.append(deposit.detach().cpu().numpy().copy())

# freeze final shape
rho_final = f.sum(dim=0)
h_final = (rho_final - rho0).clamp(min=0)
footprint = (h_final > 1e-3).float()
print(f"Stage 1 done. Footprint cells: {int(footprint.sum().item())}")


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
    (0.97, 1.00, 0.94),
    (0.75, 0.95, 0.45),
    (0.35, 0.75, 0.15),
    (0.10, 0.30, 0.02)
])

# --- 1. Fiber field ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(kappa.cpu().numpy(), cmap='bone_r', interpolation='nearest')
ax.set_title("Paper Fiber Network", fontsize=13, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("01_fiber_network.png")
plt.show()

# --- 2. Solvent snapshots ---
if len(snapshots) > 0:
    vmax_solvent = max(snapshots[0]["solvent"].max() * 0.9, 1e-3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, snap in zip(axes.flat, snapshots):
        ax.imshow(snap["solvent"], cmap='Blues', vmin=0, vmax=vmax_solvent, interpolation='bilinear')
        ax.set_title(f"Solvent — step {snap['step']}")
        ax.axis('off')
    fig.suptitle("Solvent film height over time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig("02_solvent_snapshots.png")
    plt.show()

# --- 3. Deposit snapshots ---
if len(snapshots) > 0:
    vmax_dep = max(max(s["deposit"].max() for s in snapshots) * 0.9, 1e-3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, snap in zip(axes.flat, snapshots):
        ax.imshow(snap["deposit"], cmap=dep_cmap, vmin=0, vmax=vmax_dep, interpolation='bilinear')
        ax.set_title(f"Deposit — step {snap['step']}")
        ax.axis('off')
    fig.suptitle("Pigment deposited on paper", fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig("03_deposit_snapshots.png")
    plt.show()

# --- 4. Final deposit ---
deposit_np = deposit.detach().cpu().numpy()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(deposit_np, cmap=dep_cmap, vmin=0, interpolation='bilinear')
ax.set_title("Final pigment deposit", fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("04_final_deposit.png")
plt.show()


# --- 5. Composite on paper ---
paper_base = 1.0 - 0.05 * kappa_np
paper_grain = np.random.normal(0, 0.000001, (Nx, Ny))
paper = np.clip(paper_base + paper_grain, 0.86, 1.05)

dep_norm = deposit_np / max(deposit_np.max(), 1e-6)
dep_norm = deposit_np / max(deposit_np.max(), 1e-6)
dep_norm = dep_norm ** 0.8  # gamma correction — makes faint areas darker
# build RGB: paper is white, deposit is lime green
rgb = np.ones((Nx, Ny, 3))
rgb[:,:,0] = paper * (1.0 - 0.90 * dep_norm)  # was 0.65
rgb[:,:,1] = paper * (1.0 - 0.40 * dep_norm)  # was 0.15
rgb[:,:,2] = paper * (1.0 - 0.95 * dep_norm)  # was 0.85
rgb = np.clip(rgb, 0, 1)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb, interpolation='bilinear')
ax.set_title("Deposit on paper (composite)", fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
savefig("05_composite_on_paper.png")
plt.show()

# --- 6. Radial profiles ---
dep_t = torch.tensor(deposit_np, device=device)
prof_deposit = radial_profile(dep_t, cx0, cy0)
prof_h = radial_profile(h_final, cx0, cy0)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(prof_deposit, 'b-', linewidth=2, label='Deposit')
ax1.set_xlabel('Radial distance (cells)')
ax1.set_ylabel('Deposit (avg)', color='b')
ax2 = ax1.twinx()
ax2.plot(prof_h.detach().cpu().numpy() if hasattr(prof_h, 'detach') else prof_h,
         'r--', linewidth=1.5, label='Film height', alpha=0.7)
ax2.set_ylabel('Film height', color='r')
fig.suptitle("Radial profiles", fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
savefig("06_radial_profiles.png")
plt.show()

# ============================================================
# Animations
# ============================================================
print("\nGenerating animations...")

# --- 7. Solvent animation ---
vmax_s = max(solvent_frames[0].max() * 0.9, 1e-3)
fig_a, ax_a = plt.subplots(figsize=(6, 6))
im_a = ax_a.imshow(solvent_frames[0], cmap='Blues', vmin=0, vmax=vmax_s, interpolation='bilinear')
title_a = ax_a.set_title("Solvent — step 0", fontsize=13)
ax_a.axis('off')
plt.tight_layout()

def update_solvent(frame_idx):
    im_a.set_data(solvent_frames[frame_idx])
    title_a.set_text(f"Solvent — step {frame_idx * anim_stride}")
    return [im_a, title_a]

anim_solvent = FuncAnimation(fig_a, update_solvent, frames=len(solvent_frames), interval=80, blit=True)
solvent_gif_path = os.path.join(out_dir, "07_solvent_animation.gif")
anim_solvent.save(solvent_gif_path, writer=PillowWriter(fps=12))
print(f"Saved: {solvent_gif_path}")
plt.close(fig_a)

# --- 8. Deposit animation ---
vmax_d = max(deposit_frames[-1].max() * 0.9, 1e-3)
fig_b, ax_b = plt.subplots(figsize=(6, 6))
im_b = ax_b.imshow(deposit_frames[0], cmap=dep_cmap, vmin=0, vmax=vmax_d, interpolation='bilinear')
title_b = ax_b.set_title("Deposit — step 0", fontsize=13)
ax_b.axis('off')
plt.tight_layout()

def update_deposit(frame_idx):
    im_b.set_data(deposit_frames[frame_idx])
    title_b.set_text(f"Deposit — step {frame_idx * anim_stride}")
    return [im_b, title_b]

anim_deposit = FuncAnimation(fig_b, update_deposit, frames=len(deposit_frames), interval=80, blit=True)
deposit_gif_path = os.path.join(out_dir, "08_deposit_animation.gif")
anim_deposit.save(deposit_gif_path, writer=PillowWriter(fps=12))
print(f"Saved: {deposit_gif_path}")
plt.close(fig_b)

# --- 9. Side-by-side animation (solvent + deposit) ---
fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(12, 6))
im_c1 = ax_c1.imshow(solvent_frames[0], cmap='Blues', vmin=0, vmax=vmax_s, interpolation='bilinear')
ax_c1.set_title("Solvent")
ax_c1.axis('off')
im_c2 = ax_c2.imshow(deposit_frames[0], cmap=dep_cmap, vmin=0, vmax=vmax_d, interpolation='bilinear')
ax_c2.set_title("Deposit")
ax_c2.axis('off')
title_c = fig_c.suptitle("Step 0", fontsize=14, fontweight='bold')
plt.tight_layout()

def update_both(frame_idx):
    im_c1.set_data(solvent_frames[frame_idx])
    im_c2.set_data(deposit_frames[frame_idx])
    title_c.set_text(f"Step {frame_idx * anim_stride}")
    return [im_c1, im_c2, title_c]
    
print(f"Saved: {both_gif_path}")
plt.close(fig_c)


print("\nAll figures and animations saved.")
print("Key outputs: 04_final_deposit.png, 05_composite_on_paper.png")
print("Animations: 07_solvent_animation.gif, 08_deposit_animation.gif, 09_solvent_deposit_animation.gif")