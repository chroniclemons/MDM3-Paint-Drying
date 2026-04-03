# ================================================================
# Physically-Based Coffee Ring Simulation (FIXED)
# Thin Film + Vapor Diffusion + Particle Transport
# Proper Contact Line Pinning (no hacks)
# ================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
Nx, Ny = 256, 256
dx = 1.0
dt = 0.02

steps = 1500
snap_at = [0, 200, 500, 1000, 1499]

R = 40
h0 = 1.0
evap_coeff = 0.002

# ─────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────
Y, X = torch.meshgrid(
    torch.arange(Nx, device=device),
    torch.arange(Ny, device=device),
    indexing='ij'
)

cx, cy = Nx // 2, Ny // 2
r = torch.sqrt((X - cx)**2 + (Y - cy)**2)

# Contact line (pinned footprint)
footprint = r < R

# ─────────────────────────────────────────────
# Fiber Network
# ─────────────────────────────────────────────
def generate_fiber_network(nx, ny, n_fibers=1500, seed=0):
    np.random.seed(seed)
    field = np.zeros((nx, ny), dtype=np.float32)

    for _ in range(n_fibers):
        x0 = np.random.uniform(0, nx)
        y0 = np.random.uniform(0, ny)
        theta = np.random.uniform(0, np.pi)
        length = np.random.uniform(10, 30)

        for t in np.linspace(-length/2, length/2, int(length)):
            x = int((x0 + t*np.cos(theta)) % nx)
            y = int((y0 + t*np.sin(theta)) % ny)
            field[x, y] += 1

    field /= max(field.max(), 1)
    return torch.tensor(field, device=device)

kappa = generate_fiber_network(Nx, Ny)
kappa = 0.1 + 0.5 * kappa

# ─────────────────────────────────────────────
# Initial droplet (spherical cap)
# ─────────────────────────────────────────────
h = h0 * torch.clamp(1 - (r / R)**2, min=0)

# particle concentration
c = torch.zeros_like(h)
c[footprint] = 1.0

deposit = torch.zeros_like(h)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def laplacian(f):
    return (
        torch.roll(f, 1, 0) + torch.roll(f, -1, 0) +
        torch.roll(f, 1, 1) + torch.roll(f, -1, 1) -
        4*f
    ) / dx**2

def gradient(f):
    gx = (torch.roll(f, -1, 0) - torch.roll(f, 1, 0)) / (2*dx)
    gy = (torch.roll(f, -1, 1) - torch.roll(f, 1, 1)) / (2*dx)
    return gx, gy

# ─────────────────────────────────────────────
# Vapor diffusion (Laplace solver)
# ─────────────────────────────────────────────
def compute_evaporation(h, iters=40):
    wet = h > 1e-4

    c_v = torch.ones_like(h) * 0.3
    c_v[wet] = 1.0

    wet_f = wet.float()

    for _ in range(iters):
        c_new = 0.25 * (
            torch.roll(c_v, 1, 0) + torch.roll(c_v, -1, 0) +
            torch.roll(c_v, 1, 1) + torch.roll(c_v, -1, 1)
        )
        c_new = c_new * (1 - wet_f) + wet_f * 1.0
        c_v = c_new

    gx, gy = gradient(c_v)
    J = torch.sqrt(gx**2 + gy**2)

    # porous resistance (physical)
    J *= (1.0 - 0.3 * kappa)

    return evap_coeff * J * wet_f

# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────
snapshots = []

print("Running simulation...")
for step in range(steps):

    # --- Evaporation ---
    J = compute_evaporation(h)

    # --- Thin film flow ---
    lap_h = laplacian(h)
    gx, gy = gradient(lap_h)

    mobility = h**3

    flux_x = -mobility * gx
    flux_y = -mobility * gy

    # ✅ Proper contact line pinning (no flux outside)
    outside = ~footprint
    flux_x[outside] = 0
    flux_y[outside] = 0

    div_flux = (
        torch.roll(flux_x, -1, 0) - torch.roll(flux_x, 1, 0) +
        torch.roll(flux_y, -1, 1) - torch.roll(flux_y, 1, 1)
    ) / (2*dx)

    # --- Height update ---
    h = h + dt * (-div_flux - J)
    h = torch.clamp(h, min=0)

    # Enforce dry outside region
    h[outside] = 0

    # --- Velocity field ---
    u_x = flux_x / (h + 1e-8)
    u_y = flux_y / (h + 1e-8)

    # --- Particle advection (semi-Lagrangian) ---
    Xf = (X - u_x * dt).clamp(0, Nx-1)
    Yf = (Y - u_y * dt).clamp(0, Ny-1)

    X0 = Xf.long()
    Y0 = Yf.long()

    c = c[X0, Y0]

    # --- Deposition from evaporation ---
    deposit += c * J * dt
    c = c * (h > 1e-6)

    if step in snap_at:
        snapshots.append((step, deposit.detach().cpu().numpy()))
        print(f"Step {step} | max deposit {deposit.max().item():.4f}")

print("Done.")

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, (step, img) in zip(axes.flat, snapshots):
    ax.imshow(img, cmap='inferno')
    ax.set_title(f"Step {step}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# Final result
plt.figure(figsize=(6,6))
plt.imshow(deposit.cpu().numpy(), cmap='inferno')
plt.title("Final Coffee Ring Deposit")
plt.axis('off')
plt.show()