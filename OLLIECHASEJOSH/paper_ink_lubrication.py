import numpy as np
import matplotlib.pyplot as plt

"""Lubrication-theory model for ink droplet spreading and diffusion on paper.

This 1D axisymmetric model evolves:
  h(r,t): liquid film thickness
  c(r,t): solute (ink) concentration in film

Equations (non-dimensionalized):
  dh/dt = -1/r d/dr [ r q ] - k_evap * h
  q = h^3 / 3 * d/dr [ (1/r) d/dr (r dh/dr)]
  d(hc)/dt + 1/r d/dr [ r q c ] = D/h * 1/r d/dr [ r h dc/dr ]

The film is allowed to permeate into paper as an optional sink.
"""

# Physical and numerical parameters
R = 1.0           # droplet initial radius (nondim)
N = 300
r = np.linspace(0, R, N)
dr = r[1] - r[0]

dt = 2e-5
steps = 7000

gamma = 1.0       # surface tension (nondim)
mu = 1.0          # viscosity (nondim)
D = 1e-3          # solute diffusivity in film
k_evap = 1e-3     # evaporation sink
k_sink = 2e-4     # paper absorption rate

# Initial droplet shape (parabolic cap) and concentration
h0 = 0.08 * (1 - (r/R)**2)
h0[h0 < 0] = 0
c0 = np.ones_like(r)

h = h0.copy()
phi = h * c0

# Finite difference helpers (axisymmetric)
def laplacian(f):
    d2f = np.zeros_like(f)
    d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dr**2
    d2f[0] = 2*(f[1] - f[0]) / dr**2
    d2f[-1] = 2*(f[-2] - f[-1]) / dr**2
    return d2f


def radial_gradient(f):
    g = np.zeros_like(f)
    g[1:-1] = (f[2:] - f[:-2]) / (2*dr)
    g[0] = (f[1] - f[0]) / dr
    g[-1] = (f[-1] - f[-2]) / dr
    return g

# Time-stepping
snapshots = []
for n in range(steps):
    # capillary pressure p = -gamma * curvature
    dh_dr = radial_gradient(h)
    curvature = (1/r) * dh_dr + laplacian(h)
    curvature[0] = 4 * (h[1] - h[0]) / dr**2  # regularized at center
    p = -gamma * curvature

    dp_dr = radial_gradient(p)
    q = -h**3 / (3*mu) * dp_dr

    # flux divergence in axisymmetric geometry
    q_r = np.zeros_like(q)
    q_r[1:-1] = (r[2:]*q[2:] - r[:-2]*q[:-2]) / (2*dr)
    q_r[0] = 2 * q[1]  # symmetry at r=0
    q_r[-1] = (r[-1]*q[-1] - r[-2]*q[-2]) / dr

    dh_dt = -q_r / r - k_evap * h - k_sink * h
    dh_dt[0] = - (3 * q[1] / dr) - k_evap*h[0] - k_sink*h[0]

    h_new = h + dt * dh_dt
    h_new = np.maximum(h_new, 1e-8)

    # Solute conservation with advection + diffusion
    phi_old = phi
    c = np.where(h > 1e-8, phi_old / h_old, 0) if (h_old := h) is not None else c0

    # advective flux for solute
    j_adv = q * c
    j_adv_r = np.zeros_like(j_adv)
    j_adv_r[1:-1] = (r[2:]*j_adv[2:] - r[:-2]*j_adv[:-2]) / (2*dr)
    j_adv_r[0] = 2 * j_adv[1]
    j_adv_r[-1] = (r[-1]*j_adv[-1] - r[-2]*j_adv[-2]) / dr

    # diffusion term for solute in film
    dc_dr = radial_gradient(c)
    diffusion = D * (laplacian(c) + (1/r)*dc_dr)
    diffusion[0] = 4*D*(c[1]-c[0]) / dr**2

    dphi_dt = -j_adv_r / r + h * diffusion

    phi = np.maximum(phi + dt * dphi_dt, 0)

    h = h_new

    if n % 1000 == 0 or n == steps-1:
        snapshots.append((n, h.copy(), np.where(h>0, phi / h, 0).copy()))

# Plot results
plt.figure(figsize=(10, 4))
for idx, (n, h_snap, c_snap) in enumerate(snapshots):
    plt.plot(r, h_snap, label=f't={n}')
plt.title('Height profile h(r,t)')
plt.xlabel('r (nondim)')
plt.ylabel('h')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 4))
for idx, (n, h_snap, c_snap) in enumerate(snapshots):
    plt.plot(r, c_snap, label=f't={n}')
plt.title('Ink concentration c(r,t)')
plt.xlabel('r (nondim)')
plt.ylabel('c')
plt.legend()
plt.grid(True)

plt.show()
