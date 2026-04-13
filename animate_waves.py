import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# -----------------------
# PARAMETERS
# -----------------------
c = 1.0
freq = 0.5
omega = 2 * np.pi * freq
wavelength = c / freq
k = 2 * np.pi / wavelength

# geometry
source_pos = np.array([0.0, 0.0])
reflector_center = np.array([7.0, 0.0])
reflector_radius = 2.0

# beam / pulse
beam_axis_deg = 0.0
beam_width_deg = 25.0
n_cycles = 5.0
pulse_T = n_cycles / freq
pulse_center = pulse_T / 2.0
pulse_width = pulse_T / 6.0

# reflectivity / phase
reflectivity = 0.9
phase_shift = 0.0   # 0 = hard wall reflection, π = soft

# grid
size = 10.0
points = 220
x = np.linspace(-size, size, points)
y = np.linspace(-size, size, points)
X, Y = np.meshgrid(x, y)

# circular reflector discretization
N_surf = 120
phis = np.linspace(0, 2 * np.pi, N_surf, endpoint=False)
surf_x = reflector_center[0] + reflector_radius * np.cos(phis)
surf_y = reflector_center[1] + reflector_radius * np.sin(phis)
surf_pts = np.stack([surf_x, surf_y], axis=1)

# beam shape
R_source = np.sqrt((X - source_pos[0])**2 + (Y - source_pos[1])**2)
theta = np.degrees(np.arctan2(Y - source_pos[1], X - source_pos[0]))
sigma = beam_width_deg / 2.0
gain = np.exp(-0.5 * ((theta - beam_axis_deg) / sigma)**2)

# source → surface
dist_src_to_surf = np.sqrt((surf_x - source_pos[0])**2 + (surf_y - source_pos[1])**2)
angle_src_to_surf = np.degrees(np.arctan2(surf_y - source_pos[1], surf_x - source_pos[0]))
surf_gain = np.exp(-0.5 * ((angle_src_to_surf - beam_axis_deg) / sigma)**2)

dphi = 2 * np.pi / N_surf
surf_elem_weight = reflector_radius * dphi

obs_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
M = obs_pts.shape[0]

def envelope(t):
    return np.exp(-0.5 * ((t - pulse_center) / pulse_width)**2)

# precompute distances from each surface element to every observation point
surf_to_obs_dists = np.empty((N_surf, M))
for i in range(N_surf):
    dx = obs_pts[:, 0] - surf_x[i]
    dy = obs_pts[:, 1] - surf_y[i]
    surf_to_obs_dists[i, :] = np.sqrt(dx*dx + dy*dy) + 1e-12

amplitude_scale = 6.0

# --- custom colormap: white→lightblue→blue
colors = [(1,1,1), (0.6,0.8,1), (0.0,0.3,0.9)]
whiteblue = LinearSegmentedColormap.from_list("whiteblue", colors, N=256)

# -----------------------
# plotting setup
# -----------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(np.zeros_like(R_source),
               origin='lower', extent=[-size, size, -size, size],
               cmap=whiteblue, vmin=0, vmax=2)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Cone-shaped pulse reflecting off circular surface')

circle_patch = plt.Circle(reflector_center, reflector_radius, fill=False, color='k', lw=2)
ax.add_patch(circle_patch)
ax.plot(source_pos[0], source_pos[1], 'yo', ms=9)

# -----------------------
# animation update
# -----------------------
def update(frame):
    t = frame / 15.0

    # initialize field to zero before pulse starts
    if t < 0.2:
        im.set_data(np.zeros_like(R_source))
        return [im]

    # Incident pulse
    t_source = t - R_source / c
    inc_mask = t_source > 0
    incident = np.zeros(M)
    if np.any(inc_mask):
        env_vals = envelope(t_source.ravel())
        ph = np.sin(k * R_source.ravel() - omega * t)
        incident = amplitude_scale * gain.ravel() * env_vals * ph / (R_source.ravel() + 1e-6)
        incident[~inc_mask.ravel()] = 0.0

    # Reflected pulse (surface integration)
    reflected = np.zeros(M)
    for i in range(N_surf):
        L_i = dist_src_to_surf[i] + surf_to_obs_dists[i, :]
        tau = t - L_i / c
        mask_tau = tau > 0
        if not np.any(mask_tau):
            continue
        env_tau = envelope(tau[mask_tau])
        ph_i = np.sin(k * L_i[mask_tau] - omega * t + phase_shift)
        contrib = (reflectivity * surf_gain[i] * surf_elem_weight *
                   env_tau * ph_i / (surf_to_obs_dists[i, mask_tau] + 1e-6))
        reflected[mask_tau] += contrib

    Z = (incident + reflected).reshape(R_source.shape)

    # Mask inside reflector
    inside = ((X - reflector_center[0])**2 + (Y - reflector_center[1])**2) < reflector_radius**2
    Z[inside] = np.nan

    # emphasize compression (positive) only
    Z = np.maximum(Z, 0)

    im.set_data(Z)
    im.set_clim(0, np.nanmax(Z) * 0.9)
    return [im]

ani = FuncAnimation(fig, update, frames=600, interval=30, blit=True)
plt.show()
