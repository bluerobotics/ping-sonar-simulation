import numpy as np
import matplotlib.pyplot as plt



def get_peak_frequency(signal, fs):
    N = len(signal)
    N_pad = 5 * N
    fft_spectrum = np.fft.rfft(signal, n=N_pad)
    freqs = np.fft.rfftfreq(N_pad, d=1/fs)
    
    magnitude = np.abs(fft_spectrum)
    magnitude[0] = 0 
    
    # sort indices by magnitude 
    k = np.argmax(magnitude)

    # quadratic interpolation of peak frequency
    alpha = magnitude[k-1]
    beta = magnitude[k]
    gamma = magnitude[k+1]

    p = k + 0.5 * (gamma - alpha) / (2 * beta - gamma - alpha)
    peak_freq = p * fs / N_pad

    return peak_freq, freqs, magnitude

def draw_sphere(ax, center, radius, color='r'):
    """Helper to draw a sphere on a 3D axis."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color=color, alpha=0.6)

def plot_world_3d(transducer, reflector_list):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # A. Draw Reflectors
    for i, ref in enumerate(reflector_list):
        draw_sphere(ax, ref.position, ref.radius, color='grey')
        # Add label near the sphere
        ax.text(ref.position[0], ref.position[1], ref.position[2]+ref.radius*2, 
                f'Reflector {i+1}', color='black')

    # B. Draw Transducer Path
    # We assume ping_positions is (N, 3)
    path = transducer.ping_positions
    
    # We slice [::10] to avoid plotting too many points if sample rate is high
    ax.scatter(path[::5, 0], path[::5, 1], path[::5, 2], 
               c='blue', s=5, label='Transducer Path')
    
    # Mark start and end
    ax.scatter(path[0,0], path[0,1], path[0,2], c='green', s=50, marker='x', label='Start')
    ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='black', s=50, marker='^', label='End')

    # C. Formatting
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D World Simulation')
    ax.legend()

    # D. Force Equal Aspect Ratio (Crucial for 3D spheres to look spherical)
    # Calculate the limits of all data
    all_x = np.concatenate([path[:,0], [r.position[0] for r in reflector_list]])
    all_y = np.concatenate([path[:,1], [r.position[1] for r in reflector_list]])
    all_z = np.concatenate([path[:,2], [r.position[2] for r in reflector_list]])

    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0

    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.axis('equal')
    plt.show()