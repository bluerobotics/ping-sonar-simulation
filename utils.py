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


def plot_transmitted_signal(ping_object):
    signal = ping_object.signal
    time_axis = np.arange(len(signal)) / ping_object.fs

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal)
    plt.title("Transmitted Ping Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arb. units)")
    plt.grid()
    plt.show()

def plot_received_signal(transducer_object, ping_object):
    signal = transducer_object.receive_signal
    time_axis = np.arange(len(signal)) / ping_object.fs

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal)
    plt.title("Received Signal with Single Reflector")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arb. units)")
    plt.grid()
    plt.show()

def plot_transducer_history(transducer_object, ping_object):
    signal = transducer_object.transducer_history
    time_axis = np.arange(len(signal)) / ping_object.fs

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal)
    plt.title("Transducer Signal History")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arb. units)")
    plt.grid()
    plt.show()

def plot_transducer_position(transducer_object, ping_object):
    signal = transducer_object.ping_positions
    time_axis = np.arange(len(signal)) / ping_object.fs

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- Plot X ---
    ax1.plot(time_axis, signal[:, 0], color='tab:blue')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title("Transducer Position History")
    ax1.grid(True)

    # --- Plot Y ---
    ax2.plot(time_axis, signal[:, 1], color='tab:orange')
    ax2.set_ylabel('Y Position (m)')
    ax2.grid(True)

    # --- Plot Z ---
    ax3.plot(time_axis, signal[:, 2], color='tab:green')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()



def compute_velocity(f_r, f_t, c, sigma_f_t=0):
    """
    Compute the velocity of the transducer in the direction of the reflector
    based on the observed Doppler shift.

    Parameters:
    - f_r: Received frequency (Hz)
    - f_t: Original transmitted frequency (Hz)
    - c: Speed of sound in the medium (m/s)

    Returns:
    - velocity: Estimated
    """
    # dopppler shift amount
    f_d = f_r - f_t
    sigma_f_d = sigma_f_t
    
    # v = (f_d * c) / (2 * f_t + f_d)
    v = (f_d * c) / (2 * f_t)

    # error propagation for velocity estimate
    sigma_v = sigma_f_d * (2 * f_t * c) / (2 * f_t + f_d)**2

    return v, sigma_v