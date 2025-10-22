import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

class Ping:
    # speed of sound in water
    c = 1500 # m/s
    water_density = 997 # kg/m^3

    beam_angle = 25 # degrees

    def __init__(self, 
                 frequency: float | list, 
                 duration: float, 
                 amplitude: float,
                 phase: float = 0,
                 sample_rate: int = 4e6,
                 window_mode: str = 'gaussian'):
        self.frequency = np.atleast_1d(frequency)
        self.duration = duration
        self.fs = sample_rate
        self.amplitude = amplitude
        self.phase = phase
        
        self.window_mode = window_mode

        self.t = np.arange(0, self.duration, 1/self.fs)
        self.signal = self._generate_signal()

        self.position = np.zeros(3)
        self.initial_distance = 1.0

    def _generate_window(self, N):
        """Generate a window function."""
        if self.window_mode == 'gaussian':
            std = (0.2 * N)
            return get_window(('gaussian', std), N)
        
    def _generate_signal(self):
        base_signal = np.zeros_like(self.t)
        for f in self.frequency:
            component = np.sin(2 * np.pi * f * self.t + self.phase)
            base_signal += component
        base_signal /= len(self.frequency)  # normalize multi-tone sum

        window = self._generate_window(len(self.t))
        return self.amplitude * base_signal * window

    def apply_transmission_loss(self, distance_r: float):
        r0 = self.initial_distance
        r_total = np.linalg.norm(self.position) + distance_r

        attenuation = (r0 / r_total)
        self.amplitude *= attenuation
        self.signal *= attenuation
        self.position += np.array([0, 0, distance_r])  # assumes propagation along +z
        return self.signal
    
    def add_noise(self, snr_db):
        signal_power = np.mean(self.signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(self.signal))
        self.signal += noise

    def get_intensity(self):
        p_e = self.amplitude/np.sqrt(2)
        self.intensity = p_e**2/(Ping.water_density*Ping.c) 
        return self.intensity
    
    def show(self):
        plt.figure(figsize=(8, 3))
        plt.plot(self.t * 1e3, self.signal)
        plt.title("Sonar Ping Pulse")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()


    