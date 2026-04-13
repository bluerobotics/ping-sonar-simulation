import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, hilbert
import copy

class Ping:
    # speed of sound in water
    c = 1500 # m/s
    water_density = 997 # kg/m^3
    # acoustic impedance
    Z_water = c * water_density

    beam_angle = 25 # degrees

    p_ref = 1e-6  # reference pressure in water (1 uPa)

    def __init__(self, 
                 frequency: float | list, 
                 duration: float, 
                 amplitude: float,
                 position: np.ndarray,
                 phase: float = 0,
                 sample_rate: int = 4e6,
                 window_mode: str = 'gaussian'):
        self.frequency = np.atleast_1d(frequency)
        self.duration = duration
        self.fs = sample_rate
        self.window_mode = window_mode

        # Original source amplitude (peak pressure at 1m)
        self.original_amplitude = amplitude
        self.original_phase = phase

        # Current amplitude and phase
        self.amplitude = amplitude
        self.phase = phase

        self.transducer_position = position
        self.transducer_angle = 0

        # Generate the signal based on the original amplitude
        self.t = np.arange(0, self.duration, 1/self.fs)
        self.original_signal = self._generate_signal()
        self.signal = self.original_signal

        # Calculate and store the Source Level (SL)
        self.SL = 20*np.log10((self.original_amplitude/np.sqrt(2)) / Ping.p_ref)

        self.position = np.zeros(3)
        self.initial_distance = 1.0

    def _generate_window(self, N):
        """Generate a window function."""
        if self.window_mode == 'gaussian':
            std = (0.2 * N)
            return get_window(('gaussian', std), N)
        
    def _generate_signal(self, time_delay=0.0, amplitude=None, phase=None):
        """
        Generate the ping signal with optional time delay, amplitude, and phase.
        """
        # Use provided parameters or default to the original source's
        amp = amplitude if amplitude is not None else self.original_amplitude
        ph = phase if phase is not None else self.original_phase
        
        t_delayed = self.t - time_delay
        
        base_signal = np.zeros_like(self.t)
        for f in self.frequency:
            component = np.sin(2 * np.pi * f * t_delayed + ph)
            base_signal += component
        base_signal /= len(self.frequency)
        
        window = self._generate_window(len(self.t))
        return amp * base_signal * window

    
    def add_noise(self, snr_db):
        signal_power = np.mean(self.signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(self.signal))
        self.signal += noise

    def get_intensity_level(self):
        # effective acoustic pressure
        p_e = self.amplitude/np.sqrt(2)

        # acoustic intensity level
        self.intensity_level = 20*np.log10(p_e/Ping.p_ref)
        
        return self.intensity_level
    
    def get_propagated_ping(self, r: float):
        """
        Creates a new Ping object representing the signal at distance 'r'.
        
        This is a "factory" method called on the SOURCE ping.
        """
        # 1. Calculate new properties
        r0 = self.initial_distance
        attenuation_factor = (r0 / r) if r >= r0 else 1.0
        propagated_amplitude = self.original_amplitude * attenuation_factor
        time_delay = r / Ping.c
        
        # 2. Generate the new signal
        new_signal = self._generate_signal(time_delay=time_delay, 
                                           amplitude=propagated_amplitude)
        
        # 3. Create the new Ping object
        new_ping = copy.copy(self) # Shallow copy is efficient
        
        # 4. Set the new state
        new_ping.signal = new_signal
        new_ping.amplitude = propagated_amplitude
        # Assuming propagation along Z-axis from source
        new_ping.position = self.transducer_position + np.array([0, 0, r]) 
        
        return new_ping


    def get_echo_ping(self, r_oneway: float, reflectivity_factor: float, phase_shift_deg: float):
        """
        Creates a new Ping object representing the echo from distance 'r'.
        
        This method is called on the SOURCE ping.
        """
        # 1. Calculate amplitude
        r0 = self.initial_distance
        spreading_loss_factor = (r0 / r_oneway)**2 if r_oneway >= r0 else 1.0
        final_amplitude = self.original_amplitude * spreading_loss_factor * reflectivity_factor
        
        # 2. Calculate time delay
        r_total_path = 2 * r_oneway
        time_delay = r_total_path / Ping.c
        
        # 3. Generate the base echo signal (with travel delay)
        base_echo_signal = self._generate_signal(time_delay=time_delay,
                                                 amplitude=final_amplitude)
        
        # 4. Apply the reflection phase shift
        phase_shift_rad = np.deg2rad(phase_shift_deg)
        if np.any(base_echo_signal):
            analytic_sig = hilbert(base_echo_signal)
            shifted_analytic = analytic_sig * np.exp(1j * phase_shift_rad)
            final_echo_signal = np.real(shifted_analytic)
        else:
            final_echo_signal = base_echo_signal
            
        # 5. Create the new Ping object
        new_ping = copy.copy(self)
        
        # 6. Set the new state
        new_ping.signal = final_echo_signal
        new_ping.amplitude = final_amplitude
        new_ping.position = self.transducer_position # Echo is back at the source
        
        return new_ping


    # --- REFLECTION METHOD ---
    def reflect(self, reflectivity_factor, phase_shift_deg):
        """
        Creates a new Ping object by reflecting the *current* ping.
        """
        # 1. Calculate new properties based on *current* state
        reflected_amplitude = self.amplitude * reflectivity_factor
        
        # 2. Apply phase shift
        phase_shift_rad = np.deg2rad(phase_shift_deg)
        if np.any(self.signal):
            analytic_sig = hilbert(self.signal)
            shifted_analytic = analytic_sig * np.exp(1j * phase_shift_rad)
            reflected_signal = np.real(shifted_analytic) * reflectivity_factor
        else:
            reflected_signal = self.signal
            
        # 3. Create new Ping
        reflected_ping = copy.copy(self)
        
        # 4. Set new state
        reflected_ping.signal = reflected_signal
        reflected_ping.amplitude = reflected_amplitude
        # Position is unchanged by reflection
        
        return reflected_ping

    # def get_rl(self):
    #     """ Calculates the RL of the *current* ping's signal. """
    #     if not np.any(self.signal):
    #         return -np.inf
    #     p_rms = np.sqrt(np.mean(self.signal**2))
    #     rl = 20 * np.log10(p_rms / Ping.p_ref)
    #     return rl
    
    def show(self, set_title=None, ylim_scale=1.1): # Changed ylim to ylim_scale
        plt.figure(figsize=(10, 4))
        plt.plot(self.t * 1e3, self.signal)
        title = set_title if set_title else f"Ping at {self.position}"
        plt.title(f"{title}")
        
        # Get the peak amplitude of this *specific* signal
        max_amp = np.max(np.abs(self.signal))
        if max_amp == 0:  # Handle an all-zero signal
            max_amp = 1.0 # Use a default plot range
            
        # Set the ylim based on this signal's peak and the scale factor
        plt.ylim(-max_amp * ylim_scale, max_amp * ylim_scale)
            
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (Pascals)")
        plt.grid(True)
        plt.show()