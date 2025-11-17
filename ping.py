import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, hilbert
import copy



class Transducer:
    def __init__(
            self, 
            position: np.ndarray, 
            angle_deg: float = 0.0,
            beam_width_deg: float = 25.0):
        self.position = position
        self.angle_deg = angle_deg
        self.beam_width_deg = beam_width_deg

    def update_transmit_signal(self, ping: 'Ping'):
        """
        Resets the transducer history and appends the ping to be sent out.
        """
        self.transducer_history = ping.signal.copy()

    def update_history_echo(self, return_signal: np.ndarray):
        self.transducer_history = np.concatenate(
            (self.transducer_history, return_signal)
            )
    
    def update_receive_signal(self, return_signal: np.ndarray):
        self.receive_signal = return_signal.copy()
    


class Ping:
    def __init__(
            self, 
            frequency: float | list, 
            duration: float, 
            amplitude: float,
            phase: float = 0,
            sample_rate: int = 4e6,
            window_mode: str = 'gaussian'):
        self.frequency = np.atleast_1d(frequency)
        self.duration = duration
        self.fs = sample_rate
        self.window_mode = window_mode

        self.amplitude = amplitude
        self.phase = phase

        self.t = np.arange(0, self.duration, 1/self.fs)
        self.signal = self._generate_signal()


    def _generate_window(self, N):
        """
        Generate a window function.
        """
        if self.window_mode == 'gaussian':
            std = (0.2 * N)
            return get_window(('gaussian', std), N)
        

    def _generate_signal(self):
        signal = np.zeros_like(self.t)
        for f in self.frequency:
            signal += np.sin(2 * np.pi * f * self.t + self.phase)
        if len(self.frequency) > 0:
            signal /= len(self.frequency)
            
        window = self._generate_window(len(self.t))
        
        return self.amplitude * signal * window
    


class Reflector:
    def __init__(
            self,
            position: np.ndarray,
            radius: float,
            reflectivity: float = 0.8,
            phase_shift_deg: float = 180.0):
        self.position = position
        self.radius = radius
        self.reflectivity = reflectivity
        self.phase_shift_deg = phase_shift_deg

    def reflect(self, incident_signal):
        """
        Applies 180 deg phase shift and reflectance loss.
        """
        phase_shifted_signal = incident_signal * -1
        return phase_shifted_signal * self.reflectivity



class World:
    # speed of sound in water
    c = 1500 # m/s
    water_density = 997 # kg/m^3
    
    # acoustic impedance
    Z_water = c * water_density

    p_ref = 1e-6  # reference pressure in water (1 uPa)

    def __init__(
            self,
            transducer: Transducer,
            reflectors: list[Reflector]):
        self.transducer = transducer
        self.reflectors = reflectors

    def run_simulation(self, ping: Ping):
        """
        Runs the simulation of the ping in the world with the transducer and reflectors.
        """
        # Transmit the ping
        self.transducer.update_transmit_signal(ping)

        ping_len = len(ping.signal)

        superposed_echo = np.zeros(0)
        # For each reflector, calculate the echo received at the transducer
        for reflector in self.reflectors:
            centre_dist = np.linalg.norm(
                reflector.position - self.transducer.position
                )
            path_length = centre_dist - reflector.radius

            # Apply outgoing spreading loss
            outgoing_signal = ping.signal / path_length

            # Apply reflection
            reflected_signal = reflector.reflect(outgoing_signal)

            # Apply incoming spreading loss
            received_signal = reflected_signal / path_length

            # Create pause for time delay between transmit and receive
            time_of_flight = 2 * path_length / World.c
            travel_samples = int(time_of_flight * ping.fs)
            delay_samples = travel_samples - ping_len
            signal_delay = np.zeros(delay_samples)

            superposed_echo = left_add_arrays(
                superposed_echo, np.concatenate((signal_delay, received_signal))
                )

        self.transducer.update_history_echo(superposed_echo)
        self.transducer.update_receive_signal(superposed_echo)


def left_add_arrays(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    out = np.zeros(max(len(a), len(b)))
    out[:len(a)] += a
    out[:len(b)] += b
    return out