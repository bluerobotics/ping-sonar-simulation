import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, hilbert
import copy



class Transducer:
    def __init__(
            self, 
            position: np.ndarray,
            velocity: np.ndarray = np.zeros(3), 
            angle_deg: float = 0.0,
            beam_width_deg: float = 25.0):
        self.initial_position = position
        self.velocity = velocity
        self.angle_deg = angle_deg
        self.beam_width_deg = beam_width_deg

    def update_transmit_signal(self, ping:'Ping'):
        """
        Resets the transducer history and appends the ping to be sent out.
        
        Parameters:
        - world: World object containing simulation parameters.
        - ping: Ping object containing the signal to be transmitted.
        - d_max: Maximum reflector distance to consider for timing.

        Returns the trajectory of the transducer up until the expected round
        trip time for a target at d_max.
        """
        self.transducer_history = ping.signal.copy()
        
        displacement = np.outer(ping.ping_time, self.velocity) 
        self.ping_positions = self.initial_position + displacement


    def update_history_echo(self, return_signal: np.ndarray, ping: 'Ping'):
        """
        Adds the returned signal to the transducer history.

        Parameters:
        - return_signal: The signal received at the transducer.
        """
        self.transducer_history = left_add_arrays(
            self.transducer_history, return_signal
            )
        
        time_arr = np.arange(len(self.transducer_history)) / ping.fs
        displacement = np.outer(time_arr, self.velocity)
        self.ping_positions = self.initial_position + displacement
    
    def update_receive_signal(self, return_signal: np.ndarray):
        """
        Stores the received signal separately from the transducer history.

        Parameters:
        - return_signal: The signal received at the transducer.
        """
        self.receive_signal = return_signal.copy()
        
    


class Ping:
    def __init__(
            self, 
            frequency: float | list, 
            n_cycles: int | None, 
            amplitude: float,
            phase: float = 0,
            duration: float | None = None,
            sample_rate: int = 4e6,
            window_mode: str = 'gaussian'):
        self.frequency = np.atleast_1d(frequency)

        # Determine duration of the signal, n_cycles takes priority
        if n_cycles is not None:
            if not self.frequency.any():
                raise ValueError("Cannot calculate duration from n_cycles: frequency is not set.")
            f_max = np.max(self.frequency)
            self.duration = n_cycles / f_max
        elif duration is not None:
            self.duration = duration
        else:
            raise ValueError("Must provide either 'duration' or 'n_cycles'.")

        self.fs = sample_rate
        self.window_mode = window_mode

        self.amplitude = amplitude
        self.phase = phase

        self.ping_time = np.arange(int(self.duration * self.fs)) / self.fs
        self.signal = self._generate_signal()


    def _generate_window(self, N):
        """
        Generate a window function. Availble modes: 'gaussian'
        """
        if self.window_mode == 'gaussian':
            std = (0.2 * N)
            return get_window(('gaussian', std), N)
        else:
            return np.ones(N)
        

    def _generate_signal(self):
        """
        Generate the ping signal from attributes.

        Returns:
        - signal: The generated ping signal as a numpy array, windowed as 
        pecified.
        """
        signal = np.zeros_like(self.ping_time)
        for f in self.frequency:
            signal += np.sin(2 * np.pi * f * self.ping_time + self.phase)
        if len(self.frequency) > 0:
            signal /= len(self.frequency)
            
        window = self._generate_window(len(self.ping_time))
        
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

        Parameters:
        - incident_signal: The signal incident on the reflector.

        Returns:
        - reflected_signal: The signal after reflection.
        """
        phase_shifted_signal = incident_signal * -1
        return phase_shifted_signal * self.reflectivity



class World:
    # speed of sound in water
    c = 1500 # m/s

    water_density = 997 # kg/m^3
    
    # acoustic impedance
    Z_water = c * water_density

    # reference pressure in water (1 uPa)
    p_ref = 1e-6 # Pa

    def __init__(
            self,
            transducer: Transducer,
            reflectors: list[Reflector]):
        self.transducer = transducer
        self.reflectors = reflectors

    def compute_receive_positions(self, points_A:list, reflector:Reflector) -> np.ndarray:
        """
        Computes the transducer positions during the receive phase.
        """
        point_P = reflector.position
        v = np.linalg.norm(self.transducer.velocity)
        v_hat = self.transducer.velocity/v if v != 0 else np.zeros(3) 
        k = v / World.c
        if k >= 1:
            raise ValueError("Transducer velocity must be less than the " \
            "speed of sound.")
        
        AP_vec = point_P - points_A
        p = np.linalg.norm(AP_vec, axis=1) # shape (N,)
        AP_hat = AP_vec / p[:, np.newaxis] # shape (N,3)
    
        cos_theta = np.dot(AP_hat, v_hat) # shape (N,)
        
        a = 1 - k**-2
        b = (2*p/k) - (2*p*cos_theta) - (4*reflector.radius/k)
        c = (4*p*reflector.radius) - (4*reflector.radius**2)

        discriminant = b**2 - 4*a*c

        travel_distances = (-b - np.sqrt(discriminant)) / (2*a) # NEW
        travel_times = travel_distances/v if v != 0 else np.zeros(len(points_A)) 

        return points_A + (travel_distances[:, np.newaxis] * v_hat), travel_times


    def run_simulation(self, ping: Ping):
        """
        Runs the simulation world with the transducer and reflectors.

        Parameters:
        - ping: Ping object containing the signal to be transmitted.
        """
        
        # Transmit the ping
        self.transducer.update_transmit_signal(ping)

        superposed_echo = np.zeros(0)
        # For each reflector, calculate the echo received at the transducer
        for reflector in self.reflectors:
            # Compute transducer position at receive, and corresponding delay
            receive_positions, receive_delays = self.compute_receive_positions(
                self.transducer.ping_positions, reflector
                )

            # Compute path lengths (two arms)
            out_distances = np.linalg.norm(
                reflector.position - self.transducer.ping_positions,axis=1
                )
            return_distances = np.linalg.norm(
                reflector.position - receive_positions, axis=1
                )
            
            # print(f"Time: {ping.ping_time}")
            # print(f"Receive positions: {receive_positions}")
            # print(f"Out distances: {out_distances}")
            # print(f"Return distances: {return_distances}")
            # print(f"Receive delays (s): {receive_delays}")

            # Apply outgoing spreading loss
            outgoing_signal_values = ping.signal \
                / (out_distances-reflector.radius)

            # Apply reflection
            reflected_signal_values = reflector.reflect(outgoing_signal_values)

            # Apply returning spreading loss
            received_signal_values = reflected_signal_values \
                / (return_distances - reflector.radius)

            # Map received signal to time axis with delays
            idx = np.rint((ping.ping_time+receive_delays) * ping.fs).astype(int)
            binning_signal = np.full(idx.max()+1, np.nan)
            sums = np.bincount(idx, weights=received_signal_values)
            counts = np.bincount(idx)

            binning_signal[counts>0] = sums[counts>0] / np.maximum(counts[counts>0], 1)
            coords = np.arange(len(binning_signal))
            non_empty_mask = ~np.isnan(binning_signal)
            x_valid = coords[non_empty_mask]
            y_valid = binning_signal[non_empty_mask]
            received_signal = np.interp(coords, x_valid, y_valid)

            superposed_echo = left_add_arrays(superposed_echo, received_signal)

        self.transducer.update_history_echo(superposed_echo, ping)
        self.transducer.update_receive_signal(superposed_echo)


def left_add_arrays(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    Adds two arrays of possibly different lengths by aligning them to the left.
    
    Parameters:
    - a: First input array.
    - b: Second input array.

    Returns:
    - out: The element-wise sum of the two arrays, aligned to the left.
    """
    out = np.zeros(max(len(a), len(b)))
    out[:len(a)] += a
    out[:len(b)] += b
    return out