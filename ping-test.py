from ping import *

ping_object = Ping(
    frequency=1e3,
    duration=0.01,
    amplitude=100.0,
    phase=0,
    sample_rate=4e6,
    window_mode='gaussian'
)

transducer_object = Transducer(
    position=np.array([0.0, 0.0, 0.0]),
    angle_deg=0.0,
    beam_width_deg=25.0
)

reflectors = [
    Reflector(
        position=np.array([0.0, 0.0, 10]),
        radius=0.5,
        reflectivity=0.8,
        phase_shift_deg=180.0
    ),
]

simulated_world = World(
    transducer=transducer_object,
    reflectors=reflectors,
)

simulated_world.run_simulation(ping_object)

history = transducer_object.transducer_history
time_axis = np.arange(len(history)) / ping_object.fs

plt.figure(figsize=(12, 5))
plt.plot(time_axis, history)
plt.title("Transducer Signal History")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (arb. units)")
plt.grid()
plt.show()
