"""
particle.py

Module for the Particle class and ODE-based propagation logic.
"""

import numpy as np
from scipy.integrate import solve_ivp

class Particle:
    """
    Represents a single particle in the Rutherford scattering simulation.

    Parameters
    ----------
    position : array_like, shape (3,)
        Initial position vector.
    velocity : array_like, shape (3,)
        Initial velocity vector.
    mass : float
        Particle mass.

    Attributes
    ----------
    position : ndarray
        Current position vector.
    velocity : ndarray
        Current velocity vector.
    scattered : bool
        True if the particle has scattered.
    theta : float or None
        Scattering angle (set after scattering event).
    """

    def __init__(self, position, velocity, mass=1.0):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.mass = mass
        self.scattered = False
        self.theta = None

    def propagate(self, t_end, dt=0.1, force_func=None):
        """
        Propagate the particle's trajectory using a 2nd order ODE solver.

        Parameters
        ----------
        t_end : float
            Total integration time.
        dt : float, optional
            Time step for output (default is 0.1).
        force_func : callable, optional
            Function to compute force given position and velocity, or None for free motion.
        """
        def odefunc(t, y):
            # y = [x, y, z, vx, vy, vz]
            pos = y[:3]
            vel = y[3:]
            if force_func is None:
                acc = np.zeros(3)
            else:
                acc = force_func(pos, vel, t) / self.mass
            return np.concatenate((vel, acc))

        y0 = np.concatenate((self.position, self.velocity))
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end + dt, dt)

        sol = solve_ivp(odefunc, t_span, y0, t_eval=t_eval, method='RK45')

        # Update to final position and velocity
        self.position = sol.y[:3, -1]
        self.velocity = sol.y[3:, -1]

def initialize_particles(N, start_point=None, initial_speed=1.0, mass=1.0):
    """
    Initialize N particles at a common starting position and initial velocity along +z.

    Parameters
    ----------
    N : int
        Number of particles.
    start_point : array_like or None
        Starting position (default: [0,0,0]).
    initial_speed : float
        Initial velocity magnitude (default: 1.0).
    mass : float
        Particle mass.

    Returns
    -------
    particles : list of Particle
        List of Particle instances.
    """
    if start_point is None:
        start_point = np.zeros(3)
    particles = []
    for _ in range(N):
        position = np.array(start_point)
        velocity = np.array([0, 0, initial_speed])
        particles.append(Particle(position, velocity, mass=mass))
    return particles

# Example/test usage
if __name__ == "__main__":
    # Parameters
    N = 5
    t_end = 10.0
    dt = 0.1

    # Initialize particles
    particles = initialize_particles(N)

    # Propagate each particle to the target (no force)
    for p in particles:
        p.propagate(t_end, dt=dt)
        print(f"Final position: {p.position}, Final velocity: {p.velocity}")
