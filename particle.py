"""
particle.py

Module for the Particle class and ODE-based propagation logic.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

class Particle2D:
    """
    Represents a single particle in 2D Rutherford scattering simulation.
    
    Parameters
    ----------
    position : array_like, shape (2,)
        Initial position vector [x, y].
    velocity : array_like, shape (2,)
        Initial velocity vector [vx, vy].
    mass : float
        Particle mass.
    charge : float
        Particle charge.
    """
    
    def __init__(self, position, velocity, mass=1.0, charge=1.0):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.mass = mass
        self.charge = charge
        self.trajectory = [self.position.copy()]
        self.scattered = False
        self.theta = None
        self.impact_parameter = position[1]  # y-coordinate as impact parameter
        
    def update_position(self, dt):
        """Update position based on current velocity."""
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())
        
    def get_trajectory(self):
        """Return the particle trajectory as numpy array."""
        return np.array(self.trajectory)

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

def initialize_particles_2d(N, x_start=-10.0, y_range=(-5.0, 5.0), v_initial=1.0):
    """
    Initialize N particles for 2D Rutherford scattering.
    
    Parameters
    ----------
    N : int
        Number of particles.
    x_start : float
        Starting x-position (far from target).
    y_range : tuple
        Range of impact parameters (y-positions).
    v_initial : float
        Initial velocity magnitude in +x direction.
        
    Returns
    -------
    particles : list of Particle2D
        List of initialized particles.
    """
    particles = []
    y_positions = np.linspace(y_range[0], y_range[1], N)
    
    for y in y_positions:
        position = np.array([x_start, y])
        velocity = np.array([v_initial, 0.0])
        particles.append(Particle2D(position, velocity))
    
    return particles

def plot_trajectories_2d(particles, target_pos=(0, 0), target_radius=0.1):
    """
    Plot 2D particle trajectories.
    
    Parameters
    ----------
    particles : list of Particle2D
        List of particles with trajectories.
    target_pos : tuple
        Position of scattering center.
    target_radius : float
        Visual radius of target for plotting.
    """
    plt.figure(figsize=(12, 8))
    
    for i, p in enumerate(particles):
        traj = p.get_trajectory()
        if p.scattered:
            color = 'red'
            alpha = 0.8
        else:
            color = 'blue'
            alpha = 0.5
            
        plt.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=1)
        
        # Mark starting point
        plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=4)
        
        # Mark final point
        plt.plot(traj[-1, 0], traj[-1, 1], 'ro' if p.scattered else 'bo', markersize=4)
    
    # Draw target
    circle = plt.Circle(target_pos, target_radius, color='black', fill=True, zorder=10)
    plt.gca().add_patch(circle)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position (Impact Parameter)')
    plt.title('2D Rutherford Scattering Trajectories')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(['Scattered', 'Unscattered', 'Start', 'End', 'Target'])
    plt.tight_layout()
    plt.show()

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

    # Example usage for 2D particles
    particles_2d = initialize_particles_2d(10)
    print(f"Initialized {len(particles_2d)} particles")
    for i, p in enumerate(particles_2d):
        print(f"Particle {i}: position={p.position}, impact_parameter={p.impact_parameter:.2f}")
