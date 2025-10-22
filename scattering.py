"""
scattering.py

Monte Carlo scattering logic for Rutherford simulation.
"""

import numpy as np

def sample_rutherford_theta(size=1, theta_min=1e-2, theta_max=np.pi-1e-2):
    """
    Sample polar scattering angles theta from the Rutherford distribution
    P(theta) ∝ sin(theta) / sin^4(theta/2)
    using rejection sampling.

    Parameters
    ----------
    size : int
        Number of samples to generate.
    theta_min : float
        Minimum angle to avoid singularity at 0.
    theta_max : float
        Maximum angle (typically just below pi).

    Returns
    -------
    thetas : ndarray, shape (size,)
        Sampled theta angles (radians).
    """
    def p_theta(theta):
        return np.sin(theta) / np.sin(theta/2)**4

    samples = []
    n_attempts = 0
    max_prob = p_theta(theta_min)
    while len(samples) < size:
        theta = np.random.uniform(theta_min, theta_max)
        y = np.random.uniform(0, max_prob)
        if y < p_theta(theta):
            samples.append(theta)
        n_attempts += 1
        if n_attempts > 1e5 and len(samples) == 0:
            raise RuntimeError("Rejection sampling failed: adjust theta_min/theta_max.")
    return np.array(samples)

def sample_uniform_phi(size=1):
    """
    Sample azimuthal angles phi from a uniform distribution [0, 2*pi]
    using the inverse CDF method.

    Parameters
    ----------
    size : int
        Number of samples to generate.

    Returns
    -------
    phis : ndarray, shape (size,)
        Sampled phi angles (radians).
    """
    u = np.random.uniform(0, 1, size)
    return 2 * np.pi * u

def scatter_particle(particle, theta, phi):
    """
    Update the velocity of the particle after scattering by theta and phi.

    Parameters
    ----------
    particle : Particle
        The particle object to update.
    theta : float
        Scattering angle (radians).
    phi : float
        Azimuthal angle (radians).
    """
    # Initial velocity direction assumed to be +z
    v = np.linalg.norm(particle.velocity)
    # New velocity direction in spherical coordinates
    vx = v * np.sin(theta) * np.cos(phi)
    vy = v * np.sin(theta) * np.sin(phi)
    vz = v * np.cos(theta)
    particle.velocity = np.array([vx, vy, vz])
    particle.theta = theta
    particle.scattered = True

def record_scattering_angles(particles):
    """
    Collect scattering angles from all particles.

    Parameters
    ----------
    particles : list of Particle

    Returns
    -------
    thetas : ndarray
        Array of scattering angles (radians).
    """
    return np.array([p.theta for p in particles if p.theta is not None])

# Example/test usage
if __name__ == "__main__":
    from particle import Particle, initialize_particles

    N = 10
    particles = initialize_particles(N)
    thetas = sample_rutherford_theta(N)
    phis = sample_uniform_phi(N)

    for i, p in enumerate(particles):
        scatter_particle(p, thetas[i], phis[i])

    # Data recording
    recorded_thetas = record_scattering_angles(particles)
    print("Scattering angles (degrees):", np.degrees(recorded_thetas))
