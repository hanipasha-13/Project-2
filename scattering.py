"""
scattering.py

2D Rutherford scattering with both rejection and inverse CDF sampling methods.
"""

import numpy as np
import matplotlib.pyplot as plt

def rutherford_pdf_2d(theta):
    """
    Rutherford scattering probability density function in 2D.
    P(theta) ∝ sin(theta) / sin^4(theta/2)
    
    Parameters
    ----------
    theta : array_like
        Scattering angles in radians.
        
    Returns
    -------
    pdf : array_like
        Unnormalized probability density values.
    """
    eps = 1e-10  # Avoid division by zero
    return np.sin(theta) / (np.sin(0.5 * theta + eps) ** 4)

def sample_rutherford_rejection(size=1, theta_min=1e-2, theta_max=np.pi-1e-2):
    """
    Sample scattering angles using rejection sampling method.
    
    Parameters
    ----------
    size : int
        Number of samples to generate.
    theta_min : float
        Minimum scattering angle.
    theta_max : float
        Maximum scattering angle.
        
    Returns
    -------
    samples : ndarray
        Array of sampled scattering angles.
    n_attempts : int
        Total number of attempts (for efficiency analysis).
    """
    def pdf_func(theta):
        return rutherford_pdf_2d(theta)
    
    # Find maximum of PDF for rejection sampling
    theta_test = np.linspace(theta_min, theta_max, 1000)
    pdf_max = np.max(pdf_func(theta_test))
    
    samples = []
    n_attempts = 0
    
    while len(samples) < size:
        # Sample uniformly in theta range
        theta_candidate = np.random.uniform(theta_min, theta_max)
        
        # Sample uniformly in y range
        y = np.random.uniform(0, pdf_max)
        
        # Accept or reject
        if y <= pdf_func(theta_candidate):
            samples.append(theta_candidate)
        
        n_attempts += 1
        
        # Safety check
        if n_attempts > size * 1000:
            raise RuntimeError("Rejection sampling failed - too many attempts")
    
    return np.array(samples), n_attempts

def sample_rutherford_inverse_cdf(size=1, theta_min=1e-2, theta_max=np.pi-1e-2):
    """
    Sample scattering angles using inverse CDF method.
    
    For Rutherford scattering, the exact inverse CDF transformation is:
    theta = 2 * arccos(sqrt(u_scaled))
    where u_scaled accounts for the bounded range.
    
    Parameters
    ----------
    size : int
        Number of samples to generate.
    theta_min : float
        Minimum scattering angle.
    theta_max : float
        Maximum scattering angle.
        
    Returns
    -------
    samples : ndarray
        Array of sampled scattering angles.
    """
    # Generate uniform random variables
    u = np.random.uniform(0, 1, size)
    
    # Convert angle bounds to CDF bounds
    u_min = np.cos(theta_max/2)**2
    u_max = np.cos(theta_min/2)**2
    
    # Scale u to the appropriate range
    u_scaled = u_min + u * (u_max - u_min)
    
    # Apply inverse CDF transformation
    samples = 2 * np.arccos(np.sqrt(u_scaled))
    
    return samples

def scatter_particle_2d(particle, alpha=1.0, target_pos=(0, 0)):
    """
    Simulate 2D Rutherford scattering for a particle.
    
    Parameters
    ----------
    particle : Particle2D
        The particle to scatter.
    alpha : float
        Scattering strength parameter.
    target_pos : tuple
        Position of scattering center.
    """
    x0, y0 = particle.position
    vx0, vy0 = particle.velocity
    v0 = np.sqrt(vx0**2 + vy0**2)
    
    # Impact parameter
    b = abs(y0 - target_pos[1])
    
    # Rutherford scattering formula: theta = 2 * arctan(alpha / (2 * E * b))
    # For simplicity, assume E = 0.5 * m * v^2 and alpha = 1
    E = 0.5 * particle.mass * v0**2
    
    if b < 1e-6:  # Avoid division by zero
        b = 1e-6
    
    # Scattering angle
    theta = 2 * np.arctan(alpha / (2 * E * b))
    
    # Determine scattering direction (up or down)
    scatter_sign = np.sign(y0 - target_pos[1])
    if scatter_sign == 0:
        scatter_sign = 1
    
    # New velocity direction (rotated by theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta) * scatter_sign
    
    # Rotate velocity vector
    vx_new = vx0 * cos_theta - vy0 * sin_theta
    vy_new = vx0 * sin_theta + vy0 * cos_theta
    
    particle.velocity = np.array([vx_new, vy_new])
    particle.theta = theta
    particle.scattered = True

def simulate_2d_scattering(particles, alpha=1.0, dt=0.01, max_time=20.0, target_pos=(0, 0)):
    """
    Run full 2D scattering simulation.
    
    Parameters
    ----------
    particles : list of Particle2D
        List of particles to simulate.
    alpha : float
        Scattering strength.
    dt : float
        Time step.
    max_time : float
        Maximum simulation time.
    target_pos : tuple
        Position of scattering center.
    """
    target_x, target_y = target_pos
    
    for particle in particles:
        time = 0
        scattered = False
        
        while time < max_time and not scattered:
            # Check if particle is near target
            x, y = particle.position
            distance_to_target = np.sqrt((x - target_x)**2 + (y - target_y)**2)
            
            # Scatter when close to target
            if distance_to_target < 0.5 and x > target_x - 1.0 and not scattered:
                scatter_particle_2d(particle, alpha, target_pos)
                scattered = True
            
            # Update position
            particle.update_position(dt)
            time += dt
            
            # Stop if particle moves too far away
            if x > 15.0:
                break

def compare_sampling_methods(N=10000, bins=50):
    """
    Compare rejection sampling vs inverse CDF methods.
    
    Parameters
    ----------
    N : int
        Number of samples for each method.
    bins : int
        Number of histogram bins.
    """
    # Sample using both methods
    print("Sampling with rejection method...")
    samples_rejection, n_attempts = sample_rutherford_rejection(N)
    efficiency_rejection = N / n_attempts * 100
    
    print("Sampling with inverse CDF method...")
    samples_inverse_cdf = sample_rutherford_inverse_cdf(N)
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rejection method histogram
    theta_deg_rej = np.degrees(samples_rejection)
    ax1.hist(theta_deg_rej, bins=bins, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    ax1.set_xlabel('Scattering Angle (degrees)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'Rejection Sampling (N={N}, Efficiency={efficiency_rejection:.1f}%)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inverse CDF method histogram
    theta_deg_inv = np.degrees(samples_inverse_cdf)
    ax2.hist(theta_deg_inv, bins=bins, density=True, alpha=0.7, 
             color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Scattering Angle (degrees)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(f'Inverse CDF Sampling (N={N}, Efficiency=100%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overlay comparison
    ax3.hist(theta_deg_rej, bins=bins, density=True, alpha=0.5, 
             color='skyblue', label='Rejection', edgecolor='black')
    ax3.hist(theta_deg_inv, bins=bins, density=True, alpha=0.5, 
             color='lightcoral', label='Inverse CDF', edgecolor='black')
    
    # Add theoretical curve
    theta_theory = np.linspace(1, 179, 100)
    pdf_theory = rutherford_pdf_2d(np.radians(theta_theory))
    pdf_theory_normalized = pdf_theory / (np.trapz(pdf_theory, np.radians(theta_theory)) * 180/np.pi)
    ax3.plot(theta_theory, pdf_theory_normalized, 'r-', linewidth=3, label='Theory')
    
    ax3.set_xlabel('Scattering Angle (degrees)')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('Method Comparison vs Theory')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency comparison
    sample_sizes = np.array([100, 500, 1000, 5000, 10000])
    efficiencies = []
    
    for n in sample_sizes:
        _, attempts = sample_rutherford_rejection(n)
        eff = n / attempts * 100
        efficiencies.append(eff)
    
    ax4.plot(sample_sizes, efficiencies, 'bo-', label='Rejection Efficiency')
    ax4.axhline(y=100, color='red', linestyle='--', label='Inverse CDF (100%)')
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('Sampling Efficiency Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n📊 Sampling Method Comparison:")
    print(f"   Rejection Method:")
    print(f"     • Efficiency: {efficiency_rejection:.1f}%")
    print(f"     • Attempts needed: {n_attempts:,}")
    print(f"     • Mean angle: {np.degrees(np.mean(samples_rejection)):.1f}°")
    print(f"   Inverse CDF Method:")
    print(f"     • Efficiency: 100.0%")
    print(f"     • Attempts needed: {N:,}")
    print(f"     • Mean angle: {np.degrees(np.mean(samples_inverse_cdf)):.1f}°")

# Example usage
if __name__ == "__main__":
    # Compare sampling methods
    compare_sampling_methods(N=5000)
    
    # Test individual methods
    print("\nTesting rejection sampling...")
    samples_rej, attempts = sample_rutherford_rejection(100)
    print(f"Generated {len(samples_rej)} samples in {attempts} attempts")
    
    print("\nTesting inverse CDF sampling...")
    samples_inv = sample_rutherford_inverse_cdf(100)
    print(f"Generated {len(samples_inv)} samples (100% efficiency)")
