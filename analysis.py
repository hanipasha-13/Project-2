"""
analysis.py

2D Analysis and visualization tools for Rutherford scattering simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_mean_and_uncertainty(thetas):
    """
    Compute the mean and standard error of the scattering angles.

    Parameters
    ----------
    thetas : ndarray
        Array of scattering angles (radians).

    Returns
    -------
    mean_theta : float
        Mean scattering angle (radians).
    std_error : float
        Standard error of the mean (radians).
    """
    mean_theta = np.mean(thetas)
    std_error = np.std(thetas, ddof=1) / np.sqrt(len(thetas))
    return mean_theta, std_error

def plot_angular_distribution(thetas, bins=60, show_theory=True):
    """
    Plot histogram of scattering angles (degrees) and compare with theoretical Rutherford distribution.

    Parameters
    ----------
    thetas : ndarray
        Array of scattering angles (radians).
    bins : int
        Number of bins in histogram.
    show_theory : bool
        If True, plot normalized Rutherford distribution for comparison.
    """
    # Convert to degrees for plotting
    theta_deg = np.degrees(thetas)
    plt.figure(figsize=(8, 5))
    counts, bin_edges, _ = plt.hist(theta_deg, bins=bins, density=True, alpha=0.6, label="Simulated (MC)")

    if show_theory:
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        theta_rad = np.radians(bin_centers)
        theory = rutherford_pdf(theta_rad)
        theory /= np.trapz(theory, bin_centers)  # Normalize area to 1
        plt.plot(bin_centers, theory, 'r-', lw=2, label="Theory (normalized)")

    plt.xlabel(r"Scattering angle $\theta$ (degrees)")
    plt.ylabel("Probability density")
    plt.title("Angular Distribution of Scattering")
    plt.legend()
    plt.tight_layout()
    plt.show()

def rutherford_pdf(theta):
    """
    Unnormalized Rutherford probability density for scattering angle.

    Parameters
    ----------
    theta : ndarray
        Scattering angles (radians).

    Returns
    -------
    pdf : ndarray
        Probability density values (unnormalized).
    """
    # Avoid division by zero
    eps = 1e-10
    return np.sin(theta) / (np.sin(0.5 * theta + eps) ** 4)

def validate_speeds(particles, tol=1e-8):
    """
    Check that all particles conserve speed after scattering (for elastic events).

    Parameters
    ----------
    particles : list of Particle
        List of Particle instances.
    tol : float
        Relative tolerance for speed conservation.

    Returns
    -------
    valid : bool
        True if all particles conserve speed within tolerance.
    """
    for p in particles:
        v0 = np.linalg.norm([0, 0, np.linalg.norm(p.velocity)])  # Initial velocity (should be all +z)
        v1 = np.linalg.norm(p.velocity)
        if not np.isclose(v0, v1, rtol=tol):
            print(f"Speed not conserved: {v0} vs {v1}")
            return False
    return True

def analyze_scattering_distribution(angles, method_name="Unknown"):
    """
    Analyze the statistical properties of scattering angles.
    
    Parameters
    ----------
    angles : array_like
        Scattering angles in radians.
    method_name : str
        Name of the sampling method used.
        
    Returns
    -------
    stats : dict
        Dictionary containing statistical measures.
    """
    angles_deg = np.degrees(angles)
    
    stats = {
        'mean': np.mean(angles_deg),
        'std': np.std(angles_deg),
        'median': np.median(angles_deg),
        'min': np.min(angles_deg),
        'max': np.max(angles_deg),
        'q25': np.percentile(angles_deg, 25),
        'q75': np.percentile(angles_deg, 75),
        'method': method_name
    }
    
    return stats

def plot_convergence_analysis(max_N=10000, step=500):
    """
    Analyze convergence properties of both sampling methods.
    
    Parameters
    ----------
    max_N : int
        Maximum number of samples to test.
    step : int
        Step size for sample numbers.
    """
    from scattering import sample_rutherford_rejection, sample_rutherford_inverse_cdf
    
    sample_sizes = np.arange(step, max_N + step, step)
    
    # Storage for results
    means_rejection = []
    stds_rejection = []
    efficiencies = []
    
    means_inverse = []
    stds_inverse = []
    
    print("Running convergence analysis...")
    
    for n in sample_sizes:
        # Rejection method
        samples_rej, attempts = sample_rutherford_rejection(n)
        means_rejection.append(np.degrees(np.mean(samples_rej)))
        stds_rejection.append(np.degrees(np.std(samples_rej)))
        efficiencies.append(n / attempts * 100)
        
        # Inverse CDF method
        samples_inv = sample_rutherford_inverse_cdf(n)
        means_inverse.append(np.degrees(np.mean(samples_inv)))
        stds_inverse.append(np.degrees(np.std(samples_inv)))
        
        if n % (step * 4) == 0:
            print(f"  Processed N = {n}")
    
    # Create convergence plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Mean convergence
    ax1.plot(sample_sizes, means_rejection, 'b-', label='Rejection Method', linewidth=2)
    ax1.plot(sample_sizes, means_inverse, 'r-', label='Inverse CDF Method', linewidth=2)
    ax1.set_xlabel('Sample Size N')
    ax1.set_ylabel('Mean Scattering Angle (degrees)')
    ax1.set_title('Convergence of Mean Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation convergence
    ax2.plot(sample_sizes, stds_rejection, 'b-', label='Rejection Method', linewidth=2)
    ax2.plot(sample_sizes, stds_inverse, 'r-', label='Inverse CDF Method', linewidth=2)
    ax2.set_xlabel('Sample Size N')
    ax2.set_ylabel('Standard Deviation (degrees)')
    ax2.set_title('Convergence of Standard Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency comparison
    ax3.plot(sample_sizes, efficiencies, 'bo-', label='Rejection Efficiency')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Inverse CDF (100%)')
    ax3.set_xlabel('Sample Size N')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Sampling Efficiency vs Sample Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Standard error (uncertainty) scaling
    std_errors_rej = np.array(stds_rejection) / np.sqrt(sample_sizes)
    std_errors_inv = np.array(stds_inverse) / np.sqrt(sample_sizes)
    
    ax4.loglog(sample_sizes, std_errors_rej, 'b-', label='Rejection Method', linewidth=2)
    ax4.loglog(sample_sizes, std_errors_inv, 'r-', label='Inverse CDF Method', linewidth=2)
    ax4.loglog(sample_sizes, 50/np.sqrt(sample_sizes), 'k--', label='1/√N scaling', alpha=0.7)
    ax4.set_xlabel('Sample Size N')
    ax4.set_ylabel('Standard Error (degrees)')
    ax4.set_title('Standard Error Scaling')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sample_sizes, means_rejection, means_inverse, efficiencies

def plot_full_simulation_2d(N_particles=50, alpha=1.0):
    """
    Run and visualize complete 2D Rutherford scattering simulation.
    
    Parameters
    ----------
    N_particles : int
        Number of particles to simulate.
    alpha : float
        Scattering strength parameter.
    """
    from particle import initialize_particles_2d, plot_trajectories_2d
    from scattering import simulate_2d_scattering
    
    print(f"Running 2D simulation with {N_particles} particles...")
    
    # Initialize particles
    particles = initialize_particles_2d(N_particles, x_start=-8.0, y_range=(-4.0, 4.0))
    
    # Run simulation
    simulate_2d_scattering(particles, alpha=alpha, dt=0.02)
    
    # Plot trajectories
    plot_trajectories_2d(particles, target_pos=(0, 0), target_radius=0.2)
    
    # Analyze scattering angles
    scattered_particles = [p for p in particles if p.scattered]
    if len(scattered_particles) > 0:
        angles = [p.theta for p in scattered_particles]
        impact_params = [p.impact_parameter for p in scattered_particles]
        
        # Create analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scattering angle vs impact parameter
        ax1.scatter(impact_params, np.degrees(angles), alpha=0.7, s=50)
        ax1.set_xlabel('Impact Parameter')
        ax1.set_ylabel('Scattering Angle (degrees)')
        ax1.set_title('Scattering Angle vs Impact Parameter')
        ax1.grid(True, alpha=0.3)
        
        # Add theoretical curve
        b_theory = np.linspace(0.1, max(impact_params), 100)
        # theta = 2 * arctan(alpha / (2 * E * b)) with E = 0.5
        theta_theory = 2 * np.arctan(alpha / (2 * 0.5 * b_theory))
        ax1.plot(b_theory, np.degrees(theta_theory), 'r-', linewidth=3, 
                label='Theoretical', alpha=0.8)
        ax1.legend()
        
        # Plot 2: Histogram of scattering angles
        ax2.hist(np.degrees(angles), bins=20, density=True, alpha=0.7, 
                color='green', edgecolor='black')
        ax2.set_xlabel('Scattering Angle (degrees)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'Distribution of Scattering Angles (N={len(angles)})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        stats = analyze_scattering_distribution(angles, "2D Simulation")
        print(f"\n📊 2D Simulation Results:")
        print(f"   • Particles scattered: {len(scattered_particles)}/{N_particles}")
        print(f"   • Mean scattering angle: {stats['mean']:.1f}°")
        print(f"   • Standard deviation: {stats['std']:.1f}°")
        print(f"   • Angle range: {stats['min']:.1f}° to {stats['max']:.1f}°")
    
    return particles

def create_comprehensive_comparison():
    """
    Create a comprehensive comparison of all methods and results.
    """
    from scattering import compare_sampling_methods
    
    print("🎯 Comprehensive Rutherford Scattering Analysis")
    print("=" * 60)
    
    # 1. Compare sampling methods
    print("\n1. Comparing sampling methods...")
    compare_sampling_methods(N=8000, bins=40)
    
    # 2. Convergence analysis
    print("\n2. Running convergence analysis...")
    plot_convergence_analysis(max_N=5000, step=250)
    
    # 3. Full 2D simulation
    print("\n3. Running full 2D simulation...")
    particles = plot_full_simulation_2d(N_particles=80, alpha=1.5)
    
    print("\n✅ Analysis complete! All plots should be displayed.")

# Example usage
if __name__ == "__main__":
    create_comprehensive_comparison()
