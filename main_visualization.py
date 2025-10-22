"""
main_visualization.py

Main script to demonstrate 2D Rutherford scattering with both rejection and inverse CDF methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle import initialize_particles_2d, plot_trajectories_2d
from scattering import (sample_rutherford_rejection, sample_rutherford_inverse_cdf, 
                       compare_sampling_methods, simulate_2d_scattering)
from analysis import create_comprehensive_comparison, plot_full_simulation_2d

def quick_demo():
    """
    Quick demonstration of both sampling methods.
    """
    print("🎯 Quick Demo: Rejection vs Inverse CDF Methods")
    print("=" * 50)
    
    N = 5000
    
    # Test both methods
    print(f"\n1. Sampling {N} angles with rejection method...")
    angles_rej, attempts = sample_rutherford_rejection(N)
    efficiency = N / attempts * 100
    print(f"   • Generated {len(angles_rej)} samples in {attempts} attempts")
    print(f"   • Efficiency: {efficiency:.1f}%")
    
    print(f"\n2. Sampling {N} angles with inverse CDF method...")
    angles_inv = sample_rutherford_inverse_cdf(N)
    print(f"   • Generated {len(angles_inv)} samples (100% efficiency)")
    
    # Quick comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rejection method
    ax1.hist(np.degrees(angles_rej), bins=40, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    ax1.set_xlabel('Scattering Angle (degrees)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'Rejection Method\n(Efficiency: {efficiency:.1f}%)')
    ax1.grid(True, alpha=0.3)
    
    # Inverse CDF method
    ax2.hist(np.degrees(angles_inv), bins=40, density=True, alpha=0.7, 
             color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Scattering Angle (degrees)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Inverse CDF Method\n(Efficiency: 100%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Statistics Comparison:")
    print(f"   Rejection Method  - Mean: {np.degrees(np.mean(angles_rej)):.1f}°, Std: {np.degrees(np.std(angles_rej)):.1f}°")
    print(f"   Inverse CDF Method - Mean: {np.degrees(np.mean(angles_inv)):.1f}°, Std: {np.degrees(np.std(angles_inv)):.1f}°")

def demo_2d_trajectories():
    """
    Demonstrate 2D particle trajectories.
    """
    print("\n🚀 2D Trajectory Simulation")
    print("=" * 30)
    
    # Initialize particles
    N = 30
    particles = initialize_particles_2d(N, x_start=-6.0, y_range=(-3.0, 3.0))
    
    print(f"Simulating {N} particle trajectories...")
    
    # Run simulation
    simulate_2d_scattering(particles, alpha=1.2, dt=0.01)
    
    # Plot results
    plot_trajectories_2d(particles, target_pos=(0, 0), target_radius=0.15)
    
    # Count scattered particles
    scattered = sum(1 for p in particles if p.scattered)
    print(f"Result: {scattered}/{N} particles scattered")

def interactive_demo():
    """
    Interactive demonstration with user choices.
    """
    print("\n🎮 Interactive Demo")
    print("=" * 20)
    print("Choose what to visualize:")
    print("1. Quick sampling comparison")
    print("2. 2D trajectory simulation")
    print("3. Comprehensive analysis (all plots)")
    print("4. Method efficiency comparison")
    
    choice = input("\nEnter choice (1-4) or press Enter for all: ").strip()
    
    if choice == "1" or choice == "":
        quick_demo()
    
    if choice == "2" or choice == "":
        demo_2d_trajectories()
    
    if choice == "3" or choice == "":
        print("\nRunning comprehensive analysis...")
        create_comprehensive_comparison()
    
    if choice == "4" or choice == "":
        print("\nComparing method efficiencies...")
        compare_sampling_methods(N=8000, bins=50)

def main():
    """
    Main function - run the complete demonstration.
    """
    print("🎯 2D Rutherford Scattering Simulation")
    print("=" * 50)
    print("This demo compares rejection sampling vs inverse CDF methods")
    print("and shows 2D particle trajectory visualization.")
    
    # Run interactive demo
    interactive_demo()
    
    print("\n✅ Demo complete!")
    print("\nKey takeaways:")
    print("• Inverse CDF method is 100% efficient (no rejected samples)")
    print("• Rejection method efficiency depends on the PDF shape")
    print("• Both methods produce the same Rutherford distribution")
    print("• 2D trajectories show realistic scattering physics")

if __name__ == "__main__":
    main()