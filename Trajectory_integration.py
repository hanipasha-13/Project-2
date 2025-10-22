import numpy as np
from scipy.integrate import solve_ivp 
import matplotlib.pyplot as plt

class TrajectoryIntegrator:
    """
    Class for integrating particle trajectories in Rutherford scattering.
    Uses Coulomb force in 2D geometry.
    """

    def __init__(self, Z1, Z2, m1, m2):
        """
        Initialize integrator with nuclear properties.
        
        Parameters:
        -----------
        Z1, Z2 : float
            Atomic numbers of projectile and target
        m1, m2 : float
            Masses in amu
        """
        self.Z1 = Z1
        self.Z2 = Z2
        self.m1 = m1
        self.m2 = m2
        
        # Physical constants
        self.e = 1.6e-19  # Elementary charge (C)
        self.epsilon_0 = 8.85e-12  # Permittivity (C^2/(N*m^2))
        self.k = (self.Z1 * self.Z2 * self.e**2) / (4 * np.pi * self.epsilon_0)

    def coulomb_force(self, r):
        """
        Calculate repulsive Coulomb force.
        
        Parameters:
        -----------
        r : ndarray
            Position vector [x, y] in meters
            
        Returns:
        --------
        F : ndarray
            Force vector [Fx, Fy] in Newtons
        """
        # Target at origin
        r_vec = r - np.array([0.0, 0.0])
        r_mag = np.linalg.norm(r_vec)
        
        # Avoid singularity
        if r_mag < 1e-15:
            return np.array([0.0, 0.0])
        
        # F = k/r^2 in direction of r
        F_mag = self.k / (r_mag**2)
        F_vec = F_mag * (r_vec / r_mag)
        
        return F_vec

    def equations_of_motion(self, t, y):
        """
        ODE system for trajectory: dr/dt = v, dv/dt = F/m
        
        Parameters:
        -----------
        t : float
            Time
        y : ndarray
            State vector [x, y, vx, vy]
            
        Returns:
        --------
        dydt : ndarray
            Time derivative [vx, vy, ax, ay]
        """
        # Unpack state
        pos = y[0:2]
        vel = y[2:4]
        
        # Get force and acceleration
        F = self.coulomb_force(pos)
        m = self.m1 * 1.66054e-27  # Convert amu to kg
        acc = F / m
        
        # Return derivatives
        return np.concatenate([vel, acc])

    def integrate_trajectory(self, particle, t_max=1e-19, rtol=1e-9):
        """
        Integrate trajectory using adaptive RK method.
        Stops when particle returns to initial distance after scattering.
        
        Parameters:
        -----------
        particle : dict
            Initial conditions: 'r0', 'v0'
        t_max : float
            Maximum integration time (s)
        rtol : float
            Relative tolerance
            
        Returns:
        --------
        trajectory : dict
            Time, position, velocity arrays
        """
        # Set up initial state
        y0 = np.concatenate([particle['r0'], particle['v0']])
        
        # Store initial distance and y-position for event detection
        r_initial = np.linalg.norm(particle['r0'])
        y_initial = particle['r0'][1]
        min_distance = [r_initial]  # Track minimum distance reached
        
        def stop_condition(t, y):
            """
            Stop when particle crosses back past initial y-position after scattering.
            This is more robust than distance-based stopping.
            """
            y_current = y[1]  # Current y position
            r_current = np.linalg.norm(y[0:2])
            
            # Track minimum distance
            if r_current < min_distance[0]:
                min_distance[0] = r_current
            
            # Stop when:
            # 1. We've passed the target (y > 0)
            # 2. We've gotten reasonably close (within 2x initial distance)
            # 3. We're now past the initial y-position
            if min_distance[0] < 2.0 * r_initial and y_current > abs(y_initial):
                return 0  # Trigger event
            else:
                return 1  # Keep going
        
        stop_condition.terminal = True
        stop_condition.direction = -1  # Trigger when going from positive to zero
        
        # Integrate
        sol = solve_ivp(
            self.equations_of_motion,
            (0, t_max),
            y0,
            method='DOP853',  # 8th order Runge-Kutta
            rtol=rtol,
            atol=1e-12,
            events=stop_condition,
            dense_output=True
        )
        
        # Package results
        result = {
            't': sol.t,
            'r': sol.y[0:2, :],
            'v': sol.y[2:4, :],
            'success': sol.success
        }
        
        return result
    
    def calculate_scattering_angle(self, particle, trajectory):
        """
        Compute scattering angle from velocity change.
        
        Parameters:
        -----------
        particle : dict
            Initial conditions
        trajectory : dict
            Integrated trajectory
            
        Returns:
        --------
        theta : float
            Scattering angle (radians)
        """
        v_i = particle['v0']
        v_f = trajectory['v'][:, -1]
        
        # Normalize velocities
        v_i_hat = v_i / np.linalg.norm(v_i)
        v_f_hat = v_f / np.linalg.norm(v_f)
        
        # Angle between initial and final directions
        cos_theta = np.dot(v_i_hat, v_f_hat)
        cos_theta = np.clip(cos_theta, -1, 1)  # Handle numerical errors
        theta = np.arccos(cos_theta)
        
        return theta
    
    def validate_energy_conservation(self, particle, trajectory):
        """
        Check conservation of kinetic energy (elastic scattering).
        
        Returns:
        --------
        E_initial, E_final : float
            Initial and final kinetic energies in MeV
        percent_error : float
            Percentage error in energy conservation
        """
        m_kg = self.m1 * 1.66054e-27
        
        # Initial kinetic energy
        v_i = np.linalg.norm(particle['v0'])
        E_i = 0.5 * m_kg * v_i**2
        
        # Final kinetic energy
        v_f = np.linalg.norm(trajectory['v'][:, -1])
        E_f = 0.5 * m_kg * v_f**2
        
        # Convert to MeV
        E_i_MeV = E_i / 1.60218e-13
        E_f_MeV = E_f / 1.60218e-13
        
        # Percent error
        percent_error = abs(E_f_MeV - E_i_MeV) / E_i_MeV * 100
        
        return E_i_MeV, E_f_MeV, percent_error
    
    def validate_rutherford_formula(self, particle, trajectory):
        """
        Compare numerical scattering angle with Rutherford's analytical formula.
        
        Rutherford formula: cot(theta/2) = 2*E*b / (k*Z1*Z2*e^2)
        Or: theta = 2*arctan(k / (2*E*b))
        
        Returns:
        --------
        theta_numerical : float
            Scattering angle from simulation (degrees)
        theta_analytical : float
            Scattering angle from Rutherford formula (degrees)
        percent_error : float
            Percentage difference
        """
        # Get numerical scattering angle
        theta_num = self.calculate_scattering_angle(particle, trajectory)
        
        # Calculate analytical angle using Rutherford formula
        b = abs(particle['r0'][0])  # Impact parameter
        v0 = np.linalg.norm(particle['v0'])
        m_kg = self.m1 * 1.66054e-27
        E_kinetic = 0.5 * m_kg * v0**2  # Kinetic energy in Joules
        
        # Rutherford scattering formula
        # theta = 2 * arctan(k / (2 * E * b))
        theta_analytical = 2 * np.arctan(self.k / (2 * E_kinetic * b))
        
        # Convert to degrees
        theta_num_deg = np.degrees(theta_num)
        theta_analytical_deg = np.degrees(theta_analytical)
        
        # Percent error
        percent_error = abs(theta_analytical_deg - theta_num_deg) / theta_analytical_deg * 100
        
        return theta_num_deg, theta_analytical_deg, percent_error

    def plot_trajectory(self, trajectory, color='b', label=None, ax=None, linestyle='-'):
        """
        Visualize 2D trajectory.
        Can plot on existing axis for multiple trajectories.
        """
        # Convert to Angstroms for plotting
        r = trajectory["r"] * 1e10

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_plot = True
        else:
            show_plot = False
        
        # Plot trajectory
        if label:
            ax.plot(r[0, :], r[1, :], color=color, lw=2, label=label, alpha=0.8, linestyle=linestyle)
        else:
            ax.plot(r[0, :], r[1, :], color=color, lw=2, alpha=0.8, linestyle=linestyle)
        
        # Mark start point
        ax.plot(r[0, 0], r[1, 0], 'o', color=color, ms=6, alpha=0.6)
        
        if show_plot:
            # Mark target
            ax.plot(0, 0, 'ro', ms=12, label="Target Nucleus")
            
            ax.set_xlabel("x (Å)", fontsize=13)
            ax.set_ylabel("y (Å)", fontsize=13)
            ax.set_title("Rutherford Scattering Trajectory", fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax.axvline(0, color='k', lw=0.5, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return ax


# ==================== Main Simulation ====================

def run_single_trajectory():
    """
    Run a single Rutherford scattering calculation with validation.
    """
    
    print("\n" + "="*60)
    print("    Single Trajectory Simulation with Validation")
    print("="*60)
    
    # System parameters
    Z1 = 2      # Alpha particle (He-4)
    Z2 = 79     # Gold nucleus (Au-197)
    m1 = 4.0    # amu
    m2 = 197.0  # amu
    
    print(f"\nProjectile: He-{int(m1)} (Z = {Z1})")
    print(f"Target:     Au-{int(m2)} (Z = {Z2})")
    
    # Initialize integrator
    integrator = TrajectoryIntegrator(Z1, Z2, m1, m2)
    
    # Set up initial conditions
    b = 2e-14       # Impact parameter (m)
    y0 = -1e-13     # Initial y position (m)
    
    # Energy: 5 MeV alpha particle
    E_kin = 5.0  # MeV
    E_joules = E_kin * 1.60218e-13
    mass_kg = m1 * 1.66054e-27
    v0 = np.sqrt(2 * E_joules / mass_kg)
    
    particle = {
        'r0': np.array([b, y0]),
        'v0': np.array([0.0, v0])
    }
    
    print(f"\nInitial Conditions:")
    print(f"  Impact parameter: {b*1e10:.2f} Å")
    print(f"  Starting distance: {abs(y0)*1e10:.2f} Å")
    print(f"  Kinetic energy: {E_kin:.1f} MeV")
    
    # Run integration
    print("\nIntegrating...")
    traj = integrator.integrate_trajectory(particle, t_max=1e-18, rtol=1e-10)
    
    if not traj['success']:
        print("ERROR: Integration failed!")
        return
    
    print(f"✓ Integration successful ({len(traj['t'])} steps)")
    
    # Validation 1: Energy Conservation
    print("\n" + "-"*60)
    print("VALIDATION 1: Energy Conservation")
    print("-"*60)
    E_i, E_f, e_error = integrator.validate_energy_conservation(particle, traj)
    print(f"  Initial KE: {E_i:.6f} MeV")
    print(f"  Final KE:   {E_f:.6f} MeV")
    print(f"  Error:      {e_error:.4e} %")
    if e_error < 0.01:
        print("  ✓ Energy conserved (error < 0.01%)")
    else:
        print("  ⚠ Warning: Energy not well conserved")
    
    # Validation 2: Rutherford Formula
    print("\n" + "-"*60)
    print("VALIDATION 2: Rutherford Scattering Formula")
    print("-"*60)
    theta_num, theta_analytical, theta_error = integrator.validate_rutherford_formula(particle, traj)
    print(f"  Numerical angle:   {theta_num:.4f}°")
    print(f"  Analytical angle:  {theta_analytical:.4f}°")
    print(f"  Error:             {theta_error:.4e} %")
    if theta_error < 1.0:
        print("  ✓ Agrees with Rutherford formula (error < 1%)")
    else:
        print("  ⚠ Warning: Deviation from Rutherford formula")
    
    print("="*60 + "\n")
    
    # Plot
    integrator.plot_trajectory(traj)


def run_multiple_trajectories():
    """
    Run multiple trajectories with different impact parameters and energies.
    Plot them all on the same figure with validation.
    """
    
    print("\n" + "="*60)
    print("    Multiple Trajectory Simulation with Validation")
    print("="*60)
    
    # System setup
    Z1 = 2
    Z2 = 79
    m1 = 4.0
    m2 = 197.0
    
    integrator = TrajectoryIntegrator(Z1, Z2, m1, m2)
    mass_kg = m1 * 1.66054e-27
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define different initial conditions
    # Vary impact parameters with same energy
    impact_params = [0.5e-14, 1e-14, 2e-14, 3e-14, 4e-14]  # meters
    energies = [5.0]  # MeV - keep energy constant first
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'cyan', 'magenta']
    y0 = -1.5e-13  # Starting position - further back for larger impact params
    
    print("\nVarying impact parameter (fixed E = 5 MeV):")
    print("-"*60)
    print(f"{'b (Å)':<8} {'θ_num (°)':<12} {'θ_Ruth (°)':<12} {'Error (%)':<12} {'ΔE (%)':<10}")
    print("-"*60)
    
    color_idx = 0
    for b in impact_params:
        E_kin = energies[0]
        E_joules = E_kin * 1.60218e-13
        v0 = np.sqrt(2 * E_joules / mass_kg)
        
        particle = {
            'r0': np.array([b, y0]),
            'v0': np.array([0.0, v0])
        }
        
        # Integrate
        traj = integrator.integrate_trajectory(particle, t_max=1e-18, rtol=1e-10)
        
        if traj['success']:
            # Get scattering angles
            theta_num, theta_ruth, angle_error = integrator.validate_rutherford_formula(particle, traj)
            
            # Get energy conservation
            _, _, energy_error = integrator.validate_energy_conservation(particle, traj)
            
            label = f"b={b*1e10:.1f}Å"
            print(f"{b*1e10:<8.1f} {theta_num:<12.4f} {theta_ruth:<12.4f} {angle_error:<12.4f} {energy_error:<10.2e}")
            
            # Plot on same axis
            integrator.plot_trajectory(traj, color=colors[color_idx], label=label, ax=ax)
            color_idx += 1
    
    # Now vary energy with one impact parameter
    print("\nVarying energy (fixed b = 2.0 Å):")
    print("-"*60)
    print(f"{'E (MeV)':<8} {'θ_num (°)':<12} {'θ_Ruth (°)':<12} {'Error (%)':<12} {'ΔE (%)':<10}")
    print("-"*60)
    
    b_fixed = 2e-14
    energies_vary = [3.0, 7.0, 10.0]  # MeV
    
    for E_kin in energies_vary:
        E_joules = E_kin * 1.60218e-13
        v0 = np.sqrt(2 * E_joules / mass_kg)
        
        particle = {
            'r0': np.array([b_fixed, y0]),
            'v0': np.array([0.0, v0])
        }
        
        traj = integrator.integrate_trajectory(particle, t_max=1e-18, rtol=1e-10)
        
        if traj['success']:
            # Get scattering angles
            theta_num, theta_ruth, angle_error = integrator.validate_rutherford_formula(particle, traj)
            
            # Get energy conservation
            _, _, energy_error = integrator.validate_energy_conservation(particle, traj)
            
            label = f"E={E_kin:.1f}MeV"
            print(f"{E_kin:<8.1f} {theta_num:<12.4f} {theta_ruth:<12.4f} {angle_error:<12.4f} {energy_error:<10.2e}")
            
            integrator.plot_trajectory(traj, color=colors[color_idx], 
                                      label=label, ax=ax, linestyle='--')
            color_idx += 1
    
    # Finalize plot
    ax.plot(0, 0, 'ro', ms=15, label="Au-197 Target", zorder=10)
    ax.set_xlabel("x (Å)", fontsize=13)
    ax.set_ylabel("y (Å)", fontsize=13)
    ax.set_title("Rutherford Scattering - Multiple Trajectories", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("="*60)
    print("Simulation complete.\n")


if __name__ == "__main__":
    # Run single trajectory with validation
    # run_single_trajectory()
    
    # Run multiple trajectories with validation table
    run_multiple_trajectories()
"""
Rutherford Scattering Monte Carlo Simulation

This module implements Monte Carlo methods to simulate Rutherford scattering.
Uses both inverse CDF and rejection sampling methods.

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class TrajectoryIntegrator:
    """
    Class for integrating particle trajectories in Rutherford scattering.
    Uses Coulomb force in 2D geometry.

    Parameters
    ----------
    Z1 : float
        Atomic number of projectile
    Z2 : float
        Atomic number of target
    m1 : float
        Mass of projectile in amu
    m2 : float
        Mass of target in amu
    """

    def __init__(self, Z1, Z2, m1, m2):
        self.Z1 = Z1
        self.Z2 = Z2
        self.m1 = m1
        self.m2 = m2

        # Physical constants
        self.e = 1.6e-19  # Elementary charge (C)
        self.epsilon_0 = 8.85e-12  # Permittivity (C^2/(N*m^2))
        self.k = (self.Z1 * self.Z2 * self.e**2) / (4 * np.pi * self.epsilon_0)

    def coulomb_force(self, r):
        """
        Calculate repulsive Coulomb force.

        Parameters
        ----------
        r : ndarray
            Position vector [x, y] in meters

        Returns
        -------
        F : ndarray
            Force vector [Fx, Fy] in Newtons
        """
        # Target at origin
        r_vec = r - np.array([0.0, 0.0])
        r_mag = np.linalg.norm(r_vec)

        # Avoid singularity
        if r_mag < 1e-15:
            return np.array([0.0, 0.0])

        # F = k/r^2 in direction of r
        F_mag = self.k / (r_mag**2)
        F_vec = F_mag * (r_vec / r_mag)

        return F_vec

    def equations_of_motion(self, t, y):
        """
        ODE system for trajectory: dr/dt = v, dv/dt = F/m

        Parameters
        ----------
        t : float
            Time
        y : ndarray
            State vector [x, y, vx, vy]

        Returns
        -------
        dydt : ndarray
            Time derivative [vx, vy, ax, ay]
        """
        # Unpack state
        pos = y[0:2]
        vel = y[2:4]

        # Get force and acceleration
        F = self.coulomb_force(pos)
        m = self.m1 * 1.66054e-27  # Convert amu to kg
        acc = F / m

        # Return derivatives
        return np.concatenate([vel, acc])

    def integrate_trajectory(self, particle, t_max=1e-19, rtol=1e-9):
        """
        Integrate trajectory using adaptive RK method.
        Stops when particle returns to initial distance after scattering.

        Parameters
        ----------
        particle : dict
            Initial conditions: 'r0', 'v0'
        t_max : float
            Maximum integration time (s)
        rtol : float
            Relative tolerance

        Returns
        -------
        trajectory : dict
            Time, position, velocity arrays
        """
        # Set up initial state
        y0 = np.concatenate([particle["r0"], particle["v0"]])

        # Store initial distance and y-position for event detection
        r_initial = np.linalg.norm(particle["r0"])
        y_initial = particle["r0"][1]
        min_distance = [r_initial]  # Track minimum distance reached

        def stop_condition(t, y):
            """
            Stop when particle crosses back past initial y-position after scattering.
            This is more robust than distance-based stopping.
            """
            y_current = y[1]  # Current y position
            r_current = np.linalg.norm(y[0:2])

            # Track minimum distance
            if r_current < min_distance[0]:
                min_distance[0] = r_current

            # Stop when:
            # 1. We've passed the target (y > 0)
            # 2. We've gotten reasonably close (within 2x initial distance)
            # 3. We're now past the initial y-position
            if min_distance[0] < 2.0 * r_initial and y_current > abs(y_initial):
                return 0  # Trigger event
            else:
                return 1  # Keep going

        stop_condition.terminal = True
        stop_condition.direction = -1  # Trigger when going from positive to zero

        # Integrate
        sol = solve_ivp(
            self.equations_of_motion,
            (0, t_max),
            y0,
            method="DOP853",  # 8th order Runge-Kutta
            rtol=rtol,
            atol=1e-12,
            events=stop_condition,
            dense_output=True,
        )

        # Package results
        result = {
            "t": sol.t,
            "r": sol.y[0:2, :],
            "v": sol.y[2:4, :],
            "success": sol.success,
        }

        return result

    def calculate_scattering_angle(self, particle, trajectory):
        """
        Compute scattering angle from velocity change.

        Parameters
        ----------
        particle : dict
            Initial conditions
        trajectory : dict
            Integrated trajectory

        Returns
        -------
        theta : float
            Scattering angle (radians)
        """
        v_i = particle["v0"]
        v_f = trajectory["v"][:, -1]

        # Normalize velocities
        v_i_hat = v_i / np.linalg.norm(v_i)
        v_f_hat = v_f / np.linalg.norm(v_f)

        # Angle between initial and final directions
        cos_theta = np.dot(v_i_hat, v_f_hat)
        cos_theta = np.clip(cos_theta, -1, 1)  # Handle numerical errors
        theta = np.arccos(cos_theta)

        return theta

    def validate_energy_conservation(self, particle, trajectory):
        """
        Check conservation of kinetic energy (elastic scattering).

        Parameters
        ----------
        particle : dict
            Initial conditions
        trajectory : dict
            Integrated trajectory

        Returns
        -------
        E_initial : float
            Initial kinetic energy in MeV
        E_final : float
            Final kinetic energy in MeV
        percent_error : float
            Percentage error in energy conservation
        """
        m_kg = self.m1 * 1.66054e-27

        # Initial kinetic energy
        v_i = np.linalg.norm(particle["v0"])
        E_i = 0.5 * m_kg * v_i**2

        # Final kinetic energy
        v_f = np.linalg.norm(trajectory["v"][:, -1])
        E_f = 0.5 * m_kg * v_f**2

        # Convert to MeV
        E_i_MeV = E_i / 1.60218e-13
        E_f_MeV = E_f / 1.60218e-13

        # Percent error
        percent_error = abs(E_f_MeV - E_i_MeV) / E_i_MeV * 100

        return E_i_MeV, E_f_MeV, percent_error

    def validate_rutherford_formula(self, particle, trajectory):
        """
        Compare numerical scattering angle with Rutherford's analytical formula.

        Rutherford formula: theta = 2*arctan(k / (2*E*b))

        Parameters
        ----------
        particle : dict
            Initial conditions
        trajectory : dict
            Integrated trajectory

        Returns
        -------
        theta_numerical : float
            Scattering angle from simulation (degrees)
        theta_analytical : float
            Scattering angle from Rutherford formula (degrees)
        percent_error : float
            Percentage difference
        """
        # Get numerical scattering angle
        theta_num = self.calculate_scattering_angle(particle, trajectory)

        # Calculate analytical angle using Rutherford formula
        b = abs(particle["r0"][0])  # Impact parameter
        v0 = np.linalg.norm(particle["v0"])
        m_kg = self.m1 * 1.66054e-27
        E_kinetic = 0.5 * m_kg * v0**2  # Kinetic energy in Joules

        # Rutherford scattering formula
        theta_analytical = 2 * np.arctan(self.k / (2 * E_kinetic * b))

        # Convert to degrees
        theta_num_deg = np.degrees(theta_num)
        theta_analytical_deg = np.degrees(theta_analytical)

        # Percent error
        percent_error = (
            abs(theta_analytical_deg - theta_num_deg) / theta_analytical_deg * 100
        )

        return theta_num_deg, theta_analytical_deg, percent_error

    def plot_trajectory(
        self, trajectory, color="b", label=None, ax=None, linestyle="-"
    ):
        """
        Visualize 2D trajectory.

        Parameters
        ----------
        trajectory : dict
            Trajectory data with position array
        color : str
            Line color
        label : str
            Legend label
        ax : matplotlib axis
            Axis to plot on (creates new if None)
        linestyle : str
            Line style

        Returns
        -------
        ax : matplotlib axis
            The axis used for plotting
        """
        # Convert to Angstroms for plotting
        r = trajectory["r"] * 1e10

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_plot = True
        else:
            show_plot = False

        # Plot trajectory
        if label:
            ax.plot(
                r[0, :],
                r[1, :],
                color=color,
                lw=2,
                label=label,
                alpha=0.8,
                linestyle=linestyle,
            )
        else:
            ax.plot(r[0, :], r[1, :], color=color, lw=2, alpha=0.8, linestyle=linestyle)

        # Mark start point
        ax.plot(r[0, 0], r[1, 0], "o", color=color, ms=6, alpha=0.6)

        if show_plot:
            # Mark target
            ax.plot(0, 0, "ro", ms=12, label="Target Nucleus")

            ax.set_xlabel("x (A)", fontsize=13)
            ax.set_ylabel("y (A)", fontsize=13)
            ax.set_title(
                "Rutherford Scattering Trajectory", fontsize=14, fontweight="bold"
            )
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax.axvline(0, color="k", lw=0.5, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return ax


def run_multiple_trajectories():
    """
    Run multiple trajectories with different impact parameters and energies.
    Plot them all on the same figure with validation.
    """

    print("\n" + "=" * 60)
    print("    Multiple Trajectory Simulation with Validation")
    print("=" * 60)

    # System setup
    Z1 = 2  # Alpha particle
    Z2 = 79  # Gold nucleus
    m1 = 4.0  # amu
    m2 = 197.0  # amu

    integrator = TrajectoryIntegrator(Z1, Z2, m1, m2)
    mass_kg = m1 * 1.66054e-27

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define different initial conditions
    # Impact parameters in femtometers (fm = 1e-15 m)
    impact_params_fm = [5.0, 10.0, 20.0, 30.0, 40.0]  # femtometers
    impact_params = [b * 1e-15 for b in impact_params_fm]  # convert to meters
    energies = [5.0]  # MeV - keep energy constant first

    colors = [
        "blue",
        "green",
        "orange",
        "red",
        "purple",
        "brown",
        "pink",
        "cyan",
        "magenta",
    ]
    y0 = -1.5e-13  # Starting position

    print("\nVarying impact parameter (fixed E = 5 MeV):")
    print("-" * 75)
    print(
        f"{'b (fm)':<10} {'θ_num (°)':<12} {'θ_Ruth (°)':<12} {'Error (%)':<12} {'ΔE (%)':<10}"
    )
    print("-" * 75)

    color_idx = 0
    for b, b_fm in zip(impact_params, impact_params_fm):
        E_kin = energies[0]
        E_joules = E_kin * 1.60218e-13
        v0 = np.sqrt(2 * E_joules / mass_kg)

        particle = {"r0": np.array([b, y0]), "v0": np.array([0.0, v0])}

        # Integrate
        traj = integrator.integrate_trajectory(particle, t_max=1e-18, rtol=1e-10)

        if traj["success"]:
            # Get scattering angles
            theta_num, theta_ruth, angle_error = integrator.validate_rutherford_formula(
                particle, traj
            )

            # Get energy conservation
            _, _, energy_error = integrator.validate_energy_conservation(particle, traj)

            label = f"b={b_fm:.1f} fm"
            print(
                f"{b_fm:<10.1f} {theta_num:<12.4f} {theta_ruth:<12.4f} {angle_error:<12.4f} {energy_error:<10.2e}"
            )

            # Plot on same axis
            integrator.plot_trajectory(
                traj, color=colors[color_idx], label=label, ax=ax
            )
            color_idx += 1

    # Now vary energy with one impact parameter
    print("\nVarying energy (fixed b = 20.0 fm):")
    print("-" * 75)
    print(
        f"{'E (MeV)':<10} {'θ_num (°)':<12} {'θ_Ruth (°)':<12} {'Error (%)':<12} {'ΔE (%)':<10}"
    )
    print("-" * 75)

    b_fixed = 20.0e-15  # 20 fm in meters
    energies_vary = [3.0, 7.0, 10.0]  # MeV

    for E_kin in energies_vary:
        E_joules = E_kin * 1.60218e-13
        v0 = np.sqrt(2 * E_joules / mass_kg)

        particle = {"r0": np.array([b_fixed, y0]), "v0": np.array([0.0, v0])}

        traj = integrator.integrate_trajectory(particle, t_max=1e-18, rtol=1e-10)

        if traj["success"]:
            # Get scattering angles
            theta_num, theta_ruth, angle_error = integrator.validate_rutherford_formula(
                particle, traj
            )

            # Get energy conservation
            _, _, energy_error = integrator.validate_energy_conservation(particle, traj)

            label = f"E={E_kin:.1f} MeV"
            print(
                f"{E_kin:<10.1f} {theta_num:<12.4f} {theta_ruth:<12.4f} {angle_error:<12.4f} {energy_error:<10.2e}"
            )

            integrator.plot_trajectory(
                traj, color=colors[color_idx], label=label, ax=ax, linestyle="--"
            )
            color_idx += 1

    # Finalize plot
    ax.plot(0, 0, "ro", ms=15, label="Au-197 Target", zorder=10)
    ax.set_xlabel("x (Angstrom)", fontsize=13)
    ax.set_ylabel("y (Angstrom)", fontsize=13)
    ax.set_title(
        "Rutherford Scattering - Multiple Trajectories", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rutherford_trajectories.png", dpi=200)
    print("\nPlot saved as rutherford_trajectories.png")
    plt.show()

    print("=" * 60)
    print("Simulation complete.\n")


if __name__ == "__main__":
    # Run multiple trajectories with validation
    run_multiple_trajectories()