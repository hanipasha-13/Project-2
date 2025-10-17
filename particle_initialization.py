import numpy as np
import matplotlib.pyplot as plt

class ParticleInitializer:
    """class to initialize particles in Rutherford scattering simulations.
    Parameters:
    Z1, Z2 : float Atomic numbers of the projectile and target nuclei
    E : float kinetic energy of the projectile (in KeV)
    bmax: float maximum impact paramete r (in Angstrom)
    m1 : float mass of the projectile (in amu)"""

def __init__(self, Z1, Z2, E, bmax, m1):
    self.Z1 = Z1  # Atomic number of projectile
    self.Z2 = Z2  # Atomic number of target
    self.E = E   #Kinetic energy of projectile in KeV
    self.bmax = bmax  #Maximum impact parameter in Angstrom
    self.m1 = m1      #Mass of projectile in amu
    self.c = 2.99e8 #Speed of light in Angstrom/s

    #derived quantities
    self.V0 = self.calulate_initial_velocity() #Initial velocity of projectile (in Angstrom)
    self.a = self.calculate_distance_of_closest_approach() #Distance of closest approach (in Angstrom)
    self.b = self.initialize_impact_parameter() #Impact parameter (in Angstrom)

def calulate_initial_velocity(self):
    """calulate initial velocity of the projectile form its kinetic energy 
    E = 0.5 * m * v^2
    V = sqrt ( 2 * E / m)"""
    m1_kg =  self.m1 * 1.66e-27 #Convert mass from amu to kg
    m1_MeV = m1_kg * 5.609e29 # Convert mass from kg to MeV/c^2
    v = np.sqrt(2 * self.E / m1_MeV) * self.c 
    return v #in Angstrom/s

def calculate_distance_of_closest_approach(self):
    """ calculate distance of closest approach"""
    e = 1.6e-19 #Charge of electron in Coulomb
    epsilon_0 = 8.85e-12 #Permittivity of free space in C^2/(N m^2)
    E_joules = self.E * 1.602e-16 #Convert energy from KeV to joules
    d = (self.Z1 * self.Z2 * e**2) / (4 * np.pi * epsilon_0 * E_joules) #in meters
    d_angstrom = d * 1e10 #Convert distance from meters to Amgstrom
    return d_angstrom

def initialize_impact_parameter(self):
    """initialize impact parameter b uniformly between 0 and bmax"""
    b = np.random.uniform(0, self.bmax)
    return b

def initialize_particles(self, n_particles, b_max):
    """initialize particle positions and velocities
    n_particles : int Numbe of particles to initialize
    b_max: float Maximum impact parameter (in Angstrom)
    
    Returns:
        initial_positions (np.ndarray): Array of shape (n_particles, 3) with initial positions.
        initial_velocities (np.ndarray): Array of shape (n_particles, 3) with initial velocities.
    """
    # Example initialization 

    particles = []
    initial_positions = np.zeros((n_particles, 3))
    initial_velocities = np.zeros((n_particles, 3))

    for i in range(n_particles):
        b = np.random.uniform(0, b_max) # Impact parameter
        theta = np.random.uniform(0, 2 * np.pi) # Random angle around the z-axis
        x = -100 * self.a  # Start far away from the target nucleus
        y = b * np.cos(theta)
        z = b * np.sin(theta)
        initial_positions[i] = [x, y, z]
        initial_velocities[i] = [self.V0, 0, 0]  # Moving along x-axis

        particles.append({
            'impact_parameter' : b, 
            'theta' : theta,
            'initial_position': (x, y, z),
            'initial_velocity': (self.V0, 0, 0)
        })

    return initial_positions, initial_velocities

def get_parameters(self):
        """Return simulation parameters"""
        return {
            'Z1': self.Z1,
            'Z2': self.Z2,
            'm1': self.m1,
            'E0': self.E0,
            'v0': self.v0,
            'a': self.a,
            'k_e': self.k_e,
            'c': self.c
        }