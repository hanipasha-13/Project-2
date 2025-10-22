"""
Rutherford Scattering Monte Carlo Simulation

This module implements Monte Carlo methods to simulate Rutherford scattering.
Uses both inverse CDF and rejection sampling methods.

"""

import numpy as np
import matplotlib.pyplot as plt
import time


class RutherfordSimulation:
    """
    Monte Carlo simulation for Rutherford scattering experiment.

    This class generates scattering angle samples using two different
    Monte Carlo methods and compares them to theoretical predictions.

    Parameters
    ----------
    num_samples : int
        Number of particles to simulate
    theta_min_deg : float
        Minimum scattering angle in degrees (to avoid singularity at 0)

    Attributes
    ----------
    samples_inv : ndarray
        Samples generated using inverse CDF method
    samples_rej : ndarray
        Samples generated using rejection sampling
    mean_theory : float
        Theoretical mean scattering angle
    mean_inv : float
        Mean from inverse CDF samples
    mean_rej : float
        Mean from rejection samples

    Examples
    --------
    >>> sim = RutherfordSimulation(num_samples=10000, theta_min_deg=10)
    >>> sim.run()
    >>> sim.plot()
    """

    def __init__(self, num_samples, theta_min_deg):
        self.num_samples = num_samples
        self.theta_min_deg = theta_min_deg
        self.theta_min = np.radians(theta_min_deg)  # convert to radians

        # these get filled in when we run the simulation
        self.samples_inv = None
        self.samples_rej = None
        self.norm = None
        self.mean_theory = None
        self.mean_inv = None
        self.mean_rej = None
        self.max_pdf = None

        # benchmark data
        self.time_inv = None
        self.time_rej = None
        self.time_total = None

    def pdf(self, theta):
        """
        Calculate Rutherford scattering probability density function.

        The PDF follows: p(theta) = 1 / sin^4(theta/2)

        Parameters
        ----------
        theta : float
            Scattering angle in radians

        Returns
        -------
        float
            PDF value at given angle
        """
        # calculate sin(theta/2)
        sin_half = np.sin(theta / 2)

        # avoid division by zero
        if sin_half == 0:
            sin_half = 1e-10

        # return the Rutherford formula
        return 1 / sin_half**4

    def get_norm(self):
        """
        Calculate normalization constant for the PDF.

        Integrates the PDF from theta_min to pi to get the normalization.

        Returns
        -------
        float
            Normalization constant
        """
        # create array of theta values
        theta_vals = np.linspace(self.theta_min, np.pi, 1000)

        # calculate PDF at each point
        pdf_vals = []
        for t in theta_vals:
            pdf_vals.append(self.pdf(t))

        # integrate using trapezoidal rule
        norm = np.trapz(pdf_vals, theta_vals)

        return norm

    def get_mean(self):
        """
        Calculate theoretical mean scattering angle.

        Computes E[theta] = integral of theta * p(theta) dtheta

        Returns
        -------
        float
            Theoretical mean angle in radians
        """
        theta_vals = np.linspace(self.theta_min, np.pi, 1000)

        # get normalized PDF values
        pdf_vals = []
        for t in theta_vals:
            pdf_vals.append(self.pdf(t) / self.norm)

        # calculate mean: integral of theta * p(theta)
        mean = np.trapz(theta_vals * np.array(pdf_vals), theta_vals)

        return mean

    def sample_inverse_cdf(self):
        """
        Generate one sample using inverse CDF method.

        Uses binary search to invert the CDF numerically.

        Returns
        -------
        float
            Sampled scattering angle in radians
        """
        # generate uniform random number
        u = np.random.random()

        # binary search bounds
        low = self.theta_min
        high = np.pi - 0.001  # avoid exactly pi

        # do binary search (30 iterations is enough for good precision)
        for i in range(30):
            mid = (low + high) / 2

            # calculate CDF at midpoint by integrating from theta_min to mid
            theta_range = np.linspace(self.theta_min, mid, 100)
            pdf_vals = []
            for t in theta_range:
                pdf_vals.append(self.pdf(t))
            cdf = np.trapz(pdf_vals, theta_range) / self.norm

            # update bounds based on where u falls
            if cdf < u:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def sample_rejection(self):
        """
        Generate one sample using rejection sampling method.

        Uses uniform proposal distribution with accept/reject criterion.

        Returns
        -------
        float
            Sampled scattering angle in radians
        """
        # envelope constant (slightly higher than max PDF)
        M = self.max_pdf * 1.1

        # keep trying until we accept a sample
        for attempt in range(10000):
            # sample uniformly from [theta_min, pi]
            theta = self.theta_min + np.random.random() * (np.pi - self.theta_min)

            # sample uniform value for accept/reject
            u = np.random.random() * M

            # accept if u falls below PDF
            if u <= self.pdf(theta):
                return theta

        # fallback if we somehow don't accept anything (shouldn't happen)
        return self.theta_min + np.random.random() * (np.pi - self.theta_min)

    def run(self):
        """
        Run the full Monte Carlo simulation.

        Generates samples using both methods and calculates statistics.
        """
        print(f"\nRunning simulation with {self.num_samples} samples")
        print(f"Angle range: {self.theta_min_deg} to 180 degrees")
        print("=" * 60)

        # start total timer
        time_start_total = time.time()

        # step 1: get normalization constant
        self.norm = self.get_norm()

        # step 2: find maximum PDF value (needed for rejection sampling)
        theta_test = np.linspace(self.theta_min, np.pi, 1000)
        max_val = 0
        for t in theta_test:
            val = self.pdf(t)
            if val > max_val:
                max_val = val
        self.max_pdf = max_val

        # step 3: generate samples using inverse CDF
        print("Generating inverse CDF samples...")
        time_start_inv = time.time()
        self.samples_inv = []
        for i in range(self.num_samples):
            self.samples_inv.append(self.sample_inverse_cdf())
        self.samples_inv = np.array(self.samples_inv)
        self.time_inv = time.time() - time_start_inv

        # step 4: generate samples using rejection method
        print("Generating rejection samples...")
        time_start_rej = time.time()
        self.samples_rej = []
        for i in range(self.num_samples):
            self.samples_rej.append(self.sample_rejection())
        self.samples_rej = np.array(self.samples_rej)
        self.time_rej = time.time() - time_start_rej

        # total time
        self.time_total = time.time() - time_start_total

        print("Done!\n")

        # step 5: calculate all the means
        self.mean_inv = np.mean(self.samples_inv)
        self.mean_rej = np.mean(self.samples_rej)
        self.mean_theory = self.get_mean()

        # print results
        self.print_results()
        self.print_benchmark()

    def print_results(self):
        """
        Print simulation results to console.

        Displays mean values and errors for both methods.
        """
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Theoretical mean:  {np.degrees(self.mean_theory):.3f} degrees")
        print(f"Inverse CDF mean:  {np.degrees(self.mean_inv):.3f} degrees")
        print(f"Rejection mean:    {np.degrees(self.mean_rej):.3f} degrees")
        print()
        # calculate errors
        error_inv = abs(self.mean_inv - self.mean_theory)
        error_rej = abs(self.mean_rej - self.mean_theory)
        print(f"Inverse CDF error: {np.degrees(error_inv):.3f} degrees")
        print(f"Rejection error:   {np.degrees(error_rej):.3f} degrees")
        
        # calculate standard deviations
        std_inv = np.std(self.samples_inv)
        std_rej = np.std(self.samples_rej)
        print(f"\nStandard deviation (Inverse CDF): {np.degrees(std_inv):.3f} degrees")
        print(f"Standard deviation (Rejection):   {np.degrees(std_rej):.3f} degrees")
        
        # calculate standard error
        se_inv = std_inv / np.sqrt(self.num_samples)
        se_rej = std_rej / np.sqrt(self.num_samples)
        print(f"\nStandard error (Inverse CDF): {np.degrees(se_inv):.3f} degrees")
        print(f"Standard error (Rejection):   {np.degrees(se_rej):.3f} degrees")
        print("=" * 60)

    def print_benchmark(self):
        """
        Print benchmark results showing timing and performance.
        """
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Number of samples:      {self.num_samples:,}")
        print(f"\nInverse CDF method:")
        print(f"  Total time:           {self.time_inv:.3f} seconds")
        print(f"  Time per sample:      {self.time_inv/self.num_samples*1e6:.2f} μs")
        print(f"  Samples per second:   {self.num_samples/self.time_inv:.1f}")
        
        print(f"\nRejection method:")
        print(f"  Total time:           {self.time_rej:.3f} seconds")
        print(f"  Time per sample:      {self.time_rej/self.num_samples*1e6:.2f} μs")
        print(f"  Samples per second:   {self.num_samples/self.time_rej:.1f}")
        
        print(f"\nSpeedup factor:         {self.time_inv/self.time_rej:.2f}x")
        print(f"Total simulation time:  {self.time_total:.3f} seconds")
        print("=" * 60)

    def plot(self):
        """
        Create plots comparing the two sampling methods.

        Generates three plots:
        1. PDF comparison with histograms
        2. CDF comparison
        3. Mean value comparison
        """
        # calculate theoretical PDF
        theta_theory = np.linspace(self.theta_min, np.pi, 500)
        pdf_theory = []
        for t in theta_theory:
            pdf_theory.append(self.pdf(t) / self.norm)

        # create figure with subplots
        fig = plt.figure(figsize=(12, 8))

        # ----- plot 1: PDF comparison (top, full width) -----
        ax1 = plt.subplot(2, 2, (1, 2))

        # histograms of samples
        ax1.hist(
            np.degrees(self.samples_inv),
            bins=50,
            alpha=0.6,
            density=True,
            color="purple",
            label="Inverse CDF",
        )
        ax1.hist(
            np.degrees(self.samples_rej),
            bins=50,
            alpha=0.6,
            density=True,
            color="red",
            label="Rejection",
        )

        # theoretical curve
        ax1.plot(
            np.degrees(theta_theory),
            np.array(pdf_theory) * np.pi / 180,
            "k-",
            linewidth=2,
            label="Theory",
        )

        # add vertical lines for means
        ax1.axvline(np.degrees(self.mean_theory), color="black", linestyle="--")
        ax1.axvline(np.degrees(self.mean_inv), color="purple", linestyle=":")
        ax1.axvline(np.degrees(self.mean_rej), color="red", linestyle=":")

        ax1.set_xlabel("Scattering angle (degrees)")
        ax1.set_ylabel("Probability density")
        ax1.set_title(f"PDF Comparison (N={self.num_samples:,})")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_yscale("log")  # log scale helps see the full range

        # ----- plot 2: CDF comparison (bottom left) -----
        ax2 = plt.subplot(2, 2, 3)

        # calculate empirical CDFs
        sorted_inv = np.sort(self.samples_inv)
        sorted_rej = np.sort(self.samples_rej)
        cdf_inv = np.arange(1, len(sorted_inv) + 1) / len(sorted_inv)
        cdf_rej = np.arange(1, len(sorted_rej) + 1) / len(sorted_rej)

        # calculate theoretical CDF
        cdf_theory = []
        for t in theta_theory:
            temp_theta = np.linspace(self.theta_min, t, 100)
            temp_pdf = []
            for x in temp_theta:
                temp_pdf.append(self.pdf(x))
            cdf_theory.append(np.trapz(temp_pdf, temp_theta) / self.norm)

        # plot all three CDFs
        ax2.plot(np.degrees(sorted_inv), cdf_inv, "purple", linewidth=2, label="Inverse CDF")
        ax2.plot(np.degrees(sorted_rej), cdf_rej, "red", linewidth=2, label="Rejection")
        ax2.plot(np.degrees(theta_theory), cdf_theory, "k--", linewidth=2, label="Theory")

        ax2.set_xlabel("Scattering angle (degrees)")
        ax2.set_ylabel("CDF")
        ax2.set_title("Cumulative Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ----- plot 3: mean comparison (bottom right) -----
        ax3 = plt.subplot(2, 2, 4)

        methods = ["Theory", "Inverse CDF", "Rejection"]
        means = [
            np.degrees(self.mean_theory),
            np.degrees(self.mean_inv),
            np.degrees(self.mean_rej),
        ]
        colors = ["black", "purple", "red"]

        ax3.bar(methods, means, color=colors, alpha=0.7)
        ax3.set_ylabel("Mean angle (degrees)")
        ax3.set_title("Mean Comparison")
        ax3.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.show()


def compare_sample_sizes():
    """
    Run simulations with different sample sizes and compare results.
    """
    print("\n" + "=" * 70)
    print("SAMPLE SIZE COMPARISON STUDY")
    print("=" * 70)
    
    # Run with two different sample sizes
    sample_sizes = [5000, 10000]
    results = []
    
    for N in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Running with N = {N:,} samples")
        print('='*70)
        
        sim = RutherfordSimulation(num_samples=N, theta_min_deg=10)
        sim.run()
        
        # store results for comparison
        results.append({
            'N': N,
            'mean_theory': sim.mean_theory,
            'mean_inv': sim.mean_inv,
            'mean_rej': sim.mean_rej,
            'std_inv': np.std(sim.samples_inv),
            'std_rej': np.std(sim.samples_rej),
            'time_inv': sim.time_inv,
            'time_rej': sim.time_rej,
            'time_total': sim.time_total
        })
        
        # plot for this sample size
        sim.plot()
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\nAccuracy Comparison:")
    print("-" * 70)
    print(f"{'Sample Size':<15} {'Method':<15} {'Mean (deg)':<12} {'Error (deg)':<12} {'SE (deg)':<12}")
    print("-" * 70)
    
    for res in results:
        theory_deg = np.degrees(res['mean_theory'])
        
        # Inverse CDF
        mean_inv_deg = np.degrees(res['mean_inv'])
        error_inv = abs(mean_inv_deg - theory_deg)
        se_inv = np.degrees(res['std_inv'] / np.sqrt(res['N']))
        print(f"{res['N']:<15,} {'Inverse CDF':<15} {mean_inv_deg:<12.3f} {error_inv:<12.3f} {se_inv:<12.3f}")
        
        # Rejection
        mean_rej_deg = np.degrees(res['mean_rej'])
        error_rej = abs(mean_rej_deg - theory_deg)
        se_rej = np.degrees(res['std_rej'] / np.sqrt(res['N']))
        print(f"{res['N']:<15,} {'Rejection':<15} {mean_rej_deg:<12.3f} {error_rej:<12.3f} {se_rej:<12.3f}")
        print()
    
    print("\nPerformance Comparison:")
    print("-" * 70)
    print(f"{'Sample Size':<15} {'Method':<15} {'Time (s)':<12} {'Time/sample (μs)':<18} {'Speedup':<10}")
    print("-" * 70)
    
    for res in results:
        time_inv_per = res['time_inv'] / res['N'] * 1e6
        time_rej_per = res['time_rej'] / res['N'] * 1e6
        speedup = res['time_inv'] / res['time_rej']
        
        print(f"{res['N']:<15,} {'Inverse CDF':<15} {res['time_inv']:<12.3f} {time_inv_per:<18.2f} {'-':<10}")
        print(f"{res['N']:<15,} {'Rejection':<15} {res['time_rej']:<12.3f} {time_rej_per:<18.2f} {speedup:<10.2f}x")
        print()
    
    print("\nScaling Analysis:")
    print("-" * 70)
    ratio = results[1]['N'] / results[0]['N']
    time_ratio_inv = results[1]['time_inv'] / results[0]['time_inv']
    time_ratio_rej = results[1]['time_rej'] / results[0]['time_rej']
    
    print(f"Sample size ratio:           {ratio:.1f}x")
    print(f"Time ratio (Inverse CDF):    {time_ratio_inv:.2f}x")
    print(f"Time ratio (Rejection):      {time_ratio_rej:.2f}x")
    print(f"\nBoth methods scale approximately O(N) as expected.")
    
    # Error improvement
    se_inv_ratio = (np.degrees(results[0]['std_inv']/np.sqrt(results[0]['N'])) / 
                    np.degrees(results[1]['std_inv']/np.sqrt(results[1]['N'])))
    print(f"\nStandard error improvement:  {se_inv_ratio:.2f}x")
    print(f"Expected (√{ratio:.0f}):             {np.sqrt(ratio):.2f}x")
    print("=" * 70)


def main():
    """
    Main function to run the simulation.

    Creates a RutherfordSimulation object, runs it, and plots results.
    """
    print("\nRutherford Scattering Monte Carlo Simulation")
    print("=" * 60)
    
    # Run comparison study with two sample sizes
    compare_sample_sizes()
    
    print("\nSimulation complete!")


# run the code
if __name__ == "__main__":
    main()