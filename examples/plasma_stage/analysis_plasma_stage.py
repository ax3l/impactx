#!/usr/bin/env python3
"""
Analysis script for the PlasmaStage example.

This script analyzes the output from the ImpactX plasma stage simulation
and compares it with expected plasma wakefield effects.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpmd_viewer as opmd


def analyze_plasma_stage():
    """Analyze the plasma stage simulation results."""

    # Load simulation data
    sim_path = Path("diags")
    if not sim_path.exists():
        print("Error: No diagnostics directory found. Run the simulation first.")
        return

    # Load particle data
    ts = opmd.TimeSeries(sim_path)

    # Get initial and final particle data
    initial_data = ts.get_particle(["x", "y", "t", "px", "py", "pt"], iteration=0)
    final_data = ts.get_particle(
        ["x", "y", "t", "px", "py", "pt"], iteration=ts.iterations[-1]
    )

    if initial_data is None or final_data is None:
        print("Error: Could not load particle data.")
        return

    # Extract data
    x_init, y_init, t_init, px_init, py_init, pt_init = initial_data
    x_final, y_final, t_final, px_final, py_final, pt_final = final_data

    # Calculate beam parameters
    def calculate_beam_params(x, y, px, py, pt):
        """Calculate beam parameters."""
        # RMS sizes
        sigma_x = np.std(x)
        sigma_y = np.std(y)
        sigma_t = np.std(t_init) if "t_init" in locals() else np.std(t_final)

        # RMS momenta
        sigma_px = np.std(px)
        sigma_py = np.std(py)
        sigma_pt = np.std(pt)

        # Emittances (geometric)
        eps_x = np.sqrt(np.var(x) * np.var(px) - np.cov(x, px)[0, 1] ** 2)
        eps_y = np.sqrt(np.var(y) * np.var(py) - np.cov(y, py)[0, 1] ** 2)

        # Energy spread
        energy_spread = np.std(pt)

        return {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_t": sigma_t,
            "sigma_px": sigma_px,
            "sigma_py": sigma_py,
            "sigma_pt": sigma_pt,
            "eps_x": eps_x,
            "eps_y": eps_y,
            "energy_spread": energy_spread,
        }

    initial_params = calculate_beam_params(x_init, y_init, px_init, py_init, pt_init)
    final_params = calculate_beam_params(x_final, y_final, px_final, py_final, pt_final)

    # Print results
    print("=== Plasma Stage Analysis ===")
    print(f"Initial beam size (x): {initial_params['sigma_x'] * 1e6:.2f} μm")
    print(f"Final beam size (x): {final_params['sigma_x'] * 1e6:.2f} μm")
    print(f"Initial beam size (y): {initial_params['sigma_y'] * 1e6:.2f} μm")
    print(f"Final beam size (y): {final_params['sigma_y'] * 1e6:.2f} μm")
    print(f"Initial emittance (x): {initial_params['eps_x'] * 1e6:.2f} μm")
    print(f"Final emittance (x): {final_params['eps_x'] * 1e6:.2f} μm")
    print(f"Initial emittance (y): {initial_params['eps_y'] * 1e6:.2f} μm")
    print(f"Final emittance (y): {final_params['eps_y'] * 1e6:.2f} μm")
    print(f"Initial energy spread: {initial_params['energy_spread'] * 100:.3f}%")
    print(f"Final energy spread: {final_params['energy_spread'] * 100:.3f}%")

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Plasma Stage Simulation Results")

    # Phase space plots
    axes[0, 0].scatter(x_init * 1e6, px_init, alpha=0.5, s=1, label="Initial")
    axes[0, 0].scatter(x_final * 1e6, px_final, alpha=0.5, s=1, label="Final")
    axes[0, 0].set_xlabel("x (μm)")
    axes[0, 0].set_ylabel("px (normalized)")
    axes[0, 0].legend()
    axes[0, 0].set_title("x-px Phase Space")

    axes[0, 1].scatter(y_init * 1e6, py_init, alpha=0.5, s=1, label="Initial")
    axes[0, 1].scatter(y_final * 1e6, py_final, alpha=0.5, s=1, label="Final")
    axes[0, 1].set_xlabel("y (μm)")
    axes[0, 1].set_ylabel("py (normalized)")
    axes[0, 1].legend()
    axes[0, 1].set_title("y-py Phase Space")

    axes[0, 2].scatter(t_init * 1e15, pt_init, alpha=0.5, s=1, label="Initial")
    axes[0, 2].scatter(t_final * 1e15, pt_final, alpha=0.5, s=1, label="Final")
    axes[0, 2].set_xlabel("t (fs)")
    axes[0, 2].set_ylabel("pt (normalized)")
    axes[0, 2].legend()
    axes[0, 2].set_title("t-pt Phase Space")

    # Distribution plots
    axes[1, 0].hist(x_init * 1e6, bins=50, alpha=0.7, label="Initial", density=True)
    axes[1, 0].hist(x_final * 1e6, bins=50, alpha=0.7, label="Final", density=True)
    axes[1, 0].set_xlabel("x (μm)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].set_title("x Distribution")

    axes[1, 1].hist(y_init * 1e6, bins=50, alpha=0.7, label="Initial", density=True)
    axes[1, 1].hist(y_final * 1e6, bins=50, alpha=0.7, label="Final", density=True)
    axes[1, 1].set_xlabel("y (μm)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    axes[1, 1].set_title("y Distribution")

    axes[1, 2].hist(pt_init, bins=50, alpha=0.7, label="Initial", density=True)
    axes[1, 2].hist(pt_final, bins=50, alpha=0.7, label="Final", density=True)
    axes[1, 2].set_xlabel("pt (normalized)")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].legend()
    axes[1, 2].set_title("Energy Distribution")

    plt.tight_layout()
    plt.savefig("plasma_stage_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nAnalysis complete. Plot saved as 'plasma_stage_analysis.png'")


if __name__ == "__main__":
    analyze_plasma_stage()
