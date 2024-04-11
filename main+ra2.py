import argparse
import jax.numpy as jnp
from heat_transfer import HeatTransfer
from bayesian_opt import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np

# Number of time steps = 1000000
# Target Offset set to 0.03m
# Target backfill_permeability set to 6.1728E-13
def calculate_rayleigh_number(delta_T, k, D=0.5, alpha=2.054e-7, mu=1.00e-3, g=9.81, rhow=980, beta=8.80e-05):
    """Calculate the Rayleigh number based on provided parameters."""
    Ra = (rhow * g * D * k * beta * delta_T) / (mu * alpha)
    return Ra

def offset_bayes_opt(soil_properties, backfill_properties, ntime_steps):
    # Target offset between the cables in meters
    target_offset = 0.03

    # Create an instance of the HeatTransfer class
    sim = HeatTransfer(soil_properties, backfill_properties, target_offset, ntime_steps)

    # Run simulation
    results = sim.forward_offset(target_offset)
    u_forward = results["temperature_distribution"]
    sim.plot_heat_distribution(u_forward, "Target offset")
    # sim.plot_heat_distribution(u_forward, "forward")

    # Calculate the average temperature inside the backfill
    target_temp = jnp.mean(u_forward * results["mask_backfill"])
    print("Target temperature: ", target_temp)

    # Define an objective function that takes simulation parameters as input
    def objective_function(sim_params):
        result = sim.forward_offset(sim_params[0])

        temperature = result["temperature_distribution"]
        mask_backfill = result["mask_backfill"]

        # average_temperature_inside_box
        temp = jnp.mean(temperature * mask_backfill)

        # absolute error between the average temperature and the target
        loss = float(abs(jnp.mean(temp) - target_temp))
        print("Offset: ", sim_params[0], "Average Temperature: ", temp, "Loss: ", loss)
        return loss

    # Create an instance of BayesianOptimization
    search_space = [(0.025, 0.05)]
    opt = BayesianOptimization(objective_function, search_space)

    # Run the Bayesian optimization
    optimized_offset = opt.optimize(n_calls=30, random_state=42)

    # Create an instance of HeatTransfer with the optimized offset
    sim_opt = HeatTransfer(
        soil_properties, backfill_properties, optimized_offset, ntime_steps
    )
    u_optimized = sim_opt.forward_offset(optimized_offset)["temperature_distribution"]

    # Plotting both Forward and Bayesian
    sim_opt.plot_heat_distribution(u_optimized, "Bayesian optimized offset")
    # sim_opt.plot_heat_distribution(u_optimized, "Bayesian optimized")


def permeability_bayes_opt(soil_properties, backfill_properties, ntime_steps):
    target_perm = backfill_properties["permeability_backfill"]

    # Create an instance of the HeatTransfer class
    sim = HeatTransfer(soil_properties, backfill_properties, target_perm, ntime_steps)

    # Run simulation
    results = sim.forward_permeability(target_perm)
    u_forward = results["temperature_distribution"]
    sim.plot_heat_distribution(u_forward, "Target offset")
    # sim.plot_heat_distribution(u_forward, "forward")

    # Calculate the average temperature inside the backfill
    target_temp = jnp.mean(u_forward * results["mask_backfill"])
    print("Target temperature: ", target_temp)

    def objective_function(sim_params):
        result = sim.forward_permeability(sim_params[0])

        temperature = result["temperature_distribution"]
        mask_backfill = result["mask_backfill"]

        # average_temperature_inside_box
        temp = jnp.mean(temperature * mask_backfill)

        # absolute error between the average temperature and the target
        loss = float(abs(jnp.mean(temp) - target_temp))
        print(
            "permeability: ",
            sim_params[0],
            "Average Temperature: ",
            temp,
            "Loss: ",
            loss,
        )
        return loss

    # Create an instance of BayesianOptimization
    search_space = [(1e-16, 1e-8)]
    opt = BayesianOptimization(objective_function, search_space)

    # Run the Bayesian optimization

    optimized_perm = opt.optimize(n_calls=30, random_state=42)
    sim_opt = HeatTransfer(
        soil_properties, backfill_properties, optimized_perm, ntime_steps
    )
    u_optimized = sim_opt.forward_permeability(optimized_perm)[
        "temperature_distribution"
    ]

    # plotting both Forward and Bayesian
    sim_opt.plot_heat_distribution(u_optimized, "Bayesian optimized offset")
    # sim_opt.plot_heat_distribution(u_optimized, "Bayesian optimized")


# Add this function definition anywhere after the imports but before if __name__ == "__main__":
def plot_rayleigh_contours(offsets, permeabilities):
    """Plot Rayleigh number contours for a grid of offsets and permeabilities."""
    delta_T = 1  # Assuming a constant temperature difference for illustration
    Ra_grid = np.zeros((len(permeabilities), len(offsets)))

    for i, perm in enumerate(permeabilities):
        for j, offset in enumerate(offsets):
            Ra_grid[i, j] = calculate_rayleigh_number(delta_T, perm)
    
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(offsets, permeabilities)
    cont = plt.contourf(X, Y, Ra_grid, levels=50, cmap="RdBu")
    plt.colorbar(cont)
    plt.xlabel("Offset (m)")
    plt.ylabel("Permeability (m^2)")
    plt.title("Rayleigh Number Across Different Offsets and Permeabilities")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("rayleigh_contours.png")
    print("Plot saved as rayleigh_contours.png")

if __name__ == "__main__":
    ntime_steps = 1000000
    soil_properties = {
        "n": 0.6,  # porosity
        "lambda_soil": 1.5,  # thermal conductivity (W/m-K)
        "cp_soil": 2000,  # specific heat capacity (J/kg-K)
        "rhoS": 1850,  # density (kg/m^3)
        "permeability_soil": 1e-16,  # permeability (m^2)
    }

    backfill_properties = {
        "n_backfill": 0.4,  # porosity
        "lambda_backfill": 1.0,  # thermal conductivity (W/m-K)
        "cp_backfill": 800,  # specific heat capacity (J/kg-K)
        "rho_backfill": 1900,  # density (kg/m^3)
        "permeability_backfill": 6.1728e-13,  # permeability (m^2)
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "optimize",
        choices=["offset", "permeability"],
        help="Choose the parameter to optimize: 'offset' or 'permeability'",
    )
    args = parser.parse_args()

    if args.optimize == "offset":
        offset_bayes_opt(soil_properties, backfill_properties, ntime_steps)
        
    elif args.optimize == "permeability":
        permeability_bayes_opt(soil_properties, backfill_properties, ntime_steps)

    target_perm = backfill_properties["permeability_backfill"]

    # Create an instance of the HeatTransfer class
    sim = HeatTransfer(soil_properties, backfill_properties, target_perm, ntime_steps)

    # Run simulation
    results = sim.forward_permeability(target_perm)
    u_forward = results["temperature_distribution"]
    # sim.plot_heat_distribution(u_forward, "forward")

    # Calculate the average temperature inside the backfill
    target_temp = jnp.mean(u_forward * results["mask_backfill"])
    print("Target temperature: ", target_temp)

    offsets = np.linspace(0.025, 0.05, 50)  # Example range of offsets
    permeabilities = np.logspace(-16, -8, 50)  # Example range of permeabilities
    plot_rayleigh_contours(offsets, permeabilities)

