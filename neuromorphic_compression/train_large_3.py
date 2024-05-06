import time


# This must be done before numpy import
# import os

# threads = "4"
# os.environ["OMP_NUM_THREADS"] = threads
# os.environ["OPENBLAS_NUM_THREADS"] = threads
# os.environ["MKL_NUM_THREADS"] = threads
# os.environ["VECLIB_MAXIMUM_THREADS"] = threads
# os.environ["NUMEXPR_NUM_THREADS"] = threads
# os.environ["NUMBA_NUM_THREADS"] = threads


import numpy as np

from scipy.optimize import Bounds, minimize

import click

import reservoir.random

from reservoir import targets_3
import reservoir.maths
from reservoir.systems import vshape_charge_large_3


@click.command()
@click.argument("operation_name")
@click.option(
    "--number_of_reservoirs",
    default=10,
    help="Generate a new set of n reservoir values",
)
@click.option(
    "--number_of_initial_dms",
    default=150,
    help="Generate a new set of n reservoir values",
)
@click.option(
    "--number_of_pulses",
    default=2,
    help="Use this many pulses to approximate the unitary",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="use this seed for our random numbers",
)
@click.option(
    "--max_function_evals",
    default=15000,
    type=int,
    help="limit the optimizer",
)
@click.option(
    "--maxiter",
    default=15000,
    type=int,
    help="limit the optimizer",
)
@click.option(
    "--norm",
    default=False,
    type=bool,
    help="normalize target dms",
)
def main(
    operation_name,
    number_of_reservoirs,
    number_of_initial_dms,
    number_of_pulses,
    seed,
    max_function_evals,
    maxiter,
    norm
):
    # System under study initialization
    con = True  # False double V-shape, True X-shape
    num_qubits = 12
    if seed is None:
        # Quimb, numpy and scipy use different seed system
        # we dont try to control scipy so  training is still ranndom
        seed = reservoir.random.get_random_seed()

    A = vshape_charge_large_3.get_system_under_study(num_qubits)

    rng = np.random.default_rng(seed)

    operation = targets_3.operations[operation_name]

    initial_dms = reservoir.random.generate_random_density_matrices(
        num_qubits, number_of_initial_dms, seed
    )
    target_dms = [
        operation @ vshape_charge_large_3.trace_out_reservoir(dm) @ operation.conj().T
        for dm in initial_dms
    ]
    if norm is True:
        target_dms = [ x / np.trace(x) for x in target_dms]

    # Make the reservoir randomness
    hopping_reservoirs, onsite_reservoirs, rng = vshape_charge_large_3.get_reservoirs(
        rng, con, number_of_reservoirs
    )

    bounds = Bounds(
        [0, 0, 0, 0, 0, 0, -2 * np.pi] * number_of_pulses,  # type: ignore
        [2, 2, 2, 7, 7, 7, 2 * np.pi] * number_of_pulses,  # type: ignore
    )
    parameter_start = np.array([1.2, 1.5, 0.7, 1.3, 1.1, 1.8, 2] * number_of_pulses)

    def optimize_function(parameters):
        (
            hopping_parameters,
            onsite_parameters,
            times,
        ) = vshape_charge_large_3.get_parameters_by_pulse(parameters, number_of_pulses)

        unitaries = [
            vshape_charge_large_3.reservoir_and_parameters_to_unitary(
                h_r, o_r, hopping_parameters, onsite_parameters, A, times
            )
            for h_r, o_r in zip(hopping_reservoirs, onsite_reservoirs)
        ]

        return np.mean(
            [
                reservoir.maths.infidelity(
                    vshape_charge_large_3.trace_out_reservoir(u @ initial_dm @ u.conj().T),
                    target_dm,
                )
                # np.linalg.norm(vshape_charge_large_3.trace_out_reservoir(u @ initial_dm @ u.conj().T) - target_dm, ord = 'fro')
                for u in unitaries
                for (target_dm, initial_dm) in zip(target_dms[0:int(0.7*len(initial_dms))], initial_dms[0:int(0.7*len(initial_dms))])
            ]
        )

    start = time.time()
    opt = minimize(
        optimize_function,
        parameter_start,
        method="L-BFGS-B",
        tol=1e-10,
        bounds=bounds,
        options={
            "disp": True,
            "maxfun": max_function_evals,
            "maxiter": maxiter,
        },  # limit it for testing.
    )
    end = time.time()

    save_output = {
        "number_of_qubits": num_qubits,
        "number_of_pulses": number_of_pulses,
        "number_of_reservoirs": number_of_reservoirs,
        "operation_name": operation_name,
        "operation": operation,
        "hopping_reservoirs": hopping_reservoirs,
        "onsite_reservoirs": onsite_reservoirs,
        "seed": seed,
        "parameters": opt.x,
        "objective_function_values": opt.fun,
        "success": opt.success,
        "termination_status": opt.status,
        "termination_message": opt.message,
        "number_of_iterations": opt.nit,
        "number_of_objective_function_evals": opt.nfev,
        "wall_time_to_optimize": end - start,
    }
    filename = f"{operation_name}|{num_qubits}|{number_of_pulses}|{seed}|{number_of_reservoirs}" + "_charge_large_3"
    np.savez(filename, **save_output)
    print("result", opt.x)
    print(f"Optimization Completed in {end - start} seconds")


if __name__ == "__main__":
    main()
