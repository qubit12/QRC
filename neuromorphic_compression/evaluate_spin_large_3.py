import pathlib

import click
import numpy as np
import numpy.typing as npt

import reservoir.maths
import reservoir.random
from reservoir.systems import vshape_spin_large_3


def evaluate(
    num_pulses,
    parameters,
    coupling_reservoirs: list[npt.NDArray],
    initial_dms,
    target_dms,
):
    (
        coupling_parameters,
        onsite_parameters,
        times,
    ) = vshape_spin_large_3.get_parameters_by_pulse(parameters, num_pulses)

    unitaries = [
        vshape_spin_large_3.reservoir_and_parameters_to_unitary(
            h_r, coupling_parameters, onsite_parameters, times
        )
        for h_r in coupling_reservoirs
    ]

    def result_and_fidelity(u, init, target):
        r = vshape_spin_large_3.trace_out_reservoir(u @ init @ u.conj().T)
        f = reservoir.maths.infidelity(r, target)
        return r, f

    result_fidelities = [
        result_and_fidelity(u, initial_dm, target_dm)
        for u in unitaries
        for (target_dm, initial_dm) in zip(target_dms, initial_dms)
    ]
    # split the pairs back into two lists
    results, fidelities = zip(*result_fidelities)

    return results, fidelities


@click.command()
@click.argument("filename")
@click.option(
    "--sample_n_new_reservoirs",
    default=0,
    help="Generate a new set of n reservoir values",
)
@click.option(
    "--sample_n_new_initial_dms",
    default=100,
    help="Generate a new set of n reservoir values",
)
@click.option(
    "--seed",
    default=-1,
    help="Ignore the training seed and use this instead.",
)
@click.option(
    "--norm",
    default=False,
    help="Normalize target density matrix",
)
def main(filename, sample_n_new_reservoirs, sample_n_new_initial_dms, seed, norm):
    filename = pathlib.Path(filename).resolve()
    loaded = np.load(str(filename))
    print(list(loaded.keys()))
    num_pulses = loaded["number_of_pulses"]
    operation = loaded["operation_name"]
    number_of_reservoirs = loaded["number_of_reservoirs"]
    # con = loaded["con"]
    num_qubits = loaded["number_of_qubits"]
    operation = loaded["operation"]
    parameters = loaded["parameters"]
    coupling_reservoirs = loaded["coupling_reservoirs"]
    seed = loaded["seed"].item() if seed < 0 else seed
    rng = reservoir.random.seed_to_rng(int(seed))

    if sample_n_new_reservoirs:
        coupling_reservoirs,  rng = vshape_spin_large_3.get_reservoirs(
            rng, sample_n_new_reservoirs
        )

    #####
    #  calculate the target values
    #####
    initial_dms = reservoir.random.generate_random_density_matrices(
        num_qubits, sample_n_new_initial_dms, seed
    )
    target_dms = [
        operation @ vshape_spin_large_3.trace_out_reservoir(dm) @ operation.conj().T
        for dm in initial_dms
    ]
    if norm:
        target_dms = [ x / np.trace(x) for x in target_dms]
    
    average_target_dm = np.mean(target_dms, axis=0)


    result_dms, fidelities = evaluate(
        num_pulses,
        parameters,
        coupling_reservoirs,
        initial_dms,
        target_dms,
    )
    average_result_dm = np.mean(result_dms, axis=0)
    average_fidelity = np.mean(fidelities)

    save_data = {
        "fidelities": fidelities,
        "mean_score": average_fidelity,
        "average_target_state": average_target_dm,
        "average_result_state": average_result_dm,
    }
    updated_data = {
        k: v for k, v in loaded.items()
    } | save_data  # combine the training data with evaluated

    #####
    #  save out the results, update the input file with more info.
    #####
    np.savez(str(filename), **updated_data)


if __name__ == "__main__":
    main()
