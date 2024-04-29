import time


# This must be done before numpy import
import os

threads = "4"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
os.environ["NUMBA_NUM_THREADS"] = threads


import numpy as np

import reservoir.random
from reservoir.hamiltonian import QuantumSystemOptimized as QuantumSystem

import click

@click.command()
@click.option(
    "--dataset",
    default="narma10.npz",
    help="Time series to predict",
)
@click.option(
    "--seed",
    default=666,
    type=int,
    help="use this seed for our random numbers",
)
@click.option(
    "--norm",
    default=True,
    help="Normalize dataset",
)
def main(dataset,
         norm,
         seed
         ):
    quimb_partial_trace_dims = [2] * 6
    number_of_random_parameters = 11

    rng = reservoir.random.seed_to_rng(666)
    # res = [x for x in rng.uniform(1, 2, number_of_random_parameters)]
    res = np.ones(12)

    def reservoir_to_hopping(res = res):
        hop_list = np.zeros((6, 6))
        hop_list[0, 1] = res[0]
        hop_list[1, 0] = res[0]
        hop_list[1, 2] = res[1]
        hop_list[2, 1] = res[1]
        hop_list[4, 1] = res[2]  # conn = true
        hop_list[1, 4] = res[2]
        hop_list[3, 4] = res[3]
        hop_list[4, 3] = res[3]
        hop_list[4, 5] = res[4]
        hop_list[5, 4] = res[4]
        return hop_list


    def reservoir_to_onsite(res = res):
        onsite = np.zeros(6)
        onsite[0] = res[5]
        onsite[1] = res[6]
        onsite[2] = res[7]
        onsite[3] = res[8]
        onsite[4] = res[9]
        onsite[5] = res[10]
        return onsite


    def get_system_under_study(num_qubits=6):
        QSys = QuantumSystem(num_qubits, 1, 1, 3.1, 3.1, False)
        A = QSys.Coulombint
        # Instead of this we should use
        # https://numpy.org/doc/stable/reference/generated/numpy.triu.html
        for i in range(len(A)):
            for j in range(len(A)):
                if np.abs(i - j) > 1:
                    A[i][j] = 0
        A[2][3] = 0
        A[3][2] = A[2][3]
        # A[5][6] = 0
        # A[6][5] = A[5][6]
        # A[1][4] = A[0][1]
        # A[4][1] = A[1][4]
        # A[5][7] = A[0][1]
        # A[7][5] = A[5][7]
        return np.array(A)

    # def trace_out_reservoir(dm, q):
    #     return quimb.partial_trace(
    #         dm,
    #         quimb_partial_trace_dims,
    #         [q],
    #     )

    def U0(hopping = 0, onsite = reservoir_to_onsite(), A = get_system_under_study(), time = 7):
        systems = QuantumSystem(6, hopping, onsite, A, 3.1)
        unitary = systems.U(time)
        return unitary

    def U1(hopping = reservoir_to_hopping(), onsite = reservoir_to_onsite(), A = get_system_under_study(), times = 7):
        systems = QuantumSystem(6, hopping, onsite, A, 3.1)
        unitary = systems.U(times)
        return unitary

    data = np.load(str(dataset)+ ".npz")['targets']
    if norm is True:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))


    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0, 1], [1, 0]])
    def mult(op, i):
        return np.kron(np.kron(np.identity(2**(6-i-1)), op), np.identity(2**i))
    
    # System under study initialization
    epsilon = 0.6
    rho = np.zeros((2**6, 2**6))
    rho[45][45] = 1
    sigma = rho
    rhos = [rho]
    tr_Z = []
    tr_X = []
    for j in range(6):
        traces_Z = [np.trace(mult(Z,j) @ rho)]
        traces_X = [np.trace(mult(X,j) @ rho)]
        for k in range(len(data)):
            rhos.append(( (1-epsilon) * (U0() * data[k] + (1-data[k]) * U1()) @ rhos[k] + epsilon * sigma ) )
            traces_Z.append(np.real(np.trace(mult(Z, j) @ rhos[-1])))
            traces_X.append(np.real(np.trace(mult(X, j) @ rhos[-1])))
        tr_Z.append(traces_Z)
        tr_X.append(traces_X)

    # print(U0())
    np.savez("traces_charge_"+str(dataset), Z = tr_Z, X = tr_X)

    # save_output = {
    #     "number_of_qubits": num_qubits,
    #     "number_of_pulses": number_of_pulses,
    #     "number_of_reservoirs": number_of_reservoirs,
    #     "operation_name": operation_name,
    #     "operation": operation,
    #     "hopping_reservoirs": hopping_reservoirs,
    #     "onsite_reservoirs": onsite_reservoirs,
    #     "seed": seed,
    #     "parameters": opt.x,
    #     "objective_function_values": opt.fun,
    #     "success": opt.success,
    #     "termination_status": opt.status,
    #     "termination_message": opt.message,
    #     "number_of_iterations": opt.nit,
    #     "number_of_objective_function_evals": opt.nfev,
    #     "wall_time_to_optimize": end - start,
    # }
    # filename = f"{operation_name}|{num_qubits}|{number_of_pulses}|{seed}|{number_of_reservoirs}" + "charge_3"
    # np.savez(filename, **save_output)
    # print("result", opt.x)
    # print(f"Optimization Completed in {end - start} seconds")


if __name__ == "__main__":
    main()
