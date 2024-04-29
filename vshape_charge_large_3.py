import numpy as np
import quimb

import reservoir.maths
import reservoir.random
from reservoir.hamiltonian import QuantumSystemOptimized as QuantumSystem

quimb_partial_trace_dims = [2] * 12
number_of_parameters_by_pulse = 17


def reservoir_to_hopping(reservoir):
    hop_list = np.zeros((12, 12))
    hop_list[1, 2] = reservoir[0]
    hop_list[2, 1] = reservoir[0]
    hop_list[5, 1] = reservoir[1]  # conn = true
    hop_list[1, 5] = reservoir[1]
    hop_list[2, 3] = reservoir[2]
    hop_list[3, 2] = reservoir[2]
    hop_list[5, 6] = reservoir[3]
    hop_list[6, 5] = reservoir[3]
    hop_list[6, 7] = reservoir[4]
    hop_list[7, 6] = reservoir[4]
    hop_list[3, 7] = reservoir[5]
    hop_list[7, 3] = reservoir[5]
    hop_list[9, 10] = reservoir[6]
    hop_list[10, 9] = reservoir[6]
    hop_list[6, 9] = reservoir[7]
    hop_list[9, 6] = reservoir[7]
    hop_list[10, 11] = reservoir[8]
    hop_list[11, 10] = reservoir[8]
    hop_list[7, 11] = reservoir[9]
    hop_list[11, 7] = reservoir[9]
    return hop_list


def reservoir_to_onsite(reservoir):
    onsite = np.zeros(12)
    onsite[1] = reservoir[10]
    onsite[2] = reservoir[11]
    onsite[3] = reservoir[12]
    onsite[5] = reservoir[13]
    onsite[6] = reservoir[14]
    onsite[7] = reservoir[15]
    onsite[9] = reservoir[16]
    onsite[10] = reservoir[17]
    onsite[11] = reservoir[18]
    return onsite


def parameters_to_hopping(pulse_parameters):
    hopping = np.zeros((12, 12))
    hopping[0, 1] = pulse_parameters[0]
    hopping[1, 0] = pulse_parameters[0]
    hopping[4, 5] = pulse_parameters[1]
    hopping[5, 4] = pulse_parameters[1]
    hopping[8, 9] = pulse_parameters[2]
    hopping[9, 8] = pulse_parameters[2]
    return hopping


def parameters_to_onsite(pulse_parameters):
    onsite = np.zeros(12)
    onsite[0] = pulse_parameters[3]
    onsite[4] = pulse_parameters[4]
    onsite[8] = pulse_parameters[5]
    return onsite


def get_reservoirs(
    seed_rng=None, con=True, number_of_random_initializations=20
):  # what is con?
    number_of_random_parameters = 27

    rng = reservoir.random.seed_to_rng(seed_rng)

    reservoirs = [
        rng.uniform(0, 2, number_of_random_parameters)
        for _ in range(number_of_random_initializations)
    ]
    if not con:
        # just blank out the last value if its not con,
        reservoirs = [reservoir.maths.modify_array(r, -1, 0) for r in reservoirs]

    hopping_reservoirs = [reservoir_to_hopping(r) for r in reservoirs]
    onsite_reservoirs = [reservoir_to_onsite(r) for r in reservoirs]
    return hopping_reservoirs, onsite_reservoirs, rng


def get_system_under_study(num_qubits=12):
    QSys = QuantumSystem(num_qubits, 1, 1, 3.1, 3.1, False)
    A = QSys.Coulombint
    # Instead of this we should use
    # https://numpy.org/doc/stable/reference/generated/numpy.triu.html
    for i in range(len(A)):
        for j in range(len(A)):
            if np.abs(i - j) > 1:
                A[i][j] = 0
    A[3][4] = 0
    A[4][3] = A[3][4]
    A[1][5] = A[0][1]
    A[5][1] = A[1][5]
    A[3][7] = A[0][1]
    A[7][3] = A[3][7]

    A[7][8] = 0
    A[8][7] = A[7][8]
    A[6][9] = A[0][1]
    A[9][6] = A[6][9]
    A[7][11] = A[0][1]
    A[11][7] = A[7][11]
    return np.array(A)


def trace_out_reservoir(dm):
    return quimb.partial_trace(
        dm,
        quimb_partial_trace_dims,
        [3, 7, 11],
    )


def get_system_unitary(hopping, onsite, A, times):
    systems = [QuantumSystem(12, h, o, A, 3.1) for h, o in zip(hopping, onsite)]
    unitaries = [qs.U(t) for qs, t in zip(systems, times)]
    unitary = reservoir.maths.mult(unitaries[::-1])  #  apply back wards because unitary
    return unitary


def get_parameters_by_pulse(parameters, number_of_pulses):
    number_of_parameters_per_pulse = len(parameters) // number_of_pulses
    hopping_parameters = [
        parameters_to_hopping(pp)
        for pp in reservoir.maths.chunks(parameters, number_of_parameters_per_pulse)
    ]
    onsite_parameters = [
        parameters_to_onsite(pp)
        for pp in reservoir.maths.chunks(parameters, number_of_parameters_per_pulse)
    ]
    times = [
        pp[-1]
        for pp in reservoir.maths.chunks(parameters, number_of_parameters_per_pulse)
    ]
    return hopping_parameters, onsite_parameters, times


def reservoir_and_parameters_to_unitary(
    hopping_reservoir, onsite_reservoir, hopping_parameters, onsite_parameters, A, times
):
    hopping_pulses = [hopping_reservoir + h for h in hopping_parameters]
    onsite_pulses = [onsite_reservoir + o for o in onsite_parameters]
    U = get_system_unitary(hopping_pulses, onsite_pulses, A, times)
    return U
