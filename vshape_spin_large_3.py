import numpy as np
import quimb

import reservoir.maths
import reservoir.random
from reservoir.Spin_model import Heisenberg as QuantumSystem

quimb_partial_trace_dims = [2] * 12
number_of_parameters_by_pulse = 3


def reservoir_to_coupling(reservoir):
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


def parameters_to_coupling(pulse_parameters):
    coupling = np.zeros((12, 12))
    coupling[0, 1] = pulse_parameters[0]
    coupling[1, 0] = pulse_parameters[0]
    coupling[4, 5] = pulse_parameters[1]
    coupling[5, 4] = pulse_parameters[1]
    coupling[8, 9] = pulse_parameters[2]
    coupling[9, 8] = pulse_parameters[2]
    return coupling


def parameters_to_mag(pulse_parameters):
    B = np.zeros(3)
    B[0] = pulse_parameters[3]
    B[2] = pulse_parameters[4]
    return B


def get_reservoirs(
    seed_rng=None, con=True, number_of_random_initializations=20
):  # what is con?
    number_of_random_parameters = 23

    rng = reservoir.random.seed_to_rng(seed_rng)

    reservoirs = [
        rng.uniform(0, 2, number_of_random_parameters)
        for _ in range(number_of_random_initializations)
    ]
    if not con:
        # just blank out the last value if its not con,
        reservoirs = [reservoir.maths.modify_array(r, -1, 0) for r in reservoirs]

    coupling_reservoirs = [reservoir_to_coupling(r) for r in reservoirs]
    # mag_reservoirs = [reservoir_to_mag(r) for r in reservoirs]
    return coupling_reservoirs, rng #,mag_reservoirs


def trace_out_reservoir(dm):
    return quimb.partial_trace(
        dm,
        quimb_partial_trace_dims,
        [3, 7, 11],
    )


def get_system_unitary(coupling, mags, times):
    systems = [QuantumSystem(12, h, o) for h, o in zip(coupling, mags)]
    unitaries = [qs.U(t) for qs, t in zip(systems, times)]
    unitary = reservoir.maths.mult(unitaries[::-1])  #  apply back wards because unitary
    return unitary


def get_parameters_by_pulse(parameters, number_of_pulses):
    number_of_parameters_per_pulse = len(parameters) // number_of_pulses
    coupling_parameters = [
        parameters_to_coupling(pp)
        for pp in reservoir.maths.chunks(parameters, number_of_parameters_per_pulse)
    ]
    mag_parameters = [
        parameters_to_mag(pp)
        for pp in reservoir.maths.chunks(parameters, number_of_parameters_per_pulse)
    ]
    times = [
        pp[-1]
        for pp in reservoir.maths.chunks(parameters, number_of_parameters_per_pulse)
    ]
    return coupling_parameters, mag_parameters, times


def reservoir_and_parameters_to_unitary(
    coupling_reservoir, coupling_parameters, mag_parameters, times
):
    coupling_pulses = [coupling_reservoir + h for h in coupling_parameters]
    mag_pulses = [o for o in mag_parameters]
    U = get_system_unitary(coupling_pulses, mag_pulses, times)
    return U
