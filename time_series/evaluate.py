import reservoir.random
import pathlib

import click
import numpy as np
import numpy.typing as npt

import reservoir.maths
import reservoir.random


@click.command()
@click.argument("filename")
@click.option(
    "--sample_n_new_points",
    default=1000,
    help="Generate a new set of n reservoir values",
)
@click.option(
    "--seed",
    default=-1,
    help="Ignore the training seed and use this instead.",
)
@click.option(
    "--res",
    help="Ignore the training seed and use this instead.",
)
def main(filename, res,  sample_n_new_points, seed):
    filename = pathlib.Path(filename).resolve()
    loaded = np.load(str(filename))
    print(list(loaded.keys()))
    dataset = loaded["dataset"]
    parameters = loaded["parameters"]
    seed = loaded["seed"].item() if seed < 0 else seed



    rng = reservoir.random.seed_to_rng(int(seed))


    #####
    #  calculate the target values
    #####
    weights = parameters

    data = np.load(str(dataset) + ".npz")['targets']
    data = (data - np.min(data)/ (np.max(data) - np.min(data)))
    length = len(data)
    evaluation_indices = [range(int(.8 * length), length)]
    targets = data[int(.8 * length): -1]
    output_data =  np.load("QRC_"+ str(res) + "_output_" + str(dataset) + ".npz")['arr_0']
    output_data = (output_data - np.min(output_data))/ (np.max(output_data) - np.min(output_data))
    # from os import sys
    # print(targets.shape, output_data[0,int(.8 * length): length].shape)
    # sys.exit()



    # output_data = [(output_data[i, ::] - np.min(output_data[i, ::]))/(np.max(output_data[i,::]) - np.min(output_data[i,::])) for i in range(256)]
    mean = (1/(len(evaluation_indices)) * np.sum(targets))
    var = np.sum((targets - mean)**2)
    # from os import sys
    # print([2,3,4]**2)
    # sys.exit()
    errors = [(np.sum([weights[i] * output_data[i, int(q)].T  for i in range(256)]) + weights[-1] - data[q])**2 / len(range(int(.8 * length) + 1, length))  for q in range(int(.8 * length) + 1, length)] 
    derivatives = [(np.sum([weights[i] * (output_data[i, int(q+1)].T - output_data[i, int(q)].T ) for i in range(256)]) + weights[-1] - (data[q+1] - data[q]))**2 / len(range(int(.8 * length) + 1, length-1))  for q in range(int(.8 * length), length-1)] 
    average_error = np.sum(errors)

    #####
    #  Get the result DMS, and compute fidelity
    #####
    out = 0

    for i in range(256):
        out+= weights[i] * output_data[i,int(.8 * length) + 1: -1].T
    out = out + weights[-1]
    # from os import sys
    # print(output_data[0, 4000:5000].T.shape)
    # sys.exit()
    save_data = {
        'output': out,
        "nmse": average_error,
        "derivatives" : derivatives,
        "errors" : errors,
        "average_target_state": data[int(.8 * length): -1],
        "average_result_state": out,
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
