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

from scipy.optimize import  minimize

import click

@click.command()
@click.option(
    "--dataset",
    default="narma10",
    help="dataset to fit",
)
@click.option(
    "--seed",
    default=666,
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
    "--res",
    default='charge',
)
def main(
    dataset,
    seed,
    res,
    max_function_evals,
    maxiter
):
    
    output_data =  np.load("QRC_"+ str(res) + "_output_" + str(dataset) + ".npz")['arr_0']


    data = np.load(str(dataset) + ".npz")['targets']
    data = (data - np.min(data)/ (np.max(data) - np.min(data)))
    
    training_indices = [range(int(.09 * len(data)), int(.8 * len(data)))]
    targets = data[int(.09 * len(data)): int(.8 * len(data))]

    parameter_start = np.array( np.random.default_rng().standard_normal(257) )
    mean = (1/(len(training_indices)) * np.sum(targets))
    var = np.sum((targets - mean)**2)

    def optimize_function(weights):
        
        return  np.sum(np.abs([weights[i] * output_data[i, int(.09 * len(data)): int(.8 * len(data))].T for i in range(256)] + weights[-1] - targets )**2 ) #/ var) 

    start = time.time()
    opt = minimize(
        optimize_function,
        parameter_start,
        method="L-BFGS-B",
        tol=1e-15,
        options={
            "disp": True,
            "maxfun": max_function_evals,
            "maxiter": maxiter,
        },  # limit it for testing.
    )
    end = time.time()

    def fit(weights):
        return [np.sum([weights[i] * output_data[i, k].T for i in range(256)] + weights[-1]) for k in range(int(.09 * len(data)), int(.8 * len(data)))]
    
    save_output = {
        "seed": seed,
        "dataset": dataset,
        "parameters": opt.x,
        "fit" : fit(opt.x),
        "objective_function_values": opt.fun,
        "success": opt.success,
        "termination_status": opt.status,
        "termination_message": opt.message,
        "number_of_iterations": opt.nit,
        "number_of_objective_function_evals": opt.nfev,
        "wall_time_to_optimize": end - start,
    }
    filename = f"QRC_{res}-{seed}-{dataset}"
    filename_ev = f"eval_QRC_{res}-{seed}-{dataset}"
    np.savez(filename, **save_output)
    np.savez(filename_ev, **save_output)
    print("result", opt.x)
    print(f"Optimization Completed in {end - start} seconds")


if __name__ == "__main__":
    main()
