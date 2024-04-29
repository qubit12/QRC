import numpy as np
import click

@click.command()
@click.option(
    "--dataset",
    help="Dataset to generate",
)
@click.option(
    "--size",
    default = 5000,
    help="Dataset to generate",
)
@click.option(
    '--res',
    default = 'charge',
    help = "charge or spin reservoir"
)
def main(
    dataset,
    size,
    res
    ):
    Zs = np.load("traces_"+str(res) + "_" + str(dataset) + ".npz")['Z']
    Xs = np.load("traces_"+str(res) + "_" + str(dataset) + ".npz")['X']

    z1 = Zs[0]

    # z2 = Zs[size : 2*size]
    z3 = Zs[2]
    z4 = Zs[3]
    # z5 = Zs[4*size : 5*size]
    z6 = Zs[5]

    x1 = Xs[0]
    # x2 = Xs[size : 2*size]
    x3 = Xs[2]
    x4 = Xs[3]
    # x5 = Xs[4*size : 5*size]
    x6 = Xs[5]

    output_data = []
    for i in range(2):
        for k in range(2):
            for l in range(2):
                for q in range(2):
                    for i1 in range(2):
                        for k1 in range(2):
                            for l1 in range(2):
                                for q1 in range(2):
                                    output_data.append(z1**i * z3**k * z4**l * z6**q * x1**i1 *  x3**k1 * x4**l1 * x6**q1)
    # from os import sys
    # print(output_data)
    # sys.exit() 
    np.savez("QRC_" + str(res) + "_output_" + str(dataset), output_data) # PROBLEM WITH 0-TH POWER OF MATRIX

if __name__ == "__main__":
    main()
