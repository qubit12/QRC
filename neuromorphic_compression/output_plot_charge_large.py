import click
import matplotlib.pyplot as plt
import numpy as np


def plot_figure(data_dictionary):
    dd = data_dictionary
    fig, ax = plt.subplots(2, 3)
    
    # Set font properties
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    
    avtsr = ax[0, 0].imshow(dd["average_target_state"].real)
    ax[0, 0].set_title("average target DM (real)", fontsize=19)
    fig.colorbar(avtsr, ax=ax[0, 0])
    
    avtsi = ax[0, 1].imshow(dd["average_target_state"].imag)
    ax[0, 1].set_title("average target DM (imag)", fontsize=19)
    fig.colorbar(avtsi, ax=ax[0, 1])
    
    weights = np.ones_like(dd["fidelities"]) / len(dd["fidelities"])
    ax[0, 2].hist(dd["fidelities"], edgecolor="black", weights=weights)
    ax[0, 2].axvline(dd["mean_score"], color="r", linestyle="dashed", linewidth=2)
    ax[0, 2].grid(linestyle='--', linewidth=0.5)
    
    avrsr = ax[1, 0].imshow(dd["average_result_state"].real)
    ax[1, 0].set_title("average result DM (real)", fontsize=19)
    fig.colorbar(avrsr, ax=ax[1, 0])
    
    avrsi = ax[1, 1].imshow(dd["average_result_state"].imag)
    ax[1, 1].set_title("average result DM (imag)", fontsize=19)
    fig.colorbar(avrsi, ax=ax[1, 1])
    
    fig.suptitle(
        f"{dd['operation_name']}, pulsues = {dd['number_of_pulses']}, res = {dd['number_of_reservoirs']}, seed={dd['seed']}",
        fontsize=19
    )
    
    # Set font properties for xticks and yticks
    for a in ax.flat:
        a.tick_params(axis='x', labelsize=23)
        a.tick_params(axis='y', labelsize=23)
    
    # Set font properties for colorbar
    for cax in fig.get_axes():
        if isinstance(cax, plt.Axes):
            cax.tick_params(labelsize=15)
    
    # Save figure
    base_name = "rand_con_QRC"
    fig.set_size_inches(15, 8)
    fig.tight_layout()
    fig.savefig(
        f"{dd['number_of_pulses']}_{base_name}_{dd['operation_name']}_{dd['number_of_reservoirs']}_charge_V_large.pdf"
    )
    plt.close()


@click.command()
@click.argument("filename")
def main(filename):
    loaded = np.load(filename)
    print(list(loaded.keys()))
    plot_figure(loaded)


if __name__ == "__main__":
    main()
