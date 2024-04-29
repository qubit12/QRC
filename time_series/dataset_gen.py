import numpy as np
import click


@click.command()
@click.option(
    "--dataset",
    help="Dataset to generate",
)
@click.option(
    "--n_narma",
    default=10,
    help="Correlating time steps in the past in narma_n (narma task)",
)
@click.option(
    "--timer",
    default=100,
    help="Timer timer step (timer task)",
)
@click.option(
    "--size",
    default=5000,
    help="Dataset size",
)
@click.option(
    "--qsys",
    default = None,
    help="Quantum system",
)
@click.option(
    "--n_before",
    default=500,
    help="Time steps before step function switch in input (timer task)",
)
@click.option(
    "--tau",
    default=16,
    help="Time delay for MG differential equation",
)
def main(
    dataset,
    qsys,
    n_narma,
    timer,
    size,
    n_before,
    tau
    ):
    if dataset == "narma":

        #initialize dataset
        # for j  in range(n_narma):
        #     y.append(j)


        # create input data list
        s = []
        a = 2.11
        b = 3.73
        c = 4.11
        T = 100

        for k in range(size + 1):
            s.append(0.1 * (np.sin(2* np.pi * a * k / T) * np.sin(2* np.pi * b *k /T) * np.sin(2*np.pi*c*k/T) +1))

        # create dataset

        alpha = 0.3
        beta = 0.05
        gamma = 1.5
        delta = 0.1

        y = [0.16]
        data = y
        for k in range(size):
            prev = 0
            if k < n_narma:
                end = k
                ins = s[0] * s[k]
            else:
                end = n_narma
                ins = s[k - end + 1] * s[k]
            for j in range(end):
                    prev += data[k-j]        
            data.append(alpha * data[-1] + beta * data[-1] * prev + gamma * ins + delta)

        return np.savez("narma"+str(n_narma), input = s, targets = data)

    elif dataset == "timer":
        #create inputs
        s= np.zeros(size)
        for j in range(n_before, size):
            s[j]=1
        
        y = np.zeros(size)
        y[n_before + timer] = 1
        
        return np.savez("timer", input = s, targets = y)
    
    elif dataset == "MG":
        sigma = 0.1
        s = [q for q in np.random.normal(0, 1, int(tau/sigma) + 1)]
        data = s[0:int(tau/sigma) + 1]
        for k in range(0, 10* size):
            data.append(data[-1] + sigma * ((0.2 * data[-1 - int(tau/sigma)])/(1+data[-1 - int(tau/sigma)]**10) - .1 * data[-1]))

        ss_data = []
        for j in range(0, len(data), 10):
              ss_data.append(data[j])

        return np.savez("MG_" + str(tau),  targets = ss_data)    
    
    elif dataset == "SNA":
        # import math
        # for i in range(30000):
        np.random.seed(666)
        ksi = np.random.normal(0, 1, size)
        # ksi_k =np.concatenate(np.array([0, 0]) , ksi)
        # ksi_k.append(ksi)
        sigma = 1
        mu = 4
        g = -0.6
        t = 4/3
        d = 0.89
        data = [0.2, 0]
        # from os import sys
        # print(ksi_k)
        # sys.exit()
        # for k in range(0,size):
        #     data.append(data[-1] + (sigma/t) * ((- data[-1] + 1 - mu * data[-2]**2) + d * ksi[k] + g))

        for k in range(0,size):
            data.append(data[-1] + (sigma/t) * ((- data[-1] + 1 - mu * data[-2]**2) + 0.01 * ksi[k] + g)) 

        ss_data = []
        for j in range(0, len(data)):
            ss_data.append(data[j])
        
        # from os import sys
        # print(ss_data)
        # sys.exit()


        return np.savez("SNA", targets = ss_data)    



    elif dataset == "LE": 

        # Parameters
        np.random.seed(666)
        size = 10000  # Number of iterations for the system to evolve
        sigma = 1
        mu = 4
        g = -0.6
        t = 4/3
        d = 0.01  # Adjusted noise coefficient based on your last update
        initial_condition = [0.2, 0]

        # Initialize two trajectories with a small perturbation
        epsilon = 1e-5
        data = np.array(initial_condition)
        data_perturbed = np.array(initial_condition) + np.array([epsilon, 0])

        # Generate noise
        ksi = np.random.normal(0, 1, size)

        # Lyapunov exponent approximation
        lyapunov_sums = []

        for k in range(size):
            # Update both trajectories
            next_val = data[-1] + (sigma/t) * ((-data[-1] + 1 - mu * data[-2]**2) + d * ksi[k] + g)
            next_val_perturbed = data_perturbed[-1] + (sigma/t) * ((-data_perturbed[-1] + 1 - mu * data_perturbed[-2]**2) + d * ksi[k] + g)
            
            data = np.append(data, next_val)
            data_perturbed = np.append(data_perturbed, next_val_perturbed)
            
            # Calculate the distance between trajectories and the Lyapunov exponent approximation
            if k % 100 == 0 and k > 0:  # Recalculate at intervals to avoid numerical overflow and allow perturbation resetting
                distance = np.abs(data_perturbed[-1] - data[-1])
                lyapunov_sums.append(np.log(distance / epsilon))
                
                # Reset the perturbed trajectory
                data_perturbed = np.copy(data)
                data_perturbed[-1] += epsilon

        # Compute the average to estimate the Lyapunov exponent
        lyapunov_exponent = np.mean(lyapunov_sums)
    
        from os import sys
        print(lyapunov_exponent)
        sys.exit()

    
    # elif dataset == "Hamsim":

    #     from reservoir import targets_4
    #     from scipy.linalg import expm
    #     #create inputs
    #     H = targets_4["Kitaev"]
    #     init = np.zeros((2**4,2**4))
    #     init[0][0] = 1
    #     Zs = [np.trace(expm(-1j * H * t) @ init @ expm(1j * H * t)) for t in np.linspace(0,10,1000)]
        
    #     return np.savez("Kitaev_sim", targets = y)

if __name__ == "__main__":
    main()
