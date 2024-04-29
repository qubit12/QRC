from functools import reduce

import numpy as np
import numpy.typing as npt
import quimb as qb
from scipy.linalg import expm
from reservoir.operators import CNOT, Had, X, Z, s0, s1, QFT_circ, Ham_Kitaev, Ham_Heis, Ham_TFI, UCCSD

# Define gates/algorithms to approximate
Ghz = 1/np.sqrt(2) * (np.kron(s0,np.kron(s0,s0)) + np.kron(s1, np.kron(s1,s1)))
W = 1/np.sqrt(3) * (np.kron(s0,np.kron(s0,s1)) + np.kron(s0, np.kron(s1,s0)) + np.kron(s1, np.kron(s0,s0)) )

Wghz = np.eye(2**3) - Ghz @ Ghz.conj().T
WW = np.eye(2**3) - W @ W.conj().T

Oracle = np.identity(8)
Oracle[6,6] = -1  # oracle for entry e = 110
st = np.kron(Had, np.kron(Had,Had)) @ np.kron(s0, np.kron(s0,s0))
Diff = 2 * st @ qb.dag(st) - np.identity(2**3)
Grov_ops_list = [Diff @ Oracle] * 4 * int(np.ceil(np.pi / 4 * np.sqrt(2))) #we apply double the suggested optimal times!
Grov_ops = reduce(np.matmul, Grov_ops_list)
Grov = np.kron(Had, np.kron(Had,Had)) @ Grov_ops
# Grov = 2 * st @ qb.dag(st) - np.identity(2**3)  # redefiing

for i in [2,4,6]:
    for j in range(1,5):
         locals()["RC_" + str(i)+"U"+ str(j)] = np.load("/home/npetropo/dawgz/src/reservoir/RCz_3.npz")["RC_" + str(i)+"U"+ str(j)]

operations = {
    # "CNOT": CNOT,
    # "ZZ": np.kron(Z, Z),
    # "XX": np.kron(X, X),
    "Grover": Grov,
    "QFT": QFT_circ(3),
    "iQFT": QFT_circ(3, inv = True),
    "exp_Kitaev": expm(-1j * Ham_Kitaev(3,1,1.4,3) * 5),
    "exp_Heis": expm(-1j * Ham_Heis(3,.5,-1.4,2) * 5),
    "exp_TFI": expm(-1j * Ham_TFI(3,.5,-1.4,2) * 5),
    "Kitaev": Ham_Kitaev(3,1,1.4,3),
    "Heis": Ham_Heis(3,.5,-1.4,2),
    "TFI": Ham_TFI(3,.5,-1.4,2),
    "UCCSD" : UCCSD(3, 666),
    "WGhz" : Wghz, "WW" : WW,  "RC_2U1": RC_2U1, "RC_2U2": RC_2U2, "RC_2U3": RC_2U3,  "RC_2U4": RC_2U4,  "RC_4U1": RC_4U1, "RC_4U2": RC_4U2, "RC_4U3": RC_4U3,  "RC_4U4": RC_4U4,  "RC_6U1": RC_6U1, "RC_6U2": RC_6U2, "RC_6U3": RC_6U3,  "RC_6U4": RC_6U4
}
