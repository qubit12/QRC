import numpy as np
from scipy.linalg import expm, norm
from functools import reduce

X = np.array([[0, 1],
             [1, 0]])

Y = np.array([[0, -1j],
             [1j, 0]])

Z = np.array([[1, 0],
             [0, -1]])
    
class Heisenberg:
    def __init__(self, ndots : int, J : np.array, B : np.array):
        self.l = ndots
        self.J = J
        self.dim = 2 ** (self.l)
        try:
            self.Bx = B[0]
            self.By = B[1]
            self.Bz = B[2]
        
        except(TypeError):
            self.B = B
        # Interaction Matrix:
        J_spin = np.zeros((self.l, self.l), dtype = "complex128")
        # Magnetic Field:
        mag = np.zeros((self.l, 3), dtype = "complex128")
        # Filling Matrices:
        for i in range(self.l):
            for j in range(self.l):
                
                if(i != j):
                    
                    try:
                        J_spin[i][j] = J[i][j]
                        
                    except(TypeError):
                        J_spin[i][j] = J
            try:
                mag[i][0] = self.Bx
                mag[i][1] = self.By
                mag[i][2] = self.Bz
                
            except(AttributeError):
                mag = self.B * np.ones((self.l, 3))
            
        H_spin = np.zeros((self.dim, self.dim), dtype = "complex128")
        H_mag = np.zeros((self.dim, self.dim), dtype = "complex128")
        
        for i in range(self.l):
            for j in range(self.l):
            
                H_spin += - J_spin[i][j] * ( self.multi_Pauli(X, i) @ self.multi_Pauli(X, j) + self.multi_Pauli(Y, i) @ self.multi_Pauli(Y, j) + self.multi_Pauli(Z, i) @ self.multi_Pauli(Z, j) )
            H_mag +=  mag[i][0] * self.multi_Pauli(X, i) + mag[i][1] * self.multi_Pauli(Y, i) + mag[i][2] * self.multi_Pauli(Z, i)
        # Hamiltonians
        self.Hspin = H_spin
        self.Hmag = - H_mag
        self.Htot = H_spin - H_mag
        
        # Eigenvectors and Eigenvalues
        e_val, evec = np.linalg.eig(self.Htot)
        idx = e_val.argsort()
        e_val = np.real(e_val[idx])
        evec = np.transpose(evec)[idx]
        
        # Normalizing eigenvectors:
        for vec in evec:
            vec = vec / norm(vec)

    # Multi-qubit operations
    def multi_Pauli(self, op : np.array, n : int) -> np.array:
    
        return(reduce(np.kron, [np.identity(2 ** (self.l - n - 1)), op, np.identity(2 ** n)]))
    
    # Unitary operator:
    def U(self, t):
        return(expm(-self.Htot * t * 1j))