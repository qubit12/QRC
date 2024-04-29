import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Pauli Matrices:
sigma_1 = np.matrix([[0, 1], [1, 0]])
sigma_2 = np.matrix([[0, -1j], [1j, 0]])
sigma_3 = np.matrix([[1, 0], [0, -1]])
sigma_p = (1/2)*(sigma_1 + 1j*sigma_2)
sigma_m = (1/2)*(sigma_1 - 1j*sigma_2)

# Analogue of the Mathematica Chop function
def Chop(num, tol = 1e-6):
    return(np.where(np.abs(num) < tol, 0., num))

# Class that defines quantum state:
class QuantumState:
    def __init__(self, index, energy, vector, nelectrons, state_string):
        self.i = index
        self.energy = energy
        self.vector = vector
        self.ne = nelectrons
        self.string = state_string

# Class that defines the transition:
class QuantumTransition:
    def __init__(self, qs1, qs2):
        self.i1 = qs1.i
        self.i2 = qs2.i
        self.vec1 = qs1.vector
        self.vec2 = qs2.vector
        self.str1 = qs1.string
        self.str2 = qs2.string
        self.ne1 = qs1.ne
        self.ne2 = qs2.ne
        self.E1 = qs1.energy
        self.E2 = qs2.energy
        self.dE = qs1.energy - qs2.energy
        self.change_ne = False
        if (qs1.ne > qs2.ne):
            self.change_ne = True

def sigma_s(n, num, index):
    sigma_list = [sigma_1, sigma_2, sigma_3]
    for i in range(2):
        if index[i] > n-1:
            index[i] = 1
        else:
            index[i] = 0
    return sigma_list[num][index[0],index[1]]

# Folded Kronecker product of sigma_z
def kronk(q):
    sigma_3 = np.array([[1, 0], [0, -1]])
    if (q == 0):
        return(1)
    else:
        ans = sigma_3
        for i in range(q-1):
            ans = np.kron(sigma_3, ans)
        return(ans)
    
class QuantumSystem:
    def __init__(self, ndots, hopping_c, ons_c, coulomb_c, coulomb_intra, mag_c = 0, spin = False):
        # number of quantum dots
        self.l = ndots
        self.s = spin
        # Hoping Matrix:
        hop = np.zeros((self.l, self.l))
        # Coulomb Interaction:
        Coulombint = np.zeros((self.l, self.l))
        # Spin Interaction:
        Jspin = np.zeros((self.l, self.l))
        # Magnetic Field:
        mag = mag_c * np.ones((self.l, 3))
        # Filling Matrices:
        for i in range(self.l):
            for j in range(self.l):
                if(i != j):
                    try:
                        hop[i][j] = - hopping_c[i][j]
                    except(TypeError):
                        hop[i][j] = - hopping_c
                    try:
                        Coulombint[i][j] = coulomb_c[i][j]
                    except(TypeError):
                        Coulombint[i][j] = coulomb_c* np.exp(-np.abs(i-j)*8)/np.abs(i-j)
                        # Coulombint[i][j] = coulomb_c/np.abs(i-j)
                    Jspin[i][j] = - Coulombint[i][j] * np.exp(-np.abs(i-j))
                else:
                    try:
                        hop[i][j] = ons_c[i]
                    except(TypeError):
                        hop[i][j] = ons_c
                    Coulombint[i][j] = coulomb_intra
        self.Coulombint = Coulombint
        # The dimension of the Hamiltonian matrices:
        if self.s is True:
            self.maxn = 2*self.l
            self.Hildim = 2**(2*self.l)
            
        # Define system Hamiltonians for single e- dynamics H_single
            Hsingle = np.zeros((self.Hildim, self.Hildim), dtype="complex128")
            for k in range(self.l):
                for j in range(self.l):
                    Hsingle += (hop[k][j]*(
                                    np.matmul(self.ac(k), self.ad(j))
                                    + np.matmul(self.ac(k + self.l),
                                                self.ad(j + self.l)))
                                      + np.conj(np.transpose(
                                              hop[k][j]*(
                                                  np.matmul(
                                                      self.ac(k), self.ad(j))
                                                  + np.matmul(self.ac(k + self.l),
                                                              self.ad(j + self.l))))))/2.

            # Define system Hamiltonians for single e- dynamics H_single
            HCoulomb = np.zeros((self.Hildim, self.Hildim), dtype="complex128")
            for j in range(self.l):
                for k in range(self.l):
                    HCoulomb += Coulombint[j][k]*np.matmul(np.matmul(self.ac(k), self.ad(k)), np.matmul(self.ac(j + self.l), self.ad(j + self.l)))
            for j in range(self.l):
                for k in range(j+1):
                    HCoulomb += Coulombint[j][k]*np.matmul(np.matmul(self.ac(k + self.l), self.ad(
                        k + self.l)), np.matmul(self.ac(j + self.l), self.ad(j + self.l))) + Coulombint[j][k]*np.matmul(np.matmul(self.ac(k), self.ad(k)), np.matmul(self.ac(j),                                     self.ad(j)))
            # Define the Spin matrix:
            Hspin = np.zeros((self.Hildim, self.Hildim), dtype="complex128")
            for k in range(self.l):
                for j in range(k):
                                Hspin += Jspin[k][j] * (np.matmul(self.Xrot(k) , self.Xrot(j)) + np.matmul(self.Yrot(k) , self.Yrot(j)) + np.matmul(self.Zrot(k) , self.Zrot(j)))

            # Define the Magnetic field matrix:
            Hmag = np.zeros((self.Hildim, self.Hildim), dtype="complex128")
            for k in range(self.l):
                    Hmag += float(mag[k][0]) * self.Xrot(k) + float(mag[k][1]) * self.Yrot(k) + float(mag[k][2]) * self.Zrot(k)
            self.Htot = Hsingle + HCoulomb + Hspin + Hmag   
            
        elif self.s is False:
            self.maxn = self.l
            self.Hildim = 2**(self.l)   
            
            Hsingle = np.zeros((self.Hildim, self.Hildim), dtype="complex128")
            for k in range(self.l):
                 for j in range(self.l):
                    Hsingle +=(hop[k][j] * (
                                   np.matmul(self.ac(k), self.ad(j))) 
                                      + np.conj(np.transpose(
                                              hop[k][j] * np.matmul(
                                                      self.ac(k),  self.ad(j))
                                                )))/2.

            # Define system Hamiltonians for single e- dynamics H_single
            HCoulomb = np.zeros((self.Hildim, self.Hildim), dtype="complex128")
            for j in range(self.l):
                 for k in range(j+1):
                        HCoulomb += (Coulombint[j][k]* np.matmul(np.matmul(self.ac(k) , self.ad(k)), np.matmul(self.ac(j) , self.ad(j))))
        # Total Hamiltonian
            self.Htot = Hsingle + HCoulomb
            self.Hsingle = Hsingle
            self.HCoulomb = HCoulomb
            
        # Eigenvectors and Eigenvalues
        e_val, evec = np.linalg.eig(self.Htot)
        idx = e_val.argsort()
        e_val = np.real(e_val[idx])
        evec = np.transpose(evec)[idx]

        # Normalizing eigenvectors:
        for vec in evec:
            norm = np.sqrt(np.sum(np.abs(vec)**2))
            vec = vec/norm

        # Array of strings with states
        States_string = []
        for i in range(self.Hildim):
            temp = "Eigenstate |E"+str(i+1)+"> = "
            vector = evec[i]
            norm = 0
            for j in range(self.Hildim):
                norm += np.abs(vector[j])**2
            vector = vector/np.sqrt(norm)
            tol = 1e-13
            vector.real[abs(vector.real) < tol] = 0.0
            vector.imag[abs(vector.imag) < tol] = 0.0
            for j in range(self.Hildim):
                if(abs(vector[j]) != 0.0):
                    temp += str(vector[j]) + self.symbolicstate(j)+"\t"
            States_string.append(temp)

        # Determine the number of electrons per state
        ecount = np.zeros(self.Hildim, dtype = "int")
        for i in range(self.Hildim):
            for j in range(self.Hildim):
                if(Chop(evec[i][j])!=0):
                    ecount[i] = self.ecount(j)

        # Determine the states:
        self.states = []
        for i in range(self.Hildim):
            self.states.append(QuantumState(i, e_val[i], evec[i], ecount[i], States_string[i]))

        # Determine the transitions:
        self.all_transitions = []
        self.ne_transitions  = []
        for j in range(self.Hildim):
            for i in range(self.Hildim):
                trans = QuantumTransition(self.states[i],self.states[j])
                self.all_transitions.append(trans)
                if(trans.change_ne):
                    self.ne_transitions.append(trans)

        # Lead tunneling:
        self.tlead = np.array([[1],[1]])
        # Lenergy levels:
        self.Elead = np.zeros([len(self.tlead),len(self.all_transitions)])
        for i in range(len(self.all_transitions)):
            for j in range(len(self.Elead)):
                self.Elead[j][i] = self.all_transitions[i].dE

        # Define population number initial state:
        self.P0 = np.zeros(self.Hildim)
        self.P0[0] = 1.0

        # Define the equation matrix:
        self.eqn2_matrix = np.zeros((self.Hildim, self.Hildim))

        # Define the maximal time:
        self.tmax = 10.0
        self.nintegrate = 1001

    # Electron count function of a specific state
    # (in computation basis)
    
    def ecount(self, x):
        num = 0
        ind = 2
        if ((len(bin(x))-2) > (self.maxn)):
            ind += (len(bin(x))-2)-self.maxn
        for x in bin(x)[ind:]:
            num += int(x)
        return num
            
    # Define creation operator
    # via Jordan-Wigner decomposition

    def ac(self, i):
        i = i + 1
        
        if self.s is True:
            
            return(np.kron(np.kron(np.identity(2**(2*self.l - i)), sigma_m),
                       kronk(i - 1)))            
            # return(np.kron(np.kron(kronk(i - 1), sigma_m),
            #            np.identity(2**(2*self.l - i))))
        
        elif self.s is False:
            
            return(np.kron(np.kron(np.identity(2**(self.l - i)), sigma_m),
                       kronk(i - 1)))
            
            # return(np.kron(np.kron(kronk(i - 1), sigma_m),
            #            np.identity(2**(self.l - i))))

    # Define annihilation operator
    # via Jordan-Wigner decomposition
    
    def ad(self, i):
        i = i + 1
        
        if self.s is True:

            return(np.kron(np.kron(np.identity(2**(2*self.l - i)), sigma_p),
                       kronk(i - 1))) 
            # return(np.kron(np.kron(kronk(i - 1), sigma_p),
            #            np.identity(2**(2*self.l - i))))
        
        elif self.s is False:
            
            return(np.kron(np.kron(np.identity(2**(self.l - i)), sigma_p),
                       kronk(i - 1)))             
            # return(np.kron(np.kron(kronk(i - 1), sigma_p),
            #            np.identity(2**(self.l - i))))

    # Rotation operators
    def Xrot(self, i):
        
        q=np.zeros((self.Hildim , self.Hildim), dtype = "complex128")
        for m in range(2):
            for n in range(2):
                q += (1/2) * (np.matmul(self.ac(i + m * self.l)*sigma_s(self.l, 0 ,[i + m * self.l,i + n * self.l]),self.ad(i + n * self.l)))
        return(q)

    def Yrot(self, i):
        
        q=np.zeros((self.Hildim , self.Hildim), dtype = "complex128")
        for m in range(2):
            for n in range(2):
                q += (1/2) * (np.matmul(self.ac(i + m * self.l)*sigma_s(self.l, 1 ,[i + m * self.l,i + n * self.l]),self.ad(i + n * self.l)))
        return(q)

    def Zrot(self, i):
        
        q=np.zeros((self.Hildim , self.Hildim), dtype = "complex128")
        for m in range(2):
            for n in range(2):
                q += (1/2) * (np.matmul(self.ac(i + m * self.l) * sigma_s(self.l, 2 ,[i + m * self.l,i + n * self.l]),self.ad(i + n * self.l)))
        return(q)

    # Unitary operator:
    def U(self, t):
        return(expm(-self.Htot * t * 1j))
    
    # Transform mathematical state to symbolic basis
    # (bra- ket- functions)
    def symbolicstate(self, x):
        g = [int(x) for x in bin(x)[2:]]
        gg = np.zeros(self.maxn)
        
        if self.s is True:
        
            for i in range(len(g)):
                gg[-i-1] = g[-i-1]
            state = "$ \\left|"
            for i in range(self.l):
                if((gg[i+self.l]+gg[i]) > 0):
                    if(gg[i+self.l] == 1):
                        state += "\\uparrow"
                    if(gg[i] == 1):
                        state += "\\downarrow"
                else:
                    state += "0"
                if(i != (self.l-1)):
                    state += ";"
            
        elif self.s is not True:
            
            for i in range(len(g)):
                gg[-i-1] = g[-i-1]
            state = "$\\left|"
            for i in range(self.l):
                if((gg[i]) > 0):
                        state += "1"
                else:
                    state += "0"
        state += "\\right \\rangle $"
            
        return(state)

    # Gamma function:
    def Gamma(self, lr, n1, n2):
        G = 0.0
        itrans = 0
        for i in range(len(self.all_transitions)):
            if((self.all_transitions[i].i1 == n1)&(self.all_transitions[i].i2==n2)):
                itrans = i

        for p in range(len(self.Elead[lr])):
            for j in range(self.l):
                if(self.all_transitions[itrans].dE == self.Elead[lr][p]):
                    v1 = self.states[n1].vector
                    v2 = self.states[n2].vector
                    H = self.ad(j) + self.ad(j+self.l) if self.s is True else self.ad(j)
                    #print(2.0*np.pi*self.tlead[lr][0]*np.abs(np.matmul(v1,np.transpose(np.matmul(H,v2))))**2, j, p, )
                    G += 2.0*np.pi*self.tlead[lr][0]*np.abs(np.matmul(v1,np.transpose(np.matmul(H,v2))))**2
        return(float(G))

    # Fermi function:
    def fermif(self, mu, T):
        fermi = np.zeros((len(self.all_transitions)))
        for i in range(len(self.all_transitions)):
            fermi[i] = 1/(np.exp((self.all_transitions[i].dE - mu)*11.6/T) + 1)
        return(fermi)

    # Population equations array:
    def eqn2(self, i, mu, T):
        if(self.states[i].ne == self.maxn):
            for i1 in range(len(self.all_transitions)):
                for kappa in self.tlead:
                    if((self.all_transitions[i1].i1==i)&(self.all_transitions[i1].ne2==self.states[i].ne-1)):
                        self.eqn2_matrix[i][self.all_transitions[i1].i2] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*self.fermif(mu,T)[i1]
                        self.eqn2_matrix[i][i] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*(self.fermif(mu,T)[i1] - 1)
        elif(self.states[i].ne == 0):
            for i1 in range(len(self.all_transitions)):
                for kappa in self.tlead:
                    if((self.all_transitions[i1].i2==i)&(self.all_transitions[i1].ne1==self.states[i].ne+1)):
                        self.eqn2_matrix[i][i] += -self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*self.fermif(mu,T)[i1]
                        self.eqn2_matrix[i][self.all_transitions[i1].i1] += self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*(1-self.fermif(mu,T)[i1])
                        #print(self.Gamma(int(kappa), i, self.all_transitions[i1].i1), self.fermif(mu,T)[i1], (self.all_transitions[i1].ne1, self.all_transitions[i1].i1+1), (1 - self.fermif(mu,T)[i1]), (self.all_transitions[i1].ne2, self.all_transitions[i1].i2+1))
        else:
            for i1 in range(len(self.all_transitions)):
                for kappa in self.tlead:
                    if((self.all_transitions[i1].i1==i)&(self.all_transitions[i1].ne2==self.states[i].ne-1)):
                        self.eqn2_matrix[i][self.all_transitions[i1].i2] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*self.fermif(mu,T)[i1]
                        self.eqn2_matrix[i][i] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*(self.fermif(mu,T)[i1] - 1)
                        #print(self.Gamma(int(kappa),self.all_transitions[i1].i2,i), self.fermif(mu,T)[i1], (self.all_transitions[i1].ne1, self.all_transitions[i1].i1+1), (1 - self.fermif(mu,T)[i1]), (self.all_transitions[i1].ne2, self.all_transitions[i1].i2+1))
                    if((self.all_transitions[i1].i2==i)&(self.all_transitions[i1].ne1==self.states[i].ne+1)):
                        self.eqn2_matrix[i][i] += -self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*self.fermif(mu,T)[i1]
                        self.eqn2_matrix[i][self.all_transitions[i1].i1] += self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*(1-self.fermif(mu,T)[i1])
                        #print(self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*self.fermif(mu,T)[i1], self.fermif(mu,T)[i1], (self.all_transitions[i1].ne2, self.all_transitions[i1].i2+1), (1-self.fermif(mu,T)[i1]), (self.all_transitions[i1].ne1, self.all_transitions[i1].i1+1))

    def fill_eqn2_matrix(self, mu, T):
        self.eqn2_matrix = np.zeros((self.Hildim,self.Hildim))
        for i in range(self.Hildim):
            self.eqn2(i, mu, T)

    def solv_eqn(self, mu, T, plot = False, navplot = False):

        self.fill_eqn2_matrix(mu, T)
        def equation(y, t):
            dydt = []
            for i in range(self.Hildim):
                dydt.append(0)
                for j in range(self.Hildim):
                    dydt[-1]+=(self.eqn2_matrix[i][j]*y[j])
            return(dydt)

        t = np.linspace(0, self.tmax, self.nintegrate)
        sol = odeint(equation, self.P0, t)
        sol = np.transpose(sol)

        nav = np.zeros(len(sol[0]))

        for i in range(self.Hildim):
            nav += (np.abs(self.states[i].ne*sol[i]))

        if(plot):
            for i in range(self.Hildim):
                plt.plot(t,sol[:,i])
            plt.plot(t,np.sum(sol,axis=1))
            plt.show()

        if(navplot):
            for i in range(self.Hildim):
                plt.plot(t,self.states[i].ne*sol[i])
            plt.plot(t,nav,'k')
            plt.show()

        return(t,sol,nav)
        
    # (Superposition) statevector from binary:
    def statevec(self, prob : (list, float), state : (list, str), binary = True) -> np.array:
        sts = []
        if binary is True:
            try:
                for i in range(len(state)):
                    sts.append(int(state[i],2))
            except(TypeError):
                sts.append(int(state,2))
        else:
            try:
                for i in range(len(state)):
                    sts.append(int(state[i]))
            except(TypeError):
                sts.append(int(state))
        vec = np.zeros((self.Hildim,1))

        for i in range(len(sts)):
            vecz = np.zeros((self.Hildim,1))
            vecz[sts[i]] = prob[i]
            vec += vecz
        norm_vec = vec / np.linalg.norm(vec)
        
        return(norm_vec)
    
    
        
        
        