import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import Mamba, MambaConfig
import argparse
from DataProcessing import HankelMatrices
import scipy.io
import l4casadi as l4c
from casadi import *
import casadi as cs
from casadi.tools import *
import rbf_gauss



class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=4, n_layers=1, expand_factor = 1, d_conv = 1, d_state = 2)
        self.lin1 = nn.Linear(in_dim,4)
        self.mamba = Mamba(self.config)
        self.lin2 = nn.Linear(4,out_dim)
        
    
    def forward(self,x):
        # print("Input Shape:"+str(x.shape))
        x = self.lin1(x.unsqueeze(0))
        x = self.mamba(x)
        x = self.lin2(x)
        return x.squeeze(0)
    
class ModelKoopmanResnet(nn.Module):
    def __init__(self, Tini, N, nbasis, input_dim, output_dim, KoopmanBasis):
        super().__init__()
        self.Tini = Tini
        self.N = N
        self.nbasis = nbasis
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_linear_features = (2 * self.Tini - 1)*self.input_dim
        self.out_linear_features = (self.nbasis)
        self.in_nl_features = (2 * self.Tini - 1+self.N)*self.input_dim
        self.out_nl_features = (self.nbasis)
        self.in_2_features = self.nbasis+self.N+2*self.Tini-1
        self.out_2_features = (self.N)*self.output_dim

        #Activation Function
        self.basis_funcKoopman = getattr(rbf_gauss.RBF_gaussian,KoopmanBasis)

        #Define Network Layers
        self.RBFKoopman = rbf_gauss.RBF_gaussian(self.in_linear_features, self.out_linear_features,self.basis_funcKoopman)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)

        #Linear Layer uniformly initialized
        nn.init.uniform_(self.l_2.weight)

        
    def forward(self, data):
        #Split Data into Up, Yp and Uf 
        data_ini = data[:, 0 : (2 * self.Tini - 1)*self.input_dim]
        #data_f   = data[:, (2 * self.Tini - 1)*self.input_dim :]

        #Up,Yp into RBF layer
        x1 = self.RBFKoopman(data_ini)        
        #Add Uf
        #x2 = torch.cat((x1,data_f),1)

        #Add Resnet
        x2 = torch.cat((x1,data),1)

        #Linear Layer
        x = self.l_2(x2)
        return x
    
class ModelKoopman(nn.Module):
    def __init__(self, Tini, N, nbasis, input_dim, output_dim, KoopmanBasis):
        super().__init__()
        self.Tini = Tini
        self.N = N
        self.nbasis = nbasis
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_linear_features = (2 * Tini - 1)*input_dim
        self.out_linear_features = (nbasis)
        self.in_nl_features = (2 * Tini - 1+N)*input_dim
        self.out_nl_features = (nbasis)
        self.in_2_features = (nbasis)+N
        self.out_2_features = (N)*output_dim

        self.basis_funcKoopman = getattr(rbf_gauss.RBF_gaussian,KoopmanBasis)
        self.RBFKoopman= rbf_gauss.RBF_gaussian(self.in_linear_features, self.out_linear_features,self.basis_funcKoopman)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)
        #nn.init.uniform_(self.l_2.weight)

    def forward(self, data):
        data_ini = data[:, 0 : (2 * self.Tini - 1)*self.input_dim]
        data_f   = data[:, (2 * self.Tini - 1)*self.input_dim :]
        x1 = self.RBFKoopman(data_ini)
        x1 = torch.cat((x1,data_f),1)
        x = self.l_2(x1)
        return x
    
def VDP(x1,x2, Ts,u1, mu=1):

    # Calculate the change in angular acceleration
    x1_next = x1+Ts*x2
    x2_next = x2+Ts*(mu*(1-x1**2)*x2-x1+u1)

    return (x1_next, x2_next)

def setup_MPC( n_y, n_u, N, Tini, Q, R, Rd, S):
    """Sets up the SPC controller.

        Args:
        MambaModel: Your system dynamics model (CasADi function).
        n_y: Number of outputs.
        n_u: Number of inputs.
        N: Prediction horizon.
        Tini: Initial state length.
        Q: State tracking cost matrix (n_y x n_y).
        R: Input cost matrix (n_u x n_u).
        Rd: Input rate of change cost matrix (n_u x n_u).
        S: Terminal state cost

    Returns:
        S_spc: The CasADi solver.
        opt_x_num: Numerical instance of optimization variables (for warm-starting).
        opt_p_num: Numerical instance of parameters.
        lbx: Lower bounds on variables.
        ubx: Upper bounds on variables.
    """
    
    # Create optimization variables:
    opti=casadi.Opti()

    uN = opti.variable(N,1)
    yN = opti.variable(N,1)

    u_ini = opti.parameter(Tini-1,1)
    y_ini = opti.parameter(Tini,1)
    refN = opti.parameter(N,1)
    u_prev = opti.parameter(1,1)

        # ---- dynamic constraints --------
    y_sym = MX.sym("yN", N, 1)
    input_sym = MX.sym("uN", 2*Tini-1+N, 1)

    model = ModelKoopman(3, N, 40, 1, 1, "gaussian")
    model.load_state_dict(torch.load("RBF_Params_Koopman_Tini3_nbasisKoopman40_N10_KoopmanBasis_gaussian",weights_only=True))
    MambaModel = l4c.L4CasADi(model,device='cpu',batched=True,  generate_jac_jac=True, generate_adj1=False, generate_jac_adj1=False)
    f = cs.Function("xdot", [y_sym, input_sym], [y_sym.T - MambaModel(input_sym.T)])
    inputs = cs.vertcat(u_ini, y_ini, uN)
    opti.subject_to([f(yN, inputs) == 0, opti.bounded(-10,uN,10)])

    Psi = np.kron(np.eye(N), R);
    Omega = np.kron(np.eye(N),Q)
    # Define the objective function
      # Define the objective function
    obj = 0
    obj += (yN-refN).T@Omega@(yN-refN)+(uN[0]-u_prev)*Rd*(uN[0]-u_prev)+uN[N-1]*R*uN[N-1]
    for k in range(N-1):
        # State tracking cost
        obj += uN[k]*R*uN[k]
        obj += (uN[k+1]-uN[k])*Rd*(uN[k+1]-uN[k])

    # # # Terminal state cost
    # obj += S * cs.sumsqr(yN[N-1]  - refN[N-1]) # Correct indexing for terminal cost

    opti.minimize(obj)
    # Create the constraints:
    
    # opti.subject_to([yN == MambaModel(inputs.T).T, opti.bounded(-10,uN,10)])
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    #opts = {'fatrop.print_level': 0, 'print_time': 0}
    
    opti.solver("ipopt",opts )

    
    return opti, u_ini,y_ini,refN,u_prev,uN,yN  


#Sampling Time
Ts = 0.1

# Simulation time
time_range =16

# Simulation steps
L = round(time_range/Ts)

# Init time
t = np.arange(L+1)*Ts

#Choose the reference
f = 0.2  # Frequency in Hz
A = 2

#Model Variables
N = 10
Tini = 3

ref =  A*np.sin(2*np.pi*f*t)
# Initial state
x1_init = 1
x2_init = 1

x1 = np.zeros(L- N + 1)
x2 = np.zeros(L - N+ 1)
u = np.zeros(L)

# Init arrays to initial state
x1[0] = x1_init
x2[0] = x2_init

n_u = 1
n_y = 1




# model = ModelKoopmanResnet(Tini, N, 80, 1, 1, "gaussian")
# model.load_state_dict(torch.load("RBF_Params_KoopmanResnet_Tini5_nbasisKoopman80_N10_KoopmanBasis_gaussian",weights_only=True))
# weight = scipy.io.loadmat("Theta.mat")
# Theta = np.float64(weight["Theta"])
# model.l_2.weight.data = torch.tensor(Theta,dtype=torch.float32)
# print(weight)

# #Controller Variables
Q = 100
R =  0.5
Rd = 0
S = 100


uvec = [0]
yvec = [1]
y_solver_vec = []
u_ini_vec = np.zeros(Tini-1)
y_ini_vec = np.ones(Tini)


opti,u_ini,y_ini,refN,u_prev,uN,yN = setup_SPC(n_y, n_u, N, Tini, Q, R, Rd, S)

print(ref.shape)

for k in range(L-N):

    if k == 0:
        y_ini_vec = np.concatenate((y_ini_vec[1:], [x1[k]])) 
        u_ini_vec = u_ini_vec 

    elif k>= 1:
        y_ini_vec = np.concatenate((y_ini_vec[1:], [x1[k]])) 
        u_ini_vec = np.concatenate((u_ini_vec[1:], [uvec[k]])) 

    opti.set_value(y_ini, y_ini_vec)
    opti.set_value(u_ini, u_ini_vec)
    opti.set_value(refN, ref[k:k+N])
    opti.set_value(u_prev, uvec[k])


    # tic = time.time()

    sol = opti.solve()
    # toc = time.time()


    u0 = sol.value(uN)
    print(u0[0])
    y_solver = sol.value(yN)
    y_solver_vec.append(y_solver[0] )
    uvec.append(u0[0])
    #VDP 
    (x1[k+1], x2[k+1]) = VDP(x1[k],x2[k], Ts,u0[0], mu=1)
    yvec.append(x1[k+1])
    print(k)


fig, axs = plt.subplots(2)
fig.suptitle("SPC RBF")
axs[0].plot(ref[:230], label="$ref$")
axs[0].plot(x1, label = "$y_{model}$")
axs[0].legend()
axs[0].grid()
axs[0].set_ylim(-5, 5)

axs[1].plot(uvec)
axs[0].set(ylabel="$y$")
#axs[1].set(ylabel="theta")
axs[1].set(ylabel="$u$")
axs[1].grid()
plt.show()

fig.savefig("RBFSPC.png", format='png')
