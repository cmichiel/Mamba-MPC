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
        self.config = MambaConfig(d_model=8, n_layers=1, expand_factor = 1, d_conv = 1)
        self.lin1 = nn.Linear(in_dim,8)
        self.mamba = Mamba(self.config)
        self.lin2 = nn.Linear(8,out_dim)
        
    
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

def setup_SPC(MambaModel, n_y, n_u,nx, N, Tini, Q, R, Rd, S):
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
    opt_x = struct_symMX([
        entry('y_N', shape=(n_y), repeat=N),
        entry('u_N', shape=(n_u), repeat=N)
    ])

    # Create parameters of the optimization problem
    # opt_p = struct_symMX([
    #     entry('u_Tini', shape=(n_u), repeat=Tini-1),
    #     entry('y_Tini', shape=(n_y), repeat=Tini),
    #     entry('ref', shape=(n_y), repeat=N),
    #     entry('u_prev', shape=(n_u)) 
    # ])

    opt_p = struct_symMX([
        entry('x0', shape=(n_x), repeat=1),
        entry('ref', shape=(n_y), repeat=N),
        entry('u_prev', shape=(n_u)) 
    ])

    # Create numerical instances of the structures (holding all zeros as entries)
    opt_x_num = opt_x(0)
    opt_p_num = opt_p(0)
    
    
    # Define the objective function
      # Define the objective function
    obj = 0
    for k in range(N):
        y_k = opt_x['y_N', k]
        # State tracking cost
        obj += Q * cs.sumsqr(y_k    - opt_p['ref', 1,k])
        # Input cost
        obj += R * cs.sumsqr(opt_x['u_N', k])
        # Input rate of change cost (using u_prev for the first step)
        if k == 0:
            obj += Rd * cs.sumsqr(opt_x['u_N', k] - opt_p['u_prev'])
        else:
            obj += Rd * cs.sumsqr(opt_x['u_N', k] - opt_x['u_N', k-1])

    # # Terminal state cost
    y_N = opt_x['y_N',N-1]
    refN = opt_p['ref', N-1]
    obj += S * cs.sumsqr(y_N  - refN) # Correct indexing for terminal cost


    # Create the constraints:
    b = cs.vertcat(np.zeros((N*n_u,1)), *opt_p['x0'],  np.zeros((N*n_u,1)))
    v = cs.vertcat( *opt_x['u_N'], np.zeros((Tini*n_u,1)),  *opt_x['y_N'])
    inputs = (b+v)
    y_model = cs.horzcat(MambaModel(inputs.T))
    y_N = cs.horzcat(*opt_x['y_N'])   
    cons = cs.horzcat(y_model - y_N)

    # Create lower and upper bound structures and set all values to plus/minus infinity.
    lbx = opt_x(-np.inf)
    ubx = opt_x(np.inf)

    # Set only bounds on u_N
    lbx['u_N'] = -15
    ubx['u_N'] = 15

    opts = {'ipopt.print_level': 0, 'print_time': 0}
    #opts = {'fatrop.print_level': 0, 'print_time': 0}
    solver_opts = {
    'ipopt': {
        'print_level': 5,  # Increase verbosity
        'tol': 1e-8,
        'constr_viol_tol': 1e-6,
        'max_iter': 500
        }
    }
    # Create Optim
    nlp = {'x':cs.veccat(opt_x), 'f':obj, 'g': cons, 'p': cs.veccat(opt_p)}
    S_spc = nlpsol('S', 'ipopt', nlp, opts)
    
    return S_spc, opt_x_num, opt_p_num, lbx, ubx


#Sampling Time
Ts = 0.1

# Simulation time
time_range =24

# Simulation steps
L = round(time_range/Ts)

# Init time
t = np.arange(L+1)*Ts

#Choose the reference
f = 0.2 # Frequency in Hz
A = 1
#Model Variables
N = 10
Tini = 10

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


model = Net(29,10)
model.load_state_dict(torch.load("State_dicts/Test_Model_Thomas",weights_only=True))


# model = ModelKoopmanResnet(Tini, N, 80, 1, 1, "gaussian")
# model.load_state_dict(torch.load("RBF_Params_KoopmanResnet_Tini5_nbasisKoopman80_N10_KoopmanBasis_gaussian",weights_only=True))
# weight = scipy.io.loadmat("Theta.mat")
# Theta = np.float64(weight["Theta"])
# model.l_2.weight.data = torch.tensor(Theta,dtype=torch.float32)
# print(weight)
model.eval()
MambaModel = l4c.L4CasADi(model,device='cpu',batched=True, name='y_expr')
# #Controller Variables
Q = 100
R =  0.5
Rd = 0
S = 100


uvec = [0]
yvec = [1]
y_solver_vec = []
u_ini = np.zeros(Tini-1)
y_ini = np.ones(Tini)


S_mpc, opt_x_num, opt_p_num, lbx, ubx = setup_SPC(MambaModel, n_y, n_u, N, Tini, Q, R, Rd, S)


opt_p_num['y_Tini'] = cs.vertsplit(y_ini)
opt_p_num['u_Tini'] = cs.vertsplit(u_ini)
opt_p_num['ref'] = cs.vertsplit(ref[1:1+N])
opt_p_num['u_prev'] = cs.vertsplit(uvec[0])

# tic = time.time()




#Warm starting network
for j in range(N):
    if j == 0:
        y_ini = np.concatenate((y_ini[1:], [x1[j]])) 
        u_ini = u_ini 
    elif j>= 1:
        y_ini = np.concatenate((y_ini[1:], [x1[j]])) 
        u_ini = np.concatenate((u_ini[1:], [uvec[j]])) 

    opt_p_num['y_Tini'] = cs.vertsplit(y_ini)
    opt_p_num['u_Tini'] = cs.vertsplit(u_ini)
    opt_p_num['u_prev'] = cs.vertsplit(uvec[j])

    r = S_mpc(x0 = uvec[j], p=opt_p_num, lbg = 0, ubg = 0, lbx=lbx, ubx=ubx)
    opt_x_num.master = r['x']  
    u0 = opt_x_num['u_N',0].full().reshape(-1,1)
    y_solver = opt_x_num['y_N',0].full().reshape(-1,1)
    y_solver_vec.append(y_solver.item() )
    uvec.append(u0.item())
    #VDP 
    (x1[j+1], x2[j+1]) = VDP(x1[j],x2[j], Ts,u0.item(), mu=1)
    yvec.append(x1[j+1])
    print(j)

x1 = np.zeros(L- N + 1)
x2 = np.zeros(L - N+ 1)
x1[0] = x1_init
x2[0] = x2_init
uvec = [uvec[0]]

for k in range(L-N):

    if k == 0:
        y_ini = np.concatenate((y_ini[1:], [x1[k]])) 
        u_ini = u_ini 
    elif k>= 1:
        y_ini = np.concatenate((y_ini[1:], [x1[k]])) 
        u_ini = np.concatenate((u_ini[1:], [uvec[k]])) 

    opt_p_num['y_Tini'] = cs.vertsplit(y_ini)
    opt_p_num['u_Tini'] = cs.vertsplit(u_ini)
    opt_p_num['ref'] = cs.vertsplit(ref[k:k+N])
    opt_p_num['u_prev'] = cs.vertsplit(uvec[k])

    # tic = time.time()


    r = S_mpc(x0 = uvec[k], p=opt_p_num, lbg = -0.1, ubg = 0.1, lbx=lbx, ubx=ubx)
    # toc = time.time()

    opt_x_num.master = r['x']  
    u0 = opt_x_num['u_N',0].full().reshape(-1,1)
    y_solver = opt_x_num['y_N',0].full().reshape(-1,1)
    y_solver_vec.append(y_solver.item() )
    uvec.append(u0.item())
    #VDP 
    (x1[k+1], x2[k+1]) = VDP(x1[k],x2[k], Ts,u0.item(), mu=1)
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
