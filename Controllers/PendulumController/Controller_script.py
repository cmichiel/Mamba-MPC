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
import time

from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self,in_dim,out_dim,Tini):
        super().__init__()
        D_model  = 32
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 4, d_state = 32)
        self.lin1 = nn.Linear(in_dim,D_model)
        self.LinEnc = nn.Linear(2*Tini-1,D_model)
        self.MambaEncoder = Mamba(self.config)
        self.MambaDecoder = Mamba(self.config)
        self.MLP = nn.Linear(D_model,out_dim)
        
    
    def forward(self,x):
        Data_ini = x[:,1:2*Tini-1].unsqueeze(0)
        Data_f = x[:,0].unsqueeze(0)
        x_ini = self.LinEnc(Data_ini)
        x_ini = self.MambaEncoder(x_ini)
        x = self.lin1(Data_f)
        x = torch.cat((x,x_ini),1)
        x = self.MambaDecoder(x)
        x = self.MLP(x)
        return x.squeeze(0)


class NetSequences(nn.Module):
    def __init__(self,in_dim,out_dim,L):
        super().__init__()
        D_model  = 8
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 4, d_state = 8)
        self.lin1 = nn.Linear(in_dim,D_model)
        self.mamba = Mamba(self.config)
        self.MLP = nn.Linear(D_model,out_dim)
        

    def forward(self,x):
        # print("Input Shape:"+str(x.shape))
        x = self.lin1(x.unsqueeze(0))
        x = self.mamba(x)
        x = self.MLP(x)
        return x.squeeze(0) 
    
def Pendulum(x1,x2, Ts,u1, mu=1):
    M = 1          # mass of the pendulum
    L = 1          # lenght of the pendulum
    b = 0.1         # friction coefficient
    g = 9.81        # acceleration of gravity
    J = 1/3*M*L**2   # moment of inertia
    Ts = 1/30
    # Calculate the change in angular acceleration
    x1_next = (1-b*Ts/J)*x1+ (Ts)/J*u1- (M*L*Ts*g)/(2*J)*np.sin(x2)
    x2_next = Ts*x1+x2

    return (x1_next, x2_next)

def setup_SPC( n_y, n_u, N, Tini, Q, R, Rd, S):
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
    opt_p = struct_symMX([
        entry('u_ini', shape=(n_u), repeat=Tini-1),
        entry('y_ini', shape=(n_y), repeat=Tini),
        entry('ref', shape=(n_y), repeat=N),
        entry('u_prev', shape=(n_u)),
        entry('x1', shape=(n_u), repeat=1),
        entry('x2', shape=(n_u), repeat=1)
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
        obj += Q * cs.sumsqr(opt_x['y_N', k] - opt_p['ref', k])
        # Input cost
        obj += R * cs.sumsqr(opt_x['u_N', k])
        # Input rate of change cost (using u_prev for the first step)
        if k == 0:
            obj += Rd * cs.sumsqr(opt_x['u_N', k] - opt_p['u_prev'])
        else:
            obj += Rd * cs.sumsqr(opt_x['u_N', k] - opt_x['u_N', k-1])

    # # Terminal state cost
    obj += S * cs.sumsqr(opt_x['y_N',N-1] - opt_p['ref', N-1]) # Correct indexing for terminal cost


    # model = VDPModel(0.1,1)

    # model = ModelKoopman(Tini, N, 40, 1, 1, 'gaussian')
    # model.load_state_dict(torch.load("RBF_Params_Koopman_Tini3_nbasisKoopman40_N10_KoopmanBasis_gaussian",weights_only=True))

    model = NetSequences(3,2,1)
    model.load_state_dict(torch.load("State_dicts/RK4_D8_dstate8_Conv4_Large_N10",  map_location=torch.device('cpu'))['model_state_dict'])
    MambaModel = l4c.L4CasADi(model,batched=False, name='y_expr')
    yN = cs.vertcat(*opt_x['y_N'])
    uN = cs.vertcat(*opt_x['u_N'])
    constraint = []
    # inputs = cs.horzcat(opt_x['u_N', 0],*opt_p['u_ini'], *opt_p['y_ini']) #My Initial sequence
    #inputs = cs.horzcat(*opt_p['u_ini',0:Tini-1], *opt_p['y_ini',0:Tini], opt_x['u_N', 0]) #Mircea Sequence
    inputs = cs.vertcat(cs.horzcat(opt_x['u_N', 0],*opt_p['x1'], *opt_p['x2']))
    for i in range(1,N):
        # inputs = cs.vertcat(inputs, cs.horzcat(opt_x['u_N', i],*opt_p['u_ini'], *opt_p['y_ini']))
        #inputs = cs.vertcat(inputs, cs.horzcat(*opt_p['u_ini',i:i+Tini-1], *opt_p['y_ini',i:i+Tini], opt_x['u_N', i])) #Mircea Sequence
        inputs = cs.vertcat(inputs, cs.horzcat(opt_x['u_N', i],*opt_p['x1'], *opt_p['x2']))


    y_vdp = MambaModel(inputs)
    
    constraint = cs.vertcat(yN - y_vdp[:,1])

    # Create lower and upper bound structures and set all values to plus/minus infinity.
    lbx = opt_x(-np.inf)
    ubx = opt_x(np.inf)

    # Set only bounds on u_N
    lbx['u_N'] = -3
    ubx['u_N'] = 3

    opts = {'ipopt.print_level': 0, 'print_time': 0}
    #opts = {'fatrop.print_level': 0, 'print_time': 0}
    # Create Optim
    nlp = {'x': opt_x, 'f':obj, 'g': constraint, 'p': opt_p}
    S_spc = nlpsol('S', 'qrsqp', nlp)
    
    return S_spc, opt_x_num, opt_p_num, lbx, ubx


#Sampling Time
Ts = 1/30

# Simulation time
time_range = 12

# Simulation steps
L = round(time_range/Ts)

# Init time
t = np.arange(L+1)*Ts

#Choose the reference
f = 0.3  # Frequency in Hz
A = 1
#Model Variables
N = 10
Tini = 5

ref =  A*np.sin(2*np.pi*f*t)
# ref = 0.4*np.ones(L)

# Initial state
x1_init = 1
x2_init = 1.1

x1 = np.zeros(L- N + 1)
x2 = np.zeros(L - N+ 1)
u = np.zeros(L)

# Init arrays to initial state
x1[0] = x1_init
x2[0] = x2_init

n_u = 1
n_y = 1


# #Controller Variables
Q = 200
R =  0.5
Rd = 0
S = 0


uvec = [0]
yvec = [1]
y_solver_vec = []
u_ini = np.zeros(Tini-1)
y_ini = np.ones(Tini)
initial_guess = np.ones((2*N,1))
CompTime = []

S_mpc, opt_x_num, opt_p_num, lbx, ubx = setup_SPC(n_y, n_u, N, Tini, Q, R, Rd, S)


# opt_p_num['y_Tini'] = cs.vertsplit(y_ini)
# opt_p_num['u_Tini'] = cs.vertsplit(u_ini)
# opt_p_num['ref'] = cs.vertsplit(ref[1:1+N])
# opt_p_num['u_prev'] = cs.vertsplit(uvec[0])

# --- Enable Interactive Mode ---
plt.ion()

# --- Create Figure and Axes ---
# Do this *before* the loop, so we update the same window
fig, ax = plt.subplots(2)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Value")
ax[0].set_title("Live Updating Plot")

ax[0].legend()
ax[0].grid()
ax[0].set_ylim(-5, 5)


for k in range(L-N):

    if k == 0:
        y_ini = np.concatenate((y_ini[1:], [x2[k]])) 
        u_ini = u_ini 
    elif k>= 1:
        y_ini = np.concatenate((y_ini[1:], [x2[k]])) 
        u_ini = np.concatenate((u_ini[1:], [uvec[k]])) 
 
    opt_p_num['y_ini'] = cs.vertsplit(y_ini)
    opt_p_num['u_ini'] = cs.vertsplit(u_ini)
    opt_p_num['ref'] = cs.vertsplit(ref[k:k+N])
    opt_p_num['u_prev'] = cs.vertsplit(uvec[k])
    opt_p_num['x1'] = x1[k]
    opt_p_num['x2'] = x2[k]
    

    tic = time.time()
    r = S_mpc(x0 = initial_guess, p=opt_p_num, lbg = 0, ubg = 0, lbx=lbx, ubx=ubx)
    toc = time.time()
    

    CompTime.append(toc - tic)
    opt_x_num.master = r['x']  
    uN = np.array(opt_x_num['u_N'])
    yN = np.array(opt_x_num['y_N'])
    initial_guess = np.concatenate((uN,yN)).reshape(20,1)
    u0 = opt_x_num['u_N',0].full().reshape(-1,1)
    y_solver = opt_x_num['y_N',0].full().reshape(-1,1)
    y_solver_vec.append(y_solver.item() )
    uvec.append(u0.item())
    #VDP 
    (x1[k+1], x2[k+1]) = Pendulum(x1[k],x2[k], Ts,u0.item(), mu=1)
    yvec.append(x2[k+1])
    print(k)

    
    # 2. Clear Previous Plot
    ax[0].cla() # Clear the axes
    ax[1].cla() # Clear the axes

    # 3. Plot New Data
    ax[0].plot(ref[:L-N], label="$ref$")
    ax[0].plot(x2[:k], label = "$y_{model}$")
    ax[1].plot(uvec[:k], label="$ref$")


    # 4. Customize (optional, can be outside loop if static)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Value")
    ax[0].set_title(f"Live Updating Plot (Iteration {k+1}/{L-N})")
    ax[0].grid(True)

    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Value")
    ax[1].grid(True)

    # 5. Pause and Redraw
    plt.draw()
    plt.pause(0.1)

print("Computation Time: ", np.mean(CompTime))
fig, axs = plt.subplots(2)
fig.suptitle("SPC RBF")
axs[0].plot(ref[:L-N], label="$ref$")
axs[0].plot(x2, label = "$y_{model}$")
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

####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1,1)
#gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
ax.semilogy(ref[:L-N], '-k', lw=2.0, label="validation loss")
ax.semilogy(x1, '-b', lw=2.0, label="training loss")
ax.set_xlabel('$epoch$')
ax.set_ylabel('$loss$')
ax.grid()
ax.legend(frameon=False, loc = 'best')
ax.set_title('Mamba', fontsize = 10)
#ax.set_xlim([0,1.2])
#ax.set_ylim([-1.1,1.1])
#ax.axis('square')
plt.savefig("loss.png", dpi=300)
plt.show()

filename = input("Enter file name: ")
np.save('ControllerOutput/yvec_'+ filename+ '.npy', yvec)
np.save('ControllerOutput/u_'+ filename+ '.npy', uvec)