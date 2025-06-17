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
from scipy.integrate import solve_ivp
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset




class NetSequences(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        D_model  = 4
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 4, d_state = 4)
        self.lin1 = nn.Linear(in_dim,D_model)
        self.mamba = Mamba(self.config)
        self.MLP = nn.Linear(D_model,out_dim)
        

    def forward(self,x):
        # print("Input Shape:"+str(x.shape))
        x = self.lin1(x.unsqueeze(0))
        x = self.mamba(x)
        x = self.MLP(x)
        return x.squeeze(0) 
    

    
# Model parameters
a1, a2, a3, a4 = 1.31e-4, 1.51e-4, 9.27e-5, 8.82e-5
gamma_a, gamma_b = 0.3, 0.4
g, Sc = 9.81, 0.06

params = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'gamma_a': gamma_a, 'gamma_b': gamma_b, 'g': g, 'Sc': Sc}

# Dynamics function for simulation
def four_tank(t, x, u1, u2, params):
    x1, x2, x3, x4 = x
    a1, a2, a3, a4 = params['a1'], params['a2'], params['a3'], params['a4']
    Sc, gamma_a, gamma_b = params['Sc'], params['gamma_a'], params['gamma_b']
    g = params['g']
    dx1 = (-a1 / Sc) * np.sqrt(max(0, 2 * g * x1)) + (a3 / Sc) * np.sqrt(max(0, 2 * g * x3)) + gamma_a / (3600 * Sc) * u1
    dx2 = (-a2 / Sc) * np.sqrt(max(0, 2 * g * x2)) + (a4 / Sc) * np.sqrt(max(0, 2 * g * x4)) + gamma_b / (3600 * Sc) * u2
    dx3 = (-a3 / Sc) * np.sqrt(max(0, 2 * g * x3)) + (1 - gamma_b) / (3600 * Sc) * u2
    dx4 = (-a4 / Sc) * np.sqrt(max(0, 2 * g * x4)) + (1 - gamma_a) / (3600 * Sc) * u1
    return (dx1, dx2, dx3, dx4)



def setup_SPC(nu, nx, N, Q, R, Rd, S):
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
        entry('x', shape=(N,nx)),
        entry('u_N', shape=(N,nu))
    ])

    # Create parameters of the optimization problem
    opt_p = struct_symMX([
        entry('ref', shape=(N, nx)),
        entry('u_prev', shape=(1,nu)),
        entry('x1', shape=(1), repeat=1),
        entry('x2', shape=(1), repeat=1),
        entry('x3', shape=(1), repeat=1),
        entry('x4', shape=(1), repeat=1)
    ])

    # Create numerical instances of the structures (holding all zeros as entries)
    opt_x_num = opt_x(0)
    opt_p_num = opt_p(0)

    

    
    # Define the objective function
      # Define the objective function
    obj = 0
    for k in range(N):
        xk = opt_x['x', k,:]
        uk = opt_x['u_N', k,:]

        # State tracking cost
        error_x = xk - cs.vertcat(opt_p['ref',k,:])
        # obj += cs.sum2(cs.sum1(Q_casadi * (error_x**2).T)) # Original - functionally correct but complex
        obj += cs.mtimes([error_x, Q, error_x.T]) # Standard and clear

        # Input cost
        # obj += cs.sum2(cs.sum1(R_casadi * uk.T)) # Original - INCORRECT
        # obj += cs.mtimes([uk, R, uk.T]) # Standard and correct quadratic cost

        if k == 0:  
            du = opt_x['u_N', k,:] - opt_p['u_prev']
            obj += cs.mtimes([du, Rd, du.T])

        else:
            du = opt_x['u_N', k,:] - opt_x['u_N', k-1,:]
            obj += cs.mtimes([du, Rd, du.T])

    # # Terminal state cost
    # obj += cs.sum2(cs.sum1(S * cs.sumsqr(opt_x['x',N-1,:] - cs.vertcat(opt_p['ref'])))) # Correct indexing for terminal cost
    print("Objective function shape: ", obj.shape)
    # Create the model
    model = NetSequences(6,4)
    model.load_state_dict(torch.load("State_dicts/ZOH_D4_dstate4_Conv4_N20",  map_location=torch.device('cpu'))['model_state_dict'])
    MambaModel = l4c.L4CasADi(model,batched=False, name='y_expr')
    
    constraint = []
    # inputs = cs.horzcat(opt_x['u_N', 0],*opt_p['u_ini'], *opt_p['y_ini']) #My Initial sequence
    #inputs = cs.horzcat(*opt_p['u_ini',0:Tini-1], *opt_p['y_ini',0:Tini], opt_x['u_N', 0]) #Mircea Sequence
    inputs = cs.vertcat(cs.horzcat(opt_x['u_N', 0,0], opt_x['u_N', 0,1] ,*opt_p['x1'], *opt_p['x2'],*opt_p['x3'], *opt_p['x4']))
    for i in range(1,N):
        # inputs = cs.vertcat(inputs, cs.horzcat(opt_x['u_N', i],*opt_p['u_ini'], *opt_p['y_ini']))
        #inputs = cs.vertcat(inputs, cs.horzcat(*opt_p['u_ini',i:i+Tini-1], *opt_p['y_ini',i:i+Tini], opt_x['u_N', i])) #Mircea Sequence
        inputs = cs.vertcat(inputs, cs.horzcat(opt_x['u_N', i,0],opt_x['u_N', i,1],*opt_p['x1'], *opt_p['x2'],*opt_p['x3'], *opt_p['x4']))


    x_model = MambaModel(inputs)
    print("Model output shape: ", x_model.shape)
    # Create the constraints
    constraint = cs.vertcat(opt_x['x', :,:] - x_model)
    constraint = cs.reshape(constraint, N*nx, 1)
    print("Constraint shape: ", constraint.shape)
    # Create lower and upper bound structures and set all values to plus/minus infinity.
    lbx = opt_x(-np.inf)
    ubx = opt_x(np.inf)

    # Set only bounds on u_N
    lbx['u_N'] = 0
    ubx['u_N'] = 4

    # Set bounds on the state
    lbx['x'] = 0
    ubx['x'] = np.inf

    #opts = {'ipopt.print_level': 0, 'print_time': 0}
    opts = {'fatrop.print_level': 0, 'print_time': 0}
    # Create Optim
    nlp = {'x': opt_x, 'f':obj, 'g': constraint, 'p': opt_p}
    S_spc = nlpsol('S', 'fatrop', nlp,opts)
    
    return S_spc, opt_x_num, opt_p_num, lbx, ubx





#Sampling Time
Ts = 5

# Simulation time
time_range = 10

# Simulation steps
L = round(time_range/Ts)

# Init time
t = np.arange(L+1)*Ts

#Choose the reference
f = 0.3  # Frequency in Hz
A = 1
#Model Variables
nx = 4  # [theta, theta_dot]
ny = 4  # [theta, theta_dot]
nu = 2  # torque
N = 20
Nny = N*ny

# Define constant pieces
ref1 = np.tile([0.650, 0.650, 0.652, 0.664], (500, 1))
ref2 = np.tile([0.300, 0.300, 0.301, 0.306], (500, 1))
ref3 = np.tile([0.500, 0.750, 0.305, 1.200], (500, 1))
ref4 = np.tile([0.900, 0.750, 1.062, 0.579], (500, 1))

# Concatenate them along the first axis (time steps)
xref_traj = np.concatenate((ref1, ref2, ref3,ref4), axis=0)  # Shape (100, 2)
ksim = xref_traj.shape[0]  # Number of simulation steps

#ref = np.ones(100)
# ref = 0.4*np.ones(L)

# Initial state
x0 = np.array([0.5, 0.5, 0.5, 0.5])
x = np.zeros((2000, 4))
x[0,:] = np.array([0.5, 0.5, 0.5, 0.5])
u = np.zeros(2000)


# #Controller Variables
Q = np.diag([100, 100, 100, 100])
R = 0 * np.eye(2)  # Input cost matrix
Rd = np.diag([1, 1])
S = 0


uvec1 = [0]
uvec2 = [0]
yvec= x0
y_solver_vec = []
initial_guess = np.zeros((6*N,1))
CompTime = []

S_mpc, opt_x_num, opt_p_num, lbx, ubx = setup_SPC(nu, nx, N, Q, R, Rd, S)


# opt_p_num['y_Tini'] = cs.vertsplit(y_ini)
# opt_p_num['u_Tini'] = cs.vertsplit(u_ini)
# opt_p_num['ref'] = cs.vertsplit(ref[1:1+N])
# opt_p_num['u_prev'] = cs.vertsplit(uvec[0])

# --- Enable Interactive Mode ---
plt.ion()

# --- Create Figure and Axes ---
# Do this *before* the loop, so we update the same window
fig, ax = plt.subplots(6, figsize=(10, 8), sharex=True)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("x1")
ax[1].set_ylabel("x2")
ax[2].set_ylabel("x3")
ax[3].set_ylabel("x4")
ax[4].set_ylabel("u1")
ax[5].set_ylabel("u2")
ax[0].set_title("Live Updating Plot")

ax[0].legend()
ax[0].grid()


for k in range(ksim-N):

    # --- Update Data --- 
    opt_p_num['ref'] = np.array(xref_traj[k:k+N, :])
    opt_p_num['u_prev'] = cs.vertcat(uvec1[k], uvec2[k])
    opt_p_num['x1'] = x[k,0]
    opt_p_num['x2'] = x[k,1]
    opt_p_num['x3'] = x[k,2]
    opt_p_num['x4'] = x[k,3]
    

    

    tic = time.time()
    r = S_mpc(x0 = initial_guess, p=opt_p_num, lbg = 0, ubg = 0, lbx=lbx, ubx=ubx)
    toc = time.time()
    

    CompTime.append(toc - tic)
    opt_x_num.master = r['x']  
    uN = np.array(opt_x_num['u_N'])
    yN = np.array(opt_x_num['x'])
    initial_guess = np.concatenate((uN.reshape(nu*N,1),yN.reshape(nx*N,1))).reshape((nx+nu)*N,1)

    # uN = np.array(opt_x_num['u_N'])
    # yN = np.array(opt_x_num['y_N'])
    # initial_guess = np.concatenate((uN,yN)).reshape(2*N,1)
    u1 = opt_x_num['u_N',0,0].full().reshape(-1,1)
    u2 = opt_x_num['u_N',0,1].full().reshape(-1,1)
    # y_solver = opt_x_num['x',0,:].full()
    # y_solver_vec.append(y_solver.item() )
    uvec1.append(u1.item())
    uvec2.append( u2.item())
    #Four Tank Dynamics 
    #(x[k+1,0], x[k+1,1],x[k+1,2],x[k+1,3]) = four_tank(t, x[k,:], u1.item(), u2.item(), params)
    sol = solve_ivp(lambda t, x: four_tank(t, x, u1.item(), u2.item(), params), [0, Ts], x[k,:])

    # Take the last value as the new initial condition
    (x[k+1,0], x[k+1,1],x[k+1,2],x[k+1,3]) = sol.y[:, -1]
    print(k)
    
    # 2. Clear Previous Plot
    ax[0].cla() # Clear the axes
    ax[1].cla() # Clear the axes

    # 3. Plot New Data
    #ax[0].plot(ref[:L-N], label="$ref$")
    ax[0].plot(x[:k,0], label = "$x1$")
    ax[0].plot(xref_traj[:,0], label = "$x1$")
    ax[1].plot(x[:k,1], label = "$x2$")
    ax[1].plot(xref_traj[:,1], label = "$x2$")
    ax[2].plot(x[:k,2], label = "$x3$")
    ax[2].plot(xref_traj[:,2], label = "$x3$")
    ax[3].plot(x[:k,3], label = "$x4$")
    ax[3].plot(xref_traj[:,3], label = "$x4$")
    ax[4].plot(uvec1[:k], label="$u1$")
    ax[5].plot(uvec2[:k], label="$u2$")


    # 4. Customize (optional, can be outside loop if static)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("x1")
    ax[1].set_ylabel("x2")
    ax[2].set_ylabel("x3")
    ax[3].set_ylabel("x4")
    ax[4].set_ylabel("u1")
    ax[5].set_ylabel("u2")
    ax[0].set_title(f"Live Updating Plot (Iteration {k+1}/{ksim})")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[3].grid(True)
    ax[4].grid(True)
    ax[5].grid(True)


    # 5. Pause and Redraw
    plt.draw()
    plt.pause(0.1)

print("Computation Time: ", np.mean(CompTime))




fig.savefig("MambaMPC.png", format='png')


filename = input("Enter file name: ")
np.save('ControllerOutput/yvec_'+ filename+ '.npy', x)
np.save('ControllerOutput/u1_'+ filename+ '.npy', uvec1)
np.save('ControllerOutput/u2_'+ filename+ '.npy', uvec2)