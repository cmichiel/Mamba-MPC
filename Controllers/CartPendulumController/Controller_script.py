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
        D_model  = 182
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 16, d_state = 16)
        self.lin1 = nn.Linear(in_dim,D_model)
        self.mamba = Mamba(self.config)
        self.MLP = nn.Linear(D_model,out_dim)
        

    def forward(self,x):
        # print("Input Shape:"+str(x.shape))
        x = self.lin1(x.unsqueeze(0))
        x = self.mamba(x)
        x = self.MLP(x)
        return x.squeeze(0) 
    

    
def penulum_cart(t, y, M=2.4, m=0.23, l = 0.18, g=9.81, u=0):
    x, theta, dx, dtheta = y

    s = np.sin(theta)
    c = np.cos(theta)

    ddx = (4*u - 3*m*g*s*c + 4*m*l*s*dtheta**2)/(4*(m+M)-3*m*c**2)
    ddtheta = ((M+m)*3*g*s-3*u*c - 3*l*m*s*c*dtheta**2)/(4*l*(m+M)  - 3*l*m*c**2)

    return [dx, dtheta, ddx, ddtheta]



def setup_SPC(ny, nu, nx, N, Q, R, Rd, S):
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
        entry('u_prev', shape=(nu)),
        entry('x1', shape=(1), repeat=1),
        entry('x2', shape=(1), repeat=1),
        entry('x3', shape=(1), repeat=1),
        entry('x4', shape=(1), repeat=1),
        entry('x5', shape=(1), repeat=1)
    ])

    # Create numerical instances of the structures (holding all zeros as entries)
    opt_x_num = opt_x(0)
    opt_p_num = opt_p(0)

    print(cs.vertcat(opt_p['ref',1,:]))
    
    
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


    # Create the model
    model = NetSequences(6,5)
    model.load_state_dict(torch.load("State_dicts/ZOH_D182_dstate16_Conv16_N40",  map_location=torch.device('cpu'))['model_state_dict'])
    MambaModel = l4c.L4CasADi(model,batched=False, name='y_expr')
    
    constraint = []
    inputs = cs.vertcat(cs.horzcat(opt_x['u_N', 0,:] ,*opt_p['x1'], *opt_p['x2'],*opt_p['x3'], *opt_p['x4'], *opt_p['x5']))
    for i in range(1,N):
        inputs = cs.vertcat(inputs, cs.horzcat(opt_x['u_N', i,:],*opt_p['x1'], *opt_p['x2'],*opt_p['x3'], *opt_p['x4'], *opt_p['x5']))


    x_model = MambaModel(inputs)

    # Create the constraints
    constraint = cs.vertcat(opt_x['x', :,:] - x_model)
    constraint = cs.reshape(constraint, N*nx, 1)

    # Create lower and upper bound structures and set all values to plus/minus infinity.
    lbx = opt_x(-np.inf)
    ubx = opt_x(np.inf)

    # Set only bounds on u_N
    lbx['u_N'] = -20
    ubx['u_N'] = 20

    opts = {'ipopt.print_level': 0, 'print_time': 0}
    #opts = {'fatrop.print_level': 0, 'print_time': 0}
    # Create Optim
    nlp = {'x': opt_x, 'f':obj, 'g': constraint, 'p': opt_p}
    S_spc = nlpsol('S', 'ipopt', nlp,opts)
    
    return S_spc, opt_x_num, opt_p_num, lbx, ubx



#Sampling Time
Ts = 0.1

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
nx = 5  # [theta, theta_dot]
ny = 5  # [theta, theta_dot]
nu = 1  # torque
N = 40
Nny = N*ny



# Define constant pieces
ref1 = np.tile([0], (100, 1))
ref2 = np.tile([1], (100, 1))
ref3 = np.tile([0], (100, 1))
ref4 = np.tile([0], (100, 1))
ref5 = np.tile([0], (100, 1))

# Concatenate them along the first axis (time steps)
xref_traj = np.concatenate((ref1, ref2, ref3,ref4,ref5), axis=1)  # Shape (100, 2)
#ref = np.ones(100)
# ref = 0.4*np.ones(L)

# Initial state
x0 = np.array([0.5, -1.0, 0.0 ,0.0, 0.0])
x = np.zeros((101, 4))
x[0,:] = np.array([0.5, np.pi, 0.0 ,0.0]) 
u = np.zeros(100)


# #Controller Variables
# Q = np.array([1, 1000, 1, 1, 1])
# R = 0.01
# Rd = 0
# S = np.array([0, 0, 0, 0, 0])

Q = np.diag([1, 1000, 1, 1, 1])
R = 0.01
Rd = 0
S = 0


uvec = [0]
yvec= x0
y_solver_vec = []
initial_guess = np.zeros((6*N,1))
CompTime = []

S_mpc, opt_x_num, opt_p_num, lbx, ubx = setup_SPC(ny, nu, nx, N, Q, R, Rd, S)


# opt_p_num['y_Tini'] = cs.vertsplit(y_ini)
# opt_p_num['u_Tini'] = cs.vertsplit(u_ini)
# opt_p_num['ref'] = cs.vertsplit(ref[1:1+N])
# opt_p_num['u_prev'] = cs.vertsplit(uvec[0])

# --- Enable Interactive Mode ---
plt.ion()

# --- Create Figure and Axes ---
# Do this *before* the loop, so we update the same window
fig, ax = plt.subplots(6)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("x1")
ax[1].set_ylabel("x2")
ax[2].set_ylabel("x3")
ax[3].set_ylabel("x4")
ax[4].set_ylabel("x5")
ax[5].set_ylabel("u")
ax[0].set_title("Live Updating Plot")

ax[0].legend()
ax[0].grid()


for k in range(100):

    # --- Update Data --- 
    opt_p_num['ref'] = np.array(xref_traj[k:k+N, :])
    opt_p_num['u_prev'] = cs.vertsplit(uvec[k])
    opt_p_num['x1'] = x0[0]
    opt_p_num['x2'] = x0[1]
    opt_p_num['x3'] = x0[2]
    opt_p_num['x4'] = x0[3]
    opt_p_num['x5'] = x0[4]
    

    

    tic = time.time()
    print("Start solver")
    r = S_mpc(x0 = initial_guess, p=opt_p_num, lbg = 0, ubg = 0, lbx=lbx, ubx=ubx)
    print("Solver finished")
    toc = time.time()
    

    CompTime.append(toc - tic)
    opt_x_num.master = r['x']  
    uN = np.array(opt_x_num['u_N'])
    yN = np.array(opt_x_num['x'])
    initial_guess = np.concatenate((uN,yN.reshape(5*N,1))).reshape(6*N,1)

    # uN = np.array(opt_x_num['u_N'])
    # yN = np.array(opt_x_num['y_N'])
    # initial_guess = np.concatenate((uN,yN)).reshape(2*N,1)
    u0 = opt_x_num['u_N',0].full().reshape(-1,1)
    # y_solver = opt_x_num['x',0,:].full()
    # y_solver_vec.append(y_solver.item() )
    uvec.append(u0.item())
    #VDP 
    sol = sol = solve_ivp(lambda t, y: penulum_cart(t, y, u=u0.item()),[0, Ts], x[k,:], method='RK45')
    (x[k+1,0], x[k+1,1],x[k+1,2],x[k+1,3]) = sol.y[:, -1]
    x0 = np.array([x[k+1,0], np.cos(x[k+1,1]), np.sin(x[k+1,1]), x[k+1,2], x[k+1,3]])
    print(k)

    
    # 2. Clear Previous Plot
    ax[0].cla() # Clear the axes
    ax[1].cla() # Clear the axes

    # 3. Plot New Data
    #ax[0].plot(ref[:L-N], label="$ref$")
    ax[0].plot(x[:k,0], label = "$x1$")
    ax[1].plot(np.cos(x[:k,1]), label = "$x2$")
    ax[2].plot(np.sin(x[:k,1]), label = "$x2$")
    ax[3].plot(x[:k,2], label = "$x3$")
    ax[4].plot(x[:k,3], label = "$x4$")
    ax[5].plot(uvec[:k], label="$ref$")


    # 4. Customize (optional, can be outside loop if static)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("x1")
    ax[1].set_ylabel("x2")
    ax[2].set_ylabel("x3")
    ax[3].set_ylabel("x4")
    ax[4].set_ylabel("x5")
    ax[5].set_ylabel("u")
    ax[0].set_title(f"Live Updating Plot (Iteration {k+1}/{L-N})")
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
fig, axs = plt.subplots(6)
fig.suptitle("SPC RBF")
axs[0].plot(x[:k,0], label = "$x1$")
axs[1].plot(np.cos(x[:k,1]), label = "$x2$")
axs[2].plot(np.sin(x[:k,1]), label = "$x2$")
axs[3].plot(x[:k,2], label = "$x3$")
axs[4].plot(x[:k,3], label = "$x4$")
axs[5].plot(uvec[:k], label="$ref$")


# 4. Customize (optional, can be outside loop if static)
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("x1")
axs[1].set_ylabel("x2")
axs[2].set_ylabel("x3")
axs[3].set_ylabel("x4")
axs[4].set_ylabel("x5")
axs[5].set_ylabel("u")
axs[0].set_title(f"Live Updating Plot (Iteration {k+1}/{L-N})")
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)
axs[4].grid(True)
axs[5].grid(True)

plt.show()

fig.savefig("RBFSPC.png", format='png')


filename = input("Enter file name: ")
np.save('ControllerOutput/yvec_'+ filename+ '.npy', yvec)
np.save('ControllerOutput/u_'+ filename+ '.npy', uvec)