import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import Mamba, MambaConfig
import argparse
from DataProcessing import HankelMatrices
import scipy.io
import l4casadi as l4c
import time

class NetSequences(nn.Module):
    def __init__(self,in_dim,out_dim):
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
    return [dx1, dx2, dx3, dx4]

# CasADi dynamics
def casadi_dynamics(x, u):
    h1, h2, h3, h4 = x[0], x[1], x[2], x[3]
    h1_dot = -a1/Sc*ca.sqrt(2*g*ca.fmax(h1, 0)) + a3/Sc*ca.sqrt(2*g*ca.fmax(h3, 0)) + gamma_a/(3600*Sc)*u[0]
    h2_dot = -a2/Sc*ca.sqrt(2*g*ca.fmax(h2, 0)) + a4/Sc*ca.sqrt(2*g*ca.fmax(h4, 0)) + gamma_b/(3600*Sc)*u[1]
    h3_dot = -a3/Sc*ca.sqrt(2*g*ca.fmax(h3, 0)) + (1-gamma_b)/(3600*Sc)*u[1]
    h4_dot = -a4/Sc*ca.sqrt(2*g*ca.fmax(h4, 0)) + (1-gamma_a)/(3600*Sc)*u[0]
    return ca.vertcat(h1_dot, h2_dot, h3_dot, h4_dot)



model = NetSequences(6,4)
model.load_state_dict(torch.load("State_dicts/ZOH_D8_dstate8_Conv4_N20",  map_location=torch.device('cpu'))['model_state_dict'])
MambaModel = l4c.L4CasADi(model,batched=False, name='y_expr')


# Define constant pieces
ref1 = np.tile([0.650, 0.650, 0.652, 0.664], (500, 1))
ref2 = np.tile([0.300, 0.300, 0.301, 0.306], (500, 1))
ref3 = np.tile([0.500, 0.750, 0.305, 1.200], (500, 1))
ref4 = np.tile([0.900, 0.750, 1.062, 0.579], (500, 1))

# Concatenate them along the first axis (time steps)
xref_traj = np.concatenate((ref1, ref2, ref3,ref4), axis=0)  # Shape (100, 2)


N = 20  # horizon
ld = 20
nx = 4  # [theta, theta_dot]
ny = 4  # [theta, theta_dot]
nu = 2  # torque
Ts = 5

   
# Set up optimization
opti = ca.Opti()
y = opti.variable(ny*N,)
u = opti.variable(nu*N,)
du = opti.variable(nu*N,)
x0 = opti.parameter(nx)
u0 = opti.parameter(nu,)
ref = opti.parameter(ny*N,)


Q = np.eye(N*ny) * 100
R = np.eye(N * nu)

# print(f'Q = {Q}')
# print(f'y.shape = {y.shape}')
# print(f'u.shape = {u.shape}')

# print(f'R = {R}')


cost = ca.mtimes([(y-ref).T, Q, (y-ref)]) + ca.mtimes([du.T, R, du])


inputs = ca.vertcat(ca.horzcat(u[0],u[N], x0[0], x0[1], x0[2], x0[3]))
for i in range(1,N):
    # inputs = cs.vertcat(inputs, cs.horzcat(opt_x['u_N', i],*opt_p['u_ini'], *opt_p['y_ini']))
    #inputs = cs.vertcat(inputs, cs.horzcat(*opt_p['u_ini',i:i+Tini-1], *opt_p['y_ini',i:i+Tini], opt_x['u_N', i])) #Mircea Sequence
    inputs = ca.vertcat(inputs, ca.horzcat(u[i],u[N+i], x0[0], x0[1], x0[2], x0[3]))

x_model = MambaModel(inputs)    
x_model = x_model.reshape((N*nx,1))
opti.subject_to(y == x_model)
opti.subject_to(du[0:nu] == u[0:nu] - u0)
opti.subject_to(du[nu:] == u[nu:] - u[:-nu])
opti.subject_to(opti.bounded(0, u, 4.0))     # input bounds


# opti.subject_to(opti.bounded(-1.0, U, 1.0))  # torque bounds
opti.minimize(cost)

# Solver options
opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})

# Simulation
# sim_time = 500
# steps = int(sim_time / dt)

x_vals = []
u_vals = []
solve_times = []  # Initialize empty list to store solve times

x_current = np.array([0.5, 0.5, 0.5, 0.5])  # 90 degrees
u_mpc = np.array([0,0])

# print(u_mpc.shape)
# sys.exit()

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



for i in range(len(xref_traj)-N):
    print(f'i = {i} / {len(xref_traj)-N}')
    REF = xref_traj[i,:]
    for j in range(N-1):
        REF = np.hstack((REF,xref_traj[i+j,:]))
    
    # print(f'REF = {REF.shape}')
    # sys.exit()

    opti.set_value(x0, x_current)
    opti.set_value(u0, u_mpc)
    opti.set_value(ref, REF)

    start = time.perf_counter()
    sol = opti.solve()
    elapsed = time.perf_counter() - start
    solve_times.append(elapsed)  # Store each solve time

    # print(f'U[0] = {u[0]}')
    u_mpc = sol.value(u[0:nu])
    
    sol = solve_ivp(lambda t, x: four_tank(t, x, u_mpc[0], u_mpc[1], params), [0, Ts], x_current)

    # Take the last value as the new initial condition
    x_next = sol.y[:, -1]

    x_vals.append(x_next)
    u_vals.append(u_mpc)
    x_current = x_next
    xvec = np.array(x_vals)
    uvec = np.array(u_vals)

        # 2. Clear Previous Plot
    ax[0].cla() # Clear the axes
    ax[1].cla() # Clear the axes

    # 3. Plot New Data
    #ax[0].plot(ref[:L-N], label="$ref$")
    ax[0].plot(xvec[:,0], label = "$x1$")
    ax[0].plot(xref_traj[:,0], label = "$x1$")
    ax[1].plot(xvec[:,1], label = "$x2$")
    ax[1].plot(xref_traj[:,1], label = "$x2$")
    ax[2].plot(xvec[:,2], label = "$x3$")
    ax[2].plot(xref_traj[:,2], label = "$x3$")
    ax[3].plot(xvec[:,3], label = "$x4$")
    ax[3].plot(xref_traj[:,3], label = "$x4$")
    ax[4].plot(uvec[:,0], label="$u1$")
    ax[5].plot(uvec[:,1], label="$u2$")


    # 4. Customize (optional, can be outside loop if static)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("x1")
    ax[1].set_ylabel("x2")
    ax[2].set_ylabel("x3")
    ax[3].set_ylabel("x4")
    ax[4].set_ylabel("u1")
    ax[5].set_ylabel("u2")
    ax[0].set_title(f"Live Updating Plot (Iteration {i+1}/{len(xref_traj)-N})")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[3].grid(True)
    ax[4].grid(True)
    ax[5].grid(True)


    # 5. Pause and Redraw
    plt.draw()
    plt.pause(0.1)

solve_times = np.array(solve_times)
x_vals = np.array(x_vals)
u_vals = np.array(u_vals)

# Plot
t_vec = np.arange(len(xref_traj)-N) * Ts

print(f'xref_traj.shape = {xref_traj.shape}')
print(f'x_vals.shape = {x_vals.shape}')

print(f'u_vals.shape = {u_vals.shape}')
print(f't_vec.shape = {t_vec.shape}')

# Plotting
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(6, 1, i+1)
    plt.plot(t_vec, x_vals[:, i], label=f'$x_{i+1}(t)$')
    plt.plot(t_vec, xref_traj[:len(x_vals), i], '--', label='ref')
    plt.ylabel(f'$x_{i+1}$')
    plt.legend()
plt.subplot(6, 1, 5)
plt.plot(t_vec, u_vals[:,0])
# plt.plot(t_vec, uref_traj[:len(u_vals), 0], '--', label='ref')
plt.ylabel('$u_1$')
plt.subplot(6, 1, 6)
plt.plot(t_vec, u_vals[:,1])
# plt.plot(t_vec, uref_traj[:len(u_vals), 1], '--', label='ref')
plt.ylabel('$u_2$')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
