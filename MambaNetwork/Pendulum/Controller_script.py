import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import Mamba,MyCustomMamba, MambaConfig
import argparse
from DataProcessing import HankelMatrices
import scipy.io
import l4casadi as l4c
import casadi as cs


class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=8, n_layers=1)
        self.lin1 = nn.Linear(in_dim,8)
        self.mamba = Mamba(self.config)
        self.lin2 = nn.Linear(8,out_dim)
        
    
    def forward(self,x):
        #print("Input Shape:"+str(x.shape))
        x = self.lin1(x.unsqueeze(0))
        x = self.mamba(x)
        x = self.lin2(x)
        return x.squeeze(0)

#Sampling Time
Ts = 0.1

# Simulation time
time_range =24

# Simulation steps
L = round(time_range/Ts)

# Init time
time = np.arange(L+1)*Ts

# Initial state
x1_init = 0
x2_init = 0

x1 = np.zeros(L + 1)
x2 = np.zeros(L + 1)
u = np.zeros(L)
ref = np.zeros(L)

#Choose the reference
f = 0.2  # Frequency in Hz
ref =  np.sin(2 * np.pi * f * time)

# Init arrays to initial state
x1[0] = x1_init
x2[0] = x2_init

#Maximum input
u_max = 15

#Model Variables
N = 10
Tini =5

#Controller Variables
Q = 100
R =  0.5
Rd = 0
S = 100

#Casadi variables 
u = cs.MX.sym('u', N) 
y = cs.MX.sym('y', N)
ref = cs.MX.sym('ref', N)
u_init = cs.MX.sym('u_init') 
u_ini = cs.MX.sym('u_ini', Tini-1) 
y_ini = cs.MX.sym('y_ini', Tini)
# input = cs.vertcat(u_ini, y_ini,u)
inputs = cs.MX.sym('inputs', 19)
model_output = cs.MX.sym('model_output',10)
# gvar = cs.SX.sym('gvar', length(u_data)-N)
model = Net(19,10)
model.load_state_dict(torch.load("State_dicts/Test_model_Conv1E1"))
model.eval()
MambaModel = l4c.L4CasADi(model, device='cpu')


# Define cost function (example: quadratic tracking)
cost = 0
cost += (u[1]- u_init)*Rd*(u[1]- u_init)  # Control effort
cost +=  u[N-1]*R*u[N-1]
cost +=  (y[N-1] - ref[N-1])*S*(y[N-1] - ref[N-1])
for i in range(N-1):
    cost += (y[i] - ref[i])*Q*(y[i] - ref[i])  # Tracking error

for i in range(N-2):
    cost += (u[i+1]- u[i])*Rd*(u[i+1]- u[i])  # Control effort
    cost +=  u[i]*R*u[i]
# Create a CasADi function for the cost
f_cost = cs.Function('f_cost', [u, y, ref,u_init], [cost]) 

# Model Constraint
constraint1 =  MambaModel(inputs.T).T-y

# Define lower and upper bounds for the box constraint
lower_bound = -u_max  # Example lower bound
upper_bound = u_max  # Example upper boun

constraint2 = u <= upper_bound  # Upper bound constraint
constraint3 = u >= lower_bound  # Lower bound constraint
constraint4 = inputs- cs.vertcat(u_ini, y_ini,u)

# Create a CasADi function for the constraint
f_constraint = cs.Function('f_constraint', [inputs, y, u], [constraint1]) 

# Example usage within an optimization problem:
# ... (Your optimization problem setup) ...
nlp_prob = {'x': u, 'f': f_cost, 'g': f_constraint} 

# Define the solver parameters
opts = {}
opts['ipopt.print_level'] = 0  # Set to 0 for less output

# Create the NLP solver
solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)


# Initial guess for the optimization variable (replace with your initial guess)
u0 = 1

# Solve the optimization problem
sol = solver(x0=u0)

# Extract the solution
x_opt = sol['x'] 

print("Optimal solution:", x_opt)


# # ---------- SIMULATION ----------

# for idx in range(1):

#     # ---------- SIMULATE USER INPUT ----------
#     if idx == 0:
#         y_ini = np.concatenate((y_ini[1:], [x1[idx]])) 
#         u_ini = u_ini 

#     elif idx >= 1:
#         y_ini = np.concatenate((y_ini[1:], [x1[idx]])) 
#         u_ini = np.concatenate((u_ini[1:], [uvec[idx-1]])) 
    
#     # Increment time
#     time += Ts

#     # ---------- CONTROL SYSTEM LOOP ----------
