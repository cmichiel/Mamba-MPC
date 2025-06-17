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
import l4casadi as l4c



class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=4, n_layers=1, expand_factor = 1, d_conv = 1, d_state = 2)
        self.lin1 = nn.Linear(in_dim,4)
        self.mamba = Mamba(self.config)
        self.lin2 = nn.Linear(4,out_dim)
        
    
    def forward(self,x):
        #print("Input Shape:"+str(x.shape))
        x = self.lin1(x.unsqueeze(0))
        x = self.mamba(x)
        x = self.lin2(x)
        return x.squeeze(0)
    

uN= np.array(torch.load("uNMPC.pt", weights_only=False))
u_ini_list = np.array(torch.load("u_ini.pt", weights_only=False))
y_ini_list = np.array(torch.load("y_ini.pt", weights_only=False))

model = Net(15,10)
model.load_state_dict(torch.load("State_dicts/Test_Model_Thomas_Best", weights_only=True, map_location=torch.device('cpu')))

#Sampling Time
Ts = 0.1

# Simulation time
time_range = 24

# Simulation steps
L = round(time_range/Ts)
N = 10

# Init time
t = np.arange(L+1)*Ts

mu = 1

# Initial state
x1_init = 1
x2_init = 1

x1 = np.zeros(L + 1)
x2 = np.zeros(L + 1)
u = np.zeros(L)
ref = np.zeros(L)

#Choose the reference
f = 0.1 # Frequency in Hz
A = 2
ref =  A*np.sin(2*np.pi*f*t)


yN = []


MambaModel = l4c.L4CasADi(model, device='cpu',batched=False, name='y_expr')

for i in range(L-N):
    inputs = np.concatenate((u_ini_list[i].T, y_ini_list[i].T, uN[i]))
    x = cs.DM(inputs)
    print(MambaModel(x.T))

yN = np.array(yN).reshape(L-N,10)
print(yN.shape)