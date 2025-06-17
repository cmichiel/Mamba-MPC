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
import rbf_gauss


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

xmean = np.load("Train_mean.npy")
xstd = np.load("Train_std.npy")

MambaModel = ModelKoopman(3, N, 40, 1, 1, "gaussian")
MambaModel.load_state_dict(torch.load("RBF_Params_Koopman_Tini3_nbasisKoopman40_N10_KoopmanBasis_gaussian",weights_only=True))
MambaModel = l4c.L4CasADi(MambaModel, device='cpu',batched=True, name='y_expr')

for i in range(L-N):
    inputs = (np.concatenate((u_ini_list[i].T, y_ini_list[i].T, uN[i]))-xmean)/xstd
    x = cs.DM(inputs)
    print(x.shape)
    yN.append(MambaModel(x.T))

yN = np.array(yN).reshape(L-N,10)
fig, axs = plt.subplots(2)
fig.suptitle("SPC RBF")
axs[0].plot(ref[:230], label="$ref$")
axs[0].plot(yN[:,1] , label = "$y_{model}$")
axs[0].legend()
axs[0].grid()
axs[0].set_ylim(-5, 5)

axs[1].plot(uN[:,1])
axs[0].set(ylabel="$y$")
#axs[1].set(ylabel="theta")
axs[1].set(ylabel="$u$")
axs[1].grid()
plt.show()
