import torch
import torch.nn as nn
import numpy as np
import casadi as cs
from scipy.linalg import convolution_matrix

import torch
import torch.nn as nn
import numpy as np
import casadi as cs
from scipy.linalg import convolution_matrix

def CasadiConv1D(x_cs, ConvWeight, ConvBias):
    #Sequence Length:
    D,L = x_cs.shape
    # Get the weights and input as NumPy arrays
    # Weights shape: (out_channels, 1, kernel_size)
    kernels_np = ConvWeight
    # Bias shape: (out_channels,)
    biases_np = ConvBias
    # Input shape (squeezed): (in_channels, length)
    #x_cs = cs.DM(x_torch.data.detach().numpy().squeeze(0))

    # 1. Create a standard Python list
    Conv_list = []

    # Loop over each channel (the "depthwise" part)
    for i in range(D):
        # Get the specific kernel, signal, and bias for this channel
        k_i = kernels_np[i,  :]  # Shape: (K,)
        # 1. Flip the kernel to turn SciPy's convolution
    #    into PyTorch's cross-correlation.
        k_i_flipped = k_i[::-1]
        T_full = convolution_matrix(np.array(k_i_flipped), L, mode='full')
        T_causal = T_full[:L, :]
        Conv_list.append(T_causal)

    ConvTensor = cs.hcat(Conv_list)
    ConvOutput = []

    for j in range(D):
        x_i = x_cs[j, :]           # Shape: (L,)
        b_i = biases_np[j]         # Scalar

        y_i = (ConvTensor[:,L*j:L+L*j] @ x_i.T) + b_i
        ConvOutput.append(y_i)

    y = cs.hcat(ConvOutput)
    return y

def CasadiSSM(x, delta, A, B, C, D):
    L, ED = x.shape
    N = A.shape[1]

    A = -cs.exp(A)
    A_cs = cs.repmat(cs.reshape(A.T,1,ED*N),L,1) #(ED,N) ->(L,ED N)
    delta_flat = cs.kron(delta, cs.DM.ones(1, N))
    deltaA_cs = cs.exp(delta_flat*A_cs)# (L, ED N) 
    B_cs = cs.repmat(B,1,ED) # (L,N) -> (L,ED N)

    deltaB_cs = delta_flat*B_cs #(L, ED N)
    BX_cs = deltaB_cs * cs.kron(x, cs.DM.ones(1, N))

    h = cs.DM.zeros(1,ED*N)# (ED N) 
    hs = []
    y = []
    for i in range(L):
        h = deltaA_cs[i, :] * h + BX_cs[i, :]
        hs.append(h)
        y.append((cs.reshape(h.T, N, ED).T @ C[i, :].T).T)

    y = cs.vcat(y)
    y = y + (cs.repmat(D,1,L) * x.T).T
    return y

def CasadiSSMFast(x, delta, A, B, C, D):
    L, ED = x.shape
    N = A.shape[1]

    A = -cs.exp(A)
    A_cs = cs.repmat(cs.reshape(A.T,1,ED*N),L,1) #(ED,N) ->(L,ED N)
    delta_flat = cs.kron(delta, cs.DM.ones(1, N))
    deltaA_cs = cs.exp(delta_flat*A_cs)# (L, ED N) 
    B_cs = cs.repmat(B,1,ED) # (L,N) -> (L,ED N)

    deltaB_cs = delta_flat*B_cs #(L, ED N)
    BX_cs = deltaB_cs * cs.kron(x, cs.DM.ones(1, N))

    h = cs.DM.zeros(1,ED*N)# (ED N) 

    # Placeholders for function inputs
    h_prev = cs.SX.sym('h_prev', ED*N,1)
    a_i = cs.SX.sym('a_i',  ED*N,1)      # One row of deltaA_cs
    b_i = cs.SX.sym('b_i', ED*N,1)      # One row of BX_cs
    c_i = cs.SX.sym('c_i', N,1)         # One row of C

    # --- Logic from your loop body ---
    h_new = a_i * h_prev + b_i
    y_i = (cs.reshape(h_new.T, N, ED).T @ c_i).T
    # --- End of loop logic ---

    step_output = cs.horzcat(y_i)

    step_func = cs.Function('step_func',
                            [h_prev, a_i, b_i, c_i],  # Inputs
                            [h_new, step_output])     # Outputs

    # 3. Create the 'looper' function
    # This tells CasADi to run 'step_func' L times
    looper = step_func.mapaccum("looper", L)
    [h_final, all_outputs] = looper(h_0, deltaA_cs.T, BX_cs.T, C.T)

    y = cs.reshape(all_outputs.T, L, ED)
    y = y + (cs.repmat(D,1,L) * x.T).Ts

def CasadiDiscretization(x,SelectionWeight, DeltaWeight, DeltaBias,dt,N):

    L,ED = x.shape
    deltaBC = x @ SelectionWeight.T
    delta,B,C = deltaBC[:,:dt], deltaBC[:,dt:N+dt], deltaBC[:,dt+N:]
    delta =  delta @ DeltaWeight.T + cs.repmat(DeltaBias,1,L).T 
    delta = cs.log(1 + cs.exp(delta))
    return delta, B, C

def CasadiRMSNorm(x, RMSWeight):
    eps = 1e-5
    L,ED = x.shape
    mean_val = cs.sum2(x**2) / x.size2()
    rsqrt_mean = 1/(cs.sqrt(mean_val +  1e-5))
    output = x * cs.repmat(rsqrt_mean,1,ED) *cs.repmat(RMSWeight,1,L).T
    return output

def CasadiMambaBlock(x, model,L,dt,N):
    #RMSBlock1
    RMSWeight1 = cs.DM(model.norm.weight.detach().numpy())


    #Expansion Linear Projection
    LinearExpansionWeight  = cs.DM(model.mixer.in_proj.weight.detach().numpy())
    #No Bias
    #Convolutional Layer
    ConvWeight = model.mixer.conv1d.weight.squeeze(1).detach().numpy()
    ConvBias = model.mixer.conv1d.bias.detach().numpy()

    #DeltaBC Projection
    SelectionProjectionWeight = cs.DM(model.mixer.x_proj.weight.detach().numpy())
    #No Bias

    #Sampling Time Projection
    DeltaTProjectionWeight = cs.DM(model.mixer.dt_proj.weight.detach().numpy())
    DeltaTProjectionBias = cs.DM(model.mixer.dt_proj.bias.detach().numpy())

    ##SSM Paramters
    A = cs.DM(model.mixer.A_log.detach().numpy())
    D = cs.DM(model.mixer.D.detach().numpy())


    #Reduction Projection
    LinearReductionWeight = cs.DM(model.mixer.out_proj.weight.detach().numpy())
    #No bias



    L,ED = x.shape
    x = CasadiRMSNorm(x,RMSWeight1)
    x = x @ LinearExpansionWeight.T 
    x,x_res = x[:,:ED], x[:,ED:]
    x = x.T    
    x = CasadiConv1D(x, ConvWeight, ConvBias)    
    x = x *(1 / (1 + cs.exp(-x)))
    y_res = x_res *(1 / (1 + cs.exp(-x_res)))
    
    delta, B, C = CasadiDiscretization(x,SelectionProjectionWeight, DeltaTProjectionWeight, DeltaTProjectionBias,dt,N)
    y = CasadiSSM(x, delta, A, B, C, D)
    y = y * y_res
    y = y @LinearReductionWeight.T
    return y


def CasadiMamba(x, model,L,dt,N):
    #Input Linear Projection to embedding
    LinearInWeight = cs.DM(model.lin1.weight.detach().numpy())
    LinearInBias = cs.DM(model.lin1.bias.detach().numpy())

    #RMSBlock2
    RMSWeight2 = cs.DM(model.mamba.norm_f.weight.detach().numpy())
    #Output Projection
    LinearOutputWeight = cs.DM(model.MLP.weight.detach().numpy())
    LinearOutputBias = cs.DM(model.MLP.bias.detach().numpy())

    layers = len(model.mamba.layers)

    x = x @ LinearInWeight.T + cs.repmat(LinearInBias,1,L).T
    for i in range(layers):
        x = CasadiMambaBlock(x, model.mamba.layers[i],L,dt,N)+x

    y = CasadiRMSNorm(x,RMSWeight2)
    y =  y@ LinearOutputWeight.T + cs.repmat(LinearOutputBias,1,L).T

    return y