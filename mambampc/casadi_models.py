"""CasADi re-implementations of the trained PyTorch sequence models.

Each class mirrors the forward pass of its PyTorch counterpart in `mambampc.models`
using CasADi symbolics, so a trained network can be embedded directly as an equality
constraint in the MPC nonlinear program (see `mambampc.mpc`). Weights are read out of
the PyTorch module at construction time and frozen as `cs.DM` constants.

Every class here must stay numerically identical to its torch twin; the shipped
checkpoints agree with their CasADi counterparts to ~1e-15.

Contents
--------
Mamba        : CasadiConv1D, CasadiSSM, CasadiDiscretization, CasadiMambaBlock, CasadiMamba
Transformer  : CasadiFeedForward, CasadiAttention, CasadiBlock, CasadiTransformer, CasadiPredictor
LSTM         : CasadiLSTM
Shared       : RMSNorm, LayerNorm, SILU, SiLU, GELU, TanH, sigmoid_ca, NN
"""

import numpy as np
import torch.nn as nn
import casadi as cs
from scipy.linalg import convolution_matrix


# =============================================================================
# Shared activations and normalisations
# =============================================================================

# =============================================================================
# Mamba
# =============================================================================
class CasadiConv1D:
    def __init__(self, layer,L):
        # Extract weight: shape (ED, 1, K) -> squeeze(1) -> (ED, K)
        kernel = layer.mixer.conv1d.weight.squeeze(1).detach().cpu().numpy()
        ED = layer.mixer.D.shape[0] # Assuming D is a tensor
        
        Conv_list = []
        for i in range(ED):
            k_i = kernel[i, :]  # Shape: (K,)
            k_i_flipped = k_i[::-1]
            
            # Construct full convolution matrix using scipy or custom helper
            T_full = convolution_matrix(np.array(k_i_flipped), L, mode='full')
            T_causal = T_full[:L, :] # Shape: (L, L)
            Conv_list.append(T_causal)
        
        # Horizontally stack numpy arrays, then convert the big matrix to CasADi DM
        # Shape of ConvTensor: (N, ED * N)
        ConvTensor = cs.hcat(Conv_list)
        self.kernel = cs.DM(ConvTensor)
        
        # Shape of bias: (ED, 1)
        bias_np = layer.mixer.conv1d.bias.detach().cpu().numpy()
        self.bias = cs.DM(bias_np)

    def __call__(self, x):
            """
            x: CasADi MX/SX symbolic or DM constant of shape (D, L)
            where D == ED (number of channels), and L == N (sequence length)
            """
            D, L = x.shape
            ConvOutput = []
            
            for j in range(D):
                # Extract row j (channel j) as a column vector (L, 1) to match matrix mult
                x_i = x[j, :]  
                b_i = self.bias[j] # Scalar bias for channel j
                
                # Slice the corresponding (L, L) transformation matrix
                # Block j spans from columns L*j to L*(j+1)
                T_j = self.kernel[:,L*j:L+L*j]
                
                # (L, L) @ (L, 1) + scalar -> (L, 1)
                y_i = (T_j @ x_i.T) + b_i
                ConvOutput.append(y_i) # Transpose back to (1, L) for stacking

            # Vertically stack rows to return shape (D, L)
            y = cs.hcat(ConvOutput)
            return y
    
class CasadiSSM(nn.Module):
    def __init__(self, layer, StateReset=True):
        super().__init__()
        self.A = -cs.exp(cs.DM(layer.mixer.A_log.detach().numpy()))
        self.D = cs.DM(layer.mixer.D.detach().numpy())
        self.N = self.A.shape[1]
        ED = self.D.shape[0]
        self.h = cs.DM.zeros(1,ED*self.N)# (ED N) 
        self.StateReset = StateReset
        
    def __call__(self, x, delta, B,C):
        L, ED = x.shape

        A_cs = cs.repmat(cs.reshape(self.A.T,1,ED*self.N),L,1) #(ED,N) ->(L,ED N)
        delta_flat = cs.kron(delta, cs.DM.ones(1, self.N))
        deltaA_cs = cs.exp(delta_flat*A_cs)# (L, ED N) 
        B_cs = cs.repmat(B,1,ED) # (L,N) -> (L,ED N)
        deltaB_cs = delta_flat*B_cs #(L, ED N)
        BX_cs = deltaB_cs * cs.kron(x, cs.DM.ones(1, self.N))
        
        if self.StateReset:
            self.h =  cs.DM.zeros(1, ED*self.N)# (ED N) 
        
        hs = []
        y = []
        for i in range(L):
            h = deltaA_cs[i, :] * self.h + BX_cs[i, :]
            self.h = h
            hs.append(h)
            y.append((cs.reshape(h.T, self.N, ED).T @ C[i, :].T).T)


        y = cs.vcat(y)
        y = y + (cs.repmat(self.D, 1, L) * x.T).T
        #print(y)
        return y
    
class CasadiDiscretization:
    def __init__(self, layer):
        self.dt = layer.mixer.dt_proj.weight.detach().numpy().shape[1]
        self.SelectionWeight = cs.DM(layer.mixer.x_proj.weight.detach().numpy())
        #Sampling Time Projection
        self.dtDeltaTProjectionWeight = cs.DM(layer.mixer.dt_proj.weight.detach().numpy())
        self.dtDeltaTProjectionBias = cs.DM(layer.mixer.dt_proj.bias.detach().numpy())
        self.N = layer.mixer.A_log.detach().numpy().shape[1]

    def __call__(self, x):
        L,ED = x.shape
        deltaBC = x @ self.SelectionWeight.T
        delta,B,C = deltaBC[:,:self.dt], deltaBC[:,self.dt:self.dt+self.N], deltaBC[:,self.dt+self.N:]
        delta =  delta @ self.dtDeltaTProjectionWeight.T + cs.repmat(self.dtDeltaTProjectionBias,1,L).T 
        delta = cs.log(1 + cs.exp(delta))
        return delta, B, C
    
class RMSNorm:
    def __init__(self, weight, eps = 1e-5):
        self.weight = weight
        self.eps = eps
    def __call__(self, x):
        L,ED = x.shape
        mean_val = cs.sum2(x**2) / x.size2()
        rsqrt_mean = 1/(cs.sqrt(mean_val +  self.eps))
        output = x * cs.repmat(rsqrt_mean,1,ED) *cs.repmat(self.weight,1,L).T
        return output
    
class SILU:    
    def __init__(self):
        pass
    def __call__(self, x):
        return x *(1 / (1 + cs.exp(-x)))

class CasadiMambaBlock:
    def __init__(self, layer,N):
        self.conv = CasadiConv1D(layer,N)
        self.disc = CasadiDiscretization(layer)
        self.ssm = CasadiSSM(layer)
        self.norm = RMSNorm(layer.norm.weight.detach().numpy())
        self.ExpansionWeight = cs.DM(layer.mixer.in_proj.weight.detach().numpy())
        self.SILU = SILU()
        self.RedWeight = cs.DM(layer.mixer.out_proj.weight.detach().numpy())      

    def __call__(self, x):
        L,ED = x.shape
        x = self.norm(x)
        x = x @ self.ExpansionWeight.T 
        x,x_res = x[:,:ED], x[:,ED:]
        x = x.T
        x = self.conv(x)
        x = self.SILU(x)
        y_res = self.SILU(x_res)
        delta, B, C = self.disc(x)
        y = self.ssm(x, delta, B, C)
        y = y * y_res
        y = y @self.RedWeight.T
        return y
    

class CasadiMamba:
    def __init__(self, model,L):
        self.layers = [CasadiMambaBlock(layer,L) for layer in model.mamba.layers]
        self.lin1Weight = cs.DM(model.lin1.weight.detach().numpy())
        self.lin1Bias = cs.DM(model.lin1.bias.detach().numpy())
        self.MLPWeight = cs.DM(model.MLP.weight.detach().numpy())
        self.MLPBias = cs.DM(model.MLP.bias.detach().numpy())
        self.RMSNorm = RMSNorm(cs.DM(model.mamba.norm_f.weight.detach().numpy()))
        
    def __call__(self, x):
        L,ED = x.shape
        x = x @ self.lin1Weight.T + cs.repmat(self.lin1Bias,1,L).T
        for layer in self.layers:
            x = layer(x)+x
            
        x = self.RMSNorm(x)
        x = x @ self.MLPWeight.T + cs.repmat(self.MLPBias,1,L).T
        return x

        
# =============================================================================
# Transformer
# =============================================================================

class LayerNorm:
    """Casadi implementation of LayerNorm over the last dimension."""

    def __init__(self, weight=None, bias=None, eps=1e-5):
        self.eps = eps
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        
        L, D = x.shape
        
        mean_val = cs.sum2(x) / D  # Compute mean
        
        # Center the inputs
        x_centered = x - cs.repmat(mean_val, 1, D)
        
        # Compute the true variance of the centered inputs
        var_val = cs.sum2(x_centered**2) / D  
        
        # Compute the inverse standard deviation
        rsqrt_var = 1.0 / cs.sqrt(var_val + self.eps)  
        
        # Normalize
        x_normed = x_centered * cs.repmat(rsqrt_var, 1, D)
        
        if self.weight is None:
            return x_normed 
        else:
            output = x_normed * cs.repmat(self.weight, 1, L).T + cs.repmat(self.bias, 1, L).T    # Apply learned scale and bias
            return output


def GELU(x_res): # Gaussian Error Linear Unit activation function
    alpha = np.sqrt(2 / np.pi)
    beta = 0.044715
    inner = alpha * (x_res + beta * (x_res**3)) # Intermediate calculation
    return 0.5 * x_res * (1 + cs.tanh(inner)) # GELU formula

def TanH(x_res): # Hyperbolic Tangent activation function
    return cs.tanh(x_res) # TanH formula

def SiLU(x_res): # Sigmoid Linear Unit activation function
    return x_res *(1 / (1 + cs.exp(-x_res))) # SiLU formula


class CasadiFeedForward():
    """Casadi implementation of the FeedForward block."""

    def __init__(self, layer): # Initialize FeedForward with layer weights and biases
        MLP = layer.mlp
        self.LayerNormWeight = cs.DM(MLP.net[0].weight.detach().numpy())
        self.LayerNormBias = cs.DM(MLP.net[0].bias.detach().numpy())
        self.norm = LayerNorm(self.LayerNormWeight, self.LayerNormBias)
        self.w1 = cs.DM(MLP.net[1].weight.detach().numpy())
        self.b1 = cs.DM(MLP.net[1].bias.detach().numpy())
        self.w2 = cs.DM(MLP.net[4].weight.detach().numpy())
        self.b2 = cs.DM(MLP.net[4].bias.detach().numpy())

    def __call__(self, x): # Apply FeedForward transformation
        L, D = x.shape
        x = self.norm(x) # Apply Layer Normalization
        x = x @ self.w1.T + cs.repmat(self.b1, 1, L).T # Linear transformation 1
        x = SiLU(x) # Apply SiLU activation
        x = x @ self.w2.T + cs.repmat(self.b2, 1, L).T # Linear transformation 2
        return x


class CasadiAttention:
    """Casadi implementation of the first Transformer attention block."""

    def __init__(self, layer, temperature=2.5):
        attn = layer.attn
        self.LayerNormWeight = cs.DM(attn.norm.weight.detach().numpy())
        self.LayerNormBias = cs.DM(attn.norm.bias.detach().numpy())
        self.norm = LayerNorm(self.LayerNormWeight, self.LayerNormBias)
        self.in_dim = attn.to_qkv.in_features
        self.out_dim = attn.to_out[0].in_features if isinstance(attn.to_out, nn.Sequential) else self.in_dim
        self.heads = attn.heads
        self.dim_head = self.out_dim // self.heads
        self.qkv_weight = cs.DM(attn.to_qkv.weight.detach().numpy())
        self.Q_weight = self.qkv_weight[0:self.out_dim,:]
        self.K_weight = self.qkv_weight[self.out_dim:2*self.out_dim,:]
        self.V_weight = self.qkv_weight[2*self.out_dim:3*self.out_dim,:]
        self.out_weight = cs.DM(attn.to_out[0].weight.detach().numpy().T) if isinstance(attn.to_out, nn.Sequential) else None
        self.out_bias = cs.DM(attn.to_out[0].bias.detach().numpy()) if isinstance(attn.to_out, nn.Sequential) else None
        self.temperature = temperature
        self.scale = 1.0 / (temperature * (self.dim_head ** 0.5))

    def __call__(self, x): # Apply Attention mechanism
        L,D = x.shape
        x = self.norm(x) # Apply Layer Normalization
        Q = x @ self.Q_weight.T # Compute Query
        K = x @ self.K_weight.T # Compute Key
        V = x @ self.V_weight.T # Compute Value

        
        mask = np.triu(np.ones((L, L)), k=1)  # Create upper triangular mask for causality
        # 2. Convert the boolean mask to float values (0 and -inf)
        causal_mask = np.zeros((L, L))
        causal_mask[mask == 1] = -1e9
        causal_mask = cs.DM(causal_mask)

        out_heads = []
        for h in range(self.heads): # Process each attention head
            start_col = h * self.dim_head
            end_col = (h + 1) * self.dim_head
            qh = Q[:, start_col:end_col]
            kh = K[:, start_col:end_col]
            vh = V[:, start_col:end_col]
            scores = (qh @ kh.T) * self.scale # Compute attention scores
            scores = scores + causal_mask  # Apply causal mask
            exp_scores = cs.exp(scores - cs.mmax(scores)) # Exponentiate scores
            attn_weights = exp_scores / cs.sum2(exp_scores) # Softmax to get attention weights
            out_heads.append(attn_weights @ vh) # Weighted sum of values
        attn_out = cs.hcat(out_heads) # Concatenate head outputs
        if self.out_weight is None:
            return attn_out
        return attn_out @ self.out_weight + cs.repmat(self.out_bias, 1, L).T # Apply output projection
    
class CasadiBlock:
    """Casadi implementation of the Transformer block."""

    def __init__(self, layer, temperature=2.5):
        self.attn = CasadiAttention(layer, temperature)
        self.mlp = CasadiFeedForward(layer)
        self.norm1 = LayerNorm(weight=None, bias=None, eps=1e-5) # First LayerNorm
        self.norm2 = LayerNorm(weight=None, bias=None, eps=1e-5) # Second LayerNorm

    def __call__(self, x): # Apply Transformer block operations
        x = x + self.attn(self.norm1(x)) # Attention with skip connection
        x = x + self.mlp(self.norm2(x)) # FeedForward with skip connection
        return x
    

class CasadiTransformer:
    def __init__(self, model, temperature=2.5):
        self.norm = LayerNorm(model.transformer.norm.weight.detach().numpy(), model.transformer.norm.bias.detach().numpy())
        self.layers = [CasadiBlock(layer, temperature) for layer in model.transformer.layers]
        self.InputProjWeight = model.transformer.input_proj.weight.detach().numpy().T if isinstance(model.transformer.input_proj, nn.Linear) else None
        self.InputProjBias = model.transformer.input_proj.bias.detach().numpy() if isinstance(model.transformer.input_proj, nn.Linear) else None
        self.OutputProjWeight = model.transformer.output_proj.weight.detach().numpy().T if isinstance(model.transformer.output_proj, nn.Linear) else None
        self.OutputProjBias = model.transformer.output_proj.bias.detach().numpy() if isinstance(model.transformer.output_proj, nn.Linear) else None
        self.CondProjWeight = model.transformer.cond_proj.weight.detach().numpy().T if isinstance(model.transformer.cond_proj, nn.Linear) else None
        self.CondProjBias = model.transformer.cond_proj.bias.detach().numpy() if isinstance(model.transformer.cond_proj, nn.Linear) else None


    def __call__(self, x): # Apply Transformer forward pass
        L, D = x.shape
        x = x @ self.InputProjWeight + cs.repmat(self.InputProjBias, 1, L).T # Input projection
        
        for layer in self.layers: # Apply each Transformer block
            x = layer(x)
       
        x = self.norm(x) # Apply final Layer Normalization
        if self.OutputProjWeight is not None:
            x = x @ self.OutputProjWeight + cs.repmat(self.OutputProjBias, 1, L).T # Output projection
        return x
    

class CasadiPredictor:
    def __init__(self, model, temperature=2.5):
        self.pos_embedding = cs.DM(model.pos_embedding[0].detach().numpy())
        self.transformer = CasadiTransformer(model, temperature)

    def __call__(self, x): # Apply Predictor forward pass
        L,D = x.shape
        x = x + self.pos_embedding[:L,:] # Add positional embedding
        
        x = self.transformer(x) # Apply Transformer
        return x

# =============================================================================
# LSTM
# =============================================================================

def sigmoid_ca(x):
    return 1 / (1 + cs.exp(-x))


def lstm_cell_ca(x_t, h_prev, c_prev, W_ih, W_hh, b, W_fc, b_fc):
    """One LSTM step plus the output projection.

    Gates are stacked in PyTorch's order (input, forget, cell, output) in a
    single `4*hidden` vector, matching `nn.LSTM`'s weight layout.
    """
    hidden_size = h_prev.shape[0]

    gates = cs.mtimes(W_ih, x_t) + cs.mtimes(W_hh, h_prev) + b

    i = sigmoid_ca(gates[0:hidden_size])
    f = sigmoid_ca(gates[hidden_size : 2 * hidden_size])
    g = cs.tanh(gates[2 * hidden_size : 3 * hidden_size])
    o = sigmoid_ca(gates[3 * hidden_size : 4 * hidden_size])

    c_t = f * c_prev + i * g
    h_t = o * cs.tanh(c_t)

    y_t = cs.mtimes(W_fc, h_t) + b_fc

    return y_t, h_t, c_t


class CasadiLSTM:
    """CasADi re-implementation of `mambampc.models.lstm.LSTMModel`.

    Rolls the recurrence over the sequence and returns every timestep's output,
    matching the PyTorch model's sequence-to-sequence `forward`.
    """

    def __init__(self, model):
        self.W_ih = cs.DM(model.lstm.weight_ih_l0.detach().cpu().numpy())
        self.W_hh = cs.DM(model.lstm.weight_hh_l0.detach().cpu().numpy())
        # nn.LSTM keeps two bias vectors; their sum is what the recurrence uses.
        self.b = cs.DM(
            (model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0).detach().cpu().numpy()
        )
        self.W_fc = cs.DM(model.fc.weight.detach().cpu().numpy())
        self.b_fc = cs.DM(model.fc.bias.detach().cpu().numpy())
        self.hidden_size = model.lstm.hidden_size

    def __call__(self, x):
        """x: (L, input_size) symbolic -> (L, output_size)."""
        L, _ = x.shape
        h = cs.DM.zeros(self.hidden_size, 1)
        c = cs.DM.zeros(self.hidden_size, 1)
        ys = []
        for k in range(L):
            y, h, c = lstm_cell_ca(
                x[k, :].T, h, c, self.W_ih, self.W_hh, self.b, self.W_fc, self.b_fc
            )
            ys.append(y.T)
        return cs.vcat(ys)


# =============================================================================
# Generic helper
# =============================================================================

def NN(u, Network, sigma):
    """Evaluate a plain feedforward `Network` (a module with `.linears`) on `u`."""
    y = u
    L = len(Network.linears)
    for i in range(L - 1):
        W = cs.DM(Network.linears[i].weight.detach().cpu().numpy())
        b = cs.DM(Network.linears[i].bias.detach().cpu().numpy())
        y = sigma(W @ y + b)

    W = cs.DM(Network.linears[-1].weight.detach().cpu().numpy())
    b = cs.DM(Network.linears[-1].bias.detach().cpu().numpy())
    y = W @ y + b
    return y
