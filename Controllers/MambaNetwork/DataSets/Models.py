import sys
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import rbf_gauss



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
    
class ModelKoopmanNARX(nn.Module):
    def __init__(self, Tini, N, nbasisKoopman,nbasisNARX, input_dim, output_dim, KoopmanBasis, NARXBasis):
        super().__init__()
        self.Tini = Tini
        self.N = N
        self.nbasisKoopman = nbasisKoopman
        self.nbasisNARXBasis = nbasisNARX
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_linear_features = (2 * self.Tini - 1)*self.input_dim
        self.out_linear_features = (self.nbasisKoopman)
        self.in_nl_features = (2 * self.Tini - 1+self.N)*self.input_dim
        self.out_nl_features = (self.nbasisNARXBasis)
        self.in_2_features = self.nbasisKoopman+self.nbasisNARXBasis+self.N*self.input_dim
        self.out_2_features = (self.N)*self.output_dim
        
        #Activation Functions
        self.basis_funcKoopman = getattr(rbf_gauss.RBF_gaussian,KoopmanBasis)
        self.basis_funcNARXBasis = getattr(rbf_gauss.RBF_gaussian, NARXBasis)

        #Define Network Layers
        self.RBFKoopman= rbf_gauss.RBF_gaussian(self.in_linear_features, self.out_linear_features,self.basis_funcKoopman)
        self.RBFNARXBasis = rbf_gauss.RBF_gaussian(self.in_nl_features, self.out_nl_features,self.basis_funcNARXBasis)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)
        
        #Initialize Linear layer 
        nn.init.uniform_(self.l_2.weight[:self.nbasisKoopman+self.N]) #Koopmans Basis Initialized by uniform distribution
        nn.init.zeros_(self.l_2.weight[self.nbasisKoopman+self.N:]) #Full basis initialized as 

    def forward(self, data):
        data_ini = data[:, 0 : (2 * self.Tini - 1)*self.input_dim]
        data_f   = data[:, (2 * self.Tini - 1)*self.input_dim :]
        x1 = self.RBFKoopman(data_ini)
        x1 = torch.cat((x1,data_f),1)
        x2= self.RBFNARXBasis(data)
        x = torch.cat((x1,x2),1)
        x = self.l_2(x)
        return x
    

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
        self.RBFKoopman= rbf_gauss.RBF_gaussian(self.in_linear_features, self.out_linear_features,self.basis_funcKoopman)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)

        #Linear Layer uniformly initialized
        nn.init.uniform_(self.l_2.weight)

        
    def forward(self, data):
        #Split Data into Up, Yp and Uf 
        data_ini = data[:, 0 : (2 * self.Tini - 1)*self.input_dim]
        data_f   = data[:, (2 * self.Tini - 1)*self.input_dim :]

        #Up,Yp into RBF layer
        x1 = self.RBFKoopman(data_ini)        
        #Add Uf
        #x2 = torch.cat((x1,data_f),1)

        #Add Resnet
        x2 = torch.cat((x1,data),1)

        #Linear Layer
        x = self.l_2(x2)
        return x
    
class ModelNARXResnet(nn.Module):
    def __init__(self, Tini, N, nbasis, input_dim, output_dim, NARXBasis):
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
        self.in_2_features = (nbasis)+N+2*Tini-1
        self.out_2_features = (N)*output_dim

        #Activation Function
        self.basis_funcNARXBasis = getattr(rbf_gauss.RBF_gaussian, NARXBasis)

        #Define Network Layers
        self.RBFNARXBasis = rbf_gauss.RBF_gaussian(self.in_nl_features, self.out_nl_features, self.basis_funcNARXBasis)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)

        #Initialize Linear layer uniformly
        nn.init.uniform_(self.l_2.weight)
  
    def forward(self, data):
        x2= self.RBFNARXBasis(data)
        x = torch.cat((x2,data),1)
        x = self.l_2(x)
        return x
    

class ModelNARX(nn.Module):
    def __init__(self, Tini, N, nbasis, input_dim, output_dim, NARXBasis):
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
        self.in_2_features = (nbasis)
        self.out_2_features = (N)*output_dim

        #Activation Function
        self.basis_funcNARXBasis = getattr(rbf_gauss.RBF_gaussian, NARXBasis)

        #Define Network Layers
        self.RBFNARXBasis = rbf_gauss.RBF_gaussian(self.in_nl_features, self.out_nl_features,self.basis_funcNARXBasis)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)

        #Initialize Linear layer uniformly
        nn.init.uniform_(self.l_2.weight)

    def forward(self, data):
        x2= self.RBFNARXBasis(data)
        x = self.l_2(x2)
        return x
    

class ModelKoopmanNARXResnet(nn.Module):
    def __init__(self, Tini, N, nbasisKoopman,nbasisNARX, input_dim, output_dim, KoopmanBasis, NARXBasis):
        super().__init__()
        self.Tini = Tini
        self.N = N
        self.nbasisKoopman = nbasisKoopman
        self.nbasisNARXBasis = nbasisNARX
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_linear_features = (2 * self.Tini - 1)*self.input_dim
        self.out_linear_features = (self.nbasisKoopman)
        self.in_nl_features = (2 * self.Tini - 1+self.N)*self.input_dim
        self.out_nl_features = (self.nbasisNARXBasis)
        self.in_2_features = self.nbasisKoopman+self.nbasisNARXBasis+self.N*self.input_dim+2*Tini-1+self.N*self.input_dim
        self.out_2_features = (self.N)*self.output_dim
        
        #Activation Functions
        self.basis_funcKoopman = getattr(rbf_gauss.RBF_gaussian,KoopmanBasis)
        self.basis_funcNARXBasis = getattr(rbf_gauss.RBF_gaussian, NARXBasis)

        #Define Network Layers
        self.RBFKoopman= rbf_gauss.RBF_gaussian(self.in_linear_features, self.out_linear_features,self.basis_funcKoopman)
        self.RBFNARXBasis = rbf_gauss.RBF_gaussian(self.in_nl_features, self.out_nl_features,self.basis_funcNARXBasis)
        self.l_2 = nn.Linear(self.in_2_features, self.out_2_features, bias=False)
        
        #Initialize Linear layer 
        nn.init.uniform_(self.l_2.weight[:self.nbasisKoopman+self.N]) #Koopmans Basis Initialized by uniform distribution
        nn.init.zeros_(self.l_2.weight[self.nbasisKoopman+self.N:]) #Full basis initialized as 

    def forward(self, data):
        data_ini = data[:, 0 : (2 * self.Tini - 1)*self.input_dim]
        data_f   = data[:, (2 * self.Tini - 1)*self.input_dim :]
        #Koopman Base
        x1 = self.RBFKoopman(data_ini)
        x1 = torch.cat((x1,data_f),1)

        #NARX Base
        x2= self.RBFNARXBasis(data)

        #Combine Koopman and NARX Base
        x = torch.cat((x1,x2),1)

        #Add Resnet Base
        x = torch.cat((x,data),1)
        x = self.l_2(x)
        return x