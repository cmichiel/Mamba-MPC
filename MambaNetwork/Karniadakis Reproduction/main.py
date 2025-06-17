import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import Mamba, MambaConfig
import argparse
from DataProcessing import HankelMatrices, HankelMatricesSequences, Sequence2Sequence, Sequence2SequenceStates, Sequence2SequenceFuture, Sequence2SequencePast
from DataProcessing import BestModelSaver
import scipy.io
import scipy.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, 'Utilities/')
import os
from torch.utils.data import DataLoader
 
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10001,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=4,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=1,
                    help='Num of layers')
parser.add_argument('--n-test', type=int, default=200,
                    help='Size of test set')
       

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start',series[0])
    print('Training end',series[lt-n_test-1])
    print('Testing start',series[lt-n_test])
    print('Testing end',series[lt-1])

set_seed(args.seed,args.cuda)

class ParallelNet(nn.Module):
    def __init__(self,Tini,N):
        super().__init__()
        D_model  = 4
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 1,d_state = 16)
        self.lin2N = nn.Linear(2*Tini-1,N)
        self.linDP = nn.Linear(1,D_model)
        self.linDF = nn.Linear(1,D_model)
        self.mambaP = Mamba(self.config)
        self.mambaF = Mamba(self.config)
        self.MLP = nn.Linear(D_model,1)
        
    
    def forward(self,x):
        # print("Input Shape:"+str(x.shape))
        x = self.lin1(x)
        x = self.mamba(x)
        x = self.MLP(x)
        return x.squeeze(0)
    
class NetSequences(nn.Module):
    def __init__(self,in_dim,out_dim,L,N):
        super().__init__()
        D_model  = 32
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 4, d_state = 32)
        self.lin1 = nn.Linear(in_dim,D_model)
        self.mamba = Mamba(self.config)
        self.MLP = nn.Linear(D_model,out_dim)        
        

    def forward(self,x):
        # print("Input Shape:"+str(x.shape))
        x = self.lin1(x)
        x = self.mamba(x)
        x = self.MLP(x)
        return x.squeeze(0) 

def PredictWithData(train, test):
    X_sample,Y_sample = next(iter(train))
    batch_size,SequenceLength,Input_dim = X_sample.shape
    _,N, Output_dim = Y_sample.shape 
    clf = NetSequences(Input_dim, Output_dim,SequenceLength, N)
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.99, total_iters=10)
    if args.cuda:
        clf = clf.cuda()
        print("Using GPU")
    Vallosses = []
    Train_loss = []
    filename = input("Enter file name: ")
    best_model_saver = BestModelSaver(filepath="State_dicts/" + filename)
    for e in range(args.epochs):
        clf.train()
        running_loss = 0.
        last_loss = 0
        for idx,data in enumerate(train):        
            X,Y = data
            z = clf(X)
            loss = F.mse_loss(z.squeeze(0),Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            # print every 10 epoch
        with torch.no_grad():  # turn off backpropagation
            Vloss = 0
            for idx,data in enumerate(test):
                X_val, Y_val = data
                y_eval = clf.forward(X_val) # are features from our test set
                Val_loss = F.mse_loss(y_eval, Y_val)  # Find the loss or error
                Vloss += Val_loss
            Vallosses.append(Vloss.cpu().detach().numpy()/batch_size)
        
        scheduler.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.8f | Validation Loss: %.8f' % (e, running_loss/batch_size, Vloss/batch_size))
            best_model_saver(current_valid_loss=Vloss/batch_size,
                 model=clf,
                 epoch=e, # Current epoch index (0-based)
                 optimizer=opt,
                 criterion=F.mse_loss)
        Train_loss.append(running_loss/batch_size)
    return Vallosses, Train_loss






# Load multisine data from matlab
mat = scipy.io.loadmat("DataSets/u_data.mat")
u_data = mat["u_data"]

mat = scipy.io.loadmat("DataSets/y_data.mat")
y_data = mat["y_data"]

# convert data to tensors
u_data = np.array(u_data)
y_data = np.array(y_data)
u_data = torch.FloatTensor(u_data).transpose(0,1).unsqueeze(2)
y_data = torch.FloatTensor(y_data).transpose(0,1).unsqueeze(2)
print("u_data shape:"+str(u_data.shape))
print("y_data shape:"+str(y_data.shape))
input_dim = list(u_data.shape)[1]
output_dim = list(y_data.shape)[1]


DataShuffle = False
norm_data = True

#Load in Test data
# Load multisine data from matlab
mat = scipy.io.loadmat("DataSets/u_dataVal.mat")
u_val = mat["u_data"]

mat = scipy.io.loadmat("DataSets/y_dataVal.mat")
y_val = mat["y_data"]
u_test = torch.FloatTensor(u_val).transpose(0,1).unsqueeze(2)
y_test = torch.FloatTensor(y_val).transpose(0,1).unsqueeze(2)
print("u_test shape:"+str(u_test.shape))
print("y_test shape:"+str(y_test.shape))


if args.cuda:
    X = u_data.cuda()
    Y = y_data.cuda()
    X_test = u_test.cuda()
    Y_test = y_test.cuda()
Dataset = TensorDataset(X, Y)
Dataset_test = TensorDataset(X_test, Y_test)
print("Dataset Shape:"+str(Dataset.tensors[0].shape))
print("Test Dataset Shape:"+str(Dataset_test.tensors[0].shape))
train_dataloader = DataLoader(Dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(Dataset_test, batch_size=128, shuffle=True)
# train_dataloader, test_dataloader = torch.utils.data.random_split(Dataloader, [0.8, 0.2])




# #State Data
# mat = scipy.io.loadmat("DataSets/X0_train.mat")
# X0_train = mat["X0_train"]

# mat = scipy.io.loadmat("DataSets/X0_test.mat")
# X0_test = mat["X0_test"]

# mat = scipy.io.loadmat("DataSets/Y_test.mat")
# Y_test = mat["Y_test"]

# mat = scipy.io.loadmat("DataSets/Y_train.mat")
# Y_train = mat["Y_train"]


# mat = scipy.io.loadmat("DataSets/U_train.mat")
# U_train = mat["U_train"]

# mat = scipy.io.loadmat("DataSets/U_test.mat")
# U_test = mat["U_test"]

# #State Data to tensors
# X0_train = torch.FloatTensor(np.array(X0_train))
# X0_test = torch.FloatTensor(np.array(X0_test))
# Y_test = torch.FloatTensor(np.array(Y_test))
# Y_train = torch.FloatTensor(np.array(Y_train))
# U_train = torch.FloatTensor(np.array(U_train))
# U_test = torch.FloatTensor(np.array(U_test))


# trainX,trainY = Sequence2SequenceStates(X0_train,U_train,Y_train)
# #trainX= torch.cat((X0_train, U_train), 1).unsqueeze(1).transpose(1,2)
# #trainY = Y_train.unsqueeze(1)

# testX, testY = Sequence2SequenceStates(X0_test,U_test,Y_test)
# #testX = torch.cat((X0_test, U_test), 1).unsqueeze(1).transpose(1,2)
# #testY  = Y_test.unsqueeze(1)

# print("TrainX Shape:"+str(trainX.shape))
# print("TrainY Shape:"+str(trainY.shape))
# print("TestX Shape:"+str(testX.shape))
# if args.cuda:
#     trainX= trainX.cuda()
#     trainY = trainY.cuda()
#     testX = testX.cuda()
#     testY = testY.cuda()
#     print("Using GPU")
# train_data = TensorDataset(trainX, trainY)
# train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
# test_data = TensorDataset(testX, testY)
# test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)





Val_loss, Train_loss = PredictWithData(train_dataloader, test_dataloader)


# X_test,Y_test = test_dataloader
# Error = np.mean(np.abs(predictions  - Y_test.cpu().numpy()),1)

# fig, axs = plt.subplots(2)
# fig.suptitle("RBF VDP Output")
# axs[0].plot(testy[:2000,9], label="$y_{test}$")
# axs[0].plot(predictions[:2000,9], label = "$y_{eval}$")
# axs[0].legend()
# axs[0].grid()

# axs[1].plot(Error[:2000])
# axs[0].set(ylabel="$y_1$")
# #axs[1].set(ylabel="theta")
# axs[1].set(ylabel="$E[|y_{eval}-y_{test}|]$")
# axs[1].grid()
# plt.show()

 
# Load the loss data
# val_loss = np.load("val_loss.npy")
# train_loss = np.load("train_loss.npy")
 
####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1,1)
#gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
ax.semilogy(Val_loss, '-k', lw=2.0, label="validation loss")
ax.semilogy(Train_loss, '-b', lw=2.0, label="training loss")
ax.set_xlabel('$epoch$')
ax.set_ylabel('$loss$')
ax.legend(frameon=False, loc = 'best')
ax.set_title('Training and validation loss', fontsize = 10)
#ax.set_xlim([0,1.2])
#ax.set_ylim([-1.1,1.1])
#ax.axis('square')
plt.savefig("loss.png", dpi=300)
plt.show()