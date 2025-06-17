import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import Mamba, MambaConfig
import argparse
from DataProcessing import HankelMatrices
import scipy.io
import scipy.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, 'Utilities/')
import os
 
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec





parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=40000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-3,
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

class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=4, n_layers=1, expand_factor = 2  , d_conv = 1,d_state = 2)
        self.lin1 = nn.Linear(in_dim-10,4)
        self.mamba = Mamba(self.config)
        self.lin2 = nn.Linear(4+10,out_dim)

    def forward(self,data_ini,data_f):
        x = self.lin1(data_ini)
        x = self.mamba(x)
        x = torch.cat((x,data_f),2)
        x = self.lin2(x)
        return x

def PredictWithData(trainX, trainy, testX,Tini):
    N_pred = 1 #Prediction Horizon
    clf = Net(len(trainX[1]), len(trainy[1]))
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr, weight_decay=args.wd)
    xt = trainX.float().unsqueeze(0)
    xv = testX.unsqueeze(0)
    yt = trainy.float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    data_ini = xt[:,:, 0 : (2 * Tini - 1)]
    data_f   = xt[:,:, (2 * Tini - 1) :]

    data_ini_val = xv[:,:, 0 : (2 * Tini - 1)]
    data_f_val   = xv[:,:, (2 * Tini - 1) :]
    Vallosses = []
    Train_loss = []
    for e in range(args.epochs):
        clf.train()
        z = clf(data_ini,data_f)
        loss = F.mse_loss(z.squeeze(0),yt)
        with torch.no_grad():  # turn off backpropagation
            y_eval = clf .forward(data_ini_val, data_f_val) # are features from our test set
            Val_loss = F.mse_loss(y_eval, testy.cuda())  # Find the loss or error
            Vallosses.append(Val_loss.cpu().detach().numpy())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f | Validation Loss: %.4f' % (e, loss.item(), Val_loss))
        Train_loss.append(loss.item())

    clf.eval()
    data_ini_v = xv[:,:, 0 : (2 * Tini - 1)]
    data_f_v   = xv[:,:, (2 * Tini - 1) :]
    mat = clf(data_ini_v,data_f_v )
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().squeeze(0)
    print("yhat Shape:"+str(yhat.shape))
    ##Save Model
    filename = input("Enter file name: ")
    torch.save(clf.state_dict(),"State_dicts/" + filename)
    return yhat, Vallosses, Train_loss



# Load multisine data from matlab
mat = scipy.io.loadmat("DataSets/u_dataThomas.mat")
u_data = mat["u_data"]

mat = scipy.io.loadmat("DataSets/y_dataThomas.mat")
y_data = mat["y_data"]

# convert data to tensors
u_data = np.array(u_data)
y_data = np.array(y_data)

u_data = torch.FloatTensor(u_data).transpose(0,1)
#u_data = u_data[0,:1000].unsqueeze(0).transpose(0,1)
y_data = torch.FloatTensor(y_data).transpose(0,1)
#y_data = y_data[0,:1000].unsqueeze(0).transpose(0,1)
#y_data = y_data[0,:].unsqueeze(1)
#torch.reshape(y_data,(1000, 1))
input_dim = list(u_data.shape)[1]
output_dim = list(y_data.shape)[1]

# Hyper parameters
Tini = 3
T = y_data.shape[0]
N = 10
DataShuffle = False
norm_data = True


u_data_train = u_data[0:round(0.8*T)]
y_data_train = y_data[0:round(0.8*T)]
U_ini, Y_ini,U_0_Nm1, Y_1_N = HankelMatrices(u_data_train,y_data_train,Tini,N,round(0.8*T) , input_dim, output_dim )

trainX= torch.cat((U_ini, Y_ini, U_0_Nm1), 1)
trainy = Y_1_N


u_data_test = u_data[round(0.8*T):T]
y_data_test = y_data[round(0.8*T):T]
U_ini, Y_ini,U_0_Nm1, Y_1_N = HankelMatrices(u_data_test,y_data_test,Tini,N,round(0.2*T) , input_dim, output_dim )

testX = torch.cat((U_ini, Y_ini, U_0_Nm1), 1)
testy = Y_1_N
print("TrainX Shape:"+str(trainX.shape))
print("TrainY Shape:"+str(trainy.shape))
print("TestX Shape:"+str(testX.shape))

predictions, Val_loss, Train_loss = PredictWithData(trainX, trainy , testX,Tini)



Error = np.mean(np.abs(predictions  - testy.numpy()),1)

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