import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import Mamba, MambaConfig
import argparse
from DataProcessing import HankelMatrices, HankelMatricesStates,  HankelMatricesSequences, Sequence2Sequence, Sequence2SequenceStates, Sequence2SequenceFuture, Sequence2SequencePast
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
import wandb
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset
from numpy import linalg as LA


parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=8000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=5e-5,
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
def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

set_seed(args.seed,args.cuda)


    
class NetSequences(nn.Module):
    def __init__(self,in_dim,out_dim,L,N):
        super().__init__()
        D_model  = 96
        self.config = MambaConfig(d_model=D_model, n_layers=1, expand_factor = 1  , d_conv = 16, d_state = 16)
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
    #clf.load_state_dict(torch.load("State_dicts/Tustin_D8_dstate8_Conv4_Thomas_N10_States",map_location=torch.device('cpu'))['model_state_dict'])
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.99, total_iters=10)
    if args.cuda:
        clf = clf.cuda()
        print("Using GPU")
    Vallosses = []
    Train_loss = []
    filename = input("Enter file name: ")
    best_model_saver = BestModelSaver(filepath="State_dicts/" + filename)
    
    # Start a new wandb run to track this script.   
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="MambaDeePC",
        project = "Inverted Cart Pendulum",
        name= filename,
        # Set the wandb project where this run will be logged.
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 1e-3,
            "architecture": "Mamba",
            "dataset": "Large Mix",
            "epochs": 1000,
        },
    )
    #weight = torch.ones((batch_size,SequenceLength,Output_dim))
    for e in range(args.epochs):
        clf.train()
        running_loss = 0.
        last_loss = 0
        for idx,data in enumerate(train):        
            X,Y = data
            z = clf(X)
            loss =  F.mse_loss(z.squeeze(0),Y)   #weighted_mse_loss(z.squeeze(0),Y, weight)
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
            print('Epoch %d | Lossp: %.5f | Validation Loss: %.5f' % (e, running_loss/batch_size, Vloss/batch_size))
            best_model_saver(current_valid_loss=Vloss/batch_size,
                 model=clf,
                 epoch=e, # Current epoch index (0-based)
                 optimizer=opt,
                 criterion=F.mse_loss)
            wandb.log({
                "epoch": e,
                "train_loss": running_loss/batch_size,
                "val_loss": Vloss/batch_size,
                "learning_rate": opt.param_groups[0]['lr'],
                "A log": np.exp(clf.mamba.layers[0].mixer.A_log.cpu().detach().numpy()),
                "A Conditioning Number": LA.cond(np.exp(clf.mamba.layers[0].mixer.A_log.cpu().detach().numpy())),
                "Conditioning number of SelectionProjectionWeight": LA.cond(clf.mamba.layers[0].mixer.x_proj.weight.cpu().detach().numpy()),
                "Conditioning number of DeltaTProjectionWeight": LA.cond(clf.mamba.layers[0].mixer.dt_proj.weight.cpu().detach().numpy()),
                "Conditioning Number of Expansion Layer Weight": LA.cond(clf.mamba.layers[0].mixer.in_proj.weight.cpu().detach().numpy()),
                "Conditioning Number of Reduction Layer Weight": LA.cond(clf.mamba.layers[0].mixer.out_proj.weight.cpu().detach().numpy()),
                })
        Train_loss.append(running_loss/batch_size)
    wandb.save(filepath="State_dicts/" + filename)
    wandb.finish()        
    return Vallosses, Train_loss





#Load in Test data
X = np.load("X_train.npy")
Y = np.load("Y_train.npy")

X = torch.FloatTensor(X)
Y = torch.FloatTensor(Y)

print("u_train shape:"+str(X.shape))
print("y_train shape:"+str(Y.shape))

X_test = np.load("X_val.npy")
Y_test = np.load("Y_val.npy")

X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)

print("u_test shape:"+str(X_test.shape))
print("y_test shape:"+str(Y_test.shape))




if args.cuda:
    X = X.cuda()
    Y = Y.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()
Dataset = TensorDataset(X, Y)
Dataset_test = TensorDataset(X_test, Y_test)
print("Dataset Shape:"+str(Dataset.tensors[0].shape))
print("Test Dataset Shape:"+str(Dataset_test.tensors[0].shape))
train_dataloader = DataLoader(Dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(Dataset_test, batch_size=32, shuffle=True)



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