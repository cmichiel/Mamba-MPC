import torch
import torch.nn as nn
import numpy as np
import os

class BestModelSaver:
    """
    Class to save the best model during training, overwriting previous best.
    Monitors validation loss and saves the model if it improves.
    """
    def __init__(self, filepath='best_model.pth'):
        """
        Initializes the BestModelSaver.

        Args:
            filepath (str, optional): Path to save the best model checkpoint.
                                      Defaults to 'best_model.pth'.
        """
        self.filepath = filepath
        # Ensure the directory for the filepath exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.best_valid_loss = float('inf')
        print(f"Initialized BestModelSaver. Will save best model to: {self.filepath}")

    def __call__(self, current_valid_loss, model, epoch=None, optimizer=None, criterion=None):
        """
        Checks if the current model is the best and saves it if it is.

        Args:
            current_valid_loss (float): The validation loss of the current epoch.
            model (torch.nn.Module): The PyTorch model to save.
            epoch (int, optional): The current epoch number.
            optimizer (torch.optim.Optimizer, optional): The optimizer instance.
            criterion (torch.nn.Module, optional): The loss function instance.
        """
        if current_valid_loss < self.best_valid_loss:
            previous_best = self.best_valid_loss
            self.best_valid_loss = current_valid_loss
            print(f"Validation loss improved from {previous_best:.4f} to {self.best_valid_loss:.4f}.")
            #print(f"Saving new best model to {self.filepath}...")

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_validation_loss': self.best_valid_loss,
            }
            if epoch is not None:
                checkpoint['epoch'] = epoch + 1 # Typically 1-indexed for human readability
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if criterion is not None and hasattr(criterion, 'state_dict'):
                # Some loss functions might not have a state_dict
                try:
                    checkpoint['criterion_state_dict'] = criterion.state_dict()
                except AttributeError:
                    print("Warning: Criterion does not have a state_dict method.")
            
            torch.save(checkpoint, self.filepath)
            
            #print(f"Successfully saved new best model to {self.filepath}")

def  HankelMatrices(u_data,y_data,T_ini,N, T,  input_dim, output_dim ):
    U_ini = torch.transpose(u_data[0 : T_ini - 1,0].unsqueeze(1), 0, 1)
    U_0_Nm1 = torch.transpose(u_data[T_ini - 1 : T_ini + N - 1,0].unsqueeze(1), 0, 1)
    if input_dim > 1:
        for j in range(input_dim-1):
            U_ini = torch.cat((U_ini, torch.transpose(u_data[0 : T_ini - 1,j].unsqueeze(1), 0, 1)),1)
            U_0_Nm1 = torch.cat((U_0_Nm1, torch.transpose(u_data[T_ini - 1 : T_ini + N - 1,1].unsqueeze(1), 0, 1)),1)
    # print(f"U_ini = {(U_ini).shape}")           
    # print(f"U_0_Nm1 = {(U_0_Nm1).shape}")        

    Y_ini = torch.transpose(y_data[1 : T_ini + 1,0].unsqueeze(1), 0, 1)
    Y_1_N = torch.transpose(y_data[T_ini + 1 : T_ini + 1 + N,0].unsqueeze(1), 0, 1)
    if output_dim > 1:
        for p in range(output_dim-1):
            Y_ini = torch.cat((Y_ini, torch.transpose(y_data[1 : T_ini + 1,p].unsqueeze(1), 0, 1)),1)
            Y_1_N = torch.cat((Y_1_N, torch.transpose(y_data[T_ini + 1 : T_ini + 1 + N,p].unsqueeze(1), 0, 1)),1)

    # print(f"Y_ini = {(Y_ini).shape}")           
    # print(f"Y_1_N = {(Y_1_N).shape}")       

    for i in range(T - T_ini - 1 - N):
        U_ini_part =  torch.transpose(u_data[i + 1 : T_ini + i,0].unsqueeze(1), 0, 1)
        U_0_Nm1_part = torch.transpose(u_data[T_ini + i : T_ini + i + N,0].unsqueeze(1), 0, 1)
        if input_dim > 1:
            for j in range(input_dim-1):
                U_ini_part =  torch.cat((U_ini_part, torch.transpose(u_data[i + 1 : T_ini + i,j].unsqueeze(1), 0, 1)), 1)
                U_0_Nm1_part = torch.cat((U_0_Nm1_part ,torch.transpose(u_data[T_ini + i : T_ini + i + N,j].unsqueeze(1), 0, 1)),1)
        
        Y_ini_part =  torch.transpose(y_data[i + 2 : T_ini + 2 + i,0].unsqueeze(1), 0, 1)
        Y_1_N_part = torch.transpose(y_data[T_ini + 2 + i : T_ini + 2 + i + N,0].unsqueeze(1), 0, 1)
        if output_dim > 1:
            for p in range(output_dim-1):
                Y_ini_part =  torch.cat((Y_ini_part, torch.transpose(y_data[i + 2 : T_ini + 2 + i,p].unsqueeze(1), 0, 1)), 1)
                Y_1_N_part = torch.cat((Y_1_N_part ,torch.transpose(y_data[T_ini + 2 + i : T_ini + 2 + i + N,p].unsqueeze(1), 0, 1)),1)
    #             print(f"Y_ini_part = {(Y_ini_part).shape}")
    #             print(f"Y_1_N_part = {(Y_1_N_part).shape}")
                                    
        U_ini =  torch.cat((U_ini, U_ini_part), 0)        
        U_0_Nm1 = torch.cat((U_0_Nm1,U_0_Nm1_part ))
    #     print(f"U_ini = {(U_ini).shape}")
    #     print(f"U_0_Nm1 = {(U_0_Nm1).shape}")
        Y_ini = torch.cat((Y_ini, Y_ini_part), 0)
        Y_1_N = torch.cat((Y_1_N, Y_1_N_part), 0)    
    #     print(f"Y_ini = {(Y_ini).shape}")
    #     print(f"Y_1_N = {(Y_1_N).shape}") 
    print("Hankel Matrices Generated")
    return U_ini, Y_ini,U_0_Nm1, Y_1_N



def  HankelMatricesStates(u_data,x_data, N):
    #Find the length of dataset
    T= x_data.shape[0]

    #Initialize datasets, X0 should start at 0 and yN at time instance 1
    U_0_Nm1 = u_data[0:N,:].unsqueeze(0)    
    X0 = x_data[0,:].unsqueeze(0).unsqueeze(0)
    Y_1_N = x_data[1:N+1,:].unsqueeze(0)
    
    #Loop over dataset and concatenate
    for i in range(T - N-1):
        U_0_Nm1_part = u_data[i+1 : i + 1+ N,:].unsqueeze(0)    
        X0_part = x_data[i+1,:].unsqueeze(0).unsqueeze(0)
        Y_1_N_part = x_data[i+2:i+N+2,:].unsqueeze(0)

        U_0_Nm1 = torch.cat((U_0_Nm1,U_0_Nm1_part ),0)
        X0 = torch.cat((X0, X0_part),0)
        Y_1_N = torch.cat((Y_1_N, Y_1_N_part), 0)    
    print("Hankel Matrices Generated")
    return U_0_Nm1, Y_1_N, X0

def  HankelMatricesSequences(U_ini, Y_ini,U_0_Nm1, Y_1_N, L):

    SetLength, _  = U_ini.shape

    SequenceDataSet = torch.tensor(())
    OutputSequence = torch.tensor(())

    for k in range(L, SetLength):
        
        SequenceX = torch.tensor(())
        SequenceY = torch.tensor(())
        for i in range(L):   
            TokenX = torch.cat((U_ini[k-i,:], Y_ini[k-i,:], U_0_Nm1[k-i,:])).unsqueeze(0).unsqueeze(0)
            SequenceY = torch.cat((SequenceY,Y_1_N[k-i,:].unsqueeze(0).unsqueeze(0)),1)
            SequenceX = torch.cat((SequenceX,TokenX),1)
        if k==L:
            SequenceDataSetX = torch.flip(SequenceX,dims=[1])
            SequenceDataSetY = torch.flip(SequenceY,dims=[1])
        else:
            SequenceDataSetX = torch.cat((SequenceDataSetX, torch.flip(SequenceX, dims=[1])),0)
            SequenceDataSetY = torch.cat((SequenceDataSetY, torch.flip(SequenceY,dims=[1])),0)

    Y_1_N = Y_1_N[L:SetLength,:].unsqueeze(1)

    print("Sequences Produced")
    return SequenceDataSetX,SequenceDataSetY

def Sequence2Sequence(U_ini, Y_ini,U_0_Nm1, Y_1_N):
    Batch, SequenceLength = U_0_Nm1.shape
    SequenceSet = torch.tensor(())

    for j in range(Batch):
        Sample = U_0_Nm1[j,:].unsqueeze(1)
        U_vec = U_ini[j,:].unsqueeze(0).repeat(SequenceLength,1)
        Y_vec = Y_ini[j,:].unsqueeze(0).repeat(SequenceLength,1)
        Sample = torch.cat((Sample,U_vec),1)
        Sample = torch.cat((Sample,Y_vec),1).unsqueeze(0)
        SequenceSet = torch.cat((SequenceSet,Sample),0)

    Y_1_N = Y_1_N.unsqueeze(2)
    return SequenceSet, Y_1_N

def Sequence2SequencePast(U_ini, Y_ini,U_0_Nm1, Y_1_N):
    Batch, SequenceLength = U_0_Nm1.shape
    SequenceSet = torch.tensor(())

    for j in range(SequenceLength, Batch):
        Sample = U_0_Nm1[j,:].unsqueeze(1)
    
        U_vec = torch.tensor(())
        Y_vec = torch.tensor(())
        for i in range(SequenceLength):                        
            U_vec = torch.cat((U_vec, U_ini[j-i,:].unsqueeze(0)),0)
            Y_vec = torch.cat((Y_vec, Y_ini[j-i,:].unsqueeze(0)),0)
        Sample = torch.cat((Sample,U_vec),1)
        Sample = torch.cat((Sample,Y_vec),1).unsqueeze(0)
        SequenceSet = torch.cat((SequenceSet,Sample),0)

    Y_1_N = Y_1_N[SequenceLength:Batch].unsqueeze(2)
    return SequenceSet, Y_1_N

def Sequence2SequenceStates(X0,U_0_Nm1,Y_N):
    Batch, SequenceLength,_ = U_0_Nm1.shape
    _ , PredictionHorizon,_ = Y_N.shape 

    SequenceSet = torch.tensor(())

    for j in range(Batch):
        Sample = U_0_Nm1[j,:,:]
        X_vec = X0[j,:,:].repeat(SequenceLength,1)
        Sample = torch.cat((Sample,X_vec),1).unsqueeze(0)
        SequenceSet = torch.cat((SequenceSet,Sample),0)
    return SequenceSet,Y_N

def Sequence2SequenceFuture(U_ini, Y_ini,U_0_Nm1,Y_1_N):
    Batch, SequenceLength = U_0_Nm1.shape
    SequenceSet = torch.tensor(())

    for i in range(Batch-SequenceLength):
        SequenceSample = torch.tensor(())
        for j in range(SequenceLength):
            SequenceBlock = torch.tensor(())
            SequenceBlock = torch.cat((U_ini[i+j].unsqueeze(0), Y_ini[i+j].unsqueeze(0), U_0_Nm1[i,j].unsqueeze(0).unsqueeze(1)), 1)
            SequenceSample = torch.cat((SequenceSample,SequenceBlock), 0)
        SequenceSet = torch.cat((SequenceSet, SequenceSample.unsqueeze(0)), 0)
    Y_1_N = Y_1_N[:Batch-SequenceLength,:].unsqueeze(2)
    return SequenceSet, Y_1_N

