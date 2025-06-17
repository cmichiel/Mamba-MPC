import torch
import torch.nn as nn
import numpy as np


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

def Sequence2Sequence(U_ini, Y_ini,U_0_Nm1, Y_1_N,N,Tini):
    Batch, SequenceLength = Y_1_N.shape
    ny = SequenceLength/N
    nu = U_ini.shape[1]/(Tini-1)
    SequenceSet = torch.tensor(())

    for j in range(Batch):
        for i in range(int(nu)):
            uN_vec = U_0_Nm1[j,N*i:N+N*i].unsqueeze(1)
            if i>0:
                uN_vec = torch.cat((uN_vec, U_0_Nm1[j,N*i:N+N*i].unsqueeze(1)),1)
        Sample = uN_vec
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
    Batch, SequenceLength = U_0_Nm1.shape
    _ , PredictionHorizon = Y_N.shape 
    PredictionHorizon = int(PredictionHorizon/2)
    Y_1N = Y_N[:,:PredictionHorizon].unsqueeze(2)
    Y_2N = Y_N[:,PredictionHorizon:].unsqueeze(2)
    Y_N = torch.cat((Y_1N, Y_2N),2)
    SequenceSet = torch.tensor(())

    for j in range(Batch):
        Sample = U_0_Nm1[j,:].unsqueeze(1)
        X_vec = X0[j,:].unsqueeze(0).repeat(SequenceLength,1)
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

    