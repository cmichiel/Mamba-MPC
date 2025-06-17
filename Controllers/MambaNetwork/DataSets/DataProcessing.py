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
        if i < 100:
            print(i)
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
            
    return U_ini, Y_ini,U_0_Nm1, Y_1_N


    