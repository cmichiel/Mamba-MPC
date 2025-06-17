close all; clear all; clc;  

fs = 400e3;
Ts = 1/fs;             % seconds per sample
mu = 1;                     % Nonlinearity parameter (control input)
N = 10;

tspan = [0 Ts];
y0 = [0;1];                 % Initial conditions for y(1) and y(2)

u = 1;                      % Control input


Range = [-15,15];
SineData = [300, 10, 1];
Band = [0, 1000];
NumPeriod = 1;
period = 5000;


%%

[sysC, ymin, ymax, umin, umax, Mx, Nx, Mu, Nu] = PADC(); 

sys = c2d(sysC, Ts, 'zoh');
A = sys.A;
B = sys.B;
C = sys.C;
n = size(A,1);
ny = size(C,1);
nu = size(B,2);
T=1000;
Tini = 12; 

Range = [0,1];
SineData = [30, 10, 1];
Band = [0, 1];
NumPeriod = 1;
period = 1000;
% 
% u_data1 = idinput([period 1 NumPeriod],'sine',Band,Range,SineData)';
% u_data2 = idinput([period 1 NumPeriod],'sine',Band,Range,SineData)';
% u_data = [u_data1;u_data2];

u_data = 0.8*idinput([T+N+Tini, nu], 'PRBS', [0, 1], [0, 1])';

X = zeros(n, size(u_data,2));
Y = zeros(ny,size(u_data,2));

for k=1:size(u_data,2)
    Y(:,k) = C*X(:,k);
    if(k < size(u_data,2))
        X(:,k+1) = A*X(:,k) + B*u_data(:,k);
    end
end

%%

figure;
subplot(3,1,1)
plot(Y)

subplot(3,1,2)
plot(u_data(1,:))

subplot(3,1,3)
plot(u_data(2,:))
%%
prompt = "Do you want to save? y/n: ";
x = input(prompt,"s")

if string(x) == "y"
    save('Datasets/u_dataVal.mat',"u_data")
    save('Datasets/y_dataVal.mat',"Y")
end