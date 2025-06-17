%% N-step ahead predictor
clc; close all; clear all;
%%


fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
StopTime = 24;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
F = 0.2;
N = 10;
% A = [0.25:0.5:1.25,1.75,1.6:-0.75:-1.6,-1.75,-1.75:0.5:1.75,1.75,0,0,-1.75,-1.75,-0.25 ].*1.5;
% A = nonzeros(A)
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r, A(j)*ones(1,len)];
% end
% r = r;
% r = 0.5*ones(1,length(t));
% 
r = 2*sin(2*pi*F*t)';% works 
k_sim = length(r)-N;
% net = importNetworkFromPyTorch('traced_mamba.pt', PyTorchInputSizes=[1,1,15])
% InputSize = [1 1 15];
% inputLayer = imageInputLayer(InputSize,Normalization="none");
% net = addInputLayer(net,inputLayer,Initialize=true);
% X = dlarray(rand(InputSize),"CBT");
% net = initialize(net,X);
%%

%% Build YALMIP SPC problem


Q = 100; 
R =  0.5;
Rd =  0;
S = 100;
N = 10;
Tini = 3;


Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);
load 'RBF_Params_Koopman_Tini3_nbasisKoopman40_N10_KoopmanBasis_gaussian.mat'
data_mean_1 = double(data_mean(1:2*Tini-1));
data_mean_2 = double(data_mean(2*Tini:length(data_mean)));
data_std_1 = double(data_std(1:2*Tini-1));
data_std_2 = double(data_std(2*Tini:length(data_std)));

%%
% Bounds on u
lb = -15 * ones(N, 1); % Lower bounds
ub = 15 * ones(N, 1); % Upper bounds

Q = 100; 
R =  0.5;
Rd = 0;
S = 100;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);


%% initial conditions
y(1) = 1;
y_data = y(1);
iter = 1;
x0 = [1;1]
xk = x0;
xvec1 = [xk];
u_mpc = 0;
uvec = zeros(1,10);
u_data = 0;
th=[];
y_ini = ones(Tini,1)*y(1)
u_ini = zeros(Tini-1,1)


%%
load 'NMPC.mat'

uN = reshape(uN, 230,10);
yN = []
for i = 1:length(u_ini_list)
    %inputs = [u_ini_list(i,:), y_ini_list(i,:), uN(i,:) ];
    %inputs = reshape(inputs,1,1,length(inputs));
    [y_output] = RBF(uN(i,:), u_ini_list(i,:)',y_ini_list(i,:)', weight, centersKoopman, log_sigmasKoopman, data_mean_1,data_std_1,data_mean_2,data_std_2);
    yN = [yN, y_output(1)];
end
figure;
plot(yN);
hold on
plot(r(1:length(yN)))
grid on
%%

%%
figure;
subplot(2,1,1);
plot(r(1:k_sim),'LineWidth',3);
hold on;
yplot = plot(iter,y_data,'LineWidth',3);
legend('Reference','RBF-SPC');
title ('SPC: Reference vs Closed Loop Output');
grid on
subplot(2,1,2);
uplot = plot(iter,u_data,'LineWidth',3)
ylabel("Control Input")
grid on
%saveas(fig,'RBF-SPC\Figures\Closed Loop Output\VDP_NL-RBF-SPC_output_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')
yplot.YDataSource = 'y_data';
yplot.XDataSource = 'iter';
uplot.YDataSource = 'u_data';
uplot.XDataSource = 'iter';
%%
for i = 1:k_sim;
i
tic;
%%% u_ini & y_ini   
if i == 1
y_ini = [y_ini(2:end);y_data(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y_data(i)];
u_ini = [u_ini(2:end);uvec(i-1)];
end

% Objective function (cost function)
objective = @(u) cost_functionRBF(u, r(i+1:i+N), u_ini, y_ini, N, Ts,Omega,R,Rd,S, uvec(i+N-1),  weight, centersKoopman, log_sigmasKoopman, data_mean_1,data_std_1,data_mean_2,data_std_2)

options = optimoptions('fmincon');
u_optimal = fmincon(objective, uN(i,:), [], [], [], [], lb, ub, [], options);
uvec = [uvec, u_optimal(1)];


%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_optimal(1))];
y(i+1) = xk(1);
xvec1 = [xvec1 xk];
y_data(i+1) = xk(1);
iter(i+1) = i+1;
u_data(i+1) = u_optimal(1);
refreshdata
drawnow
end

e = abs(y-r(1:length(y))');
%% Plots
fig = figure;
plot(r(1:length(y)),'LineWidth',3);
hold on;
plot(y,'LineWidth',3);
legend('Reference','RBF-SPC');
title ('SPC: Reference vs Closed Loop Output');
grid on
saveas(fig,'RBF-SPC\Figures\Closed Loop Output\KoopmanResnet-SPC\VDP_KoopmanResnet-RBF_STEPREF-SPC_output_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_Tini'+string(Tini)+'_'+Basis_funcKoopman+'.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('SPC: Control Input');
grid on
xlabel('Iterations');
saveas(fig2,'RBF-SPC\Figures\Control Input\KoopmanResnet-SPC\VDP_KoopmanResnet-RBF_STEPREF-SPC_input_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_Tini'+string(Tini)+'_'+Basis_funcKoopman+'.png')


fig3 = figure;
plot(e','LineWidth',3);
title ('SPC Tracking Error');
grid on;
xlabel('Iterations');
saveas(fig3,'RBF-SPC\Figures\Tracking Error\KoopmanResnet-SPC\VDP_KoopmanResnet-RBF_STEPREF-SPC_Error_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_Tini'+string(Tini)+'_'+Basis_funcKoopman+'.png')
