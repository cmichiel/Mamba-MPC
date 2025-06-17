%% N-step ahead predictor
clc; close all; clear all;



fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
StopTime = 24;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
F = 0.2;
% A = [ 0.25:0.5:1.25,1.75,1.6:-0.75:-1.6,-1.75,-1.75:0.5:1.75,1.75,0,0,-1.75,-1.75,-0.25 ].*1.5;
% A = nonzeros(A)
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r, A(j)*ones(1,len)];
% end
% r = -0.25*ones(1,length(t));

r = 1*sin(2*pi*F*t)';% works 

% use this for noiseless 
load('DataSets/u_data40sinsSweptAmpRed.mat')
load('DataSets/y_data40sinsSweptAmpRed.mat')

nu = 1;
ny = 1;

%% load centers and sigmas
N = 10; %prediction horizon
Tini = 20;
n_basisNARX =60;
Basis_funcNARX = 'gaussian';
file_string = 'RBFs/'+string(Basis_funcNARX)+'/RBF_Params_NARX_Tini'+string(Tini)+'_nbasisNARX'+string(n_basisNARX)+'_N'+string(N)+'_NARXBasis_'+string(Basis_funcNARX)+'.mat';
load(file_string)
centers_sigmas = struct('centers',centersNARX,'log_sigmas',log_sigmasNARX)
data_mean = double(data_mean);
data_std = double(data_std);


%%

k_sim = length(r)-N;
Phi = []; Y = [];
y_ini = ones(Tini,1)*y_data(1)
u_ini = zeros(Tini-1,1)

disp("Simulation for VDP system SPC controller with: N="+string(N)+", Tini="+string(Tini)+" and Nbasis="+string(n_basisNARX)+" and Basis function:"+Basis_funcNARX)
%% recompute Theta 
for i = 1:length(y_data)-N;
if i == 1
y_ini = [y_ini(2:end);y_data(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y_data(i)];
u_ini = [u_ini(2:end);u_data(i-1)];
end
uf = u_data(i:i+(N-1))'; 


%Phi = [Phi rbf_before_out_layer(centers_sigmas,u_ini,uf,y_ini)];
Phi = [Phi rbf_before_out_layer_NL(centers_sigmas,u_ini,uf,y_ini,data_mean,data_std,Basis_funcNARX)];
out_vector = [];
for j = 1:(N)
out_vector = [out_vector; y_data(i+j)];
end

Y = [Y out_vector];
end
Phi_approx = pinv(weight)*Y;
Theta = Y*pinv(Phi_approx);

P    = Theta(:,1);
Gamma = Theta(:,2);

%%
[Y_sne,loss] =tsne([Phi]);
gscatter(Y_sne(:,1),Y_sne(:,2))
grid on
title('T-SNE Clustering of \Phi for NARX')

%%
%Orthogonality
LI = norm(Phi*Phi' - eye(size(Phi*Phi')), 'fro')
%%
%Smallest Singular Value
[U,S,V] = svd(Phi);
SVals = diag(S);
CondNumb = max(SVals)/min(SVals)
MinSval = min(SVals)
%% Build YALMIP SPC problem


centers = double(centers_sigmas.centers);
log_sigmas = double(centers_sigmas.log_sigmas);


Q = 100; 
R =  0.5;
Rd =  0;
S= 100;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
u_init = sdpvar(1,1);
u_ini_var = sdpvar(Tini-1, 1);
y_ini_var = sdpvar(Tini, 1);
nl_part = sdpvar(length(centers(:,1)),1);

constraints = [y == Theta*rbf_before_out_layer_NL(centers_sigmas,u_ini_var,u,y_ini_var,data_mean,data_std,Basis_funcNARX) ];


objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u(1)-u_init)'*Rd*(u(1)-u_init) + u(N)'*R*u(N);

for k = 1:N-1
    objective = objective + u(k)'*R*u(k);
    objective = objective +(u(k+1)-u(k))'*Rd*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k+1)<=15];
end

objective = objective+(y(N)-ref(N))'*S*(y(N)-ref(N));
Parameters = {u_init, ref,u_ini_var,y_ini_var};
Outputs = {u};

%options = sdpsettings('solver', 'fmincon', 'verbose', 0, 'debug', 0);
%ops = sdpsettings('solver','fmincon','sdpa.maxIteration',100);
%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective,[], Parameters, Outputs);

%% initial conditions
y(1) = 1;
y_data = y(1);
iter = 1;
x0 = [1;1];
xk = x0;
xvec1 = [xk];
u_mpc = 0;
uvec = [];
u = [];
th=[];
y_ini = ones(Tini,1)*y(1);
u_ini = zeros(Tini-1,1);
uf = zeros(N,1);
%%
figure;
plot(r(1:k_sim),'LineWidth',3);
hold on;
yplot = plot(iter,y_data,'LineWidth',3);
legend('Reference','RBF-SPC');
title ('SPC: Reference vs Closed Loop Output');
grid on
%saveas(fig,'RBF-SPC\Figures\Closed Loop Output\VDP_NL-RBF-SPC_output_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')
yplot.YDataSource = 'y_data';
yplot.XDataSource = 'iter';

for i = 1:k_sim;
i
tic;
%%% u_ini & y_ini   
if i == 1
y_ini = [y_ini(2:end);y(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y(i)];
u_ini = [u_ini(2:end);uvec(i-1)];
end


%%% mpc sol
[Uk ] = controller({u_mpc(1), r(i+1:i+N)',u_ini,y_ini});
u_mpc = Uk;
%u_mpc = (Uk+data_mean(2*Tini:2*Tini-1+N)')./data_std(2*Tini:2*Tini-1+N)';
uf = u_mpc ;
uvec = [uvec u_mpc(1)];
th=[th;toc];

%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_mpc(1))];
y(i+1) = xk(1);
y_data(i) = xk(1);
iter(i) = i;
xvec1 = [xvec1 xk];
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
saveas(fig,'RBF-SPC\Figures\Closed Loop Output\NARX-SPC\VDP_NARX-RBF_STEPREF-SPC_output_N'+string(N)+'_n_basis'+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX+'.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('SPC: Control Input');
grid on
xlabel('Iterations');
saveas(fig2,'RBF-SPC\Figures\Control Input\NARX-SPC\VDP_NARX-RBF_STEPREF-SPC_input_N'+string(N)+'_n_basis'+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX+'.png')


fig3 = figure;
plot(e,'LineWidth',3);
title ('SPC Tracking Error');
grid on;
xlabel('Iterations');
saveas(fig3,'RBF-SPC\Figures\Tracking Error\NARX-SPC\VDP_NARX-RBF_STEPREF-SPC_Error_N'+string(N)+'_n_basis'+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX+'.png')

%%

save('Mircea-Plotting/Data/u_dataNARXSPC_SINREF_N'+string(N)+'_n_basisNARX_'+string(n_basisNARX )+'_Tini'+string(Tini)+'_'+Basis_funcNARX +'.mat',"uvec")
save('Mircea-Plotting/Data/y_dataNARXSPC_SINREF_N'+string(N)+'_n_basisNARX_'+string(n_basisNARX )+'_Tini'+string(Tini)+'_'+Basis_funcNARX +'.mat',"y")