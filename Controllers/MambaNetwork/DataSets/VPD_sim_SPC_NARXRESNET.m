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
% r = r;
% r = 1.5*ones(1,length(t));

r = 1*sin(2*pi*F*t)';% works 



nu = 1;
ny = 1;

%% load centers and sigmas
N = 10;
Tini =5;
n_basisKoopman =40;
n_basisNARX =40;
Basis_funcKoopman = 'gaussian';
Basis_funcNARX = 'gaussian';
file_string = 'RBFs/'+string(Basis_funcNARX)+'/RBF_Params_NARXResnet_Tini'+string(Tini)+'_nbasisNARX'+string(n_basisNARX)+'_N'+string(N)+'_NARXBasis_'+string(Basis_funcNARX)+'.mat';
%file_string = 'RBFs/'+string(Basis_func)+'/RBF_Params_MIXED_Tini'+string(Tini)+'_nbasis'+string(n_basis)+'_N'+string(N)+'_'+string(Basis_func)+'.mat';
load(file_string)
data_mean = double(data_mean);
data_std = double(data_std);
% centersKoopman = centersL;
% log_sigmasKoopman = log_sigmasL;
% centersFullBasis = centersNL;
% log_sigmasFullBasis = log_sigmasNL;
centers_sigmasNARX = struct('centersNARX',centersNARX,'log_sigmasNARX',log_sigmasNARX)
data_mean_1 = double(data_mean(1:2*Tini-1));
data_mean_2 = double(data_mean(2*Tini:length(data_mean)));
data_std_1 = double(data_std(1:2*Tini-1));
data_std_2 = double(data_std(2*Tini:length(data_std)));

%% recompute Theta 
load('DataSets/u_data40sinsSweptAmpRed.mat')
load('DataSets/y_data40sinsSweptAmpRed.mat')
y_ini = ones(Tini,1)*y_data(1)
u_ini = zeros(Tini-1,1)
Phi = []; Y = [];

for i = 1:length(y_data)-N
if i == 1
y_ini = [y_ini(2:end);y_data(i)];
u_ini = u_ini;
end

if i >= 2 
y_ini = [y_ini(2:end);y_data(i)];
u_ini = [u_ini(2:end);u_data(i-1)];
end
uf = u_data(i:i+(N-1))'; 
uf1 = (uf-data_mean(2*Tini:2*Tini-1+N)')./data_std(2*Tini:2*Tini-1+N)';

ResPart = ([u_ini;y_ini;uf]-data_mean')./data_std';
Phi_part = [rbf_before_out_layer_NARX(centers_sigmasNARX,u_ini,uf,y_ini,data_mean,data_std,Basis_funcNARX); ResPart];

Phi = [Phi Phi_part];
out_vector = [];
for j = 1:(N)
out_vector = [out_vector; y_data(i+j)];
end

Y = [Y out_vector];
end

Theta = Y*pinv(Phi);
%%
figTSNE = figure;
[Y_sne,loss] =tsne([Phi]);
gscatter(Y_sne(:,1),Y_sne(:,2))
grid on
title('T-SNE Clustering of \Phi for NARXResnet')
saveas(figTSNE,'T-SNE\T-SNE_NARXResnet_Phi_N'+string(N)+'_n_basisNARX'+string(n_basisNARX )+'_Tini'+string(Tini)+'_NARX'+Basis_funcNARX+'.png')

%%
%Orthogonality
LI = norm(Phi*Phi' - eye(size(Phi*Phi')), 'fro')
%%
%Smallest Singular Value
[U,S,V] = svd(Phi);
SVals = diag(S);
CondNumb = max(SVals)/min(SVals)
MinSval = min(SVals)
%%
figure;
plot(mean(weight,1),'LineWidth',2)
title("Mean weighting of RBF")
grid on;
xlabel("Radial basis Function")
%%
k_sim = length(r)-N;

%disp("Simulation for VDP system SPC controller with: N="+string(N)+", Tini="+string(Tini)+" and Nbasis="+string(n_basis)+" and Basis function:"+Basis_func)

%% Build YALMIP SPC problem


centers = double(centers_sigmasNARX.centersNARX);
log_sigmas = double(centers_sigmasNARX.log_sigmasNARX);

Q = 100; 
R =  0.5;
Rd =  0;
S = 10;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
u_init = sdpvar(1,1);
u_ini_var = sdpvar(Tini-1, 1);
y_ini_var = sdpvar(Tini, 1);
% nl_part = sdpvar(length(centersNARX(:,1)),1);

constraints = [y == Theta*[rbf_before_out_layer_NARX(centers_sigmasNARX,u_ini_var,u,y_ini_var,data_mean,data_std,Basis_funcNARX); ([u_ini_var; y_ini_var;u]-data_mean')./data_std';  ]];

objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u(1)-u_init)'*Rd*(u(1)-u_init) + u(N)'*R*u(N);

for k = 1:N-1
    objective = objective + u(k)'*R*u(k);
    objective = objective +(u(k+1)-u(k))'*Rd*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k+1)<=15];
end

objective = objective+(y(1:N)-ref(1:N))'*S*(y(1:N)-ref(1:N));

Parameters = {u_init, ref,u_ini_var,y_ini_var};
Outputs = {u,y};

%options = sdpsettings('solver', 'ipopt', 'verbose', 0, 'debug', 0);
%ops = sdpsettings('solver','fmincon','sdpa.maxIteration',100);
%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective,[] , Parameters, Outputs);

%% initial conditions
y(1) = 1;
y_data = y(1);
iter = 1;
x0 = [1;1]
xk = x0;
xvec1 = [xk];
u_mpc = 0;
uvec = [0];
u = [];
th=[];
y_ini = ones(Tini,1)*y(1)
u_ini = zeros(Tini-1,1)
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
[UYK ] = controller({u_mpc(1), r(i+1:i+N)',u_ini,y_ini});
UYK = cell2mat(UYK);
Uk = UYK(:,1);
Yk(:,i) = UYK(:,2);
u_mpc = Uk(1);
uvec = [uvec Uk(1)];
th=[th;toc];

%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+Uk(1))];
y(i+1) = xk(1);
xvec1 = [xvec1 xk];
y_data(i+1) = xk(1);
iter(i+1) = i+1;
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
saveas(fig,'RBF-SPC\Figures\Closed Loop Output\NARXResnet-SPC\VDP_NARXRESNET-RBF_SINREF-SPC_output_N'+string(N)+'_n_basis'+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX+'.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('SPC: Control Input');
grid on
xlabel('Iterations');
saveas(fig2,'RBF-SPC\Figures\Control Input\NARXResnet-SPC\VDP_NARXRESNET-RBF_SINREF-SPC_input_N'+string(N)+'_n_basis'+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX+'.png')


fig3 = figure;
plot(e','LineWidth',3);
title ('SPC Tracking Error');
grid on;
xlabel('Iterations');
saveas(fig3,'RBF-SPC\Figures\Tracking Error\NARXResnet-SPC\VDP_NARXRESNET-RBF_SINREF-SPC_Error_N'+string(N)+'_n_basis'+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX+'.png')

%%

save('Mircea-Plotting/Data/u_dataNARXResnetSPC_SINREF_N'+string(N)+'_n_basisNARX '+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX +'.mat',"uvec")
save('Mircea-Plotting/Data/y_dataNARXResnetSPC_SINREF_N'+string(N)+'_n_basisNARX '+string(n_basisNARX)+'_Tini'+string(Tini)+'_'+Basis_funcNARX +'.mat',"y")