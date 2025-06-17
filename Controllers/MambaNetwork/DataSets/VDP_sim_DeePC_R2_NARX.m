%% N-step ahead predictor
clc; close all; clear all;



fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
StopTime = 24;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
F = 0.2;                       % Sine wave frequency (hertz)
% A = [0:0.1:0.5, 0.5:-0.1:-0.5,-0.5:0.1:0];
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r; A(j)*ones(len,1)];
% end  
% StopTime = 24;                % seconds
% t = (0:Ts:StopTime-Ts)';        % seconds
% F = 0.1;
% A = [ 0:0.75:1.5,2,1.5:-0.75:-1.5,-2,-2:0.5:2,2,0,0,-2,-2 ];
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r, A(j)*ones(1,len)];
% end

Ir = 150;
r = 1*sin(2*pi*F*t)'; %Working


load('DataSets/u_dataStepRef.mat')
load('DataSets/y_dataStepRef.mat')

nu = size(u_data,1);
ny = size(y_data,1);
%% load centers and sigmas
N = 10; %prediction horizon
Tini =5;
n_basis =40;
Basis_func = 'gaussian';
file_string = 'RBFs/'+string(Basis_func)+'/RBF_Params_NL_Tini'+string(Tini)+'_nbasis'+string(n_basis)+'_N'+string(N)+'_'+string(Basis_func)+'.mat';
load(file_string)
centers_sigmas = struct('centers',centers,'log_sigmas',log_sigmas)
data_mean = double(data_mean);
data_std = double(data_std);
%%
T = length(u_data(1,:))-N-Tini;
k_sim = length(r)-N;
Phi = []; Y = [];

y_ini = ones(Tini,1)*y_data(1);
u_ini = zeros(Tini-1,1);

disp("Simulation for VDP system $\Phi$ DeePC-R2 controller with: N="+string(N)+", Tini="+string(Tini)+" and Nbasis="+string(n_basis)+" and Basis function:"+Basis_func)
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
Phi = [Phi rbf_before_out_layer_NL(centers_sigmas,u_ini,uf,y_ini,data_mean,data_std,Basis_func)];
out_vector = [];
for j = 1:(N)
out_vector = [out_vector; y_data(i+j)];
end

Y = [Y out_vector];
end

Theta = Y*pinv(Phi);

P    = Theta(:,1);
Gamma = Theta(:,2);
[~,~,V] = svd([Phi;Y]);
PHIY_reduced = [Phi;Y]*V(:,1:Ir);
PHI_reduced = PHIY_reduced(1:40,:);
Y_reduced = PHIY_reduced(41:50,:);
%% Build YALMIP SPC problem

disp("Constructing the YALMIP DeePC-R2 problem")
centers = double(centers_sigmas.centers);
log_sigmas = double(centers_sigmas.log_sigmas);

Q = 100; 
R =  0.5;
Rd =  0;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
u_init = sdpvar(1,1);
u_ini_var = sdpvar(Tini-1, 1);
y_ini_var = sdpvar(Tini, 1);
gvar = sdpvar(Ir,1);
nl_part = sdpvar(length(centers(:,1)),1);


%constraints = [constraints,  -15<=u(1)<=15];
objective = 0;
lambda = 1e-4;
objective = objective + lambda*norm(gvar,2)^2;
objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u(1)-u_init)'*Rd*(u(1)-u_init) + u(N)'*R*u(N);

Phi_approx = pinv(weight)*Y;
constraints = [PHI_reduced *gvar == 0];
constraints = [constraints,  Theta*rbf_before_out_layer_NL(centers_sigmas,u_ini_var,u,y_ini_var,data_mean,data_std,Basis_func)+Y_reduced*gvar==y];
%objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u-u_init)'*Psi*(u-u_init);
%objective = objective +  (y(1)-ref(1))'*Q*(y(1)-ref(1))+(u(1)-u_init )'*R*(u(1)-u_init );

for k = 1:N-1
    objective = objective + u(k)'*R*u(k);
    objective = objective +(u(k+1)-u(k))'*Rd*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k)<=15];
end
constraints = [constraints,  -15<=u(N)<=15];

Parameters = {u_init, ref,u_ini_var,y_ini_var};
Outputs = {u,y};

%options = sdpsettings('solver', 'fmincon', 'verbose', 0, 'debug', 0);

options = sdpsettings('solver', 'fmincon', 'verbose', 0, 'debug', 0);
controller = optimizer(constraints, objective,options, Parameters, Outputs);
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


% clear nl_part;
% nl_part = [];
% for im = 1:length(centers(:,1))
% input_data = ([u_ini;y_ini;uf]-double(dataset_mean'))./double(dataset_std');
% out = input_data - centers(im,:)';  
% out = sqrtm(sum(out.^2))./ exp(log_sigmas(im));
% out = exp(-1*out^2);
% nl_part= [nl_part;out];
% end


%%% mpc sol
[UYK ] = controller({u_mpc(1), r(i+1:i+N)',u_ini,y_ini});
UYK = cell2mat(UYK);
Uk = UYK(:,1);
Yk(:,i) = UYK(:,2);
uf = Uk ;
uvec = [uvec u_mpc(1)];
th=[th;toc];

%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_mpc(1))];
y(i+1) = xk(1);
xvec1 = [xvec1 xk];
y_data(i) = xk(1);
iter(i) = i;
refreshdata
drawnow
end

e = y-r(1:length(y))';
%% Plots
fig = figure;
plot(r(1:length(y)),'LineWidth',3);
hold on;
plot(y,'LineWidth',3);
legend('Reference','RBF-\phi-DeePC-R2');
title ('\phi-DeePC-R2: Reference vs Closed Loop Output');
grid on
saveas(fig,'RBF-DeePC-R2\Figures\Closed Loop Output\VDP_RBF-DeePC-R2_output_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+"_lambda"+string(lambda)+'_step.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('\phi-DeePC-R2: Control Input');grid on
xlabel('Iterations');
saveas(fig2,'RBF-DeePC-R2\Figures\Control Input\VDP_RBF-DeePC-R2_input_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+"_lambda"+string(lambda)+'_step.png')


fig3 = figure;
plot(e,'LineWidth',3);
title ('DeePC-R2 Tracking Error');
grid on;
xlabel('Iterations');
saveas(fig3,'RBF-DeePC-R2\Figures\Tracking Error\VDP_RBF-DeePC-R2_Error_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+"_lambda"+string(lambda)+'_step.png')


save('Mircea-Plotting/u_dataDeePC_R2NL_sin_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.mat',"uvec")
save('Mircea-Plotting/y_dataDeePC_R2NL_sin_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.mat',"y")