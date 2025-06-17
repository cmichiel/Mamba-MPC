%% N-step ahead predictor
clc; close all; clear all;



fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
StopTime = 20;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
F = 1/3;                       % Sine wave frequency (hertz)
% A = [0:0.1:0.5, 0.5:-0.1:-0.5,-0.5:0.1:0];
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r; A(j)*ones(len,1)];
% end  
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


r = 1*sin(2*pi*F*t)';% works 
% r = 0.5*ones(1,length(t));

% use this for noiseless case
load('DataSets/u_data40sinsSweptAmpRed.mat')
load('DataSets/y_data40sinsSweptAmpRed.mat')

nu = size(u_data,1);
ny = size(y_data,1);

% use this for noisy case 
 % u_data = load('train_1000_noise.mat').input;  %input training data
 % y_data = load('train_1000_noise.mat').output; %output training data
%% load centers and sigmas
N = 10; %prediction horizon
Tini =5;
n_basis = 80;
Basis_func = 'gaussian';
file_string = 'RBFs/'+string(Basis_func)+'/RBF_Params_Koopman_Tini'+string(Tini)+'_nbasisKoopman'+string(n_basis)+'_N'+string(N)+'_KoopmanBasis_'+string(Basis_func)+'.mat';
load(file_string)
centers_sigmas = struct('centers',centersKoopman,'log_sigmas',log_sigmasKoopman)
data_mean_1 = double(data_mean(1:2*Tini-1));
data_mean_2 = double(data_mean(2*Tini:length(data_mean)));
data_std_1 = double(data_std(1:2*Tini-1));
data_std_2 = double(data_std(2*Tini:length(data_std)));

%%

T = length(u_data(1,:))-N-Tini;
k_sim = length(r)-N;
Phi = []; Y = [];

y_ini = ones(Tini,1)*y_data(1)
u_ini = zeros(Tini-1,1)

disp("Simulation for VDP system SPC controller with: N="+string(N)+", Tini="+string(Tini)+" and Nbasis="+string(n_basis)+" and Basis function:"+Basis_func)
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
uf = (uf-data_mean_2')./data_std_2';

%Phi = [Phi rbf_before_out_layer(centers_sigmas,u_ini,uf,y_ini)];
Phi_part = [rbf_before_out_layer_norm(centers_sigmas,u_ini,uf,y_ini,data_mean_1,data_std_1,Basis_func); uf];
Phi = [Phi Phi_part];
out_vector = [];
for j = 1:(N)
out_vector = [out_vector; y_data(i+j)];
end

Y = [Y out_vector];
end

Theta =  Y*pinv(Phi);

P    = Theta(:,1);
Gamma = Theta(:,2);
%%
figTSNE = figure;
[Y_sne,loss] =tsne([Phi]);
gscatter(Y_sne(:,1),Y_sne(:,2))
grid on
title('T-SNE Clustering of \Phi for Koopman Basis')
saveas(figTSNE,'T-SNE\T-SNE_Koopman_Phi_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')

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
Rd = 0;
S = 100;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
u_init = sdpvar(1,1);
%u_ini = sdpvar(Tini-1, 1);
%y_ini = sdpvar(Tini, 1);
gvar = sdpvar(length(u_data)-N,1);
nl_part = sdpvar(length(centers(:,1)),1);



constraints = [y == Theta*[nl_part;(u-data_mean_2')./data_std_2']];
%constraints = [constraints,  -15<=u(1)<=15];
% objective = 0;
% 
% %objective = (y-ref)'*Omega*(y-ref)+(u)'*Psi*(u);
% objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u(1)-u_init)'*R*(u(1)-u_init);
% %objective = objective +  (y(1)-ref(1))'*Q*(y(1)-ref(1))+(u(1)-u_init )'*R*(u(1)-u_init );
% 
% for k = 1:N-1
%     objective = objective +(u(k+1)-u(k))'*R*(u(k+1)-u(k));
%     constraints = [constraints,  -15<=u(k+1)<=15];
% end


objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u(1)-u_init)'*Rd*(u(1)-u_init) + u(N)'*R*u(N);

for k = 1:N-1
    objective = objective + u(k)'*R*u(k);
    objective = objective +(u(k+1)-u(k))'*Rd*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k+1)<=15];
end
objective = objective+(y(N)-ref(N))'*S*(y(N)-ref(N));

Parameters = {nl_part,u_init, ref};
Outputs = {u,y};

options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0);

%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

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

clear nl_part;
nl_part = [];
for im = 1:length(centers(:,1))
out = ([u_ini;y_ini]-data_mean_1')./data_std_1' - centers(im,:)';  
out = sqrtm(sum(out.^2))./ exp(log_sigmas(im));
if string(Basis_func) == 'gaussian'
    out = exp(-1*out^2);
elseif string(Basis_func) == 'spline'
    out = (out.^2 * log(out + 1));
elseif string(Basis_func) == 'inverse_multiquadratic'
    out = 1 /( 1 + out^2);
elseif string(Basis_func) == 'matern52'
   %out = exp(-1*out^2);
   out = (1 + sqrt(5) * out + (5/3) * out .^2) .* exp(-sqrt(5) * out );
end
nl_part= [nl_part;out];
end


%%% mpc sol
[UYK] = controller({nl_part,u_mpc, r(i+1:i+N)'});
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
saveas(fig,'RBF-SPC\Figures\Closed Loop Output\Koopman-SPC\VDP_Koopman_RBF-SPC_output_STEPREF_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('SPC: Control Input');
grid on
xlabel('Iterations');
saveas(fig2,'RBF-SPC\Figures\Control Input\Koopman-SPC\VDP_Koopman_RBF-SPC_input_STEPREF_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')


fig3 = figure;
plot(e,'LineWidth',3);
title ('SPC Tracking Error');
grid on;
xlabel('Iterations');
saveas(fig3,'RBF-SPC\Figures\Tracking Error\Koopman-SPC\VDP_Koopman_RBF-SPC_Error_STEPREF_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')

save('Mircea-Plotting/Data/u_dataKoopmanSPC_SINREF_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.mat',"uvec")
save('Mircea-Plotting/Data/y_dataKoopmanSPC_SINREF_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.mat',"y")