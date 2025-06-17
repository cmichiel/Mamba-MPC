%% N-step ahead predictor
clc; close all; clear all;



fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
F = 0.1;                       % Sine wave frequency (hertz)
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
% r = r;


% 
r = 1*sin(2*pi*F*t)'; %Working

% use this for noiseless case
load('DataSets/u_data40sinsSweptAmpRed.mat')
load('DataSets/y_data40sinsSweptAmpRed.mat')
%UkNMPC = load('PredictionSets/UkNMPC_SINREF_FULL_N10_n_basis40_Tini3_Gaussian.mat')
nu = size(u_data,1);
ny = size(y_data,1);

% use this for noisy case 
 % u_data = load('train_1000_noise.mat').input;  %input training data
 % y_data = load('train_1000_noise.mat').output; %output training data
%% load centers and sigmas
N = 10; %prediction horizon
Tini =3;
n_basis =60;
n_basisKoopman =40;
n_basisNARX =40;
Basis_funcKoopman = 'gaussian';
Basis_funcNARX = 'gaussian';
lambda = 1e-5;
file_string = 'RBFs/'+string(Basis_funcKoopman)+'/'+string(Basis_funcNARX)+ '/RBF_Params_KoopmanNARX_Tini'+string(Tini)+'_nbasisKoopman'+string(n_basisKoopman)+'_nbasisFullBasis'+string(n_basisNARX)+'_N'+string(N)+'_KoopmanBasis_'+string(Basis_funcKoopman)+'_FullBasis_'+string(Basis_funcNARX)+'.mat';
%file_string = 'RBFs/'+string(Basis_funcKoopman)+'/'+string(Basis_funcNARX)+ '/RBF_Params_KoopmanFullBasis_Tini'+string(Tini)+'_nbasisKoopman'+string(n_basisKoopman)+'_nbasisFullBasis'+string(n_basisNARX)+'_N'+string(N)+'_KoopmanBasis_'+string(Basis_funcKoopman)+'_FullBasis_'+string(Basis_funcNARX)+'.mat';
%file_string = 'RBFs/gaussian/RBF_Params_MIXED_Tini10_nbasis80_N10_gaussian.mat';
load(file_string)
% centers_sigmasNL = struct('centersFullBasis',centersFullBasis,'log_sigmasFullBasis',log_sigmasFullBasis)
data_mean = double(data_mean);
data_std = double(data_std);
% centersKoopman = centersL;
% log_sigmasKoopman = log_sigmasL;
% centersFullBasis = centersNL;
% log_sigmasFullBasis = log_sigmasNL;
centers_sigmasNL = struct('centersFullBasis',centersNARX,'log_sigmasFullBasis',log_sigmasNARX)
centers_sigmasL = struct('centers',centersKoopman,'log_sigmas',log_sigmasKoopman)
%% recompute Theta 
load('DataSets/u_data40sinsSweptAmpRed.mat')
load('DataSets/y_data40sinsSweptAmpRed.mat')
y_ini = ones(Tini,1)*y_data(1)
u_ini = zeros(Tini-1,1)
Phi = []; Y = [];

data_mean_1 = double(data_mean(1:2*Tini-1));
data_mean_2 = double(data_mean(2*Tini:length(data_mean)));
data_std_1 = double(data_std(1:2*Tini-1));
data_std_2 = double(data_std(2*Tini:length(data_std)));

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
uf1 = (uf-data_mean_2')./data_std_2';

% PhiL = [];
% PhiL = [PhiL rbf_before_out_layer_norm(centersc_sigmasL,u_ini,uf1,y_ini,data_mean_1,data_std_1,Basis_funcKoopman); uf1];
% 
% 
% PhiNL = [];
% PhiNL = [PhiNL rbf_before_out_layer_MIXED(centers_sigmasNL,u_ini,uf,y_ini,data_mean,data_std,Basis_funcNARX)];
PhiPart = [rbf_before_out_layer_norm(centers_sigmasL,u_ini,uf1,y_ini,data_mean_1,data_std_1,Basis_funcKoopman); uf1; rbf_before_out_layer_MIXED(centers_sigmasNL,u_ini,uf,y_ini,data_mean,data_std,Basis_funcNARX)];
Phi = [Phi,PhiPart];
out_vector = [];
for j = 1:(N)
out_vector = [out_vector; y_data(i+j)];
end

Y = [Y out_vector];
end
Theta = Y*pinv(Phi);
%%
% [~,~,V] = svd([Phi;Y]);
% PHIY_reduced = [Phi;Y]*V(:,1:45);
% Phi_app = PHIY_reduced(1:50,:); 
% Y_red = PHIY_reduced(51:60,:);
% 
% Theta = Y_red*pinv(Phi_app);
%%
figTSNE = figure;
[Y_sne,loss] =tsne([Phi]);
gscatter(Y_sne(:,1),Y_sne(:,2))
grid on
title('T-SNE Clustering of \Phi for Koopman/NARX')
saveas(figTSNE,'T-SNE\T-SNE_KoopmanNARX_Phi_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_Koopman'+Basis_funcKoopman+'_NARX'+Basis_funcNARX+'.png')

%%
%Orthogonality
LI = norm(Phi*Phi' - eye(size(Phi*Phi')), 'fro')
%%
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


%% Build YALMIP SPC problem
k_sim = length(r)-N;
disp("Constructing the YALMIP DeePC-R2 problem")

centers = double(centers_sigmasNL.centersFullBasis);
log_sigmas = double(centers_sigmasNL.log_sigmasFullBasis);

Q = 100; 
R =  0.5;
Rd =  0;
S = 100;


Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
u_init = sdpvar(1,1);
u_ini_var = sdpvar(Tini-1, 1);
y_ini_var = sdpvar(Tini, 1);
% gvar = sdpvar(length(u_data)-N,1);
gvar = sdpvar(60,1);
nl_part = sdpvar(length(centersKoopman(:,1)),1);

%constraints = [constraints,  -15<=u(1)<=15];
objective = 0;



%%
Phi_inv = pinv(Y)*weight;
Phi_approx = pinv(weight)*Y;
[~,~,V] = svd([Phi;Y]);
PHIY_reduced = [Phi;Y]*V(:,1:60);
Phi_app = PHIY_reduced(1:90,:); 
Y_red = PHIY_reduced(91:100,:);
%%

constraints = [Phi_app *gvar == 0];
constraints = [constraints,  Theta*[nl_part;(u-data_mean_2')./data_std_2';
    rbf_before_out_layer_MIXED(centers_sigmasNL,u_ini_var,u,y_ini_var,data_mean,data_std,Basis_funcNARX) ]+Y_red*gvar==y];

objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u(1)-u_init)'*Rd*(u(1)-u_init) + u(N)'*R*u(N);

for k = 1:N-1
    objective = objective + u(k)'*R*u(k);
    objective = objective +(u(k+1)-u(k))'*Rd*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k+1)<=15];
end
objective = objective+(y(N)-ref(N))'*S*(y(N)-ref(N));

objective = objective + lambda*norm(gvar,2)^2;
constraints = [constraints,  -15<=u(1)<=15];

Parameters = {u_init, ref,u_ini_var,y_ini_var,nl_part};
Outputs = {u,y};


options = sdpsettings('verbose', 0, 'debug', 0, 'warning',0,'solver','fmincon','fmincon.maxiter',100 );
controller = optimizer(constraints, objective,options , Parameters, Outputs);

%% initial conditions
% y(1) = 1;
y_data = 1;
iter = 1;
x0 = [1;1];
xk = x0;
xvec1 = [xk];
u_mpc = 0;
uvec = [];
u = [];
th=[];
y_ini = ones(Tini,1)*y_data(1);
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
y_ini = [y_ini(2:end);y_data(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y_data(i)];
u_ini = [u_ini(2:end);uvec(i-1)];
end

clear nl_part;
nl_part = [];
for im = 1:length(centersKoopman(:,1))
out = ([u_ini;y_ini]-data_mean(1:2*Tini-1)')./data_std(1:2*Tini-1)' - centersKoopman(im,:)';  
out = sqrtm(sum(out.^2))./ exp(log_sigmasKoopman(im));
if string(Basis_funcKoopman) == 'gaussian'
    out = exp(-1*out^2);
elseif string(Basis_funcKoopman) == 'spline'
    out = (out.^2 * log(out + 1));
elseif string(Basis_funcKoopman) == 'inverse_multiquadratic'
    out = 1 /( 1 + out^2);
elseif string(Basis_funcKoopman) == 'matern52'
   %out = exp(-1*out^2);
   out = (1 + sqrt(5) * out + (5/3) * out .^2) .* exp(-sqrt(5) * out );
end
nl_part= [nl_part;out];
end


%%% mpc sol
UYK = controller({u_mpc(1), r(i:i+N-1)',u_ini,y_ini,nl_part});
UYK = cell2mat(UYK);
Uk = UYK(:,1);
Yk(:,i) = UYK(:,2);
u_mpc = Uk(1);
uf = u_mpc ;
uvec = [uvec u_mpc(1)];
th=[th;toc];

%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_mpc(1))];
y(i+1) = xk(1);
y_data(i+1) = xk(1);
iter(i+1) = i;
xvec1 = [xvec1 xk];
refreshdata
drawnow
end

e = abs(y_data-r(1:length(y_data)));
%% Plots
fig = figure;
plot(r(1:length(y_data)),'LineWidth',3);
hold on;
plot(y_data,'LineWidth',3);
legend('Reference','RBF-\phi-DeePC-R2');
title ('\phi-DeePC-R2: Reference vs Closed Loop Output');
grid on
%saveas(fig,'RBF-DeePC-R2\Figures\Closed Loop Output\VDP_RBF-DeePC_MIXED-R2_output_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisFullBasis'+string(n_basisFullBasis)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_FullBasis'+Basis_funcFullBasis+"_lambda"+string(lambda)+'_step.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('\phi-DeePC-R2: Control Input');grid on
xlabel('Iterations');
%saveas(fig2,'RBF-DeePC-R2\Figures\Control Input\VDP_RBF-DeePC_MIXED-R2_input_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisFullBasis'+string(n_basisFullBasis)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_FullBasis'+Basis_funcFullBasis+"_lambda"+string(lambda)+'_step.png')


fig3 = figure;
plot(e,'LineWidth',3);
title ('DeePC-R2 Tracking Error');
grid on;
xlabel('Iterations');
%saveas(fig3,'RBF-DeePC-R2\Figures\Tracking Error\VDP_RBF-DeePC-R2_MIXED_Error_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisFullBasis'+string(n_basisFullBasis)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_FullBasis'+Basis_funcFullBasis+"_lambda"+string(lambda)+'_step.png')


saveas(fig,'RBF-DeePC-R2\Figures\Closed Loop Output\KoopmanNARX-DeePC-R2\VDP_KoopmanNARX-RBF_SINREF-DeePC-R2_output_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_NARXBasis'+Basis_funcNARX+"_lambda"+string(lambda)+'.png')
saveas(fig2,'RBF-DeePC-R2\Figures\Control Input\KoopmanNARX-DeePC-R2\VDP_KoopmanNARX-RBF_SINREF-DeePC-R2_input_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_NARXBasis'+Basis_funcNARX+"_lambda"+string(lambda)+'.png')
saveas(fig3,'RBF-DeePC-R2\Figures\Tracking Error\KoopmanNARX-DeePC-R2\VDP_KoopmanNARX-RBF_SINREF-DeePC-R2_Error_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_NARXBasis'+Basis_funcNARX+"_lambda"+string(lambda)+'.png')




save('Mircea-Plotting/Data/u_dataDeePC_R2_MIXED_SINREF_FULL_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_NARXBasis'+Basis_funcNARX+"_lambda"+string(lambda)+'.mat',"uvec")
save('Mircea-Plotting/Data/y_dataDeePC_R2_MIXED_SINREF_FULL_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_NARXBasis'+Basis_funcNARX+"_lambda"+string(lambda)+'.mat',"y")
%%
save('PredictionSets/YkDeePC_R2_MIXED_SINREF_NMPCInput_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_KoopmanBasis'+Basis_funcKoopman+'_FullBasis'+Basis_funcNARX+"_lambda"+string(lambda)+'.mat',"Yk")