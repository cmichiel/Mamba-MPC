function [y_output] = Mamba(inputs,RMSWeight1, RMSWeight2, LinearInWeight, LinearInBias, LinearExpansionWeight,ConvWeight,ConvBias, SelectionProjectionWeight, DeltaTProjectionWeight, DeltaTProjectionBias, A,D, LinearReductionWeight, LinearOutputWeight, LinearOutputBias,d_inner, d_state, d_model)


[L,Dim] = size(inputs);
x_embed = inputs*LinearInWeight'+ LinearInBias;

%RMS Block
mean_squared = mean(x_embed .^2, 2);
rsqrt_term = 1 ./ sqrt(mean_squared + 1e-5); 
rsqrt_term_reshaped = repmat(rsqrt_term, 1, size(x_embed , 2));
x_RMS = x_embed  .* rsqrt_term_reshaped .* RMSWeight1;

%Expansion
x_expanded = x_RMS*LinearExpansionWeight' % B x L x ED
x_ssm = x_expanded(:,1:d_inner)

%Convolutional Layer
% x_conv = diag( x_ssm'*ConvWeight'+ConvBias)';
% x_conv = x_ssm(1:d_model);
x_conv = conv1d(x_ssm', ConvWeight, ConvBias)'


%Residual Branch
x_res = x_expanded(:,d_inner+1:2*d_inner);
Residual = x_res./(1+exp(-x_res))

%Silu Activation
x_ssm = x_conv ./(1+exp(-x_conv )) % B x L x ED

%SSM Block
DeltaBC = x_ssm*SelectionProjectionWeight' % 1 x dt_rank+2*d_state = 1 x 17
Delta = DeltaBC(:,1) % 1 x dt_rank = 1 x 1
B = DeltaBC(:, 2:d_state+1); % 1 x d_state = 1 x 8 
C = DeltaBC(:, d_state+2:d_state*2+1); % 1 x d_state = 1 x 8 
DeltaT = log(1+exp(Delta*DeltaTProjectionWeight'+ DeltaTProjectionBias)) % 1 x ED = 1 x 8

%A = -A
%DeltaT = reshape(DeltaT,1,d_inner,1);
% A_delta = 0;
% for i = 1:d_inner
%    A_delta = A_delta + DeltaT'*A(i,:);
% end
A_bar = DeltaT'.*A
B_bar = DeltaT'*B;
% BX  = 0;
% for i = 1:d_state
%    BX = BX + B_bar(:,i)*x_ssm;
% end

States = 1*ones(d_inner,d_state);
States = A_bar.*States+BX;
% end
for i = 1:L
x_seq = x_ssm(i,:)
BX =B_bar.*x_seq';
States =  A_bar.*States+BX;
end

y_ssm = (States*C')';
y_ssm = diag(y_ssm + (D.*x_ssm)')';

%OutputProjection
y_mamba = (Residual.*y_ssm);
y_mamba = y_mamba*LinearReductionWeight';

y_mamba = y_mamba+x_embed;
%RMS Block
mean_squared = mean(y_mamba.^2, 2);
rsqrt_term = 1 ./ sqrt(mean_squared + 1e-5); 
rsqrt_term_reshaped = repmat(rsqrt_term, 1, size(y_mamba , 2));
y_RMS2 = y_mamba  .* rsqrt_term_reshaped .* RMSWeight2;

y_output = y_RMS2*LinearOutputWeight'+LinearOutputBias;

end