function [phi1_nn] =rbf_before_out_layer(param,uu,u,y,e,sig,Basis_func)
centers = double(param.centers);
log_sigmas = double(param.log_sigmas);
rbf = [];
for i = 1:length(centers(:,1))
out = ([uu;y;u]-e')./sig' - centers(i,:)';  
out = sqrtm(sum(out.^2))./ exp(log_sigmas(i));
if string(Basis_func) == 'gaussian'
    out = exp(-1*out^2);
elseif string(Basis_func) == 'spline'
    out = (out.^2 * log(out + 1));
elseif string(Basis_func) == 'inverse multiquadratic'
    out = 1 /( 1 + out^2);
elseif string(Basis_func) == 'matern52'
   out = (1 + sqrt(5) * out + (5/3) * out .^2) .* exp(-sqrt(5) * out );
  
end

rbf = [rbf;out];
end
phi1_nn = rbf;
end