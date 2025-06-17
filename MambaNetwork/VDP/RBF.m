function [y_output] = RBF(u, u_ini,y_ini, weight, centers, log_sigmas, data_mean_1,data_std_1,data_mean_2,data_std_2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
clear nl_part;
nl_part = [];
Basis_func = 'gaussian';
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

y_output = weight*[nl_part;(u'-data_mean_2')./data_std_2']

end