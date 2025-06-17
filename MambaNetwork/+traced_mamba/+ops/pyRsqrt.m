function Y = pyRsqrt(X)
%PYRSQRT computes the reciprocal square root of each element of the input
%X.
% at::Tensor at::rsqrt(const at::Tensor &self)

% Copyright 2024 The MathWorks, Inc.

import traced_mamba.ops.*

% For inputs comprised of real numbers the output is set to NaN
% for negative values.
Xval = X.value;
Xval(Xval<0) = NaN;
Yval = 1./sqrt(Xval);
Y = struct('value', Yval, 'rank', X.rank);
end