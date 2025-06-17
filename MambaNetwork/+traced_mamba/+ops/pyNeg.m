function [Y] = pyNeg(X)
%PYNEGOPERATOR Negates the input Tensor

%   Copyright 2022 The MathWorks, Inc.

import traced_mamba.ops.*
Xval = X.value;
Yval = -Xval;
Y = struct('value',Yval,'rank',X.rank);
end

