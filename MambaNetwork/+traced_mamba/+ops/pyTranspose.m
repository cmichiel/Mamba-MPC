function Y = pyTranspose(X, dim0, dim1)
% at::transpose(const at::Tensor &self, int64_t dim0, int64_t dim1)
% Swaps the dimensions dim0 and dim1 in X

%Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

dim0 = dim0.value;
dim1 = dim1.value;

% Input dlarray is expected to be in reverse-PyTorch Ordering
Xval = X.value;

% Perform Transpose only if input rank >= 2
if X.rank >= 2
    mlDim0 = transformDim(dim0, X.rank);
    mlDim1 = transformDim(dim1, X.rank);
    
    % Create permutation vector
    perm = 1:X.rank;
    perm(mlDim0) = mlDim1;
    perm(mlDim1) = mlDim0;
    
    Yval = dlarray(permute(Xval, perm), repmat('U', 1, X.rank));
    Y = struct('value', Yval, 'rank', X.rank);
else
    Y = struct('value', Xval, 'rank', X.rank);
end
    function revPTDim = transformDim(dimRef ,XRank)
        if dimRef < 0
            revPTDim = -dimRef;
        else
            revPTDim = XRank - dimRef;
        end
    end
end