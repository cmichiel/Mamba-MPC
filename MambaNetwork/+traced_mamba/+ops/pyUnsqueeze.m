function Y = pyUnsqueeze(X, dim)
%PYUNSQUEEZE Inserts a singleton dimension at the position given by dim.
% at::Tensor at::unsqueeze(const at::Tensor &self, int64_t dim)

%   Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

dim = dim.value;

Xval = X.value;
Xrank = X.rank;

% Convert dim to reverse-pytorch
if (dim<0)
    dim = -dim;
else
    dim = Xrank - dim + 1; 
end

% Reshape the data, inserting a singleton dim
Yrank = Xrank + 1;
if Yrank == 1
    newShape = size(Xval);
elseif Yrank == 2
    if dim==0
        % X is a 1D vector, which will be a col in MATLAB ([N 1]).
        % The new singleton dim should be inserted at the front if dim=0
        newShape = flip(size(Xval));
    elseif dim==1
        % The new singleton dim should be inserted at the end if dim=1.
        % X is already [N 1], so no need to flip it.
        newShape = size(Xval);
    end
else
    newShape = ones(1, Yrank);
    knownSizes = setdiff(1:Yrank, dim);
    newShape(knownSizes) = size(Xval, 1:numel(knownSizes));
end

Yval = reshape(Xval, newShape);
Yval = dlarray(Yval, repmat('U', 1, max(2,Yrank)));
Y = struct('value', Yval, 'rank', Yrank);
end