function Y = pyMean(X, dim, keepdim)
%PYMEAN Calculates the mean of the input tensor along dimension dim.

%Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

dim = [dim.value];
keepdim = keepdim.value;

% If dim is empty, get all dims
if isempty(dim)
    dim = 0:X.rank-1; % 0-indexed, as in PyTorch
elseif (dim < 0)
    dim = X.rank + dim;
end

Xval = X.value;
mlDims = X.rank - dim;

% Compute the mean
Yval = mean(Xval, mlDims);

% If keepdim is false (the default), squeeze out dims
if ~keepdim || isempty(keepdim)
    % Get size vector (including singletons up to rank)
    sz = ones(1, X.rank);
    ySz = size(Yval);
    sz(1:numel(ySz)) = ySz;
    % Remove dims that were averaged across
    for i=1:numel(mlDims)
        sz(mlDims(i)) = [];
    end
    % If sz has less than 2 elements, append trailing singletons
    if numel(sz) < 2
        sz = [sz ones(1, 2-numel(sz))];
    end
    % Reshape to remove dims
    Yrank = X.rank - numel(dim);
    Yval = dlarray(reshape(Yval, sz));   
    
else
    Yrank = X.rank;    
end

Yval = dlarray(Yval, repmat('U', 1, max(Yrank, 2)));
Y = struct('value', Yval, 'rank',Yrank);
end