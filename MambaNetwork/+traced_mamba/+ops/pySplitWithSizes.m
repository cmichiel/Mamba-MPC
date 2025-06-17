function varargout = pySplitWithSizes(X, split_sizes, dim)
%PYSPLITWITHSIZES implements aten:split_with_sizes.
% ::std::vector<at::Tensor> at::split_with_sizes(const at::Tensor
% &self, at::IntArrayRef split_sizes, int64_t dim = 0).
%
% Y is a Kx1 struct array

%Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

split_sizes = extractdata(split_sizes.value);
dim = extractdata(dim.value);

format = dims(X.value);
Xval = X.value;
Xrank = X.rank;
Yrank = Xrank;

% Convert negative index to positive
if dim < 0
    dim = dim + Xrank;
end

MLdim = Xrank - dim;

% If X is a vector, ensure it is a column vector
if Xrank==1
    Xval = [Xval(:)];
    MLdim = 1;
end

% Calls to subsref
S      = struct;
S.type = '()';
S.subs = repmat({':'}, 1, ndims(Xval));
splitIndices = [0 cumsum(split_sizes(:)')];
numY = numel(splitIndices)-1;
for i = 1:numY
    from            = splitIndices(i) + 1;
    to              = splitIndices(i+1);
    S.subs{MLdim}	= from:to;
    Yval_i          = subsref(Xval, S);
    Yval_i = dlarray(Yval_i, repmat('U',1,max(2,Yrank)));
    % Assign into result array
    varargout{i} = struct('value', Yval_i, 'rank', Xrank);
end
end