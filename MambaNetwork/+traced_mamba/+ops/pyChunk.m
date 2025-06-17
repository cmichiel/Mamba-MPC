function varargout = pyChunk(X, chunks, dim)
%PYCHUNK Splits a tensor into chunks
% ::std::vector<at::Tensor> at::chunk(const at::Tensor &self, int64_t chunks, int64_t dim = 0)

% Copyright 2022-2023 The MathWorks, Inc.


import traced_mamba.ops.*

%Get value from structs
chunks = chunks.value;
dim = dim.value;

% X is already in reverse-python
Xval = X.value;

% Convert dim to reverse-python
dim = transformDim(dim ,X.rank);

% Calculate chunks
C       = size(Xval, dim);
sz      = ceil(C/chunks);
splits	= repmat(sz, 1, chunks);

% If C is not evenly divisible by chunks, the last chunk will be smaller
% than the rest.
r = rem(C, chunks);
if r > 0
    splits(end) = r;
end

S      = struct;
S.type = '()';
S.subs = repmat({':'}, 1, ndims(Xval));
splitIndices = [0 cumsum(splits(:)')];
numY = numel(splitIndices)-1;
Ychunks = cell(1, numY);

for i = 1:numY
    from            = splitIndices(i) + 1;
    to              = splitIndices(i+1);
    S.subs{dim}	= from:to;
    % The first numY outputs are the Y's. The second numY outputs are their
    % numDims. We assume all the outputs of Split have the same numDims as
    % the input.
    Yi_val        = subsref(Xval, S);
    Yi_val = dlarray(Yi_val, dims(X.value));
    Ychunks{i} = struct('value', Yi_val, 'rank', X.rank);
end
varargout = Ychunks;

function revPTDim = transformDim(dimRef ,XRank)
    if dimRef < 0
        revPTDim = -dimRef;
    else
        revPTDim = XRank - dimRef;
    end
end
end
