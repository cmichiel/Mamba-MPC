function Y = pyStack(varargin)
%PYSTACK Concatenates tensors together along a new dimension dim
%   at::Tensor at::stack(at::TensorList tensors, int64_t dim = 0)

%   Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

numInputs = numel(varargin);
if numInputs == 2
    Xs = varargin{1};
    dim = varargin{2};
else
    Xs = [varargin{1:end-1}];
    dim = varargin{end};
end
dim = dim.value;

% Convert dim to reverse-pytorch
if dim < 0 
    dim = -dim;
else
    dim = Xs(1).rank - dim;
end

reshapedXVals = cell(1,numel(Xs));
for i=1:numel(Xs)
    Xval = Xs(i).value;
    XValShape = size(Xval,1:Xs(i).rank);
    Xval = reshape(Xval, [XValShape(1:dim) 1 XValShape(dim+1:end)]);
    reshapedXVals{i} = Xval;
end
%Concatenate on the new dimension (dim +1) in reverse Pytorch
Yval = cat(dim + 1 , reshapedXVals{:});

Yval = dlarray(Yval, repmat('U', 1, Xs(1).rank + 1));

Y = struct('value', Yval, 'rank', Xs(1).rank+1);
end