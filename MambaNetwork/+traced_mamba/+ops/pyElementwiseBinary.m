function Yout = pyElementwiseBinary(Xin1,Xin2,matlabFcn,alpha)
%Calculates elementise binary 'matlabFcn' for Xin1 and Xin2

%Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

%'matlabFcn' can be plus, minus, div, floor_divide, mul, or power
functionHandle = str2func(matlabFcn);

%Inputs are already in reverse-pytorch
Xin1ValRevPyTorch = Xin1.value;
Xin2ValRevPyTorch = Xin2.value;

Xin1Rank = Xin1.rank;
Xin2Rank = Xin2.rank;

if ~isdlarray(Xin1ValRevPyTorch)
    Xin1ValRevPyTorch = single(Xin1ValRevPyTorch);
end


if ~isdlarray(Xin2ValRevPyTorch)
    Xin2ValRevPyTorch = single(Xin2ValRevPyTorch);
end

if matlabFcn == "idivide"
    Xin1ValRevPyTorch = int64(floor(extractdata(Xin1ValRevPyTorch)));
    Xin2ValRevPyTorch = int64(floor(extractdata(Xin2ValRevPyTorch)));
end

% Function handle for floor_divide
if matlabFcn == "@(x, y)floor(rdivide(x, y))" 
    isNeg = any(Xin1ValRevPyTorch < 0, 'all') || any(Xin2ValRevPyTorch < 0, 'all');
    % Warn the user that floor_divide operator can produce inference
    % mismatch in PyTorch 1.12 and earlier for negative valued tensors
    if isNeg
        warning(message('nnet_cnn_pytorchconverter:pytorchconverter:NumericalMismatchInOperatorForSpecifiedReason', ...
        "pyElementwiseBinary", "aten::floor_divide", "tensors with negative values"));
    end
end

% Warn the user if the power operator can produce complex number outputs
if matlabFcn == "power"
    isFracExponent = any(Xin2ValRevPyTorch<1, 'all') && any(Xin2ValRevPyTorch>-1, 'all');
    isNegBase = any(Xin1ValRevPyTorch < 0, 'all');
    if isNegBase && isFracExponent
        warning(message('nnet_cnn_pytorchconverter:pytorchconverter:NumericalMismatchInOperatorForSpecifiedReason', ...
        "pyElementwiseBinary", "aten::pow", "tensors with negative values and fractional exponents"));
    end
end

% alpha is a scaling factor that belongs to the "aten::add" and "aten::sub" operators
if nargin == 4 && alpha.value ~= 1
    alphaVal = alpha.value;
    YValRevPyTorch = functionHandle(Xin1ValRevPyTorch,Xin2ValRevPyTorch*single(alphaVal));
else
    YValRevPyTorch = functionHandle(Xin1ValRevPyTorch,Xin2ValRevPyTorch);
end

%Get labels and reverse permutation from the max rank input

%When input ranks are equal, we get the output rank from
%the inputs
%Output will be in reverse PyTorch format

if Xin1Rank == Xin2Rank
    Yrank = Xin1Rank;
else
    %When rank is not equal we get the rank from the input
    %with max rank. 
    [Yrank, ~] = max([Xin1Rank,Xin2Rank]);
end

YVal = dlarray(single(YValRevPyTorch),repmat('U',1,max(2,Yrank)));
Yout = struct("value",YVal,"rank",Yrank);

end
