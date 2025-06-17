function dlY = pyToUnary(varargin)
    %PYTOUNARY performs casting a Tensor with the specified dtype. 
    % torch.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) → Tensor
    % torch.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) → Tensor
    % Copyright 2023 The MathWorks, Inc.
    
    import traced_mamba.ops.*

    numInputs = numel(varargin);
    dlX = varargin{1};
    Xval = dlX.value;
    
    if numInputs == 6
        dtype = varargin{3};
    else 
        dtype = varargin{2};
    end
    dtype = dtype.value;
    % input rank and the output rank should be the same
    Xrank = dlX.rank;
    Yrank = Xrank;

    % Enumerators of c10 scalars "int" data type. 
    dtypeList = [0,1,2,3,4,12,13,14,16,17];
    % If dtype is "int", round to int using "fix"
    if(ismember(dtype, dtypeList))
        Yval = fix(Xval);
    else
        Yval = Xval;
    end
    
    dlY = struct('value', Yval, 'rank', Yrank);
    
    end