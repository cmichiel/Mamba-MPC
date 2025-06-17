function Z = pyMatmul(X, Y)
%PYMATMUL  The result is always returned in reverse-pytorch ordering and
%labeled with all U's.
% at::Tensor at::matmul(const at::Tensor &self, const at::Tensor &other)

%   Copyright 2022-2023 The MathWorks, Inc.

import traced_mamba.ops.*

% Convert inputs to reverse-pytorch
Xrev = stripdims(X.value);
Yrev = stripdims(Y.value);

if X.rank==1
    Xrev = Xrev(:);
end
if Y.rank==1
    Yrev = Yrev(:);
end

% Do the Matmul. See
% https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul
% for the rules of behavior.
if X.rank==0 || Y.rank==0
    Zrev = Xrev*Yrev;                                             % at least one of them is a scalar
    Zrank = max([X.rank, Y.rank]);
elseif X.rank==1 && Y.rank==1
    Zrev = Xrev'*Yrev;                                            % 1d-1d dot product xy
    Zrank = 0;
elseif X.rank==2 && Y.rank==2
    Zrev = Yrev*Xrev;                                             % 2d-2d matrix product XY becomes Y'X' = Yrev*Xrev
    Zrank = 2;
elseif X.rank==1 && Y.rank==2
    Zrev = Yrev*Xrev;                                             % 1d-2d product xY becomes Y'x' = Yrev*Xrev, then columnize.
    Zrev = Zrev(:);
    Zrank = 1;
elseif X.rank==2 && Y.rank==1
    % Yrev is a column vector but should be a row vector to multiply
    Zrev = Yrev'*Xrev;                                             % 2d-1d product Xy becomes y'X' = Yrev'*Xrev, then columnize.
    Zrev = Zrev(:);
    Zrank = 1;
else
    % At this point, "both arguments are at least 1-dimensional and at
    % least one argument is N-dimensional (where N > 2)".
    if Y.rank==1
        % "If the second argument is 1-dimensional, a 1 is appended to its
        % dimension." Since we're using reverse-pt ordering, PREpend a 1
        % to Yrev:
        Yrev = Yrev';
    end
    % Do batched 2d-2d matrix product with broadcasting.
    Zrev = pagemtimes(Yrev, Xrev);  
    Zrank = max(X.rank, Y.rank);

    % If one of the ranks was 1, we must remove an appended singleton
    % dimension. If X.rank was 1, we implicitly appended a trailing
    % singleton to X, which is now in dimension 2. If Y.rank==1, the
    % appended dimension is in dim 1
    if X.rank==1
        loc = 2;
    elseif Y.rank==1
        loc = 1;
    else
        loc = [];
    end
    if ~isempty(loc)
        szZrev = size(Zrev, 1:max(X.rank, Y.rank));
        szZrev(loc) = [];
        Zrev = reshape(Zrev, szZrev);
        Zrank = numel(szZrev);
    end
end

% Apply output label of all U's
Zrev = dlarray(Zrev, repmat('U', 1, max(2,Zrank)));

% Return Z
Z = struct('value', Zrev, 'rank', Zrank);
end