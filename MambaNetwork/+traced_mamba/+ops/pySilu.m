function Y = pySilu(X)
%PYSILU Applies a silu (swish) transformation to the input data.

import traced_mamba.ops.*

Yval = X.value .* sigmoid(X.value);
Y = struct('value', Yval, 'rank', X.rank);
end