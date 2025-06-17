function output = apply_conv1d(input_matrix, weights, bias)
  % Applies a 1D convolution to an input matrix using Matlab's conv function.
  %
  % Args:
  %   input_matrix (double): Input matrix of size [in_channels, width].
  %   weights (double): Convolution weights of size [out_channels, in_channels, kernel_size].
  %   bias (double): Convolution bias of size [out_channels, 1].
  %
  % Returns:
  %   output (double): Output matrix of size [out_channels, width].

  [out_channels, in_channels, kernel_size] = size(weights);
  [input_in_channels, input_width] = size(input_matrix);

  if in_channels ~= input_in_channels
    error('Input matrix and weights must have the same number of input channels.');
  end

  output = zeros(out_channels, input_width);

  for o = 1:out_channels
    for i = 1:in_channels
      % Apply convolution for each input channel and accumulate
      output(o, :) = output(o, :) + conv(input_matrix(i, :), weights(o, i, :), 'same');
    end
    % Add bias
    output(o, :) = output(o, :) + bias(o);
  end
end