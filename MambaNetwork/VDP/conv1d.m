function output = conv1d(input_matrix, weights, bias)
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


  output = zeros(out_channels, input_width);
  for o = 1:out_channels
    convolution_sum = zeros(1, input_width); % Initialize sum for each output channel
    for i = 1:in_channels
      % Apply convolution for each input channel and accumulate
      convolution_sum = convolution_sum + conv(input_matrix(i, :), squeeze(weights(o, i, :)), 'same'); % Use squeeze
    end
    % Add bias
    output(o, :) = convolution_sum + bias(o);
  end
  output = flipud(output');
end