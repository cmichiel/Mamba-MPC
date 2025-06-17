function output = apply_conv1d_matlab_pytorch_groups(input_matrix, weights, bias, groups)
  % Applies a 1D convolution to an input matrix using Matlab's conv function,
  % ensuring parity with PyTorch's Conv1d behavior, including the 'groups' parameter.
  %
  % Args:
  %   input_matrix (double): Input matrix of size [in_channels, width].
  %   weights (double): Convolution weights of size [out_channels, in_channels, kernel_size].
  %   bias (double): Convolution bias of size [out_channels, 1].
  %   groups (int): Number of groups.
  %
  % Returns:
  %   output (double): Output matrix of size [out_channels, width].

  [out_channels, in_channels, kernel_size] = size(weights);
  [input_in_channels, input_width] = size(input_matrix);

  % if in_channels ~= input_in_channels
  %   error('Input matrix and weights must have the same number of input channels.');
  % end
  % if mod(in_channels, groups) ~= 0
  %     error('in_channels must be divisible by groups');
  % end
  % if mod(out_channels, groups) ~= 0
  %     error('out_channels must be divisible by groups');
  % end

  output = zeros(out_channels, input_width);
  in_channels_per_group = in_channels / groups;
  out_channels_per_group = out_channels / groups;

  for g = 1:groups
      input_start_channel = (g - 1) * in_channels_per_group + 1;
      input_end_channel = g * in_channels_per_group;
      output_start_channel = (g - 1) * out_channels_per_group + 1;
      output_end_channel = g * out_channels_per_group;

      for o = output_start_channel:output_end_channel
          convolution_sum = zeros(1, input_width);
          for i = input_start_channel:input_end_channel
              convolution_sum = convolution_sum + conv(input_matrix(i, :), rot90(squeeze(weights(o, i, :)), 2), 'same');
          end
          output(o, :) = convolution_sum + bias(o);
      end
  end
end
load 'MambaParameters.mat'
LinearInWeight = double(LinearInWeight);
LinearInBias = double(LinearInBias);
LinearExpansionWeight = double(LinearExpansionWeight);
SelectionProjectionWeight = double(SelectionProjectionWeight);
DeltaTProjectionWeight = double(DeltaTProjectionWeight);
DeltaTProjectionBias = double(DeltaTProjectionBias);
if exist('ConvWeight','var') == 1
ConvWeight = double(ConvWeight);
ConvBias = double(ConvBias);
end
% Example usage:
input_matrix = [-0.0393   , 0.7507  ,  0.2586   , 0.3931;
   -0.0393  ,  0.7507  ,  0.2586   , 0.3931;
   -0.0393 ,   0.7507  ,  0.2586   , 0.3931]';
weights = randn(4, 1, 2);
bias = randn(4, 1);
groups = 4;

output_conv = apply_conv1d_matlab_pytorch_groups(input_matrix, ConvWeight, ConvBias, groups);

disp('Convolution Output (PyTorch Parity with Groups):');
disp(output_conv);

% Verification with convn
output_convn = zeros(size(output_conv));
in_channels_per_group = size(input_matrix, 1) / groups;
out_channels_per_group = size(weights, 1) / groups;

for g = 1:groups
    input_start_channel = (g - 1) * in_channels_per_group + 1;
    input_end_channel = g * in_channels_per_group;
    output_start_channel = (g - 1) * out_channels_per_group + 1;
    output_end_channel = g * out_channels_per_group;
    for o = output_start_channel:output_end_channel
        convolution_sum_convn = zeros(1, size(input_matrix, 2));
        for i = input_start_channel:input_end_channel
            convolution_sum_convn = convolution_sum_convn + convn(input_matrix(i, :), rot90(reshape(weights(o, i, :), [1, size(weights, 3)]), 2), 'same');
        end
        output_convn(o, :) = convolution_sum_convn + bias(o);
    end
end

disp('Output using convn (for comparison):');
disp(output_convn);

disp('Difference:');
disp(output_conv - output_convn);
