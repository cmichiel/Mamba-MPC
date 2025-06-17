% Define the input range
x = linspace(-5, 5, 100);

% Calculate the softplus function
soft = log(1+exp(x));

% Calculate the silu function
silu = x ./ (1 + exp(-x));

% Create the plot
figure;
plot(x, soft, 'b-', 'LineWidth', 2, 'DisplayName', 'Softplus (log(1+exp(x)))');
hold on;
plot(x, silu, 'r-', 'LineWidth', 2, 'DisplayName', 'SiLU (x/(1+exp(-x)))');

% Add labels and title
xlabel('x');
ylabel('y');
title('Softplus and SiLU Functions');
legend('show');
grid on;