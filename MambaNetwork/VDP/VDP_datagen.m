%% N-step ahead predictor
clc; close all; clear all;


fs = 10;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 100;                % seconds
t = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*t);           % Reference




%                  % seconds per sample
% StopTime = 100;                % seconds
% time = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*time);           % Reference

Range = [-15, 15];
SineData = [25,10, 1];
Band = [0, 1];
NumPeriod = 1;
Period = 10000;
Nu = 1;


u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';
u_data = [u_data, 4*randn(size(u_data))];
u_data = [u_data, idinput([Period 1 NumPeriod],'sine',[-10,10],Range,[100,100,1])'];

% u_data = zeros(1,length(u_data))
% SNR = 10;
% noise_u = randn(size(u_data))*std(u_data)/db2mag(SNR);
% u_data = u_data+noise_u ;


% u_data = [u_data, 15*chirp(t,0,2e1,20)'];
u_data = [u_data, idinput([Period 1 NumPeriod],'prbs',Band,[-8,8],[10,40,1])'];
%u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_ID = iddata([],u_data,dt);

time = (0:length(u_data)-1) * dt;

%%
x_initial = [1,1];
x0 =  [x_initial(1);
    x_initial(2)];

mu = 1;


y_train(1) = x_initial(1);
xk = x0;
x = [x0];


for i = 1:length(u_data)
xk = [xk(1) + dt*xk(2);  xk(2)+dt*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_data(i))];
x = [x,xk];
y_train(i) = xk(1);
end



SNR = 0;
noise_y = 0%randn(size(y_train))*std(y_train)/db2mag(SNR);

y = [y_train+noise_y]
% y = [x(1)';x(2)];

% for i = 1:length(u_data)
% xk = [xk(1) + dt*xk(2);  xk(2)+dt*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_data(i))];
% y_train(i) = xk(1);
% end
% y = y_train;

figure()
subplot(2,1,1)
title("VDP System")
plot(time,u_data)
ylabel("u_1")
xlabel("time in [s]")
subplot(2,1,2)
plot(time,y')
ylabel('x_1')
xlabel("time in [s]")

figure()
plot(x(1,:), x(2,:))

%u_data = [u_data1;u_data2].'
y_data = y;
%%
save('Datasets/u_data.mat',"u_data")
save('Datasets/y_dataMix.mat',"y_data")
save('Datasets/x_dataMix.mat', "x")
%%
% [coeff,score,latent,tsquared] = pca([u_data; y_data]')
% scatter(score(:,1),score(:,2))
% 
% xlabel('1st Principal Component')
% ylabel('2nd Principal Component')


% [Y,loss] = tsne([u_data; y_data]');
% figure;
% gscatter(Y(:,1),Y(:,2))