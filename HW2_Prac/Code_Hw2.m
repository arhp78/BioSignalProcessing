% HW2
% Amirreza Hatamipour
% 97101507
%% Part One
% part 1-c 
% signal 1
clc; close all;
L = 10000;
X_1 = randn(1,L);

% Periodogram method
S1_perio = periodogram_calculator(X_1, 'rec') ;
% BT method
S_BT = BT_calculator(X_1, 'rec', length(X_1)) ;
% Real spectrum
S1_real = pwelch(X_1);
S1_real = [flip(S1_real) ;S1_real];
% Welch method
S_welch = welch_calculator(X_1, 'rec', 100 , 50);


figure()
subplot(2,2,1)
w = linspace(-1,1,length(S1_real));
plot(w*pi, 10*log10(S1_real),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Real Spectrum of x[n]')
grid on

subplot(2,2,2)
w2 = linspace(-1,1,length(S1_perio));
plot(w2*pi,10*log10(S1_perio),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by Periodogram method')
grid on

subplot(2,2,3)
w3 = linspace(-1,1,length(S_BT));
plot(w3*pi,10*log10(S_BT),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by BT method')
grid on


subplot(2,2,4)
w3 = linspace(-1,1,length(S_welch));
plot(w3*pi,10*log10(S_welch),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by welch method')
grid on


%%  part 1-c 
%signal 2
clc; close all;
L = 10000;
n = 0:L-1;
thetha_1 = unifrnd(0,2*pi,1,1);
thetha_2 = unifrnd(0,2*pi,1,1);
thetha_3 = unifrnd(0,2*pi,1,1);
f1 = 0.05; f2 = 0.40; f3 = 0.45; 
X_2 = 2*cos(f1*2*pi*n + thetha_1) + 2*cos(f2*2*pi*n + thetha_2) + 2*cos(f3*2*pi*n + thetha_3);

% Periodogram method
S2_perio = periodogram_calculator(X_2, 'rec') ;
% BT method
S_BT = BT_calculator(X_2, 'rec', length(X_2)) ;
% Real spectrum
S2_real = pwelch(X_2);
S2_real = [flip(S2_real) ;S2_real];
% Welch method
S_welch = welch_calculator(X_2, 'rec', 100 , 50);

figure()
subplot(2,2,1)
w = linspace(-1,1,length(S2_real));
plot(w*pi, 10*log10(S2_real),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Real Spectrum of x[n]')
grid on

subplot(2,2,2)
w2 = linspace(-1,1,length(S2_perio));
plot(w2*pi,10*log10(S2_perio),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by Periodogram method')
grid on

subplot(2,2,3)
w3 = linspace(-1,1,length(S_BT));
plot(w3*pi,10*log10(abs(S_BT)),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by BT method')
grid on


subplot(2,2,4)
w3 = linspace(-1,1,length(S_welch));
plot(w3*pi,10*log10(S_welch),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by welch method')
grid on

%%  part 1-c 
%signal 3 AR(2)
clc; close all;
L = 2000;
N = randn(1,L);
b = 1;
a = [1 -0.5 0.4];

X_3 = filter(b,a,N);


% Periodogram method
S3_perio = periodogram_calculator(X_3, 'rec') ;

% BT method
S_BT = BT_calculator(X_3, 'rec', length(X_3)) ;
% Real spectrum
S3_real = pwelch(X_3);
S3_real = [flip(S3_real) ;S3_real];
% Welch method
S_welch = welch_calculator(X_3, 'rec', 100 , 50);

figure()
subplot(2,2,1)
w = linspace(-1,1,length(S3_real));
plot(w*pi, 10*log10(S3_real),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Real Spectrum of x[n]')
grid on

subplot(2,2,2)
w2 = linspace(-1,1,length(S3_perio));
plot(w2*pi,10*log10(S3_perio),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by Periodogram method')
grid on

subplot(2,2,3)
w3 = linspace(-1,1,length(S_BT));
plot(w3*pi,10*log10(abs(S_BT)),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by BT method')
grid on


subplot(2,2,4)
w3 = linspace(-1,1,length(S_welch));
plot(w3*pi,10*log10(S_welch),'k')
xlabel('w')
ylabel('10*log(Power)')
title('Spectrum of x[n] by welch method')
grid on
%% Part one
% section d
clc; clear; close all
load('Q_1/PCG_1.mat');
load('Q_1/PCG_2.mat');
load('Q_1/PCG_3.mat');
load('Q_1/PCG_4.mat');
Fs = 2000;
fpass = 800;
X1 = lowpass(PCG_1,fpass,Fs);
X2 = lowpass(PCG_2,fpass,Fs);
X3 = lowpass(PCG_3,fpass,Fs);
X4 = lowpass(PCG_4,fpass,Fs);

figure()
subplot(2,2,1)
t1 = 0:1/Fs: length( PCG_1)/Fs;
plot( t1(1:end-1), PCG_1)
xlabel('time(s)')
grid on 
title('signal 1')

subplot(2,2,2)
t1 = 0:1/Fs: length( PCG_2)/Fs;
plot( t1(1:end-1), PCG_2)
xlabel('time(s)')
grid on 
title('signal 2')

subplot(2,2,3)
t1 = 0:1/Fs: length( PCG_3)/Fs;
plot( t1(1:end-1), PCG_3)
xlabel('time(s)')
grid on 
title('signal 3')


subplot(2,2,4)
t1 = 0:1/Fs: length( PCG_4)/Fs;
plot( t1(1:end-1), PCG_4)
xlabel('time(s)')
grid on 
title('signal 4')

X = zeros(20,2000);
Y = zeros(20,1);

for i= 1:5
   X((i-1)*4+1,:) = X1( (i-1)*7000+1: (i-1)*7000+2000,1)';
   X((i-1)*4+2,:) = X2( (i-1)*7000+1: (i-1)*7000+2000,1)';
   X((i-1)*4+3,:) = X3( 1, (i-1)*7000+1: (i-1)*7000+2000);
   X((i-1)*4+4,:) = X4( 1, (i-1)*7000+1: (i-1)*7000+2000);
   Y((i-1)*4+1,1) = 1;
   Y((i-1)*4+2,1) = 1;
   Y((i-1)*4+3,1) = 0;
   Y((i-1)*4+4,1) = 0;
end
r = randi([1 20],1,4);
r_all = 1:1:20;
X_test =  X( r , : );
Y_test =  Y( r , : );
X_train = X(setdiff(r_all,r),:);
Y_train = Y(setdiff(r_all,r),:);


figure()
subplot(2,2,1)
t1 = 0:1/Fs: 1;
plot( t1(1:end-1), X_test(1,:))
xlabel('time(s)')
grid on 
title(['signal 1, label:',num2str(Y_test(1))])

subplot(2,2,2)
plot( t1(1:end-1), X_test(2,:))
xlabel('time(s)')
grid on 
title(['signal 2, label:',num2str(Y_test(2))])

subplot(2,2,3)
plot( t1(1:end-1), X_test(3,:))
xlabel('time(s)')
grid on 
title(['signal 3, label:',num2str(Y_test(3))])

subplot(2,2,4)
plot( t1(1:end-1), X_test(4,:))
xlabel('time(s)')
grid on 
title(['signal 4, label:',num2str(Y_test(4))])

figure()
for i = 1:16
    subplot(4,4,i)
    [s,w] = pwelch( X_train(i,:));
    plot(w,10*log10(s),'k')
    xlabel('w')
    title( [ ' label:',num2str(Y_train(i))])
    grid on

end

for i = 1:4
    [s,w] = pwelch( X_test(i,:));
    if mean(10*log10(s(195:210)))- mean(10*log10(s(212:227)))<20
        sprintf('test %d: estimate:0, real, %d', i, Y_test(i))
    else
       sprintf('test %d: estimate:1, real, %d', i, Y_test(i))   
    end
end
%% Part one
% section e
clc; clear; close all
[y1,Fs1] = audioread('Extra/a0014.wav');
[y2,Fs2] = audioread('Extra/a0007.wav');
[y3,Fs3] = audioread('Extra/a0008.wav');
[y4,Fs4] = audioread('Extra/a0009.wav');
[y5,Fs5] = audioread('Extra/a0010.wav');
[y6,Fs6] = audioread('Extra/a0011.wav');
[y7,Fs7] = audioread('Extra/a0013.wav');
[y8,Fs8] = audioread('Extra/a0012.wav');


subplot(4,2,1)
t1 = 0:1/Fs1: length(y1)/Fs1;
plot( t1(1:end-1), y1)
xlabel('time(s)')
grid on 
title('normal')

subplot(4,2,2)
t1 = 0:1/Fs1: length(y2)/Fs1;
plot( t1(1:end-1), y2)
xlabel('time(s)')
grid on 
title('abnormal')

subplot(4,2,3)
t1 = 0:1/Fs1: length(y3)/Fs1;
plot( t1(1:end-1), y3)
xlabel('time(s)')
grid on 
title('normal')

subplot(4,2,4)
t1 = 0:1/Fs1: length(y4)/Fs1;
plot( t1(1:end-1), y4)
xlabel('time(s)')
grid on 
title('abnormal')

subplot(4,2,5)
t1 = 0:1/Fs1: length(y5)/Fs1;
plot( t1(1:end-1), y5)
xlabel('time(s)')
grid on 
title('normal')

subplot(4,2,6)
t1 = 0:1/Fs1: length(y6)/Fs1;
plot( t1(1:end-1), y6)
xlabel('time(s)')
grid on 
title('abnormal')

subplot(4,2,7)
t1 = 0:1/Fs1: length(y7)/Fs1;
plot( t1(1:end-1), y7)
xlabel('time(s)')
grid on 
title('normal')

subplot(4,2,8)
t1 = 0:1/Fs1: length(y8)/Fs1;
plot( t1(1:end-1), y8)
xlabel('time(s)')
grid on 
title('abnormal')



X = zeros(24,2000);
Y = zeros(24,1);

for i= 1:3
   X((i-1)*8+1,:) = y1( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+2,:) = y2( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+3,:) = y3( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+4,:) = y4( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+5,:) = y5( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+6,:) = y6( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+7,:) = y7( (i-1)*2000+1: (i-1)*2000+2000,1)';
   X((i-1)*8+8,:) = y8( (i-1)*2000+1: (i-1)*2000+2000,1)';
   
   Y((i-1)*8+1,1) = 1;
   Y((i-1)*8+2,1) = 0;
   Y((i-1)*8+3,1) = 1;
   Y((i-1)*8+4,1) = 0;
   Y((i-1)*8+5,1) = 1;
   Y((i-1)*8+6,1) = 0;
   Y((i-1)*8+7,1) = 1;
   Y((i-1)*8+8,1) = 0;
end
r = randi([1 24],1,4);
r_all = 1:1:24;
X_test =  X( r , : );
Y_test =  Y( r , : );
X_train = X(setdiff(r_all,r),:);
Y_train = Y(setdiff(r_all,r),:);


figure()
subplot(2,2,1)
t1 = 0:1/Fs1: 1;
plot( t1(1:end-1), X_test(1,:))
xlabel('time(s)')
grid on 
title(['signal 1, label:',num2str(Y_test(1))])

subplot(2,2,2)
plot( t1(1:end-1), X_test(2,:))
xlabel('time(s)')
grid on 
title(['signal 2, label:',num2str(Y_test(2))])

subplot(2,2,3)
plot( t1(1:end-1), X_test(3,:))
xlabel('time(s)')
grid on 
title(['signal 3, label:',num2str(Y_test(3))])

subplot(2,2,4)
plot( t1(1:end-1), X_test(4,:))
xlabel('time(s)')
grid on 
title(['signal 4, label:',num2str(Y_test(4))])

figure()
for i = 1:20
    subplot(4,5,i)
    [s,w] = pwelch( X_train(i,:));
    plot(w,10*log10(s),'k')
    xlabel('w')
    title( [ ' label:',num2str(Y_train(i))])
    grid on

end

for i = 1:4
    [s,w] = pwelch( X_test(i,:));
    if mean(10*log10(s(195:210))) > mean(10*log10(s(130:140)))
        sprintf('test %d: estimate:1, real, %d', i, Y_test(i))
    else
       sprintf('test %d: estimate:0, real, %d', i, Y_test(i))   
    end
end
%% Part two
% Part 2-d
% signal AR
clear; clc; close all;
L = 1000;
N = randn(1,L);
b = 1;
a = [1 -1.352 1.338 -0.662 0.24];
X_1 = filter(b,a,N);
[A, p]= AR_estimator(X_1)
% compare result
b = A;
a = 1;
N_est = filter(b,a,X_1);

figure()
subplot(1,2,1)
n = -length(N)/2 :1:length(N)/2;
plot(n(1,1:end-1), N )
hold on
plot(n(1,1:end-1), N_est )
xlabel('n')
title('noise')
grid on
legend( 'white noise' , 'estimated white noise')

subplot(1,2,2)
R_n = xcorr(N,N)/L;
R_nest = xcorr(N_est,N_est)/L;
n = -length(R_n)/2 :1:length(R_n)/2;
plot( n(1,1:end-1), R_n )
hold on
plot( n(1,1:end-1), R_nest )
xlabel('n')
title('Autocorrelation')
grid on
legend( 'white noise' , 'estimated white noise')

[b_est, q_est]= MA_estimator(X_1)
%% signal MA
clear; clc; close all;
L = 1000;
N = randn(1,L);
b = [1 -2.76 3.809 -2.654 0.924];
a = 1;
X_2 = filter(b,a,N);
[b_est, q_est]= MA_estimator(X_2)

b = 1;
a = b_est;
N_est = filter(b,a,X_2);


figure()
subplot(1,2,1)
n = -length(N)/2 :1:length(N)/2;
plot(n(1,1:end-1), N )
hold on
plot(n(1,1:end-1), N_est )
xlabel('n')
title('noise')
grid on
legend( 'white noise' , 'estimated white noise')

subplot(1,2,2)
R_n = xcorr(N,N)/L;
R_nest = xcorr(N_est,N_est)/L;
n = -length(R_n)/2 :1:length(R_n)/2;
plot( n(1,1:end-1), R_n )
hold on
plot( n(1,1:end-1), R_nest )
xlabel('n')
title('Autocorrelation')
grid on
legend( 'white noise' , 'estimated white noise')
[a_est, p_est] = AR_estimator(X_2)
%% ARMA 
clear; clc; close all;
L = 1000;
N = randn(1,L);
b = [1 -0.2 0.04];
a = [1 -1.352 1.338 -0.662 0.24];
X_3 = filter(b,a,N);

[a_est, p_est]= AR_estimator(X_3)
% compare result
b = a_est;
a = 1;
v = filter(b,a,X_3);
[b_est, q_est]= MA_estimator(v)


b = a_est;
a = b_est;
N_est = filter(b,a,X_3);


figure()
subplot(1,2,1)
n = -length(N)/2 :1:length(N)/2;
plot(n(1,1:end-1), N )
hold on
plot(n(1,1:end-1), N_est )
xlabel('n')
title('noise')
grid on
legend( 'white noise' , 'estimated white noise')

subplot(1,2,2)
R_n = xcorr(N,N)/L;
R_nest = xcorr(N_est,N_est)/L;
n = -length(R_n)/2 :1:length(R_n)/2;
plot( n(1,1:end-1), R_n )
hold on
plot( n(1,1:end-1), R_nest )
xlabel('n')
title('Autocorrelation')
grid on
legend( 'white noise' , 'estimated white noise')


%% Part two
% section d
clc; clear; close all
load('Q_1/PCG_1.mat');
load('Q_1/PCG_3.mat');
[y1,Fs1] = audioread('Q_2/crowd-talking-138493.mp3');
[y2,Fs2] = audioread('Q_2/female-babble-45052.mp3');
Fs = 2000;
fpass = 800;
X1 = lowpass(PCG_1,fpass,Fs);
X2 = lowpass(PCG_3,fpass,Fs);

X1_1 = X1' + 0.004*randn(1, length(X1));
X1_2 = X1' + 0.4 * y1(1:length(X1),1)';
X1_3 = X1' + 0.4 * y2(1:length(X1),1)';


X2_1 = X2 + 0.04*randn(1, length(X2));
X2_2 = X2 + 0.8 * y1(1:length(X2),1)';
X2_3 = X2(1,1:length(y2)) + 0.8 * y2(:,1)';

figure()
subplot(3,2,1)
plot( X1 )
hold on
plot( X1_1 )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy')
xlim ([4000 4200])

subplot(3,2,3)
plot( X1 )
hold on
plot( X1_2 )
xlabel('n')
title('crowd noise')
grid on
legend( 'pure' , 'noisy')
xlim ([4000 4200])


subplot(3,2,5)
plot( X1 )
hold on
plot( X1_3 )
xlabel('n')
title('female talking noise')
grid on
legend( 'pure' , 'noisy')
xlim ([4000 4200])


subplot(3,2,2)
plot( X2 )
hold on
plot( X2_1 )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy')
xlim ([4000 4200])

subplot(3,2,4)
plot( X2 )
hold on
plot( X2_2 )
xlabel('n')
title('crowd noise')
grid on
legend( 'pure' , 'noisy')
xlim ([4000 4200])


subplot(3,2,6)
plot( X2 )
hold on
plot( X2_3 )
xlabel('n')
title('female talking noise')
grid on
legend( 'pure' , 'noisy')
xlim ([4000 4200])
%% denoising
close all;
clc;
[a_est, p_est]= AR_estimator(X1_1);
b = a_est;
a = 1;
v = filter(b,a,X1_1);
RRMSE_X1_1_de = RRMSE(X1',X1_1-0.5*v)
RRMSE_X1_1 = RRMSE(X1',X1_1)

figure()
subplot(3,2,1)
plot( X1 )
hold on
plot( X1_1 )
hold on
plot(  X1_1-0.5*v )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy', 'denoise')
xlim ([4000 4200])

[a_est, p_est]= AR_estimator(X1_2);
b = a_est;
a = 1;
v = filter(b,a,X1_2);
RRMSE_X1_2_de = RRMSE(X1',X1_2-0.5*v)
RRMSE_X1_2 = RRMSE(X1',X1_2)

subplot(3,2,3)
plot( X1 )
hold on
plot( X1_2 )
hold on
plot(  X1_2-0.5*v )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy', 'denoise')
xlim ([4000 4200])

[a_est, p_est]= AR_estimator(X1_3);
b = a_est;
a = 1;
v = filter(b,a,X1_3);
RRMSE_X1_3_de = RRMSE(X1',X1_3-0.5*v)
RRMSE_X1_3 = RRMSE(X1',X1_3)

subplot(3,2,5)
plot( X1 )
hold on
plot( X1_3 )
hold on
plot(  X1_3-0.5*v )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy', 'denoise')
xlim ([4000 4200])


[a_est, p_est]= AR_estimator(X2_1);
b = a_est;
a = 1;
v = filter(b,a,X2_1);
RRMSE_X2_1_de = RRMSE(X2,X2_1-0.5*v)
RRMSE_X2_1 = RRMSE(X2,X2_1)

subplot(3,2,2)
plot( X2 )
hold on
plot( X2_1 )
hold on
plot(  X2_1-0.5*v )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy', 'denoise')
xlim ([4000 4200])

[a_est, p_est]= AR_estimator(X2_2);
b = a_est;
a = 1;
v = filter(b,a,X2_2);
RRMSE_X2_2_de = RRMSE(X2,X2_2-0.5*v)
RRMSE_X2_2 = RRMSE(X2,X2_2)

subplot(3,2,4)
plot( X2 )
hold on
plot( X2_2 )
hold on
plot(  X2_2-0.5*v )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy', 'denoise')
xlim ([4000 4200])

[a_est, p_est]= AR_estimator(X2_3)
b = a_est;
a = 1;
v = filter(b,a,X2_3);
RRMSE_X2_3_de = RRMSE(X2(1,1:length(y2)),X2_3-0.5*v)
RRMSE_X2_3 = RRMSE(X2(1,1:length(y2)),X2_3)

subplot(3,2,6)
plot( X2 )
hold on
plot( X2_3 )
hold on
plot(  X2_3-0.5*v )
xlabel('n')
title('guassian noise')
grid on
legend( 'pure' , 'noisy', 'denoise')
xlim ([4000 4200])



%% function
function S =  periodogram_calculator(X, win) 
    L = length(X);
    if win == 'rec'
       W = ones(1,L);
    else 
       W = window(win,L);
    end
    X_w = X .* W;
    X_w_fft = fft(X_w);
    S = (1/L) * X_w_fft.^2;
end
function S =  BT_calculator(X, win, L) 
    if win == 'rec'
       W = ones(1,L);
    else 
       W = window(win,L);
    end
    R = xcorr(X(1:L).*W,X(1:L).*W)/L;
    S = abs(fftshift(fft(R,2*L+1)))/(L);
end

function S =  welch_calculator(X, win, L , corres) 
    N = length(X);
    S = 0;
    if win == 'rec'
       W = ones(1,L);
    else 
       W = window(win,L);
    end
    for i= 1: floor(N/corres)-1
        X_w = X(1+(i-1)*(L-corres): 100+(i-1)*(L-corres)) .* W;
        X_w_fft = fft(X_w);
        S_i = (1/L) * X_w_fft.^2;
        S = S + S_i;
    end
    S = S / floor(N/corres);
end

function [A, p]= AR_estimator(X)
% Estimate degree of AR model
L = length(X);
E = zeros(1,L);
k = zeros(1,L);
% AR model with Levinson-Durbin
r = xcorr(X,X)/L;
R0 = r(L);
R = r(L+1:end);
% step 0
E0 = R0;
% step 1
k(1) = - R(1)/E0;
a(1,1) = k(1);
E(1) = (1-k(1)^2) * E0;
m = 2;
%step m
while(E(m-1))> 1
        temp = 0;
        for j=1:m-1
            temp = temp + a(j,m-1) * R(m-j);
        end
        k(m) = -(R(m)+temp)/E(m-1);
        a(m,m) = k(m);
        for j=1:m-1
            a(j,m) = a(j,m-1) + k(m) * a(m-j,m-1);
        end
        E(m) = (1-k(m)^2) * E(m-1);
        m = m + 1;
    end
    
p = m-1;
A = [1 , a(:,p)'];



end

function  [b_est, q_est]= MA_estimator(X)
% Estimate degree of model
L = length(X);
r = xcorr(X,X)/L;
R = r(L:end);
B = zeros(11);
Var = zeros(1,11);
for q=1:10
    bs(1,1:q+1) = abs(randn(1,q+1));
    m = 2;
 
    while 1
         bs(m , 1) =   sqrt(R(1) - sum(bs(m-1,2:q+1).^2));   %b0
        for i = 2:q+1
           bs(m , i) = (1/bs(m-1 , 1)) * (R(i) - sum( bs(m-1,2:q+1-i).*bs(m-1,2+i:q+1)));    
        end
       if m > 50000
           break
       end
       if sum(( bs(m,:) - bs(m-1,:)).^2) < 0.1
           break
       end
       m = m + 1; 
    end
    b = 1;
    a = bs(m,:);
    N_est = filter(b,a,X);
    Var(q) = var(N_est);
    B( q ,1:q+1)  = bs(m,:);
     
    
    
end
w = abs(Var-1);
q_est = find( w == min(abs(Var-1)));
b_est = B( q_est ,1:q_est+1);



end

function output = RRMSE(X_org,X_den)

output=sqrt(sum((X_org-X_den).^2,'all'))/sqrt(sum(X_org.^2,'all'));

end