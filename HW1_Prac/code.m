%Amirreza Hatamipour
%97101507
%HW1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part zero
clc;
clear;
close all
load('Data_0/EEG.mat');
Fs_EEG = 512;
t_EEG = 0:1/Fs_EEG : 5000/Fs_EEG ;
figure()
plot(t_EEG(1:5000),eeg_signal)
title('EEG signal')
xlabel('time')
ylabel('abs')
%% part zero - Question 1 - section 1-a 
clc;
mean_EEG = mean(eeg_signal)
var_EEG = var(eeg_signal)
% Normalize the signal
eeg_signal_norm = eeg_signal -mean_EEG ;
eeg_signal_norm = eeg_signal_norm / sqrt(var(eeg_signal_norm));

mean_EEG_norm = mean(eeg_signal_norm)
var_EEG_norm = var(eeg_signal_norm)

figure()
subplot(1,2,1)
plot(t_EEG(1:5000),eeg_signal)
title('EEG signal')
xlabel('time')
ylabel('abs')

subplot(1,2,2)
plot(t_EEG(1:5000),eeg_signal_norm)
title('EEG signal norm')
xlabel('time')
ylabel('abs')
%% part zero - Question 1 - section 1-b
clc;
fft_EEG = fftshift(fft(eeg_signal_norm , Fs_EEG));
w = linspace(-Fs_EEG/2,Fs_EEG/2,512);
figure()
subplot(1,3,1)
plot(w, abs(fft_EEG))
grid on
title('EEG signal in ferquency domain')
xlabel('F(Hz)')
ylabel('abs(FFT)')

% Design a bandpass filter
bpFilt = designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',14,'HalfPowerFrequency2',30, ...
         'SampleRate',512);

eeg_signal_norm_BP_filtfilt = filtfilt(bpFilt,double(eeg_signal_norm'));
fft_EEG_BP_filtfilt = fftshift(fft(eeg_signal_norm_BP_filtfilt , Fs_EEG));
eeg_signal_norm_BP_filter = filter(bpFilt,eeg_signal_norm');
fft_EEG_BP_filter = fftshift(fft(eeg_signal_norm_BP_filter , Fs_EEG));
subplot(1,3,2)
plot(w, abs(fft_EEG_BP_filtfilt))
grid on
title('BP filtfilt')
xlabel('F(Hz)')
ylabel('abs(FFT)')
subplot(1,3,3)
plot(w, abs(fft_EEG_BP_filter))
grid on
title('BP filter')
xlabel('F(Hz)')
ylabel('abs(FFT)')

fvtool(bpFilt)

%% part zero - Question 2
clc;
clear;
close all;

load('Data_0/ECG.mat');
Fs_ECG = 128;
t_ECG = 0:1/Fs_ECG : 12800/Fs_ECG ;
figure()
plot(t_ECG(1:12800),ecg_singal)
title('ECG signal')
xlabel('time')
ylabel('abs')
%% part zero - Question 2 - section 2-a
% Design a bandpass filter

ecg_signal_BP_filtfilt = bandpass(ecg_singal,[5 12],Fs_ECG);
figure()
subplot(2,2,1)
plot( t_ECG(1:12800) , ecg_singal)
grid on
title( 'ECG signal BP')
xlabel('Time')
ylabel('abs')

subplot(2,2,2)
plot( t_ECG(1:12800) , ecg_singal)
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal BP')
xlabel('Time')
ylabel('abs')

subplot(2,2,3)
plot( t_ECG(1:12800) , ecg_signal_BP_filtfilt)
grid on
title( 'ECG signal BP')
xlabel('Time')
ylabel('abs')

subplot(2,2,4)
plot( t_ECG(1:12800) , ecg_signal_BP_filtfilt)
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal BP')
xlabel('Time')
ylabel('abs')
%% part zero - Question 2 - section 2-b
clc;

ecg_signal_BP_derivative  = diff(ecg_signal_BP_filtfilt);
figure()
subplot(2,2,1)
plot( t_ECG(1:12800) , ecg_signal_BP_filtfilt)
grid on
title( 'ECG signal BP')
xlabel('Time')
ylabel('abs')

subplot(2,2,2)
plot( t_ECG(1:12800) , ecg_signal_BP_filtfilt)
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal BP')
xlabel('Time')
ylabel('abs')

subplot(2,2,3)
plot( t_ECG(1:12800-1) , ecg_signal_BP_derivative)
grid on
title( 'ECG signal derivative ')
xlabel('Time')
ylabel('abs')

subplot(2,2,4)
plot( t_ECG(1:12800-1) , ecg_signal_BP_derivative)
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal derivative ')
xlabel('Time')
ylabel('abs')
%% part zero - Question 2 - section 2-c
clc;
ecg_signal_BP_derivative_pow2 = ecg_signal_BP_derivative.^2 ;
M = movmean(ecg_signal_BP_derivative_pow2,3);
figure()
subplot(2,2,1)
plot( t_ECG(1:12800-1) , ecg_signal_BP_derivative)
grid on
title( 'ECG signal derivative ')
xlabel('Time')
ylabel('abs')

subplot(2,2,2)
plot( t_ECG(1:12800-1) , ecg_signal_BP_derivative)
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal derivative ')
xlabel('Time')
ylabel('abs')

subplot(2,2,3)
plot( t_ECG(1:12800-1) , M)
grid on
title( 'ECG signal derivative after mean filter')
xlabel('Time')
ylabel('abs')

subplot(2,2,4)
plot( t_ECG(1:12800-1) , M)
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal derivative after mean filter ')
xlabel('Time')
ylabel('abs')

%% part zero - Question 2 - section 2-d
clc;
X_peak = find (M > 600);
Y_peak = M(X_peak);

figure()
subplot(2,2,1)
plot( t_ECG(1:12800-1) , M, t_ECG(X_peak),Y_peak,'or')
grid on
title( 'ECG signal derivative after mean filter')
xlabel('Time')
ylabel('abs')


subplot(2,2,2)
plot( t_ECG(1:12800-1) , M, t_ECG(X_peak),Y_peak,'or')
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal derivative after mean filter ')
xlabel('Time')
ylabel('abs')


Y_peak = ecg_singal(X_peak);
subplot(2,2,3)
plot(t_ECG(1:12800),ecg_singal, t_ECG(X_peak),Y_peak,'or')
grid on
title('ECG signal')
xlabel('time')
ylabel('abs')

subplot(2,2,4)
plot(t_ECG(1:12800),ecg_singal, t_ECG(X_peak),Y_peak,'or')
grid on
xlim([ t_ECG(1) t_ECG(800) ])
title('ECG signal')
xlabel('time')
ylabel('abs')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part one
clc;
clear;
close all
Fs = 100;
t = 0: 1/Fs : 10;
x = sin ( 2*pi*(2*t.^2+ 5*t)) ;

figure()
subplot(1,3,1)
plot( t, x)
grid on
title( 'signal')
xlabel('Time(s)')
ylabel('abs')

subplot(1,3,2)
plot( t, x)
xlim([ t(100) t(122) ])
grid on
title( 'signal')
xlabel('Time(s)')
ylabel('abs')

subplot(1,3,3)
plot( t, x)
xlim([ t(400) t(422) ])
grid on
title( 'signal')
xlabel('Time(s)')
ylabel('abs')
%% part one - Question 1 - section 1-b

X_fft = fftshift(fft(x , Fs));
w = linspace(-Fs/2,Fs/2,Fs);

figure()
subplot(1,2,1)
plot(w, abs(X_fft))
grid on
title('abs signal in ferquency domain')
xlabel('F(Hz)')
ylabel('abs(FFT)')

subplot(1,2,2)
plot(w, angle(X_fft))
grid on
title('angle signal in ferquency domain')
xlabel('F(Hz)')
ylabel('angle(FFT)')
%% part one - Question 1 -section 1-c
x_noise = x + sqrt(1) * randn(1,1001);

x_noise_fft = fftshift(fft(x_noise , Fs));

figure()
subplot(2,2,1)
plot(w, abs(X_fft))
grid on
title('abs signal in ferquency domain')
xlabel('F(Hz)')
ylabel('abs(FFT)')

subplot(2,2,2)
plot(w, angle(X_fft))
grid on
title('angle signal in ferquency domain')
xlabel('F(Hz)')
ylabel('angle(FFT)')

subplot(2,2,3)
plot(w, abs(x_noise_fft))
grid on
title('abs of noisy signal in ferquency domain')
xlabel('F(Hz)')
ylabel('abs(FFT)')

subplot(2,2,4)
plot(w, angle(x_noise_fft))
grid on
title('angle of noisy signal in ferquency domain')
xlabel('F(Hz)')
ylabel('angle(FFT)')

%% part one - Question 1 - section 1-d:
clc;
X_periodic = zeros(40,50);
for i= 1:39
    X_periodic(i,:) = x_noise((i-1)*25+1: (i-1)*25 + 50);
end
X_periodic(40,:) = x_noise(end-49: end);

freqrange_delta = [0.5 4] ;
freqrange_teta = [4 8] ;
freqrange_alpha = [8 13] ;
freqrange_beta = [13 30] ;

p_delta = bandpower(X_periodic,Fs,freqrange_delta);
p_teta = bandpower(X_periodic,Fs,freqrange_teta);
p_alpha = bandpower(X_periodic,Fs,freqrange_alpha);
p_beta = bandpower(X_periodic,Fs,freqrange_beta);

figure()
plot(p_delta)
hold on
plot(p_teta)
hold on
plot(p_alpha)
hold on
plot(p_beta)
grid on
title('Bandpower in different frequencies')
xlabel('time')
ylabel('PSD')
legend('delta', 'teta' ,'alpha' ,'beta')
%% part one - Question 1 - section 1-f:
clc;
clear;
close all;
Fs = 512;
load('Data_1/EEG_rest.mat');

EEG_rest_periodic = zeros(20,500);
for i= 1:19
    EEG_rest_periodic(i,:) = EEG_rest((i-1)*250+1: (i-1)*250 + 500);
end
EEG_rest_periodic(20,:) = EEG_rest_periodic(end-499: end);

freqrange_delta = [0.5 4] ;
freqrange_teta = [4 8] ;
freqrange_alpha = [8 13] ;
freqrange_beta = [13 30] ;

p_delta = bandpower(EEG_rest_periodic,Fs,freqrange_delta);
p_teta = bandpower(EEG_rest_periodic,Fs,freqrange_teta);
p_alpha = bandpower(EEG_rest_periodic,Fs,freqrange_alpha);
p_beta = bandpower(EEG_rest_periodic,Fs,freqrange_beta);

figure()
plot(p_delta)
hold on
plot(p_teta)
hold on
plot(p_alpha)
hold on
plot(p_beta)
grid on
title('Bandpower in different frequencies')
xlabel('time')
ylabel('PSD')
legend('delta', 'teta' ,'alpha' ,'beta')
%% part one - Question 2
clc;
clear;
close all;

load('Data_0/ECG.mat');
Fs_ECG = 128;
t_ECG = 0:1/Fs_ECG : 12800/Fs_ECG ;
% Design a bandpass filter

ecg_signal_BP_filtfilt = bandpass(ecg_singal,[5 12],Fs_ECG);
ecg_signal_BP_derivative  = diff(ecg_signal_BP_filtfilt);
ecg_signal_BP_derivative_pow2 = ecg_signal_BP_derivative.^2 ;
M = movmean(ecg_signal_BP_derivative_pow2,3);
X_peak = find (M > 600);
Y_peak = M(X_peak);

figure()
subplot(2,2,1)
plot( t_ECG(1:12800-1) , M, t_ECG(X_peak),Y_peak,'or')
grid on
title( 'ECG signal derivative after mean filter')
xlabel('Time')
ylabel('abs')


subplot(2,2,2)
plot( t_ECG(1:12800-1) , M, t_ECG(X_peak),Y_peak,'or')
xlim([ t_ECG(1) t_ECG(800) ])
grid on
title( 'ECG signal derivative after mean filter ')
xlabel('Time')
ylabel('abs')


Y_peak = ecg_singal(X_peak);
subplot(2,2,3)
plot(t_ECG(1:12800),ecg_singal, t_ECG(X_peak),Y_peak,'or')
grid on
title('ECG signal')
xlabel('time')
ylabel('abs')

subplot(2,2,4)
plot(t_ECG(1:12800),ecg_singal, t_ECG(X_peak),Y_peak,'or')
grid on
xlim([ t_ECG(1) t_ECG(800) ])
title('ECG signal')
xlabel('time')
ylabel('abs')

bpm= length(X_peak)/3 ;
bpm = bpm / (length(ecg_singal)/ (60*Fs_ECG))
%% part one - Question 2 -section 2-b
close all
pxx = pwelch(ecg_singal);
[pks,locs] = findpeaks(pxx);
pwelch(ecg_singal)

bpm2 = length(pks)/(4*length(ecg_singal)/ (60*Fs_ECG))
%% part one - Question 2 - section 2-c
clc; close all
n = length(ecg_singal)/(2 * Fs_ECG);
BPM = zeros(2,50);
for i=1:n
    signal = ecg_singal((i-1)*120+1:(i-1)*120+120);
    ecg_s = bandpass(signal,[5 12],Fs_ECG);
    ecg_s_derivative  = diff(ecg_s);
    ecg_s_derivative_pow2 = ecg_s_derivative.^2 ;
    M = movmean(ecg_s_derivative_pow2,3);
    X_peak = find (M > 600);
    BPM(1,i) = length(X_peak)*30 ; 
    
    pxx = pwelch(signal);
    [pks,locs] = findpeaks(pxx);
    BPM(2,i) = length(pks)*30;
end
mean(BPM,2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part Two - Question 1-b
clc;
clear;
close all
X = rand(7000,1) ;

figure()
subplot(3,1,1)
plot(X)
title('random signal')
xlabel('sample')
ylabel('amp')

subplot(3,1,2)
histogram(X)
title('histogram')
xlabel('bins')

subplot(3,1,3)
pwelch(X);

[X_acf,lags]  = xcorr(X);
figure()
subplot(1,3,1)
plot(lags,X_acf)
grid on
xlabel('Lag')
title('Auto correlation')

X_acf_fft = fftshift(fft(X_acf));

subplot(1,3,2)
plot( abs(X_acf_fft))
grid on
title('abs signal in ferquency domain')
xlabel('samples')
ylabel('abs(FFT)')

subplot(1,3,3)
plot( abs(X_acf_fft))
xlim([6950 7050])
grid on
title('abs signal in ferquency domain')
xlabel('samples')
ylabel('abs(FFT)')
%% part Two -  Question 1-b
clc;
clear;
close all
Xn = randn(7000,1) ;

figure()
subplot(3,1,1)
plot(Xn)
title('random signal')
xlabel('sample')
ylabel('amp')

subplot(3,1,2)
histogram(Xn)
title('histogram')
xlabel('bins')

subplot(3,1,3)
pwelch(Xn);

[Xn_acf,lags]  = xcorr(Xn);
figure()
subplot(1,2,1)
plot(lags,Xn_acf)
grid on
xlabel('Lag')
title('Auto correlation')

Xn_acf_fft = fftshift(fft(Xn_acf));

subplot(1,2,2)
plot( abs(Xn_acf_fft))
grid on
title('abs signal in ferquency domain')
xlabel('samples')
ylabel('abs(FFT)')


%% part Two -  Question 1-c
clc;
clear;
close all
mean_x = 1 ;
var_x = 2;
X = zeros(1,7000);
for i=1:7000
    a = rand(1);
    b = rand(1);
    X(i) = sqrt(var_x) * cos(2*pi*b) * sqrt(-log(1-a)) + mean_x;
end

figure()
subplot(3,1,1)
plot(X)
title('random signal')
xlabel('sample')
ylabel('amp')

subplot(3,1,2)
histogram(X)
title('histogram')
xlabel('bins')

subplot(3,1,3)
pwelch(X);

[X_acf,lags]  = xcorr(X);
figure()
subplot(1,3,1)
plot(lags,X_acf)
grid on
xlabel('Lag')
title('Auto correlation')

X_acf_fft = fftshift(fft(X_acf));

subplot(1,3,2)
plot( abs(X_acf_fft))
grid on
title('abs signal in ferquency domain')
xlabel('samples')
ylabel('abs(FFT)')

subplot(1,3,3)
plot( abs(X_acf_fft))
xlim([6950 7050])
grid on
title('abs signal in ferquency domain')
xlabel('samples')
ylabel('abs(FFT)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part Three - Question 1-a
clc;
clear;
close all
% Y = a*u[n]
a = 0.4 ;
n = 0:1:500 ;
A = (-a).^n ;
[X_acf,lags] = xcorr(A) ; 
figure()
subplot(1,2,1)
plot(n,A)
grid on
xlim([-1 80] )
ylim([-0.5 1.25])
title('signal')

subplot(1,2,2)
plot(lags,X_acf)
grid on
xlabel('Lag')
xlim([-70 70])
ylim([-0.5 1.5])
title('Auto correlation')

%% part Three - Question 1-b
clc;
a = 0.4 ;
n = 0:1:10000 ;
A = (-a).^n ;

N1 = 100 ;
N2 = 500 ;
N3 = 1000;
N4 = 5000;

M = N1;
R_xx_1 = zeros(1,2*M+1);
for m=0:M-1
    temp = A(1:M);
    first_arr = temp(1:M-1-m);
    second_arr = temp(m+1:M-1);
    R_xx_1(M+1+m) = sum(first_arr.*second_arr);
    R_xx_1(M+1-m) = R_xx_1(M+1+m);
end

M = N2;
R_xx_2 = zeros(1,2*M+1);
for m=0:M-1
    temp = A(1:M);
    first_arr = temp(1:M-1-m);
    second_arr = temp(m+1:M-1);
    R_xx_2(M+1+m) = sum(first_arr.*second_arr);
    R_xx_2(M+1-m) = R_xx_2(M+1+m);
end

M = N3;
R_xx_3 = zeros(1,2*M+1);
for m=0:M-1
    temp = A(1:M);
    first_arr = temp(1:M-1-m);
    second_arr = temp(m+1:M-1);
    R_xx_3(M+1+m) = sum(first_arr.*second_arr);
    R_xx_3(M+1-m) = R_xx_3(M+1+m);
end


M = N4;
R_xx_4 = zeros(1,2*M+1);
for m=0:M-1
    temp = A(1:M);
    first_arr = temp(1:M-1-m);
    second_arr = temp(m+1:M-1);
    R_xx_4(M+1+m) = sum(first_arr.*second_arr);
    R_xx_4(M+1-m) = R_xx_4(M+1+m);
end


figure()
subplot(2,2,1)
plot(lags,X_acf)
hold on
grid on
xlabel('Lag')
xlim([-50 50])
ylim([-0.5 1.5])
title('Auto correlation with 100 points')
n1 = -N1:1:N1;
scatter(n1,R_xx_1)
legend('Orginal','estimated')

subplot(2,2,2)
plot(lags,X_acf)
hold on
grid on
xlabel('Lag')
xlim([-50 50])
ylim([-0.5 1.5])
title('Auto correlation with 500 points')
n2 = -N2:1:N2;
scatter(n2,R_xx_2)
legend('Orginal','estimated')

subplot(2,2,3)
plot(lags,X_acf)
hold on
grid on
xlabel('Lag')
xlim([-50 50])
ylim([-0.5 1.5])
title('Auto correlation with 1000 points')
n3 = -N3:1:N3;
scatter(n3,R_xx_3)
legend('Orginal','estimated')

subplot(2,2,4)
plot(lags,X_acf)
hold on
grid on
xlabel('Lag')
xlim([-50 50])
ylim([-0.5 1.5])
title('Auto correlation with 5000 points')
n4 = -N4:1:N4;
scatter(n4,R_xx_4)
legend('Orginal','estimated')

%% part Three - Question 2-a
clc;
clear;
close all
load('Data_0/EEG.mat');
Fs_EEG = 512;
t_EEG = 0:1/Fs_EEG : 5000/Fs_EEG ;

[X_acf,lags] = xcorr(eeg_signal) ; 
figure()
plot(lags,X_acf)
grid on
xlabel('Lag')
title('Auto correlation of EEG signal')
%% part Three - Question 2-a
clc;
clear;
close all
load('Data_2/ECGPCG.mat');

figure()
subplot(2,2,1)
plot(ECG)
xlabel('sample')
grid on
title('ECG signal')

subplot(2,2,2)
plot(PCG)
xlabel('sample')
grid on
title('PCG signal')

subplot(2,2,3)
plot(ECG)
xlabel('sample')
xlim([0 12000])
grid on
title('ECG signal')

subplot(2,2,4)
plot(PCG)
xlabel('sample')
xlim([0 12000])
grid on
title('PCG signal')


[ECG_acf,ECG_lags] = xcorr(ECG) ; 
[PCG_acf,PCG_lags] = xcorr(PCG) ; 

figure()
subplot(1,2,1)
plot(ECG_lags,ECG_acf)
grid on
xlabel('Lag')
title('Auto correlation of ECG signal')

subplot(1,2,2)
plot(PCG_lags,PCG_acf)
grid on
xlabel('Lag')
title('Auto correlation of PCG signal')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part Four - Question 1-b
clc;
clear;
close all
load('Data_3/EEG_p300.mat');
signal = EEG_p300.data;
Fs = EEG_p300.srate;
t = 0 : 1/Fs : length(signal)/Fs;
markers_seq = EEG_p300.markers_seq;
markers_target = EEG_p300.markers_target;
markers_seq_loc = find(markers_seq) ;
markers_seq_clk = markers_seq(markers_seq_loc);
index_target = find(markers_target==1);
index_nontarget = find(markers_target==2);


title_plot = ['First target in index ', num2str(index_target(1))];  
plotEEG(signal( :, index_target(1)- floor(0.2*Fs): index_target(1) + Fs),title_plot)

title_plot = ['Second target in index', num2str(index_nontarget(2))];
plotEEG(signal( :, index_nontarget(2)- floor(0.2*Fs): index_nontarget(2) + Fs),title_plot)

%% part Four - Question 1-d
index_target_8 = index_target(find(markers_seq(index_target)== 8));
index_target_12 = index_target(find(markers_seq(index_target)== 12));
index_nontarget_12 = index_nontarget(find(markers_seq(index_nontarget)== 12));
index_nontarget_8 = index_nontarget(find(markers_seq(index_nontarget)== 8));


mean_target_12 = zeros(1,615);
for i= 1:length(index_target_12)
    mean_target_12 = mean_target_12 + signal( 31, index_target_12(i)- floor(0.2*Fs): index_target_12(i) + Fs);
end
mean_target_12 = mean_target_12 / length(index_target_12);

mean_nontarget_12 = zeros(1,615);
for i= 1:length(index_nontarget_12)
    mean_nontarget_12 = mean_nontarget_12 + signal( 31, index_nontarget_12(i)- floor(0.2*Fs): index_nontarget_12(i) + Fs);
end
mean_nontarget_12 = mean_nontarget_12 / length(index_nontarget_12);

mean_target_8 = zeros(1,615);
for i= 1:length(index_target_8)
    mean_target_8 = mean_target_12 + signal( 31, index_target_8(i)- floor(0.2*Fs): index_target_8(i) + Fs);
end
mean_target_8 = mean_target_8 / length(index_target_8);

mean_nontarget_8 = zeros(1,615);
for i= 1:length(index_nontarget_8)
    mean_nontarget_8 = mean_nontarget_8 + signal( 31, index_nontarget_8(i)- floor(0.2*Fs): index_nontarget_8(i) + Fs);
end
mean_nontarget_8 = mean_nontarget_8 / length(index_nontarget_8);

figure()
subplot(4,1,1)
plot(t(1:615) ,mean_target_12)
xlabel('Time')
grid on
title('Mean of target 12 in Fz channel')

subplot(4,1,2)
plot(t(1:615),mean_nontarget_12)
xlabel('Time')
grid on
title('Mean of nontarget 12 in Fz channel')

subplot(4,1,3)
plot(t(1:615) ,mean_target_8)
xlabel('Time')
grid on
title('Mean of target 8 in Fz channel')

subplot(4,1,4)
plot(t(1:615),mean_nontarget_8)
xlabel('Time')
grid on
title('Mean of nontarget 8 in Fz channel')
%% part Four - Question 1-f
org_target =  signal( 31, index_target_12(1)- floor(0.2*Fs): index_target_12(1) + Fs);

delay_target_12 = zeros(1,length(index_target_12));
for j= 1: length(index_target_12)
    R_xx = zeros(1,200);
    for i= -100:100
         R_xx(i+101) = sum (org_target .*  signal( 31, index_target_12(j)- floor(0.2*Fs)+i : index_target_12(j) + Fs+i),'all') ;
    end
delay_target_12(j+1) = find( R_xx == max(R_xx)) - 100;
end 

org_target =  signal( 31, index_target_8(1)- floor(0.2*Fs): index_target_8(1) + Fs);
delay_target_8 = zeros(1,length(index_target_8));
for j= 1: length(index_target_8)
    R_xx = zeros(1,200);
    for i= -100:100
         R_xx(i+101) = sum (org_target .*  signal( 31, index_target_8(j)- floor(0.2*Fs)+i : index_target_8(j) + Fs+i),'all') ;
    end
delay_target_8(j+1) = find( R_xx == max(R_xx)) - 100;
end 

org_target =  signal( 31, index_nontarget_12(1)- floor(0.2*Fs): index_nontarget_12(1) + Fs);
delay_nontarget_12 = zeros(1,length(index_nontarget_12));
for j= 1: length(index_nontarget_12)
    R_xx = zeros(1,200);
    for i= -100:100
         R_xx(i+101) = sum (org_target .*  signal( 31, index_nontarget_12(j)- floor(0.2*Fs)+i : index_nontarget_12(j) + Fs+i),'all') ;
    end
delay_nontarget_12(j+1) = find( R_xx == max(R_xx)) - 100;
end 

org_target =  signal( 31, index_nontarget_8(1)- floor(0.2*Fs): index_nontarget_8(1) + Fs);
delay_nontarget_8 = zeros(1,length(index_nontarget_8));
for j= 1: length(index_nontarget_8)
    R_xx = zeros(1,200);
    for i= -100:100
         R_xx(i+101) = sum (org_target .*  signal( 31, index_nontarget_8(j)- floor(0.2*Fs)+i : index_nontarget_8(j) + Fs+i),'all') ;
    end
delay_nontarget_8(j+1) = find( R_xx == max(R_xx)) - 100;
end 

mean_target_12 = zeros(1,615);
for i= 1:length(index_target_12)
    mean_target_12 = mean_target_12 + signal( 31, index_target_12(i)- floor(0.2*Fs)+ delay_target_12(i): index_target_12(i) + Fs+ delay_target_12(i));
end
mean_target_12 = mean_target_12 / length(index_target_12);

mean_nontarget_12 = zeros(1,615);
for i= 1:length(index_nontarget_12)
    mean_nontarget_12 = mean_nontarget_12 + signal( 31, index_nontarget_12(i)- floor(0.2*Fs)+ delay_nontarget_12(i): index_nontarget_12(i) + Fs+ delay_nontarget_12(i));
end
mean_nontarget_12 = mean_nontarget_12 / length(index_nontarget_12);

mean_target_8 = zeros(1,615);
for i= 1:length(index_target_8)
    mean_target_8 = mean_target_12 + signal( 31, index_target_8(i)- floor(0.2*Fs)+ delay_target_8(i): index_target_8(i) + Fs + delay_target_8(i));
end
mean_target_8 = mean_target_8 / length(index_target_8);

mean_nontarget_8 = zeros(1,615);
for i= 1:length(index_nontarget_8)
    mean_nontarget_8 = mean_nontarget_8 + signal( 31, index_nontarget_8(i)- floor(0.2*Fs)+delay_nontarget_8(i): index_nontarget_8(i) + Fs +delay_nontarget_8(i));
end
mean_nontarget_8 = mean_nontarget_8 / length(index_nontarget_8);

figure()
subplot(4,1,1)
plot(t(1:615) ,mean_target_12)
xlabel('Time')
grid on
title('Mean of target 12 in Fz channel')

subplot(4,1,2)
plot(t(1:615),mean_nontarget_12)
xlabel('Time')
grid on
title('Mean of nontarget 12 in Fz channel')

subplot(4,1,3)
plot(t(1:615) ,mean_target_8)
xlabel('Time')
grid on
title('Mean of target 8 in Fz channel')

subplot(4,1,4)
plot(t(1:615),mean_nontarget_8)
xlabel('Time')
grid on
title('Mean of nontarget 8 in Fz channel')
%% part Four - Question 1-d - all characters
clc; close all;
mean_target = zeros(1,615);
for i= 1:150
    mean_target = mean_target + signal( 31, index_target(i)- floor(0.2*Fs): index_target(i) + Fs);
end
mean_target = mean_target / 150;

mean_nontarget = zeros(1,615);
for i= 1:750
    mean_nontarget = mean_nontarget + signal( 31, index_nontarget(i)- floor(0.2*Fs): index_nontarget(i) + Fs);
end
mean_nontarget = mean_nontarget / 750;

figure()
subplot(2,1,1)
plot(t(1:615) ,mean_target)
xlabel('Time')
grid on
title('Mean of target all chracters in Fz channel')

subplot(2,1,2)
plot(t(1:615),mean_nontarget)
xlabel('Time')
grid on
title('Mean of nontarget all chracters in Fz channel')

%% part Four - Question 1-f  - all characters
clc; close all;
org_target =  signal( 31, index_target(1)- floor(0.2*Fs): index_target(1) + Fs);

delay_target = zeros(1,150);
for j= 1: 149
    R_xx = zeros(1,200);
    for i= -100:100
         R_xx(i+101) = sum (org_target .*  signal( 31, index_target(j)- floor(0.2*Fs)+i : index_target(j) + Fs+i),'all') ;
    end
delay_target(j+1) = find( R_xx == max(R_xx)) - 100;
end 

org_nontarget =  signal( 31, index_nontarget(1)- floor(0.2*Fs): index_nontarget(1) + Fs);
delay_nontarget = zeros(1,750);
for j= 1: 749
    R_xx = zeros(1,200);
    for i= -100:100
         R_xx(i+101) = sum (org_nontarget .*  signal( 31, index_nontarget(j)- floor(0.2*Fs)+i : index_nontarget(j) + Fs+i),'all') ;
    end
delay_nontarget(j+1) = find( R_xx == max(R_xx)) - 100;
end 

mean_target = zeros(1,615);
for i= 1:150
    mean_target = mean_target + signal( 31, index_target(i)- floor(0.2*Fs): index_target(i) + Fs);
end
mean_target = mean_target / 150;

mean_nontarget = zeros(1,615);
for i= 1:750
    mean_nontarget = mean_nontarget + signal( 31, index_nontarget(i)- floor(0.2*Fs): index_nontarget(i) + Fs);
end
mean_nontarget = mean_nontarget / 750;

figure()
subplot(2,1,1)
plot(t(1:615) ,mean_target)
xlabel('Time')
grid on
title('Mean of target all chracters in Fz channel')

subplot(2,1,2)
plot(t(1:615),mean_nontarget)
xlabel('Time')
grid on
title('Mean of nontarget all chracters in Fz channel')
