% HW3
% Amirreza Hatamipour
% 97101507
%% Section One
% Part 1-a
clc; clear; close all;
load('Q1/ECG.mat');
n = 0 : 1 : length(ecg_singal)-1;
A = 25;
F = 50;
Fs = 128;
M = 100;


X_n = A * sin( (F/(Fs/2))*pi * n + unifrnd(0,2*pi));
d = ecg_singal + X_n; 


figure()
subplot(1,3,1)
t1 = 0:1/Fs: length(ecg_singal)/Fs;
plot( t1(1:end-1) , ecg_singal)
%plot(ecg_singal)
%xlabel('sample(n)')
xlabel('time(s)')
grid on 
title('pure ECG signal')
xlim ([t1(4000) t1(4400)])
%xlim ([4000 4400])

subplot(1,3,2)
plot( t1(1:end-1) , X_n)
xlabel('time(s)')
grid on 
title('noise')
xlim ([t1(4000) t1(4400)])

subplot(1,3,3)
plot( t1(1:end-1) , ecg_singal)
hold on
plot( t1(1:end-1) ,d)
legend( 'pure ECG signal' , 'ECG noisy')
xlabel('time(s)')
grid on 
title('ECG noisy')
xlim ([t1(4000) t1(4400)])

% Adaptive filter
M= 200; 
A2 = 10;
X = A2 * sin( (F/(Fs/2))*pi * n + unifrnd(0,2*pi));
w0 = randn(M+1,1);
mu = 0.005;
[X2 , w_AF] = AdaptiveFilter(X,d,M,w0,mu);

RRMSE_before = RRMSE(ecg_singal,d)
RRMSE_after = RRMSE(ecg_singal,d - X2')
RRMSE_improvment = RRMSE_before - RRMSE_after



figure()
subplot(4,1,1)
t1 = 0:1/Fs: length(ecg_singal)/Fs;
plot( t1(1:end-1) , X)
%xlabel('sample(n)')
xlabel('time(s)')
grid on 
title('Primary signal')
xlim ([t1(4000) t1(4600)])

subplot(4,1,2)
plot( t1(1:end-1) , d)
xlabel('time(s)')
grid on 
title('Refrence signal')
xlim ([t1(4000) t1(4600)])

subplot(4,1,3)
plot( t1(1:end-1) , X2)
xlabel('time(s)')
grid on 
title('Estimated signal')
xlim ([t1(4000) t1(4600)])

subplot(4,1,4)
plot( t1(1:end-1) , d - X2')
xlabel('time(s)')
grid on 
title('Denoised signal')
xlim ([t1(4000) t1(4600)])

figure()
plot(t1(1:end-1) ,ecg_singal)
hold on
plot(t1(1:end-1) ,d - X2')
hold on
plot(t1(1:end-1) ,d)
xlim ([t1(4000) t1(4600)])
title('ECG signal in different stages')
xlabel('time(s)')
legend( 'pure ECG signal' , 'after denoising','ECG noisy')
grid on

%% Part 1-b
clear; clc; close all;
load('Q1/ECG.mat');
n = 0 : 1 : length(ecg_singal)-1;
Fs = 128;

figure()
subplot(3,1,1)
t1 = 0:1/Fs: length(ecg_singal)/Fs;
plot(t1(4663:4752) , ecg_singal(4663:4752))
title('one period')
xlabel('time(s)')
grid on

S = [];
for i = 1:21
    S = [S ecg_singal(4663:4752)];
end
shift = 90;
N = numel(S);


subplot(3,1,2)
t2 = 0:1/Fs: length(S)/Fs;
plot(t2(1:end-shift-1) , S(1,1:N-shift))
title('concatenate')
xlabel('time(s)')
grid on


X_n = 15*randn(1,N);
d = S + X_n ;

subplot(3,1,3)
plot(t2(1:end-shift-1) , S(1,1:N-shift))
hold on
plot(t2(1:end-shift-1) , d(1,1:N-shift) )
title('guassian noise')
xlabel('time(s)')
grid on
legend( 'pure ECG signal' ,'ECG noisy')

% Adaptive filter
S = S(1:end-shift);
M= 100; 


X = zeros(1,N-shift-1);
X =  d(shift+1:end);
d = d(1:end-shift);

w0 = randn(M+1,1);
mu = 0.05;
[X2 , w_AF] = AdaptiveFilter(X,d,M,w0,mu);

RRMSE_before = RRMSE(S,d)
RRMSE_after = RRMSE(S, X2')
RRMSE_improvment = RRMSE_before - RRMSE_after

figure()
subplot(5,1,1)
t1 = 0:1/Fs: length(S)/Fs;
plot( t1(1:end-1) , X)
xlabel('time(s)')
grid on 
title('Primary signal')
xlim ([t1(180) t1(630)])

subplot(5,1,2)
plot( t1(1:end-1) , d)
xlabel('time(s)')
grid on 
title('Refrence signal')
xlim ([t1(180) t1(630)])

subplot(5,1,3)
plot( t1(1:end-1) , X2)
xlabel('time(s)')
grid on 
title('Estimated signal')
xlim ([t1(180) t1(630)])

subplot(5,1,4)
plot( t1(1:end-1) , X_n(1,1:N-shift))
xlabel('time(s)')
grid on 
title('Guassian noise')
xlim ([t1(180) t1(630)])

subplot(5,1,5)
plot( t1(1:end-1) , d - X2')
xlabel('time(s)')
grid on 
title('Estimated noise after denoising: d-S_h')
xlim ([t1(180) t1(630)])

figure()
plot(t1(1:end-1) ,S)
hold on
plot(t1(1:end-1) , X2')
hold on
plot(t1(1:end-1) ,d)
xlim ([t1(180) t1(540)])
title('ECG signal in different stages')
xlabel('time(s)')
legend( 'pure ECG signal' , 'after denoising','ECG noisy')
grid on


%% Part 1-c
clear; clc; close all;
load('Q1/ECG.mat');
n = 0 : 1 : length(ecg_singal)-1;
Fs = 128;

figure()

shift = 90;
N = numel(ecg_singal);


subplot(2,1,1)
t2 = 0:1/Fs: length(ecg_singal)/Fs;
plot(t2(1:end-shift-1) , ecg_singal(1,1:N-shift))
title('Signal')
xlabel('time(s)')
grid on
xlim ([t2(4000) t2(4800)])


X_n = 15*randn(1,N);
d = ecg_singal + X_n ;

subplot(2,1,2)
plot(t2(1:end-shift-1) , ecg_singal(1,1:N-shift))
hold on
plot(t2(1:end-shift-1) , d(1,1:N-shift) )
title('guassian noise')
xlabel('time(s)')
grid on
legend( 'pure ECG signal' ,'ECG noisy')
xlim ([t2(4000) t2(4800)])

% Adaptive filter
ecg_singal = ecg_singal(1:end-shift);
M= 100; 


X = zeros(1,N-shift-1);
X =  d(shift+1:end);
d = d(1:end-shift);

w0 = randn(M+1,1);
mu = 0.05;
[X2 , w_AF] = AdaptiveFilter(X,d,M,w0,mu);

RRMSE_before = RRMSE(ecg_singal,d)
RRMSE_after = RRMSE(ecg_singal, X2')
RRMSE_improvment = RRMSE_before - RRMSE_after

figure()
subplot(5,1,1)
t1 = 0:1/Fs: length(ecg_singal)/Fs;
plot( t1(1:end-1) , X)
xlabel('time(s)')
grid on 
title('Primary signal')
xlim ([t2(4000) t2(4800)])

subplot(5,1,2)
plot( t1(1:end-1) , d)
xlabel('time(s)')
grid on 
title('Refrence signal')
xlim ([t2(4000) t2(4800)])

subplot(5,1,3)
plot( t1(1:end-1) , X2)
xlabel('time(s)')
grid on 
title('Estimated signal')
xlim ([t2(4000) t2(4800)])

subplot(5,1,4)
plot( t1(1:end-1) , X_n(1,1:N-shift))
xlabel('time(s)')
grid on 
title('Guassian noise')
xlim ([t2(4000) t2(4800)])

subplot(5,1,5)
plot( t1(1:end-1) , d - X2')
xlabel('time(s)')
grid on 
title('Estimated noise after denoising: d-S_h')
xlim ([t2(4000) t2(4800)])

figure()
plot(t1(1:end-1) ,ecg_singal)
hold on
plot(t1(1:end-1) , X2')
hold on
plot(t1(1:end-1) ,d)
xlim ([t2(4000) t2(4800)])
title('ECG signal in different stages')
xlabel('time(s)')
legend( 'pure ECG signal' , 'after denoising','ECG noisy')
grid on


%% ############## Section two ##############
clear; clc; close all;
load('Q2/Normal.mat');
load('Q2/PVC.mat');
load('Q2/PVC1.mat');
load('Q2/tinv.mat');

figure()
subplot(2,2,1)
plot(ecg_signal)
xlabel('Sample(n)')
grid on 
title('ECG Signal')

subplot(2,2,2)
plot(PVC)
xlabel('Sample(n)')
grid on 
title('PVC')

subplot(2,2,3)
plot(PVC1)
xlabel('Sample(n)')
grid on 
title('PVC1')

subplot(2,2,4)
plot(tinv)
xlabel('Sample(n)')
grid on 
title('tinv')
% min phase
% normal signal
[y_ecg,y_min_ecg] = rceps(ecg_signal);
[y_min_hat_ecg,nd_min_ecg] = cceps(y_min_ecg);
[y_hat_ecg,nd_ecg] = cceps(ecg_signal);
y_max_ecg = icceps(y_hat_ecg - y_min_hat_ecg,nd_ecg);
y_recon_ecg = conv(y_min_ecg,y_max_ecg);

figure()
subplot(4,3,1)
plot(ecg_signal)
xlabel('Sample(n)')
grid on 
title('ECG Signal')

subplot(4,3,2)
plot(y_min_ecg)
xlabel('Sample(n)')
grid on 
title('Min phase part of normal signal')

subplot(4,3,3)
plot(y_max_ecg)
xlabel('Sample(n)')
grid on 
title('Max phase part of normal signal')

%PVC
[y_PVC,y_min_PVC] = rceps(PVC);
[y_min_hat_PVC,nd_min_PVC] = cceps(y_min_PVC);
[y_hat_PVC,nd_PVC] = cceps(PVC);
y_max_PVC = icceps(y_hat_PVC - y_min_hat_PVC,nd_PVC);
y_recon_PVC = conv(y_min_PVC,y_max_PVC);


subplot(4,3,4)
plot(PVC)
xlabel('Sample(n)')
grid on 
title('PVC Signal')

subplot(4,3,5)
plot(y_min_PVC)
xlabel('Sample(n)')
grid on 
title('Min phase part of PVC signal')

subplot(4,3,6)
plot(y_max_PVC)
xlabel('Sample(n)')
grid on 
title('Max phase part of PVC signal')


%PVC1

[y_PVC1,y_min_PVC1] = rceps(PVC1);
[y_min_hat_PVC1,nd_min_PVC1] = cceps(y_min_PVC1);
[y_hat_PVC1,nd_PVC1] = cceps(PVC1);
y_max_PVC1 = icceps(y_hat_PVC1 - y_min_hat_PVC1 , nd_PVC1);
y_recon_PVC1 = conv(y_min_PVC1,y_max_PVC1);


subplot(4,3,7)
plot(PVC1)
xlabel('Sample(n)')
grid on 
title('PVC1 Signal')

subplot(4,3,8)
plot(y_min_PVC1)
xlabel('Sample(n)')
grid on 
title('Min phase part of PVC1 signal')

subplot(4,3,9)
plot(y_max_PVC1)
xlabel('Sample(n)')
grid on 
title('Max phase part of PVC1 signal')


%tinv

[y_tinv,y_min_tinv] = rceps(tinv);
[y_min_hat_tinv,nd_min_tinv] = cceps(y_min_tinv);
[y_hat_tinv,nd_tinv] = cceps(tinv);
y_max_tinv = icceps(y_hat_tinv - y_min_hat_tinv , nd_tinv);
y_recon_tinv = conv(y_min_tinv,y_max_tinv);


subplot(4,3,10)
plot(tinv)
xlabel('Sample(n)')
grid on 
title('tinv Signal')

subplot(4,3,11)
plot(y_min_tinv)
xlabel('Sample(n)')
grid on 
title('Min phase part of tinv signal')

subplot(4,3,12)
plot(y_max_tinv)
xlabel('Sample(n)')
grid on 
title('Max phase part of tinv signal')


figure()
subplot(4,2,1)
plot(ecg_signal)
xlabel('Sample(n)')
grid on 
title('orginal ECG Signal')

subplot(4,2,2)
plot(y_recon_ecg(1,1:length(ecg_signal)))
xlabel('Sample(n)')
grid on 
title('reconstruction ECG Signal: Xmin * Xmax')

subplot(4,2,3)
plot(PVC)
xlabel('Sample(n)')
grid on 
title('orginal PVC Signal')

subplot(4,2,4)
plot(y_recon_PVC(1,1:length(PVC)))
xlabel('Sample(n)')
grid on 
title('reconstruction PVC Signal: Xmin * Xmax')

subplot(4,2,5)
plot(PVC1)
xlabel('Sample(n)')
grid on 
title('orginal PVC1 Signal')

subplot(4,2,6)
plot(y_recon_PVC1(1,1:length(PVC1)))
xlabel('Sample(n)')
grid on 
title('reconstruction PVC1 Signal: Xmin * Xmax')

subplot(4,2,7)
plot(tinv)
xlabel('Sample(n)')
grid on 
title('orginal tinv Signal')

subplot(4,2,8)
plot(y_recon_tinv(1,1:length(tinv)))
xlabel('Sample(n)')
grid on 
title('reconstruction tinv Signal: Xmin * Xmax')

%% ############## Section three ##############
% part 3-a

clear; clc; close all;
load('Q3/train_recording.mat');
load('Q3/train_annotations.mat');

figure()
for i = 1:4
    subplot(2,2,i)
    plot(train_recordings{i})
    hold on
    plot(20*train_annotations{i,1},train_recordings{i}(20*train_annotations{i,1}),'r*')
    labels = {'S1'};
    text(20*train_annotations{i,1},train_recordings{i}(20*train_annotations{i,1}),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
    hold on
    plot(20*train_annotations{i,2},train_recordings{i}(20*train_annotations{i,2}),'g*')
    labels = {'S2'};
    text(20*train_annotations{i,2},train_recordings{i}(20*train_annotations{i,2}),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
    grid on
    xlim([0 2000])
    title( [ ' ECG',num2str(i)])
    xlabel('Sample(n)')

end
%% part 3-b
load('Q3/PCG_Features_all.mat');
load('Q3/PCG_states_all.mat');
PCG_Features = PCG_Features_all;
PCG_states = PCG_states_all;
%% part 3-c
% class 1 vs class 2,3,4
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states == 1);
class2_indices = setdiff(all_class,class1_indices)';

% fisher
for i=1:4
    u1 = mean(PCG_Features(class1_indices,i)) ;
    S1 = var(PCG_Features(class1_indices,i));
    u2 = mean(PCG_Features(class2_indices,i)) ;
    S2 = var(PCG_Features(class2_indices,i));
    Sw = S1+S2 ;
    u0 = mean(PCG_Features(:,i)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    J(i) = Sb/Sw ;
end
J

subplot(3,2,1)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
legend('class1','class2,3,4')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('Feature 1 & 2')

subplot(3,2,2)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,3),'g*')
legend('class1','class2,3,4')
xlabel('Feature 1')
ylabel('Feature 3')
grid on 
title('Feature 1 & 3')

subplot(3,2,3)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,4),'g*')
legend('class1','class2,3,4')
xlabel('Feature 1')
ylabel('Feature 4')
grid on 
title('Feature 1 & 4')

subplot(3,2,4)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,3),'g*')
legend('class1','class2,3,4')
xlabel('Feature 2')
ylabel('Feature 3')
grid on 
title('Feature 2 & 3')

subplot(3,2,5)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,4),'g*')
legend('class1','class2,3,4')
xlabel('Feature 2')
ylabel('Feature 4')
grid on 
title('Feature 2 & 4')

subplot(3,2,6)
plot(PCG_Features(class1_indices,3),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,3),PCG_Features(class2_indices,4),'g*')
legend('class1','class2,3,4')
xlabel('Feature 3')
ylabel('Feature 4')
grid on 
title('Feature 3 & 4')
sgtitle('class 1 v.s. class 2,3,4')

% classifier 
mx1_1 = mean(PCG_Features(class1_indices,1)); %mean_feature1_class1
my1_1 = mean(PCG_Features(class1_indices,2)); %mean_feature2_class1

mx2 = mean(PCG_Features(class2_indices,1)); %mean_feature1_class2
my2 = mean(PCG_Features(class2_indices,2)); %mean_feature2_class2

figure()
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('class 1 v.s. class 2,3,4  , Feature 1 & 2')
hold on
plot([mx1_1 mx2],[my1_1 my2],'s',...
    'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
hold on
labels = {'mu_1','mu_2'};
text([mx1_1 mx2],[my1_1 my2] ,labels,...
    'VerticalAlignment','bottom','HorizontalAlignment','right')
a = (my2 - my1_1) / (mx2 - mx1_1);
a_1 = - 1/a;
b_1 = (my2 + my1_1) /2 - a_1 * (mx2 + mx1_1) /2 ;
x = -2:0.001:5;
hold on
plot( x, a_1*x+b_1,'--','LineWidth' ,2)
ylim([min(PCG_Features(:,2)) max(PCG_Features(:,2))])
legend('class1','class2,3,4','mean of classes','classifier')

%% class 2 vs class 1,3,4
clc; close all
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states==2);
class2_indices = setdiff(all_class,class1_indices)';

% fisher
for i=1:4
    u1 = mean(PCG_Features(class1_indices,i)) ;
    S1 = var(PCG_Features(class1_indices,i));
    u2 = mean(PCG_Features(class2_indices,i)) ;
    S2 = var(PCG_Features(class2_indices,i));
    Sw = S1+S2 ;
    u0 = mean(PCG_Features(:,i)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    J(i) = Sb/Sw ;
end
J

subplot(3,2,1)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
legend('class2','class1,3,4')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('Feature 1 & 2')

subplot(3,2,2)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,3),'g*')
legend('class2','class1,3,4')
xlabel('Feature 1')
ylabel('Feature 3')
grid on 
title('Feature 1 & 3')

subplot(3,2,3)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,4),'g*')
legend('class2','class1,3,4')
xlabel('Feature 1')
ylabel('Feature 4')
grid on 
title('Feature 1 & 4')

subplot(3,2,4)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,3),'g*')
legend('class2','class1,3,4')
xlabel('Feature 2')
ylabel('Feature 3')
grid on 
title('Feature 2 & 3')

subplot(3,2,5)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,4),'g*')
legend('class2','class1,3,4')
xlabel('Feature 2')
ylabel('Feature 4')
grid on 
title('Feature 2 & 4')

subplot(3,2,6)
plot(PCG_Features(class1_indices,3),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,3),PCG_Features(class2_indices,4),'g*')
legend('class2','class1,3,4')
xlabel('Feature 3')
ylabel('Feature 4')
grid on 
title('Feature 3 & 4')
sgtitle('class 2 v.s. class 1,3,4')
%
mx1_2 = mean(PCG_Features(class1_indices,1)); %mean_feature1_class1
my1_2 = mean(PCG_Features(class1_indices,2)); %mean_feature2_class1

mx2 = mean(PCG_Features(class2_indices,1)); %mean_feature1_class2
my2 = mean(PCG_Features(class2_indices,2)); %mean_feature2_class2

figure()
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('class 2 v.s. class 1,3,4  , Feature 1 & 2')
hold on
plot([mx1_2 mx2],[my1_2 my2],'s',...
    'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
hold on
labels = {'mu_1','mu_2'};
text([mx1_2 mx2],[my1_2 my2] ,labels,...
    'VerticalAlignment','bottom','HorizontalAlignment','right')
a = (my2 - my1_2) / (mx2 - mx1_2);
a_2 = - 1/a;
b_2 = (my2 + my1_2) /2 - a_2 * (mx2 + mx1_2) /2 ;
x = -2:0.001:5;
hold on
plot( x, a_2*x+b_2,'--','LineWidth' ,2)
ylim([min(PCG_Features(:,2)) max(PCG_Features(:,2))])
legend('class2','class 1,3,4','mean of classes','classifier')

%% class 3 vs class 1,2,4
clc; close all
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states==3);
class2_indices = setdiff(all_class,class1_indices)';

% fisher
for i=1:4
    u1 = mean(PCG_Features(class1_indices,i)) ;
    S1 = var(PCG_Features(class1_indices,i));
    u2 = mean(PCG_Features(class2_indices,i)) ;
    S2 = var(PCG_Features(class2_indices,i));
    Sw = S1+S2 ;
    u0 = mean(PCG_Features(:,i)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    J(i) = Sb/Sw ;
end
J

subplot(3,2,1)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
legend('class 3','class 1,2,4')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('Feature 1 & 2')

subplot(3,2,2)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,3),'g*')
legend('class 3','class 1,2,4')
xlabel('Feature 1')
ylabel('Feature 3')
grid on 
title('Feature 1 & 3')

subplot(3,2,3)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,4),'g*')
legend('class 3','class 1,2,4')
xlabel('Feature 1')
ylabel('Feature 4')
grid on 
title('Feature 1 & 4')

subplot(3,2,4)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,3),'g*')
legend('class 3','class 1,2,4')
xlabel('Feature 2')
ylabel('Feature 3')
grid on 
title('Feature 2 & 3')

subplot(3,2,5)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,4),'g*')
legend('class 3','class 1,2,4')
xlabel('Feature 2')
ylabel('Feature 4')
grid on 
title('Feature 2 & 4')

subplot(3,2,6)
plot(PCG_Features(class1_indices,3),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,3),PCG_Features(class2_indices,4),'g*')
legend('class 3','class 1,2,4')
xlabel('Feature 3')
ylabel('Feature 4')
grid on 
title('Feature 3 & 4')
sgtitle('class 3 v.s. class 1,2,4')
%
mx1_3 = mean(PCG_Features(class1_indices,1)); %mean_feature1_class1
my1_3 = mean(PCG_Features(class1_indices,2)); %mean_feature2_class1

mx2 = mean(PCG_Features(class2_indices,1)); %mean_feature1_class2
my2 = mean(PCG_Features(class2_indices,2)); %mean_feature2_class2

figure()
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('class 3 v.s. class 1,2,4  , Feature 1 & 2')
hold on
plot([mx1_3 mx2],[my1_3 my2],'s',...
    'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
hold on
labels = {'mu_1','mu_2'};
text([mx1_3 mx2],[my1_3 my2] ,labels,...
    'VerticalAlignment','bottom','HorizontalAlignment','right')
a = (my2 - my1_3) / (mx2 - mx1_3);
a_3 = - 1/a;
b_3 = (my2 + my1_3) /2 - a_3 * (mx2 + mx1_3) /2 ;
x = -2:0.001:5;
hold on
plot( x, a_3*x+b_3,'--','LineWidth' ,2)
ylim([min(PCG_Features(:,2)) max(PCG_Features(:,2))])
legend('class3','class 1,2,4','mean of classes','classifier')
%% class 4 vs class 1,2,3
clc; close all
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states==4);
class2_indices = setdiff(all_class,class1_indices)';

% fisher
for i=1:4
    u1 = mean(PCG_Features(class1_indices,i)) ;
    S1 = var(PCG_Features(class1_indices,i));
    u2 = mean(PCG_Features(class2_indices,i)) ;
    S2 = var(PCG_Features(class2_indices,i));
    Sw = S1+S2 ;
    u0 = mean(PCG_Features(:,i)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    J(i) = Sb/Sw ;
end
J

subplot(3,2,1)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
legend('class 4','class 1,2,3')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('Feature 1 & 2')

subplot(3,2,2)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,3),'g*')
legend('class 4','class 1,2,3')
xlabel('Feature 1')
ylabel('Feature 3')
grid on 
title('Feature 1 & 3')

subplot(3,2,3)
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,4),'g*')
legend('class 4','class 1,2,3')
xlabel('Feature 1')
ylabel('Feature 4')
grid on 
title('Feature 1 & 4')

subplot(3,2,4)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,3),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,3),'g*')
legend('class 4','class 1,2,3')
xlabel('Feature 2')
ylabel('Feature 3')
grid on 
title('Feature 2 & 3')

subplot(3,2,5)
plot(PCG_Features(class1_indices,2),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,2),PCG_Features(class2_indices,4),'g*')
legend('class 4','class 1,2,3')
xlabel('Feature 2')
ylabel('Feature 4')
grid on 
title('Feature 2 & 4')

subplot(3,2,6)
plot(PCG_Features(class1_indices,3),PCG_Features(class1_indices,4),'r*')
hold on 
plot(PCG_Features(class2_indices,3),PCG_Features(class2_indices,4),'g*')
legend('class 4','class 1,2,3')
xlabel('Feature 3')
ylabel('Feature 4')
grid on 
title('Feature 3 & 4')
sgtitle('class 4 v.s. class 1,2,3')
%
mx1_4 = mean(PCG_Features(class1_indices,1)); %mean_feature1_class1
my1_4 = mean(PCG_Features(class1_indices,2)); %mean_feature2_class1

mx2 = mean(PCG_Features(class2_indices,1)); %mean_feature1_class2
my2 = mean(PCG_Features(class2_indices,2)); %mean_feature2_class2

figure()
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
title('class 4 v.s. class 1,2,3  , Feature 1 & 2')
hold on
plot([mx1_4 mx2],[my1_4 my2],'s',...
    'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
hold on
labels = {'mu_1','mu_2'};
text([mx1_4 mx2],[my1_4 my2] ,labels,...
    'VerticalAlignment','bottom','HorizontalAlignment','right')
a = (my2 - my1_4) / (mx2 - mx1_4);
a_4 = - 1/a;
b_4 = (my2 + my1_4) /2 - a_4 * (mx2 + mx1_4) /2 ;
x = -2:0.001:5;
hold on
plot( x, a_4*x+b_4,'--','LineWidth' ,2)
ylim([min(PCG_Features(:,2)) max(PCG_Features(:,2))])
legend('class 4','class 1,2,3','mean of classes','classifier')
%% conclusion
clc; close all
class1_indices = find(PCG_states==1);
class2_indices = find(PCG_states==2);
class3_indices = find(PCG_states==3);
class4_indices = find(PCG_states==4);

figure()
plot(PCG_Features(class1_indices,1),PCG_Features(class1_indices,2),'r*')
hold on 
plot(PCG_Features(class2_indices,1),PCG_Features(class2_indices,2),'g*')
hold on 
plot(PCG_Features(class3_indices,1),PCG_Features(class3_indices,2),'c*')
hold on 
plot(PCG_Features(class4_indices,1),PCG_Features(class4_indices,2),'y*')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
hold on
plot([mx1_1 mx1_2 mx1_3 mx1_4],[my1_1 my1_2 my1_3 my1_4],'s',...
    'MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
hold on
labels = {'mu_1','mu_2','mu_3','mu_4'};
text([mx1_1 mx1_2 mx1_3 mx1_4],[my1_1 my1_2 my1_3 my1_4] ,labels,...
    'VerticalAlignment','bottom','HorizontalAlignment','right')

x = -2:0.001:5;
hold on
plot( x, a_1*x+b_1,'--','LineWidth' ,2)
text(-1,3,'\leftarrow class1')
hold on
plot( x, a_2*x+b_2,'--','LineWidth' ,2)
text(-0.9,0.77,'\leftarrow class2')

hold on
plot( x, a_3*x+b_3,'--','LineWidth' ,2)
text(-0.9,2.278,'\leftarrow class3')
hold on
plot( x, a_4*x+b_4,'--','LineWidth' ,2)
text(-0.7,1,'\leftarrow class4')

ylim([min(PCG_Features(:,2)) max(PCG_Features(:,2))])
legend('class 1','class 2','class 3','class 4')
title('Train data')

%% part 3-d
load('Q3/PCG_Features_test.mat');
load('Q3/PCG_states_test.mat');

indices_test = zeros(length(PCG_states_test),4);
class1_indices_test = find(PCG_states_test==1);
class2_indices_test = find(PCG_states_test==2);
class3_indices_test = find(PCG_states_test==3);
class4_indices_test = find(PCG_states_test==4);
indices_test(class1_indices_test,1) = 1;
indices_test(class2_indices_test,2) = 1;
indices_test(class3_indices_test,3) = 1;
indices_test(class4_indices_test,4) = 1;

figure()
plot(PCG_Features(class1_indices_test,1),PCG_Features(class1_indices_test,2),'r*')
hold on 
plot(PCG_Features(class2_indices_test,1),PCG_Features(class2_indices_test,2),'g*')
hold on 
plot(PCG_Features(class3_indices_test,1),PCG_Features(class3_indices_test,2),'c*')
hold on 
plot(PCG_Features(class4_indices_test,1),PCG_Features(class4_indices_test,2),'y*')
xlabel('Feature 1')
ylabel('Feature 2')
grid on 
hold on

x = -2:0.001:5;
hold on
plot( x, a_1*x+b_1,'--','LineWidth' ,2)
text(-1,3,'\leftarrow class1')
hold on
plot( x, a_2*x+b_2,'--','LineWidth' ,2)
text(-0.9,0.77,'\leftarrow class2')

hold on
plot( x, a_3*x+b_3,'--','LineWidth' ,2)
text(-0.9,2.278,'\leftarrow class3')
hold on
plot( x, a_4*x+b_4,'--','LineWidth' ,2)
text(-0.7,1,'\leftarrow class4')

ylim([min(PCG_Features(:,2)) max(PCG_Features(:,2))])
legend('class 1','class 2','class 3','class 4')
title('Test data')
% Estimate the class of data
est_class = zeros(length(PCG_states_test),4);
for i = 1 : length(PCG_states_test)
     if a_1 * PCG_Features_test(i,1) + b_1 < PCG_Features_test(i,2)
         % class1 
         d1 = sqrt( (PCG_Features_test(i,1)- mx1_1)^2 + (PCG_Features_test(i,2)- my1_1)^2);
         d3 = sqrt( (PCG_Features_test(i,1)- mx1_3)^2 + (PCG_Features_test(i,2)- my1_3)^2);
         if d1 > d3
             est_class(i,1) = 1;
         else 
             est_class(i,3) = 1;
         end
     else
         d2 = sqrt( (PCG_Features_test(i,1)- mx1_2)^2 + (PCG_Features_test(i,2)- my1_2)^2);
         d4 = sqrt( (PCG_Features_test(i,1)- mx1_4)^2 + (PCG_Features_test(i,2)- my1_4)^2);
         if d2 > d4
             est_class(i,2) = 1;
         else 
             est_class(i,4) = 1;
         end
     end
    
    
end
plotconfusion(indices_test',est_class')

% plot test data
load('Q3/test_recording.mat');

i=1;
figure()
plot(test_recordings{i})
hold on
plot(20*class1_indices_test,test_recordings{i}(20*class1_indices_test),'r*')
labels = {'S1'};
text(20*class1_indices_test,test_recordings{i}(20*class1_indices_test),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
hold on
plot(20*class2_indices_test,test_recordings{i}(20*class2_indices_test),'r*')
labels = {'Systole'};
text(20*class2_indices_test,test_recordings{i}(20*class2_indices_test),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
hold on
plot(20*class3_indices_test,test_recordings{i}(20*class3_indices_test),'r*')
labels = {'S2'};
text(20*class3_indices_test,test_recordings{i}(20*class3_indices_test),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
hold on
plot(20*class4_indices_test,test_recordings{i}(20*class4_indices_test),'r*')
labels = {'diastole'};
text(20*class4_indices_test,test_recordings{i}(20*class4_indices_test),labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
grid on
xlim([1000 1500])
title('Test ECG')
xlabel('Sample(n)')

%% ############## Section Four ##############
% part 4-a
clc; clear; close all;
load('Q3/PCG_Features_all.mat');
load('Q3/PCG_states_all.mat');
PCG_Features = PCG_Features_all;
PCG_states = PCG_states_all;
Transition_Matrix = zeros(4,4);
j = PCG_states(1,1);
for i = 2:length(PCG_states)
    Transition_Matrix(j,PCG_states(i,1)) = Transition_Matrix(j,PCG_states(i,1)) + 1;
    j = PCG_states(i,1);
end
for i = 1:4
    Transition_Matrix(i,:) = Transition_Matrix(i,:)/ sum(Transition_Matrix(i,:),'all');
end
%% part 4-b mnrfit
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states == 1);
Y = 2*ones(length(PCG_states),1);
Y(class1_indices,1) = 1;
B_1 = mnrfit(PCG_Features_all,Y);
% class2
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states == 2);
Y = 2*ones(length(PCG_states),1);
Y(class1_indices,1) = 1;
B_2 = mnrfit(PCG_Features_all,Y);
% class3
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states == 3);
Y = 2*ones(length(PCG_states),1);
Y(class1_indices,1) = 1;
B_3 = mnrfit(PCG_Features_all,Y);
% class4
all_class = 1:1:length(PCG_states);
class1_indices = find(PCG_states == 4);
Y = 2*ones(length(PCG_states),1);
Y(class1_indices,1) = 1;
B_4 = mnrfit(PCG_Features_all,Y);
% part 4-b mnrfit
load('Q3/PCG_Features_test.mat');
load('Q3/PCG_states_test.mat');
pihat_1 = mnrval(B_1,PCG_Features_test);
pihat_2 = mnrval(B_2,PCG_Features_test);
pihat_3 = mnrval(B_3,PCG_Features_test);
pihat_4 = mnrval(B_4,PCG_Features_test);
pihat = [pihat_1(:,1) pihat_2(:,1) pihat_3(:,1) pihat_4(:,1)];

%% part 4-b viterbi
% step 0
T_1 = zeros(length(PCG_Features_test) ,4);
for i = 1:4
    T_1(1,i) =  pihat(1,i);    
end

% step i
for i = 2:length(PCG_Features_test)
    for j = 1:4
        t1 =  T_1(i-1,1)* Transition_Matrix(1,j)*pihat(i,j);
        t2 =  T_1(i-1,2)* Transition_Matrix(2,j)*pihat(i,j);
        t3 =  T_1(i-1,3)* Transition_Matrix(3,j)*pihat(i,j);
        t4 =  T_1(i-1,4)* Transition_Matrix(4,j)*pihat(i,j);
        T_1(i,j) = max([t1 t2 t3 t4]);
    
    end
end

[M,est_state] = max(T_1,[],2);
%
acc = length(find(PCG_states_test == est_state))/length(PCG_states_test)

indices_test = zeros(length(PCG_states_test),4);
class1_indices_test = find(PCG_states_test==1);
class2_indices_test = find(PCG_states_test==2);
class3_indices_test = find(PCG_states_test==3);
class4_indices_test = find(PCG_states_test==4);
indices_test(class1_indices_test,1) = 1;
indices_test(class2_indices_test,2) = 1;
indices_test(class3_indices_test,3) = 1;
indices_test(class4_indices_test,4) = 1;
est_state_4 = zeros(length(PCG_states_test),4);
class1_indices_test = find(est_state==1);
class2_indices_test = find(est_state==2);
class3_indices_test = find(est_state==3);
class4_indices_test = find(est_state==4);
est_state_4(class1_indices_test,1) = 1;
est_state_4(class2_indices_test,2) = 1;
est_state_4(class3_indices_test,3) = 1;
est_state_4(class4_indices_test,4) = 1;
plotconfusion(indices_test',est_state_4')

%% Function
function output = RRMSE(X_org,X_den)

output=sqrt(sum((X_org-X_den).^2,'all'))/sqrt(sum(X_org.^2,'all'));

end


function [X2 , w] = AdaptiveFilter(X,d,M,w0,mu)
    N = numel(d);
    error = 1;
    w = w0;
    while error > 0.0000001 
        w_n = w;
        for i = M+1:N
            X_v = X(1, i: -1 : i - M )';
            delta =  (d(i) - w' * X_v)/(X_v' * X_v);
            w = w + 2 * mu * delta * X_v;
        end
        error = sum((w - w_n),'all');
    end
    X2 = zeros(N,1);
    for i = M+1:N
        X2(i) =  w' * X(1, i: -1 : i - M )';
    end

end