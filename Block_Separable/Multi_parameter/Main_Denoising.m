%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Multi-parameter choices for signal denoising (block separable)
%
%     Model:Signal Denoising Model by an Orthogonal Wavelet Transform
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc,close all;
disp('-----------------------------------------------------------------')
disp('Signal Denoising Model by Multi-parameter Regularization')
disp('-----------------------------------------------------------------')

%% Initialize signal 
Lev = 12;
N = 2^Lev;
x = linspace(0,1,N)';
signal = sqrt(x.*(1-x)).*sin(2*pi*1.05./(x + 0.05)); 

%% Add noise
rng(1);
SNR = 80;
signal_noi = awgn(signal,SNR,'measured');   

%% Wavelet
dwtmode('per')     % peroidic condition to extend signal  
paraFPPA.WaveName = 'db6';
paraFPPA.DecLev = 6;     % decomposion level  (Lev-DecLev is the most coarst level)

%% Wavelet decomposition of the noisy signal
[Wx,L] = wavedec(signal_noi,paraFPPA.DecLev,paraFPPA.WaveName);   

%% Wavelet coefficient grouping
num_group = length(L) - 1;
group_info = zeros(num_group,1);   % starting index of each group
group_info(1) = 1;
for j = 2:1:num_group
    group_info(j) = sum(L(1:j-1)) + 1;
end
group_info = [group_info;L(end)+1];  % length(group_info) = num_group + 1;
paraFPPA.group_info = group_info;

%% Parameter choice stategy 
paraFPPA.TargetTSLs = [64,51,68,86,108,116,107];
paraFPPA.TargetSL = sum(paraFPPA.TargetTSLs);
Lambda = zeros(size(paraFPPA.TargetTSLs));
for i = 1:num_group
    gamma = Wx(group_info(i):group_info(i+1)-1);
    sort_gamma = sort(abs(gamma));
    if L(i)-paraFPPA.TargetTSLs(i) > 0
        lambda = sort_gamma(L(i)-paraFPPA.TargetTSLs(i));
    else
        sigma = 0.001;
        lambda = sigma*min(sort_gamma);
    end
    Lambda(i) = lambda;
end   
paraFPPA.lambda = Lambda;

%% Numerical solution by FPPA
paraFPPA.rho = 0.01;
paraFPPA.alpha = 1/paraFPPA.rho*0.99;    % convergence condition
paraFPPA.MaxIter = 1000;
[SOLUTION, ~] = Wavelet_FPPA(signal_noi,paraFPPA);

Result.SLs = zeros(num_group,1);
for j=1:1:num_group
    Result.SLs(j) = nnz(SOLUTION(paraFPPA.group_info(j):paraFPPA.group_info(j+1)-1));
end
Result.SL = nnz(SOLUTION);
Result.Ratio = Result.SL/N*100;

%% Wavelet reconstruction
RecSignal = waverec(SOLUTION,L,paraFPPA.WaveName);     % reconstructed signal 

%% Error of the denoised signal
Result.MSE = sum((RecSignal - signal).^2)/N;
disp('****** Results ******')
fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
fprintf('SLs = [%d, %d, %d, %d, %d, %d, %d]\n', Result.SLs)
fprintf('Ratio = %.2f%% \n', Result.Ratio)
fprintf('MSE = %.2e\n', Result.MSE)
fname = sprintf('Result_SL-%d.mat',Result.SL);
save(fname,'sigma','paraFPPA','SNR','Result')
