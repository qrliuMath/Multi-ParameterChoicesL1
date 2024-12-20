%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    Signal-parameter choices for signal denoising (block separable)
%         
%    Model: Signal Denoising Model by an Orthogonal Wavelet Transform
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, close all
disp('Signal denoising model by Single-parameter Regularization')
disp('----------------------------------------------')

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
[Wx,L] = wavedec(signal_noi, paraFPPA.DecLev, paraFPPA.WaveName);  

%% Parameter choice
sort_gamma = sort(abs(Wx));
paraFPPA.TargetTSLs = 1000;         % targetted sparsity levels   
paraFPPA.lambda = sort_gamma(length(sort_gamma)-paraFPPA.TargetTSLs);

%% Numerical solution by FPPA
paraFPPA.rho = 0.02;
paraFPPA.alpha = 1/paraFPPA.rho*0.99;   %  convergence condition
paraFPPA.MaxIter = 1000;
[SOLUTION, ~] = Wavelet_FPPA(signal_noi,paraFPPA);
Result.SL = nnz(SOLUTION);
Result.Ratio = Result.SL/N*100;

%% Wavelet reconstruction
RecSignal = waverec(SOLUTION,L,paraFPPA.WaveName);   

%% Error of the denoised signal
Result.MSE = sum((RecSignal - signal).^2)/N;
disp('****** Results ******')
fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
fprintf('SL = %d\n', Result.SL)
fprintf('Ratio = %.2f%% \n', Result.Ratio)
fprintf('MSE = %.2e\n', Result.MSE)
fname = sprintf('Result_SL-%d.mat',Result.SL);
save(fname, 'paraFPPA', 'SNR', 'Result')
