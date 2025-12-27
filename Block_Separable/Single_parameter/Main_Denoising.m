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
Lev = 18;
N = 2^Lev;
x = linspace(0,1,N)';
signal = sqrt(x.*(1-x)).*sin(2*pi*1.05./(x + 0.05)); 

%% Add noise
rng(1);
SNR = 20;
signal_noi = awgn(signal,SNR,'measured');   

%% Wavelet
dwtmode('per')     % peroidic condition to extend signal  
paraFPPA.WaveName = 'db6';
paraFPPA.DecLev = 9;     % decomposion level  (Lev-DecLev is the most coarst level)

%% Wavelet decomposition of the noisy signal
[Wx,L] = wavedec(signal_noi, paraFPPA.DecLev, paraFPPA.WaveName);  

%% Parameter choice
sort_gamma = sort(abs(Wx));
paraFPPA.TargetTSLs = 1100;         % targetted sparsity levels   
paraFPPA.lambda = sort_gamma(length(sort_gamma)-paraFPPA.TargetTSLs);

%% Numerical solution by FPPA
paraFPPA.rho = 0.01;
paraFPPA.alpha = 1/paraFPPA.rho*0.99;   
paraFPPA.beta = 0.0001;
paraFPPA.FGO_norm = sqrt(paraFPPA.alpha*(paraFPPA.rho+paraFPPA.beta)); %  convergence condition FGO_norm <1
paraFPPA.MaxIter = 2000;
signal_noi = gpuArray(signal_noi);
SOLUTION = Wavelet_FPPA(signal_noi,paraFPPA);
Result.SL = nnz(SOLUTION);
Result.Ratio = Result.SL/N*100;

%% Wavelet reconstruction
RecSignal = waverec(gather(SOLUTION),L,paraFPPA.WaveName);   

%% Error of the denoised signal
Result.MSE = sum((RecSignal - signal).^2)/N;
disp('****** Results ******')
fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
fprintf('SL = %d\n', Result.SL)
fprintf('Ratio = %.2f%% \n', Result.Ratio)
fprintf('MSE = %.4e\n', Result.MSE)
fname = sprintf('Result_SL-%d.mat',Result.SL);
save(fname, 'paraFPPA', 'SNR', 'Result')
