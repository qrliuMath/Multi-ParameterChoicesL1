%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%
%    Multi-parameter choices by Algorithm 1 for signal denoising (nonseparable)
%
%    Model: Signal denoising model by a biorthogonal wavelet transform
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear, clc,close all
disp('Signal denoising model by a biorthogonal wavelet transform ')
disp('----------------------------------------------')
format short
dbstop if error

%% Initialize signal 
Lev = 12;
N = 2^Lev;
x = linspace(0,1,N)';
signal = sqrt(x.*(1-x)).*sin(2*pi*1.05./(x + 0.05));   % original signal
 
%% Add noise
rng(1)
SNR = 80;
signal_noi = awgn(signal,SNR,'measured');    % add Gaussian white noise to signal

dwtmode('per')       % peroidic condition to extend signal 
paraFPPA.WaveName = 'bior2.2';
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
paraFPPA.num_group = num_group;

%% Generate matrix A
paraFPPA.RecLev = L;
RecMat = GenerateRecMat(paraFPPA);

%% Choose parameter in FPPA algorithm
paraFPPA.TargetTSLs = [64,64,124,237,295,389,427];     %  targeted sparsity levels
paraFPPA.TargetSL = sum(paraFPPA.TargetTSLs);
paraFPPA.MaxIter = 1000;
paraFPPA.rho = 0.01;
paraFPPA.alpha = 1/(norm(RecMat,2)^2)/paraFPPA.rho*0.99;  % convergence condition

%% Choose initial lambda
Initial_lambda = zeros(1,num_group);
Lambda_0 = abs(RecMat'*(signal_noi));
for k = 1:num_group
    Initial_lambda(k) = max(Lambda_0(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
end      
paraFPPA.lambda = Initial_lambda;

%% Iterative scheme choosing multiple regularization parameters
Sort_Gamma = cell(1,num_group);
Lambda = zeros(1,num_group);
a = zeros(1,num_group);
s = zeros(1,num_group);
Result.NUM = 0;
NumExp = 100;         % number of experiments
for LastTag = 0:1:NumExp
     fprintf('\n-------------------- Tag=%d -------------------\n',LastTag)
     [SOLUTION, ~] = NonOrthoWavelet_FPPA(signal_noi,RecMat,paraFPPA);
     Result.SLs = zeros(1,num_group);
     for k = 1:1:num_group
          Result.SLs(k) = nnz(SOLUTION(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
     end
     Result.SL = nnz(SOLUTION);
     Result.Ratio =  Result.SL/N*100;
     paraFPPA.Tag = LastTag;
     fname = sprintf('TargetSL%d_Tag%d-.mat',paraFPPA.TargetSL,paraFPPA.Tag);
    
     %% Show accuracy
     RecSignal = waverec(SOLUTION,L,paraFPPA.WaveName); 
     Result.MSE = sum((RecSignal - signal).^2)/N;
     disp('****** Results ******')
     fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
     fprintf('SLs = [%d, %d, %d, %d, %d, %d, %d]\n', Result.SLs)
     fprintf('Ratio = %.2f%% \n', Result.Ratio)
     fprintf('MSE = %.2e\n', Result.MSE)
     Result.NUM = Result.NUM + 1;
     save(fname,'SOLUTION','paraFPPA','Result','Sort_Gamma')
     epsilon = paraFPPA.num_group;
     if sum(abs(Result.SLs-paraFPPA.TargetTSLs))<=epsilon
         break;
     end
     for j = 1:paraFPPA.num_group
         if  Result.SLs(j) < paraFPPA.TargetTSLs(j)
             RHS = RecMat'*(RecMat*SOLUTION - signal_noi);
             Sort_Gamma{j} = sort(abs(RHS(group_info(j):group_info(j+1)-1)));
             Lambda(j) = Sort_Gamma{j}(L(j)-paraFPPA.TargetTSLs(j)+1);
             a(j) = max(Sort_Gamma{j}(Sort_Gamma{j}<paraFPPA.lambda(j)));
             paraFPPA.lambda(j) = min(Lambda(j),a(j));
         elseif Result.SLs(j) > paraFPPA.TargetTSLs(j)
             s(j) = 0;
             for i = 1:100
                 s(j) = s(j) + Result.SLs(j)- paraFPPA.TargetTSLs(j);
                 paraFPPA.lambda(j) = min(Sort_Gamma{j}(L(j)-paraFPPA.TargetTSLs(j)+1+s(j)),a(j));
                 [SOLUTION, ~] = NonOrthoWavelet_FPPA(signal_noi,RecMat,paraFPPA);
                 Result.SLs(j) = nnz(SOLUTION(paraFPPA.group_info(j):paraFPPA.group_info(j+1)-1));
                 Result.NUM = Result.NUM + 1;
                 fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
                 fprintf('SLs = [%d, %d, %d, %d, %d, %d, %d]\n', Result.SLs)
                 fname = sprintf('TargetSL%d_Tag%d-%d_%d.mat',paraFPPA.TargetSL,paraFPPA.Tag,j,i);
                 save(fname,'SOLUTION','paraFPPA','Result','Sort_Gamma')
                 if Result.SLs(j)<=paraFPPA.TargetTSLs(j)
                     break;
                 end
             end
         end
     end
end
  

     
  

