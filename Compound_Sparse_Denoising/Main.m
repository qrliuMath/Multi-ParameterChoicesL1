%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%
%    Multi-parameter choices by Algorithm 2 for compound sparse denoising model 
%
%    Model: Compound Sparse Denoising
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
clear,clc,close all
disp('Compound Sparse Denoising')
disp('----------------------------------------------')
format short
dbstop if error

%% Initialize signal 
N = 300;                                       % N : length of signal
t = (1:N)';
f = sin(0.021*pi*t);                          % f : smooth function
x = 2*(t < 0.3*N) + 1*(t > 0.6*N);            % x : step function
g = f + x;                                   %  g : total signal

%% Add noise
sigma = 0.3;                             % sigma : standard deviation of noise
rng(1)                                   % set random numbers so example is reproducible
noise = sigma*randn(N, 1);               % noise : white zero-mean Gaussian 
y = g + noise;                           % y : noisy data

%% Define high-pass filter H
% The high-pass filter is H = A\B£¬where A and B are banded matrices.
% Set filter parameters (fc, d) 
d = 2;            % d : filter order parameter (d = 1, 2, or 3)
fc = 0.022;       % fc : cut-off frequency (cycles/sample) (0 < fc < 0.5);
[A,B] = ABfilt(d,fc,N);
H = gpuArray(A\B);
c_H = gpuArray(H*y);

%% Group
num_group = 2;
group_info = [1;N+1;2*N];    % starting index of each group£»
paraFPPA.group_info = group_info;
paraFPPA.num_group = num_group;
L = [N,N-1];

%-------------------------------------------------------
Generate_B_and_B_prime
%-------------------------------------------------------

%% Choose parameter in FPPA algorithm
paraFPPA.TargetTSLs = [80,60];     %  targeted sparsity levels
paraFPPA.TargetSSL = sum(paraFPPA.TargetTSLs);
paraFPPA.MaxIter = 1000;
paraFPPA.alpha = 0.1;    
paraFPPA.beta = 4;
paraFPPA.rho = 4;
B_prime = gpuArray(B_prime);
FGO = [sqrt(paraFPPA.alpha*paraFPPA.rho)*B_prime;sqrt(paraFPPA.alpha*paraFPPA.beta)*eye(2*N-1)];
paraFPPA.FGO_norm = norm(gather(FGO),2); %  convergence condition FGO_norm <1

%% Choose initial lambda
tic;
Initial_lambda = [0.5,1.0];     
paraFPPA.lambda = gpuArray(Initial_lambda); 

%% Iterative scheme choosing multiple regularization parameters 
H_prime = gpuArray(H'*H+paraFPPA.rho*eye(size(H'*H)));
E = gpuArray(B*(B'*B)^(-1)*B');
Sort_gamma = cell(1,num_group);
Lambda = gpuArray.zeros(1,num_group);
Gamma = gpuArray.zeros(1,num_group);
s = gpuArray.zeros(1,num_group);
Result.NUM = 0;
NumExp = 100;         % number of experiments
for LastTag = 0:1:NumExp
    fprintf('\n-------------------- Tag=%d -------------------\n',LastTag)
    [w,a,b] = Differentiable_FPPA(N,H,H_prime,c_H,B_prime,E,paraFPPA);
    Result.SLs = zeros(1,num_group);
    for k = 1:1:num_group
        Result.SLs(k) = nnz(w(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
    end
    Result.Ratios = Result.SLs./L*100;
    Result.SL = nnz(w);
    paraFPPA.Tag = LastTag;
    fname = sprintf('TargetTSLs%d_%d_Tag%d-.mat',paraFPPA.TargetTSLs,paraFPPA.Tag);
    %% Show accuracy
    SOLUTION = B_prime*w;
    bn = nan + zeros(d, 1);                              % bn : nan's to extend f to length N
    f_tilde = y - SOLUTION - [bn; H*(y-SOLUTION); bn];         % f_tilde : low-pass component
    err = g - SOLUTION - f_tilde;
    err = err(d+1:N-d);  % truncate NaNs.
    Result.MSE = mean(err.^2);
    disp('****** Results ******')
    fprintf('lambda_star = %f\n', paraFPPA.lambda)
    fprintf('SLs = [%d, %d]\n', Result.SLs)
    fprintf('Ratios = [%.2f%%,%.2f%%] \n', Result.Ratios)
    fprintf('MSE = %.2e\n', Result.MSE)
    Result.NUM = Result.NUM + 1;
    save(fname,'SOLUTION','w','paraFPPA','Result')
    epsilon = paraFPPA.num_group;
    if sum(abs(Result.SLs-paraFPPA.TargetTSLs))<=epsilon
        break;
    end
    for j = 1:paraFPPA.num_group
        if  Result.SLs(j) < paraFPPA.TargetTSLs(j)
            RHS = B_prime'*a + b;
            Sort_gamma{j} = sort(abs(RHS(group_info(j):group_info(j+1)-1)));
            Lambda(j) = Sort_gamma{j}(L(j)-paraFPPA.TargetTSLs(j)+1);
            Gamma(j) = max(Sort_gamma{j}(Sort_gamma{j}<paraFPPA.lambda(j)));
            paraFPPA.lambda(j) = min(Lambda(j),Gamma(j));
        elseif Result.SLs(j) > paraFPPA.TargetTSLs(j)
            s(j) = 0;
            for i = 1:1:100
                s(j) = s(j) + Result.SLs(j)- paraFPPA.TargetTSLs(j);
                paraFPPA.lambda(j) = min(Sort_gamma{j}(L(j)-paraFPPA.TargetTSLs(j)+1+s(j)),Gamma(j));
                [w,a,b] = Differentiable_FPPA(N,H,H_prime,c_H,B_prime,E,paraFPPA);
                for k = 1:1:num_group
                    Result.SLs(k) = nnz(w(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
                end
                Result.Ratios = Result.SLs./L*100;
                Result.SL = nnz(w);
                Result.NUM = Result.NUM + 1;
                fprintf('lambda_star = %f\n', paraFPPA.lambda)
                fprintf('SLs = [%d, %d]\n', Result.SLs)
                fname = sprintf('TargetTSLs%d_%d_Tag%d-%d_%d.mat',paraFPPA.TargetTSLs,paraFPPA.Tag,j,i);
                save(fname,'SOLUTION','w','paraFPPA','Result','Sort_gamma')
                if Result.SLs(j)<=paraFPPA.TargetTSLs(j) 
                    break;
                end
            end
        end
    end
end
time = toc;
save time time