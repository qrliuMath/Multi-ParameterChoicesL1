%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%
%    Multi-parameter choices by Algorithm 2 for signal denoising (nonseparable)
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
Lev = 18;
N = 2^Lev;
x = linspace(0,1,N)';
signal = sqrt(x.*(1-x)).*sin(2*pi*1.05./(x + 0.05));   % original signal
 
%% Add noise
rng(1)
SNR = 20;
signal_noi = awgn(signal,SNR,'measured');    % add Gaussian white noise to signal

dwtmode('per')       % peroidic condition to extend signal 
paraFPPA.WaveName = 'bior2.2';
paraFPPA.DecLev = 9;     % decomposion level  (Lev-DecLev is the most coarst level)

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
% RecMat = GenerateRecMat(paraFPPA);
load RecMat;

%% Choose parameter in FPPA algorithm
paraFPPA.TargetTSLs = [512,25,10,15,14,17,37,62,146,262];     %  targeted sparsity levels
paraFPPA.TargetSL = sum(paraFPPA.TargetTSLs);
paraFPPA.MaxIter = 1000;
paraFPPA.rho = 0.02;
paraFPPA.alpha = 1/paraFPPA.rho*0.99;   
paraFPPA.beta = 0.0001;
paraFPPA.FGO_norm = sqrt(paraFPPA.alpha*(paraFPPA.rho+paraFPPA.beta)); %  convergence condition FGO_norm <1
A_rho = paraFPPA.rho*speye(size(RecMat))+ RecMat'*RecMat;
A_rho = gpuArray(A_rho);

%% Choose initial lambda
tic;
Initial_lambda = gpuArray.zeros(1,num_group);
RecMat = gpuArray(RecMat);
signal_noi = gpuArray(signal_noi);
Lambda_0 = abs(RecMat'*(signal_noi));
for k = 1:num_group
    Initial_lambda(k) = max(Lambda_0(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
end      
paraFPPA.lambda = Initial_lambda;

%% Iterative scheme choosing multiple regularization parameters
Sort_Gamma = cell(1,num_group);
Lambda = gpuArray.zeros(1,num_group);
a = gpuArray.zeros(1,num_group);
s = gpuArray.zeros(1,num_group);
Result.NUM = 0;
NumExp = 100;         % number of experiments
for LastTag = 0:1:NumExp
     fprintf('\n-------------------- Tag=%d -------------------\n',LastTag)
     SOLUTION = NonOrthoWavelet_FPPA(signal_noi,RecMat,paraFPPA,A_rho);
     Result.SLs = zeros(1,num_group);
     for k = 1:1:num_group
          Result.SLs(k) = nnz(SOLUTION(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
     end
     Result.SL = nnz(SOLUTION);
     Result.Ratio =  Result.SL/N*100;
     paraFPPA.Tag = LastTag;
     fname = sprintf('TargetSL%d_Tag%d-.mat',paraFPPA.TargetSL,paraFPPA.Tag);
    
     %% Show accuracy
     RecSignal = waverec(gather(SOLUTION),L,paraFPPA.WaveName); 
     Result.MSE = sum((RecSignal - signal).^2)/N;
     disp('****** Results ******')
     fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
     fprintf('SLs = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n', Result.SLs)
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
                 SOLUTION = NonOrthoWavelet_FPPA(signal_noi,RecMat,paraFPPA,A_rho);
                 for k = 1:1:num_group
                     Result.SLs(k) = nnz(SOLUTION(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
                 end
                 Result.SL = nnz(SOLUTION);
                 Result.Ratio =  Result.SL/N*100;
                 Result.NUM = Result.NUM + 1;
                 fprintf('lambda_star = %.2e\n', paraFPPA.lambda)
                 fprintf('SLs = [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n', Result.SLs)
                 fname = sprintf('TargetSL%d_Tag%d-%d_%d.mat',paraFPPA.TargetSL,paraFPPA.Tag,j,i);
                 save(fname,'SOLUTION','paraFPPA','Result','Sort_Gamma')
                 if Result.SLs(j)<=paraFPPA.TargetTSLs(j)
                     break;
                 end
             end
         end
     end
end
time = toc;
save time time

%% Figure
figure
plot(x,signal,'Linewidth',1);
axis([0 1 -0.6 0.6])
yticks(-0.6:0.2:0.6)
grid on;
ax = gca;
ax.FontSize = 14; 

figure
plot(x,signal_noi,'Linewidth',1);
axis([0 1 -0.6 0.6])
yticks(-0.6:0.2:0.6)
grid on;
ax = gca;
ax.FontSize = 14; 

figure
plot(x,signal,'Linewidth',1);
axis([0 1 -0.6 0.8])
yticks(-0.6:0.2:0.8)
hold on
plot(x,RecSignal,'r','Linewidth',1);
leg = legend('Original signal','Denoised signal','Location', 'northwest',...
     'Position', [0.165 0.76 0.34 0.15]);  
ax = gca;
ax.FontSize = 14; 
leg.FontSize = 12;
grid on;

% Define magnification area
zoom_x = [0.76, 0.79];     
zoom_y = [0.38, 0.44];     
rectangle('Position',[zoom_x(1) zoom_y(1) zoom_x(2)-zoom_x(1) zoom_y(2)-zoom_y(1)],...
          'EdgeColor',[0 0 0],'LineWidth',2,'LineStyle',':')
axes('Position',[0.54 0.76 0.34 0.15])
box on
hold on
plot(x,signal,'Linewidth',1)
plot(x,RecSignal,'r','Linewidth',1)
xlim([0.76, 0.79])    
ylim([0.38, 0.44])  
ax2 = gca;
ax2.FontSize = 10;
ax2.XTick = []; 
ax2.YTick = []; 
annotation('arrow', [0.725 0.70], [0.72 0.80],'Color', [0 0 0],...
           'LineWidth', 2, ...
           'HeadStyle', 'vback3', ...
           'HeadLength', 8, ...
           'HeadWidth', 8, ...
           'LineStyle', ':');
