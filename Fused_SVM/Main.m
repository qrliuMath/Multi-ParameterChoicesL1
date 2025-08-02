%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%
%    Multi-parameter choices by Algorithm 2 for fused SVM model  
%
%    Model: Classification with hinge loss  (12214 training dataset) 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clc, clear, close all
load Data
disp('Classification-Hinge Loss')
disp('----------------------------------------------')
format short

%% Goal: binary classification of handwritten digit numbers: $num1$ and $num2$
num1 = 7;
num2 = 9;
fprintf('Handwritten Digit Recognization: %d and %d\n\n', num1, num2)

ImgsTrain = ImgsTrain(:,1:1:end);
LabelsTrain = LabelsTrain(1:end);

NumTrain = length(LabelsTrain);
NumTest = length(LabelsTest);


%% Group
num_group = 2;
N = size(ImgsTrain,1);
group_info = [1;N+1;2*N];    % starting index of each group£»
paraFPPA.group_info = group_info;
paraFPPA.num_group = num_group;
L = [N,N-1];

%-------------------------------------------------------
Generate_B_and_B_prime
%-------------------------------------------------------

%% Choose parameter in FPPA algorithm
paraFPPA.TargetTSLs = [360,400];     %  targeted sparsity levels
paraFPPA.TargetSSL = sum(paraFPPA.TargetTSLs);
paraFPPA.MaxIter = 500000;
paraFPPA.alpha = 0.001;    
paraFPPA.rho = 0.002;
paraFPPA.beta = 10;
Y = diag(LabelsTrain);
X = ImgsTrain';
B_tilde = gpuArray(Y*X*B_prime);
FGO = [sqrt(paraFPPA.alpha*paraFPPA.rho)*B_tilde;sqrt(paraFPPA.alpha*paraFPPA.beta)*eye(2*N-1)];
paraFPPA.FGO_norm = norm(gather(FGO),2);   %  convergence condition FGO_norm < 1

%% Choose initial lambda
tic;
Initial_lambda = [1.0,0.2];
paraFPPA.lambda = gpuArray(Initial_lambda); 

%% Iterative scheme choosing multiple regularization parameters 
alpha_B_tilde_T = paraFPPA.alpha*B_tilde';
E = gpuArray(B*(B'*B)^(-1)*B');
Sort_gamma = cell(1,num_group);
Lambda = gpuArray.zeros(1,num_group);
Gamma = gpuArray.zeros(1,num_group);
s = gpuArray.zeros(1,num_group);
Result.NUM = 0;
NumExp = 100;         % number of experiments
for LastTag = 0:1:NumExp
    fprintf('\n-------------------- Tag=%d -------------------\n',LastTag)
    [w,a,c] = HingeLoss_FPPA(N,NumTrain,B_tilde,alpha_B_tilde_T,E,paraFPPA);
    Result.SLs = zeros(1,num_group);
    for k = 1:1:num_group
        Result.SLs(k) = nnz(w(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
    end
    Result.SL = nnz(w);
    Result.Ratios =  Result.SLs./L*100;
    paraFPPA.Tag = LastTag;
    fname = sprintf('TargetTSLs%d_%d_Tag%d-.mat',paraFPPA.TargetTSLs,paraFPPA.Tag);
    %% Show accuracy
    SOLUTION = B_prime*w;   
    Result.TrA = Accuracy(SOLUTION,ImgsTrain,LabelsTrain);    % Train data accuracy
    Result.TeA = Accuracy(SOLUTION,ImgsTest,LabelsTest);      % Test data accuracy
    disp('****** Results ******')
    fprintf('lambda_star = %f\n', paraFPPA.lambda)
    fprintf('SLs = [%d, %d]\n', Result.SLs)
    fprintf('Ratios = [%.2f%%,%.2f%%] \n', Result.Ratios)
    fprintf('TrA = %f\n', Result.TrA)
    fprintf('TeA = %f\n', Result.TeA)
    Result.NUM = Result.NUM + 1;
    save(fname,'SOLUTION','w','paraFPPA','Result','Sort_gamma')
    epsilon = paraFPPA.num_group;
    if sum(abs(Result.SLs-paraFPPA.TargetTSLs))<=epsilon
        break;
    end
    for j = 1:paraFPPA.num_group
        if  Result.SLs(j) < paraFPPA.TargetTSLs(j)
            RHS = B_tilde'*a + c;
            Sort_gamma{j} = sort(abs(RHS(group_info(j):group_info(j+1)-1)));
            Lambda(j) = Sort_gamma{j}(L(j)-paraFPPA.TargetTSLs(j)+1);
            Gamma(j) = max(Sort_gamma{j}(Sort_gamma{j}<paraFPPA.lambda(j)));
            paraFPPA.lambda(j) = min(Lambda(j),Gamma(j));
        elseif Result.SLs(j) > paraFPPA.TargetTSLs(j)
            s(j) = 0;
            for i = 1:100
                s(j) = s(j) + Result.SLs(j)- paraFPPA.TargetTSLs(j);
                paraFPPA.lambda(j) = min(Sort_gamma{j}(L(j)-paraFPPA.TargetTSLs(j)+1+s(j)),Gamma(j));
                [w,a,c] = HingeLoss_FPPA(N,NumTrain,B_tilde,alpha_B_tilde_T,E,paraFPPA);
                for k = 1:1:num_group
                    Result.SLs(k) = nnz(w(paraFPPA.group_info(k):paraFPPA.group_info(k+1)-1));
                end
                Result.SL = nnz(w);
                Result.Ratios =  Result.SLs./L*100;
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



