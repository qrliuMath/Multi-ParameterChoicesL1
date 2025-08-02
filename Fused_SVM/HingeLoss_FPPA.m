function [w1,a1,c1] = HingeLoss_FPPA(N,NumTrain,B_tilde,alpha_B_tilde_T,E,paraFPPA)

%    minimize phi(B_tilde*w) + indicator_M(w) + sum_j lambda_j|| I_prime_(j)*w ||_1;

alpha = paraFPPA.alpha;
rho = paraFPPA.rho;
beta =  paraFPPA.beta;
MaxIter = paraFPPA.MaxIter;

lambda = paraFPPA.lambda;
group_info = paraFPPA.group_info;

w0 = gpuArray.zeros(2*N-1,1);
a0 = gpuArray.zeros(NumTrain,1);
c0 = gpuArray.zeros(2*N-1,1);

% TargetValue = gpuArray.zeros(1,MaxIter);
for k = 1:MaxIter
    % w update
    w1 = prox_group_l1(w0-alpha_B_tilde_T*a0-alpha*c0,alpha*lambda,group_info);
    
    % a update
    temp_a = 1/rho*a0 + B_tilde*(2*w1-w0);
    a1 = rho*(temp_a-prox_phi(temp_a,rho));
   
    % c update
    temp_c = 1/beta*c0 + 2*w1-w0;
    c1 = beta*(temp_c-prox_indicator(temp_c,E));
    
    w0 = w1;
    a0 = a1;
    c0 = c1;
    
%     group_norm = 0;
%     for j = 1:1:length(group_info)-1
%         group_norm = group_norm + lambda(j)*norm(w1(group_info(j):group_info(j+1)-1),1);
%     end
%     TargetValue(k) = sum(max(0,1-B_tilde*w1)) + group_norm;  % Targeted Function value
%     if rem(k,100)==0
%         fprintf('Iter: %d, TargetValue: %f; \n ', k, TargetValue(k))
%     end
end

    function y = prox_group_l1(z,tau,group_info)
        %% prox operator for $tau1*||group1||_1+tau2*||group2||_1$
        y = gpuArray.zeros(length(z),1);
        for i = 1:1:length(group_info)-1
            z_sub = z(group_info(i):group_info(i+1)-1);
            y(group_info(i):group_info(i+1)-1) = (z_sub>tau(i)).*(z_sub-tau(i))+...
                (z_sub<-tau(i)).*(z_sub+tau(i));
        end
    end

    function y = prox_phi(z,rho)
        %% prox operator for $1/rho*phi$
         tau = 1/rho;
         y = (z+tau).*(z<1-tau) + z.*(z>1) + (z>=(1-tau)).*(z<=1);
    end

    function y = prox_indicator(z,E)
        %% prox operator for indicator function
        y = E*z;
    end
end

