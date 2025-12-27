function [w1,a1,c1] = Differentiable_FPPA(N,H,H_prime,c,B_prime,E,paraFPPA)

%    minimize 1/2*|| H*B_prime*w - c ||_2^2 + indicator_M(w) + sum_j lambda_j|| I_prime_(j)*w ||_1;

alpha = paraFPPA.alpha;
rho = paraFPPA.rho;
beta =  paraFPPA.beta;
MaxIter = paraFPPA.MaxIter;

lambda = paraFPPA.lambda;
group_info = paraFPPA.group_info;

w0 = gpuArray.zeros(2*N-1,1);
a0 = gpuArray.zeros(N,1);
c0 = gpuArray.zeros(2*N-1,1);

% TargetValue = gpuArray.zeros(1,MaxIter);
for k = 1:MaxIter
    % w update
    w1 = prox_group_l1(w0-alpha*B_prime'*a0-alpha*c0,alpha*lambda,group_info);
    
    % a update
    temp_a = 1/rho*a0 + B_prime*(2*w1-w0);
    a1 = rho*(temp_a-prox_square(temp_a,rho,c,H,H_prime));
   
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
%     TargetValue(k) = 1/2*sum((H*B_prime*w1-c).^2) + group_norm;  % Targeted Function value
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

    function y = prox_square(z,rho,c,H,H_prime)
       %% prox operator for $1/2*1/rho*||Hu-c||_2^2$
        y = H_prime\(rho*z+H'*c);
    end

    function y = prox_indicator(z,E)
        %% prox operator for indicator function
        y = E*z;  
    end
end

