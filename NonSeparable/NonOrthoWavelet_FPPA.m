function [u1,TargetValue] = NonOrthoWavelet_FPPA(c,RecMat,paraFPPA)

%    minimize 1/2*|| Au - c ||_2^2 + \sum_i  lambda_i * || u_i ||_1
%    A is wavelet matrix; u_i is the sub-vector of vector u

rho = paraFPPA.rho;
alpha = paraFPPA.alpha;
MaxIter = paraFPPA.MaxIter;

WaveName = paraFPPA.WaveName;
RecLev = paraFPPA.RecLev;

lambda = paraFPPA.lambda;
group_info = paraFPPA.group_info; 

u0 = gpuArray.zeros(length(c),1);
v0 = gpuArray.zeros(length(c),1);

TargetValue = gpuArray.zeros(1,MaxIter);
for k = 1:MaxIter
    % u update
    A_top_v0 = RecMat'*v0;    % A'*v0
    u1 = prox_group_l1(u0-alpha*A_top_v0,alpha*lambda,group_info);
    
    % v update
    temp = 1/rho*v0 + waverec(gather(2*u1-u0),RecLev,WaveName);  % A*(2*u1-u0)
    v1 = rho*(temp-prox_square(temp,rho,c));
    
    u0 = u1;
    v0 = v1;
    
    group_norm = 0;
    for j = 1:1:length(group_info)-1
        group_norm = group_norm + lambda(j)*norm(u0(group_info(j):group_info(j+1)-1),1);
    end
    
    TargetValue(k)=1/2*sum((waverec(gather(u0),RecLev,WaveName)-c).^2) + group_norm;  % Targeted Function value
    if rem(k,100)==0
        fprintf('Iter: %d, TargetValue: %f; \n ', k, TargetValue(k))
    end
    
end
disp(' ')
    function y = prox_group_l1(z,tau,group_info)
       %% prox operator for $tau1*||group1||_1+tau2*||group2||_1+...+tau_d*||group_d||_1$
        y = gpuArray.zeros(length(z),1);
        for i = 1:1:length(group_info)-1
            z_sub = z(group_info(i):group_info(i+1)-1);
            y(group_info(i):group_info(i+1)-1) = (z_sub>tau(i)).*(z_sub-tau(i))+...
                                                 (z_sub<-tau(i)).*(z_sub+tau(i));
        end
    end

    function y = prox_square(z,rho,c)
        %% prox operator for square function $1/rho*1/2*||variable-c||_2^2$
         y = (rho*z+c)/( rho+1);
    end

end

