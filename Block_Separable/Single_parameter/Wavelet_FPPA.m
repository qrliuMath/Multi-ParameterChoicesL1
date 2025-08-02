function u0 = Wavelet_FPPA(c,paraFPPA)

%  minimize 1/2*|| Au - c ||_2^2 + lambda* || u ||_1
%  A is an orthogona wavelet matrix;

lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
alpha = paraFPPA.alpha;

WaveName = paraFPPA.WaveName;
DecLev = paraFPPA.DecLev;

u0 = gpuArray.zeros(length(c),1);
a0 = gpuArray.zeros(length(c),1);

% TargetValue = gpuArray.zeros(1,MaxIter);
[A_top_c,L] = wavedec(gather(c),DecLev,WaveName);  % A'*c
for k = 1:MaxIter
    % u update
    u1 = prox_abs(u0-alpha*a0,alpha*lambda);
    
    % a update
    temp = 1/rho*a0 + (2*u1-u0);  
    a1 = rho*(temp-prox_square(temp,rho,A_top_c));
    
    u0 = u1;
    a0 = a1;
    
%     TargetValue(k)=1/2*sum((waverec(gather(u0),L,WaveName)-c).^2) + lambda*norm(u0,1);  % Targeted Function value
%     if rem(k,100)==0
%         fprintf('Iter: %d, TargetValue: %f; \n ', k, TargetValue(k))
%     end
    
end

disp(' ')
    function y = prox_abs(z,tau)
        %% prox operator for absolute value function $tau*||variable||_1$
        y = sign(z).*max(abs(z)-tau,0);
    end

    function y = prox_square(z,rho,A_top_c)
       %% prox operator for square function $1/rho*1/2*||A*variable-c||_2^2$       
        y = (rho*z+A_top_c)/(rho+1);
    end

end

