function [u0,TargetValue] = Wavelet_FPPA(c,paraFPPA)

%  minimize 1/2*|| Au - c ||_2^2 + lambda* || u ||_1
%  A is wavelet matrix 

lambda = paraFPPA.lambda;
MaxIter = paraFPPA.MaxIter;
rho = paraFPPA.rho;
alpha = paraFPPA.alpha;

WaveName = paraFPPA.WaveName;
DecLev = paraFPPA.DecLev;

u0 = zeros(length(c),1);
v0 = zeros(length(c),1);

TargetValue = zeros(1,MaxIter);
for k = 1:MaxIter
    % u update
    [A_top_v0,L] = wavedec(v0,DecLev,WaveName);    % A'*v0
    u1 = prox_abs(u0-alpha*A_top_v0,alpha*lambda);
    % v update
    temp = 1/rho*v0 + waverec(2*u1-u0,L,WaveName);  % A*(2*u1-u0)
    v1 = rho*(temp-prox_square(temp,rho,c));
    
    u0 = u1;
    v0 = v1;
    
    TargetValue(k)=1/2*sum((waverec(u0,L,WaveName)-c).^2) + lambda*norm(u0,1);  % Targeted Function value
    if rem(k,100)==0
        fprintf('Iter: %d, TargetValue: %f; \n ', k, TargetValue(k))
    end
    
end

disp(' ')
    function y = prox_abs(z,tau)
        %% prox operator for absolute value function $tau*||variable||_1$
        y = sign(z).*max(abs(z)-tau,0);
    end

    function y = prox_square(z,rho,c)
       %% prox operator for square function $1/rho*1/2*||variable-c||_2^2$
        y = (rho*z+c)/(rho+1);
    end

end

