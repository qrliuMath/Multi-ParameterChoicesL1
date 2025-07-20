function sigma_max = power_iteration(A)

% Power iteration method for computing the largest singular value

tol = 1e-6;
max_iter = 5000;

n = size(A,2);
v = randn(n, 1);
v = v / norm(v);

for k = 1:max_iter
    w = A * v;
    v_new = A' * w;
    v_new = v_new / norm(v_new);
    
    % Compute the Rayleigh Quotient: v'*A'*A*v
    lambda = v_new'*(A')*A*v_new;
    
    % Check for convergence
    if norm(v_new - v) < tol
        break;
    end
    
    v = v_new;
end
sigma_max = sqrt(lambda);  
fprintf('Iteration number: %d, The largest singular value: %.6f\n', k, sigma_max);
end