%% Generate matrix B
I_N  = speye(N);
D  = -1*speye(N) + sparse((1:N),[(2:N) 1],1);  % first order difference matrix
D(end,:) = [];
B = [I_N;D];

%% Compute singular value decomposition of B
B = full(B);
rank_B = rank(B);
[U,S,V] = svd(B);

%% Compute B_prime in the paper 
s = sum(S,1); 
S_prime = diag(s.^(-1));
U_r = U(:,1:rank_B);
U_prime = U_r';
B_prime = V*S_prime*U_prime;