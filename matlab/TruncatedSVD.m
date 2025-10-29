% Example of Truncated SVD reconstruction for Numerical differentiation problem
% For 02624
% Kim Knudsen, DTU Compute, 2024

% Call NumericalDifferentiation to get matrix A, yvec, yvecnoise and xvec
NumericalDifferentiation;

close all;


% SVD analysis
[U,S,V] = svd(A);
figure(3);
plot(log(diag(S)));

%True singular values
strue=1./(((1:N)-1/2)*pi);
hold on;
plot(log(strue));

legend('Computed \mu_j','True \mu_j')
xlabel('j')

% Truncated SVD
% k determines number of terms in SVD sum; should be chosen in accordance
% with noise level
k = 5;
xk = V(:,1:k)*S(1:k,1:k)^(-1)*U(:,1:k)'*yvecnoise;

%Inspect result
figure(4)
plot(s,xvec)
hold on;
plot(s,xk);
legend('x(s)','xk(s)')

