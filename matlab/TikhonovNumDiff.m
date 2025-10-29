%Tikhonov regularization for numerical diff
% Solves argmin \|Ax-y\|^2 + \alpha\|x\|^2

% Set up problem using numerical differentiation code
% Provides matrix A, xvec and yvec and N
NumericalDifferentiation;



%SVD 
[U,S,V] = svd(A);

%Filter factors
alpha = 10^(-4);
ffTikhonov = diag(S).^2./ (diag(S).^2 + alpha);
loglog(diag(S),ffTikhonov)
title('Filter factors')

 
 %xhat = A \ yvecnoise;
 alpha = 10.^(-2);
 xalpha =zeros(N,5);

 for j = 1:5
     alpha = 10^(-4+j);
     Aalpha  = [A;sqrt(alpha)*eye(N)];
     yext = [yvecnoise;zeros(N,1)];
     xalpha(:,j) = Aalpha \ yext;
 end;

 
 figure
 plot(s,xvec);
 hold on;
 plot(s,xalpha);
 %plot(s,xhat);
 legend('xtrue', 'alpha = 10^{-4}','alpha = 10^{-3}','alpha = 10^{-2}','alpha = 10^{0}','alpha = 10^{1}')
 title('Tikhonov for num diff')
 xlabel('s')
 ylabel('x(s)')
