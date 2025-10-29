%Tikhonov regularization for numerical diff
% Solves argmin \|Ax-y\|^2 + \alpha\|D1x-x0\|^2

% Set up problem using numerical differentiation code
% Provides matrix A, xvec and yvec, yvecnoise and N
NumericalDifferentiation;

%p = zeros(N,1);
%p(N/4:(3*N/4-1)) = ones(N/2,1);
%xvec = xvec + p;
%yvecnoise = yvecnoise + A*p;


% Tikhonov
 
 %xhat = A \ yvecnoise;
 alpha = 10.^(-2);
 xalpha =zeros(N,5);

 for j = 1:5
     alpha = 10^(-4+j);
     Aalpha  = [A;sqrt(alpha)*eye(N)];
     yext = [yvecnoise;zeros(N,1)];
     xalpha(:,j) = Aalpha \ yext;
 end;

% Tikhonov with first derivative

D1= -diag(ones(N,1)) + diag(ones(N-1,1),1);

xD1alpha =zeros(N,5);

for j = 1:5
     alpha = 10^(-2+j);
     Aalpha  = [A;sqrt(alpha)*D1];
     yext = [yvecnoise;zeros(N,1)];
     xD1alpha(:,j) = Aalpha \ yext;
 end;


 figure(1)
 plot(s,xvec);
 hold on;
 plot(s,xalpha);
 %plot(s,xhat);
 legend('xtrue', 'alpha = 10^{-3}','alpha = 10^{-2}','alpha = 10^{-1}','alpha = 10^{0}','alpha = 10^{1}')
 title('Tikhonov for num diff')
 xlabel('s')
 ylabel('x(s)')
 hold off

  figure(2)
 plot(s,xvec);
 hold on;
 plot(s,xD1alpha);
 %plot(s,xhat);
 legend('xtrue', 'alpha = 10^{-1}','alpha = 10^{0}','alpha = 10^{1}','alpha = 10^{2}','alpha = 10^{3}')
 title('Tikhonov with one derivative for num diff')
 xlabel('s')
 ylabel('x(s)')
 hold off

 figure(3)
plot(s,xvec);
 hold on;
  plot(s,xalpha(:,2));
  plot(s,xD1alpha(:,3));
   hold off
   xlabel('s')
 ylabel('x(s)')
  legend('xtrue', 'Ordinary Tikhonov', 'D1 Tikhonov' )
 title('Ordinary Tikhonov and D1 Tikhonov for num diff')
 
